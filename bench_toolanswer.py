"""P-2026-09-24-TOOL1: agentic answer -- the answer LLM operates a
search_memory tool (multi-round retrieval) before answering.

Targets the aggregation-statistics failure mode: answers that require
combining evidence scattered across many turns cannot be produced from a
single top-5 recall, no matter the ingest form (4 probes confirmed).
Registered prediction: research/RESEARCH.md (TOOL1).

Library: three-layer (turns + consolidated + aggregate), same as AGG1.
Run:
  DEEPSEEK_API_KEY=sk-... python bench_toolanswer.py [--sample 50]
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
try:
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 4))
except ImportError:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flymemory"))

from flymemory.v3 import SmartMemory, load  # noqa: E402
from bench_longmemeval_s import parse_lme_date  # noqa: E402
from bench_memory_judgment import ds_client, parse_decision  # noqa: E402
from bench_lme_e2e import JUDGE_SYSTEM  # noqa: E402

MODEL = "deepseek-chat"

ANSWER_SYSTEM = """You are a personal assistant answering questions from your
owner's long-term memory store. You have a search_memory tool: call it with
different phrasings as many times as needed to gather the evidence relevant
to the question. Entries carry their age. For time-bounded questions
("what did I do in March", "which phone last year"), pass a time_range to
scope the search instead of relying on one query. When you have enough
evidence, write the final answer as plain text (no tool call) in one or two
short sentences, using ONLY gathered evidence. If the evidence does not
contain the answer, say you don't know."""

CALC_TOOL = {
    "type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression (numbers, + - * / "
                       "and parentheses only). Use it to combine quantities "
                       "from memory instead of guessing.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}},
            "required": ["expression"]}}}

SEARCH_TOOL = {
    "type": "function", "function": {
        "name": "search_memory",
        "description": "Search the long-term memory store. Returns the top-5 "
                       "matching entries with ages. Entries carry dates; use "
                       "time_range to scope searches when the question is "
                       "time-bounded (e.g. 'which OS in March 2025?').",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "keyword query, rephrase as needed"},
            "time_range": {"type": "string",
                           "description": "optional 'YYYY-MM..YYYY-MM' filter "
                                          "on entry dates"}},
            "required": ["query"]}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--calculator", action="store_true",
                    help="TOOL3 probe: also expose a safe calculator tool "
                         "(L5 aggregation-law lever: cross-turn arithmetic)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    mem = load(os.path.join(here, "longmemeval_bench.pkl"), enable_hopfield=False)
    cons = json.load(open(os.path.join(here, "reports",
                                       "consolidated_entries.json"),
                          encoding="utf-8"))
    aggs = json.load(open(os.path.join(here, "reports",
                                       "aggregate_entries.json"),
                          encoding="utf-8"))
    for c in cons + aggs:
        tag = "consolidated" if c in cons else "aggregate"
        for text in c["entries"]:
            mem.remember_text(text, tags=["lme", c["sid"], tag],
                              source="model", timestamp=c["ts"], force_new=True)
    print(f"three-layer library: {mem.size} entries", flush=True)

    with open(os.path.join(here, "data_longmemeval", "longmemeval_oracle.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(data)), min(args.sample, len(data)))

    client = ds_client()
    prompt_hash = hashlib.sha1(ANSWER_SYSTEM.encode()).hexdigest()[:10]
    tally = {"correct": 0, "partial": 0, "wrong": 0, "judge_error": 0}
    per_type = {}
    traces = []
    t0 = time.time()

    def safe_calc(expression):
        expr = str(expression).strip()
        if not expr or any(ch not in "0123456789+-*/(). " for ch in expr):
            return "rejected: only numbers and + - * / ( ) are allowed"
        try:
            val = eval(expr, {"__builtins__": {}}, {})
            return f"= {val}"
        except Exception as e:
            return f"error: {e}"

    def search_mem(query, k, time_range=None):
        hits = mem.recall(query, top_k=k * 4 if time_range else k)
        if time_range and ".." in str(time_range):
            start_s, end_s = str(time_range).split("..", 1)
            try:
                start = time.mktime(time.strptime(start_s + "-01", "%Y-%m-%d"))
                end = time.mktime(time.strptime(end_s + "-01",
                                                "%Y-%m-%d")) + 86400 * 31
                hits = [h for h in hits
                        if (h[0].timestamp or 0) >= start
                        and (h[0].timestamp or 0) < end]
            except ValueError:
                pass
        hits = hits[:k]
        out = []
        for m, s, _e in hits:
            age = max(0, int((time.time() - m.last_accessed) / 86400))
            out.append(f"#-{m.text[:200]} ({age}d ago)")
        return "\n".join(out) if out else "(no matches in that time range)"

    for qi, i in enumerate(idx):
        q = data[i]
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"QUESTION: {q['question']}\n\n"
                                        "Search memory as needed, then answer."},
        ]
        tool_log = []
        for _round in range(args.max_rounds):
            try:
                tools = [SEARCH_TOOL] + ([CALC_TOOL] if args.calculator else [])
                r = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools,
                    tool_choice="auto", temperature=0, max_tokens=400)
                msg = r.choices[0].message
            except Exception as e:
                tool_log.append(f"[api error: {str(e)[:80]}]")
                break
            if not msg.tool_calls:
                tool_log.append(f"final: {msg.content}")
                break
            messages.append({"role": "assistant",
                             "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                if tc.function.name == "search_memory":
                    try:
                        qargs = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        qargs = {}
                    result = search_mem(str(qargs.get("query", "")), args.topk,
                                        qargs.get("time_range"))
                    tool_log.append(f"search({str(qargs.get('query',''))[:40]}"
                                    f" range={qargs.get('time_range')})")
                elif tc.function.name == "calculator":
                    result = safe_calc(tc.function.arguments and
                                       json.loads(tc.function.arguments or "{}").get("expression", ""))
                    tool_log.append(f"calc({str(tc.function.arguments)[:40]})")
                else:
                    result = "unknown tool"
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result})
        else:
            tool_log.append("[max rounds]")

        answer = "(no final answer)"
        for entry in reversed(tool_log):
            if entry.startswith("final: "):
                answer = entry[7:]
                break

        juser = (f"QUESTION: {q['question']}\n\nGROUND TRUTH: "
                 f"{q.get('answer', '')}\n\nASSISTANT ANSWER: {answer}\n\nGrade now.")
        verdict = "judge_error"
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": juser}],
                temperature=0, max_tokens=200)
            d, err = parse_decision(r.choices[0].message.content or "")
            if not err and d.get("score"):
                verdict = d["score"]
        except Exception:
            pass
        tally[verdict] = tally.get(verdict, 0) + 1
        qt = q.get("question_type", "?")
        per_type.setdefault(qt, {})
        per_type[qt][verdict] = per_type[qt].get(verdict, 0) + 1
        n_searches = sum(1 for x in tool_log if x.startswith("search("))
        traces.append({"idx": i, "question": q["question"], "answer": answer,
                       "gold": q.get("answer", ""), "verdict": verdict,
                       "type": qt, "n_searches": n_searches, "tool_log": tool_log})
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(idx)} ({time.time()-t0:.0f}s) {tally}", flush=True)

    n = len(idx)
    n_scored = sum(v for k, v in tally.items() if k != "judge_error")
    avg_s = sum(t.get("n_searches", 0) for t in traces) / max(n, 1)
    print(f"\n=== TOOL1 agentic answer (n={n}, three-layer, model {MODEL}, "
          f"prompt {prompt_hash}) ===")
    for k in ("correct", "partial", "wrong", "judge_error"):
        print(f"  {k:12s} {tally.get(k,0):3d}")
    if n_scored:
        acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
        print(f"  score = {acc:.3f}  (strict = {tally.get('correct',0)/n_scored:.3f})")
        print(f"  avg searches/question: {avg_s:.1f}")
        print("  registered baseline: single-round three-layer 0.400 strict "
              "(0.380 was two-layer)")
    print("\n=== per question_type ===")
    for qt, cc in sorted(per_type.items()):
        tot = sum(cc.values())
        print(f"  {qt:30s} n={tot:3d}  {dict(cc)}")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports",
                       f"toolanswer_{args.sample}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "topk": args.topk, "model": MODEL,
                   "prompt_hash": prompt_hash, "tally": tally,
                   "per_type": per_type, "traces": traces},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
