"""A1: answer model with a real calculator tool (function calling).

The CoT-prompt probe failed (30% vs 32%) because the bottleneck is not prompt
wording but the model's ability to actually compute over multi-turn evidence.
This arm gives it a REAL calculator tool through a function-calling loop:
the model may call calculator(expression) any number of times before
answering. Same 50 questions (seed 7), same judge (DeepSeek, no tools).

Run:
  DEEPSEEK_API_KEY=sk-... python bench_a1_calc_tool.py [--sample 50]
"""
import argparse
import hashlib
import json
import os
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

from flymemory.v3 import load  # noqa: E402
from bench_memory_judgment import ds_client, parse_decision  # noqa: E402
from bench_lme_e2e import ANSWER_SYSTEM, JUDGE_SYSTEM  # noqa: E402

MODEL = "deepseek-chat"

CALC_TOOLS = [{
    "type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression and return the "
                       "numeric result. Use this for any duration, total, or "
                       "difference computation instead of mental math.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string",
                           "description": "Arithmetic expression, e.g. "
                                          "'(2023-05-21 -> 2023-06-04) in days' "
                                          "is NOT valid -- use plain numbers: "
                                          "'14 - 2'"}},
            "required": ["expression"]}}},
]

CALC_SYSTEM = ANSWER_SYSTEM + """

You also have a calculator tool. Whenever the answer requires ANY arithmetic
(durations, totals, differences, counting), you MUST:
1. Extract the relevant numbers (with their dates) from the remembered facts.
2. Call the calculator with a plain arithmetic expression.
3. Use the returned result in your final answer.
Never do mental arithmetic for non-trivial computation. When done, answer in
one or two short sentences."""


def safe_eval(expr):
    import ast
    try:
        node = ast.parse(expr, mode="eval")
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
                   ast.Mod, ast.Pow)
        if not all(isinstance(n, allowed) for n in ast.walk(node)):
            return "error: only plain arithmetic is allowed"
        return str(eval(compile(node, "<expr>", "eval")))
    except Exception as e:
        return f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    mem = load(os.path.join(here, "longmemeval_bench.pkl"), enable_hopfield=False)
    with open(os.path.join(here, "data_longmemeval", "longmemeval_oracle.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    import random
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(data)), min(args.sample, len(data)))

    client = ds_client()
    prompt_hash = hashlib.sha1(CALC_SYSTEM.encode()).hexdigest()[:10]
    print(f"questions: {len(idx)} | A1 calculator-tool | prompt {prompt_hash}\n",
          flush=True)

    tally = {"correct": 0, "partial": 0, "wrong": 0, "judge_error": 0}
    traces = []
    t0 = time.time()
    n_calls = 0
    for qi, i in enumerate(idx):
        q = data[i]
        hits = mem.recall(q["question"], top_k=args.topk)
        lines = [f"- {m.text}" for m, _s, _e in hits]
        messages = [
            {"role": "system", "content": CALC_SYSTEM},
            {"role": "user", "content":
                f"REMEMBERED FACTS:\n" + "\n".join(lines)
                + f"\n\nQUESTION: {q['question']}\n\nAnswer now."},
        ]
        answer = ""
        calls_this_q = 0
        for _round in range(6):
            try:
                r = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=CALC_TOOLS,
                    temperature=0, max_tokens=400)
                msg = r.choices[0].message
            except Exception as e:
                answer = f"[api error: {e}]"
                break
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                break
            messages.append({"role": "assistant",
                             "content": msg.content or "",
                             "tool_calls": [tc.model_dump()
                                            for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    a = json.loads(tc.function.arguments or "{}")
                    result = safe_eval(str(a.get("expression", "")))
                except json.JSONDecodeError:
                    result = "error: bad JSON"
                calls_this_q += 1
                n_calls += 1
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result})
        if not answer:
            answer = "[no final answer]"
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
        traces.append({"idx": i, "question": q["question"], "answer": answer,
                       "gold": q.get("answer", ""), "verdict": verdict,
                       "n_tool_calls": calls_this_q})
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(idx)} ({time.time()-t0:.0f}s) {tally} "
                  f"tool_calls={n_calls}", flush=True)

    n = len(idx)
    n_scored = sum(v for k, v in tally.items() if k != "judge_error")
    print(f"\n=== A1: calculator tool (n={n}, top-{args.topk}) ===")
    for k in ("correct", "partial", "wrong", "judge_error"):
        print(f"  {k:12s} {tally.get(k,0):3d}")
    if n_scored:
        acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
        print(f"  score = {acc:.3f}  (strict = {tally.get('correct',0)/n_scored:.3f})")
        print("  baselines (same 50 questions): CoT prompt 0.300 strict / "
              "0.320 weighted; no-calc baseline 0.320 strict / 0.350 weighted")
        print(f"  total calculator calls: {n_calls}")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports",
                       f"a1_calc_tool_{args.sample}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "topk": args.topk, "model": MODEL,
                   "prompt_hash": prompt_hash, "tally": tally,
                   "total_tool_calls": n_calls, "traces": traces},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
