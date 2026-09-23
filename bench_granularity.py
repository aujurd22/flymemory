"""Granularity A/B: turn-granularity vs session-consolidated memory.

bench_lme_e2e.py showed the answer-conversion bottleneck: multi-turn
aggregation questions cannot be answered from turn-granularity retrieval.
This bench tests the proposed fix -- ingest each session as 1-3
DeepSeek-consolidated higher-order entries instead of raw turns -- on the
same 50 sampled LongMemEval-oracle questions (seed 7, same protocol).

Arm A (baseline): longmemeval_bench.pkl (9,729 turn entries)   -- already run
Arm B (this run): every session consolidated into up to 3 durable-fact
                   entries (force_new, session date, session tag)

Both arms: production recall top-5 -> deepseek answer -> judge vs gold.

Run:
  DEEPSEEK_API_KEY=sk-... python bench_granularity.py [--sessions 940]
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

from flymemory.v3 import SmartMemory  # noqa: E402
from bench_memory_judgment import ds_client, parse_decision  # noqa: E402
from bench_lme_e2e import ANSWER_SYSTEM, JUDGE_SYSTEM  # noqa: E402

MODEL = "deepseek-chat"
CONSOLIDATE_SYSTEM = """You distill a conversation session into durable long-term
memory entries. Extract the concrete FACTS a personal assistant would need
later: preferences, numbers, names, dates, plans, outcomes. Write 1-3 short
self-contained entries, each stating facts plainly. Do NOT invent anything
not present in the conversation. Reply with ONLY a JSON array of strings."""

ANSWER_SYSTEM = ANSWER_SYSTEM


def consolidate_session(client, turns):
    convo = "\n".join(turns)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": CONSOLIDATE_SYSTEM},
                  {"role": "user", "content": convo[:6000]}],
        temperature=0, max_tokens=500)
    d, err = parse_decision(r.choices[0].message.content or "")
    if err or not isinstance(d, list):
        return None
    return [str(x) for x in d if str(x).strip()][:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-sessions", type=int, default=10_000)
    ap.add_argument("--mode", choices=["replace", "overlay"], default="replace")
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "reports",
        "consolidated_entries.json"))
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    client = ds_client()

    # ---- load sessions from the source data (same parse as the S bench) ----
    from bench_longmemeval_s import parse_lme_date  # noqa: E402
    with open(os.path.join(here, "data_longmemeval", "longmemeval_oracle.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    sessions = {}
    for q in data:
        for sid, ds, sess in zip(q["haystack_session_ids"],
                                 q["haystack_dates"], q["haystack_sessions"]):
            if sid not in sessions:
                ts = parse_lme_date(ds)
                turns = [f"{t.get('role', 'user')}: {t.get('content', '')}"
                         for t in sess]
                sessions[sid] = (ts, turns)
    order = sorted(sessions.items(), key=lambda kv: kv[1][0] or 0)
    order = order[:args.max_sessions]
    print(f"sessions to consolidate: {len(order)} (mode {args.mode})", flush=True)

    # ---- consolidated entries (cached so overlay reruns are cheap) ----
    if os.path.exists(args.cache):
        with open(args.cache, encoding="utf-8") as f:
            cons = json.load(f)
        print(f"consolidated entries loaded from cache: {len(cons)}", flush=True)
    else:
        cons = []
        t0 = time.time()
        for si, (sid, (ts, turns)) in enumerate(order):
            entries = consolidate_session(client, turns)
            if not entries:
                continue
            cons.append({"sid": sid, "ts": ts, "entries": entries})
            if (si + 1) % 100 == 0:
                print(f"  consolidated {si+1}/{len(order)} sessions "
                      f"({time.time()-t0:.0f}s)", flush=True)
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(cons, f, ensure_ascii=False)
        print(f"consolidation done: {len(cons)} sessions cached "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- build the store per mode ----
    mem = SmartMemory(n_bits=4096)
    if args.mode == "replace":
        for c in cons:
            for text in c["entries"]:
                mem.remember_text(text, tags=["lme", c["sid"]], source="model",
                                  timestamp=c["ts"], force_new=True)
    else:  # overlay: turn store PLUS consolidated entries
        from flymemory.v3 import load
        mem = load(os.path.join(here, "longmemeval_bench.pkl"),
                   enable_hopfield=False)
        for c in cons:
            for text in c["entries"]:
                mem.remember_text(text, tags=["lme", c["sid"], "consolidated"],
                                  source="model", timestamp=c["ts"],
                                  force_new=True)
    print(f"library: {mem.size} entries ({args.mode})", flush=True)

    # ---- answer the same 50 questions ----
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(data)), min(args.sample, len(data)))
    ANSWER_SYSTEM = ("You are a personal assistant answering questions from "
                     "your long-term memory. Use ONLY the remembered entries "
                     "below. If they do not contain the answer, say you don't "
                     "know. Answer in one or two short sentences.")
    JUDGE_SYSTEM = ("You are grading an assistant's answer against the ground "
                    "truth. Reply with ONLY:\n"
                    '{"score": "correct|partial|wrong", "note": "..."}\n'
                    "- correct: same key fact(s) as the ground truth\n"
                    "- partial: some right, some missing\n"
                    "- wrong: contradicts or off topic")
    tally = {"correct": 0, "partial": 0, "wrong": 0, "judge_error": 0}
    traces = []
    t0 = time.time()
    for qi, i in enumerate(idx):
        q = data[i]
        hits = mem.recall(q["question"], top_k=args.topk)
        lines = [f"- {m.text}" for m, _s, _e in hits]
        user = ("REMEMBERED FACTS:\n" + "\n".join(lines)
                + f"\n\nQUESTION: {q['question']}\n\nAnswer now.")
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": ANSWER_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0, max_tokens=300)
            answer = (r.choices[0].message.content or "").strip()
        except Exception as e:
            answer = f"[api error: {e}]"
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
                       "gold": q.get("answer", ""), "verdict": verdict})
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(idx)} ({time.time()-t0:.0f}s) {tally}", flush=True)

    n = len(idx)
    n_scored = sum(v for k, v in tally.items() if k != "judge_error")
    print(f"\n=== Granularity A/B arm B (session-consolidated, n={n}, "
          f"top-{args.topk}) ===")
    for k in ("correct", "partial", "wrong", "judge_error"):
        print(f"  {k:12s} {tally.get(k,0):3d}")
    if n_scored:
        acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
        print(f"  score = {acc:.3f}  (strict = {tally.get('correct',0)/n_scored:.3f})")
        print("  baseline (turn granularity, same 50 questions): "
              "0.350 weighted / 0.320 strict")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports",
                       f"granularity_{args.sample}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "topk": args.topk, "model": MODEL,
                   "consolidate_prompt_hash": hashlib.sha1(
                       CONSOLIDATE_SYSTEM.encode()).hexdigest()[:10],
                   "tally": tally, "traces": traces},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
