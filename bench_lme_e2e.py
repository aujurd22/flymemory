"""LongMemEval end-to-end (retrieval -> LLM answer -> judge) -- minimal version.

Phase 2 established answer-level numbers on the 30 self-made state cases.
This bench extends the same protocol to the public LongMemEval-oracle
questions, giving an externally comparable answer-level score:

  recall (production, RRF, top-5) over the 9,729-entry oracle store
  -> deepseek answers from the recalled entries only
  -> deepseek judge vs the dataset's gold answer (correct/partial/wrong)

n=50 sampled with a fixed seed; keep the seed fixed across models.

Run:
  DEEPSEEK_API_KEY=sk-... python bench_lme_e2e.py [--sample 50]
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

from flymemory.v3 import load  # noqa: E402
from bench_memory_judgment import ds_client, parse_decision  # noqa: E402

MODEL = "deepseek-chat"
ANSWER_SYSTEM = """You are a personal assistant answering questions from your
long-term memory. Use ONLY the remembered entries below. If they do not
contain the answer, say you don't know. Answer in one or two short
sentences."""
JUDGE_SYSTEM = """You are grading an assistant's answer against the ground
truth. Reply with ONLY:
{"score": "correct|partial|wrong", "note": "one short sentence"}
- correct: the answer conveys the same key fact(s) as the ground truth
- partial: some key facts right, some missing or imprecise
- wrong: contradicts the ground truth or is off topic"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--sort-by-time", action="store_true",
                    help="present the recalled entries to the answer model in "
                         "chronological order (temporal-reasoning aid)")
    ap.add_argument("--calc-prompt", action="store_true",
                    help="v4 lever: instruct the answer model to extract "
                         "numbers/dates and compute step by step (cross-turn "
                         "arithmetic aid)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    mem = load(os.path.join(here, "longmemeval_bench.pkl"), enable_hopfield=False)
    with open(os.path.join(here, "data_longmemeval", "longmemeval_oracle.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(data)), min(args.sample, len(data)))

    client = ds_client()
    print(f"questions: {len(idx)} (seed {args.seed}) | top-k {args.topk} | "
          f"model {MODEL}\n", flush=True)

    tally = {"correct": 0, "partial": 0, "wrong": 0, "judge_error": 0}
    traces = []
    t0 = time.time()
    for qi, i in enumerate(idx):
        q = data[i]
        hits = mem.recall(q["question"], top_k=args.topk)
        if args.sort_by_time:
            hits = sorted(hits, key=lambda h: h[0].timestamp or 0)
        lines = [f"- {m.text}" for m, _s, _e in hits]
        user = (f"REMEMBERED FACTS:\n" + "\n".join(lines)
                + f"\n\nQUESTION: {q['question']}\n\nAnswer now.")
        system = ANSWER_SYSTEM
        if args.calc_prompt:
            system = (ANSWER_SYSTEM + "\n\nIf answering requires arithmetic "
                      "(durations, totals, differences), first extract the "
                      "relevant numbers with their dates from the facts, then "
                      "compute the result step by step before stating it.")
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0, max_tokens=400)
            answer = (r.choices[0].message.content or "").strip()
        except Exception as e:
            answer = f"[api error: {e}]"
        juser = (f"QUESTION: {q['question']}\n\nGROUND TRUTH: "
                 f"{q.get('answer', '')}\n\nASSISTANT ANSWER: {answer}\n\nGrade now.")
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": juser}],
                temperature=0, max_tokens=200)
            d, err = parse_decision(r.choices[0].message.content or "")
            verdict = d.get("score") if not err and d.get("score") else "judge_error"
        except Exception as e:
            verdict = "judge_error"
        tally[verdict] = tally.get(verdict, 0) + 1
        traces.append({"idx": i, "question": q["question"], "answer": answer,
                       "gold": q.get("answer", ""), "verdict": verdict})
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(idx)} ({time.time()-t0:.0f}s) {tally}", flush=True)

    n = len(idx)
    n_scored = sum(v for k, v in tally.items() if k != "judge_error")
    print(f"\n=== LongMemEval-oracle end-to-end (n={n}, top-{args.topk}) ===")
    for k in ("correct", "partial", "wrong", "judge_error"):
        print(f"  {k:12s} {tally.get(k,0):3d}")
    if n_scored:
        acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
        print(f"  score (correct + 0.5*partial) / scored = {acc:.3f}  "
              f"(strict correct = {tally.get('correct',0)/n_scored:.3f})")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports",
                       f"lme_e2e_{args.sample}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"sample_seed": args.seed, "topk": args.topk,
                   "model": MODEL, "tally": tally, "traces": traces},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
