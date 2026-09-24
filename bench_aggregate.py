"""P-2026-09-24-AGG1: aggregate-style entries overlay (three-layer store).

Layers: turn store + consolidated (prose) + AGGREGATE list-style entries
("TOPIC: item1; item2 (total: N)") targeting the aggregation/statistics
question type that dominates full-miss wrongs (85 cases, evidence rank
1708-11867). Registered prediction in research/RESEARCH.md (AGG1).

Run:
  DEEPSEEK_API_KEY=sk-... python bench_aggregate.py [--sample 50]
"""
import argparse
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
from bench_lme_e2e import ANSWER_SYSTEM, JUDGE_SYSTEM  # noqa: E402

MODEL = "deepseek-chat"
AGG_SYSTEM = """Extract AGGREGATED / COUNTABLE facts from this conversation as
list-style memory entries. Look for: items tried/visited/bought/eaten
(enumerate them), recurring people/places/providers, cumulative amounts
(total spend, total distance, counts). Format one entry per topic:
"TOPIC: item1 (date); item2 (date); item3 (date) (total/count: N if stated)"
Only facts present in the conversation. If nothing countable exists, reply
with an empty array. Reply with ONLY a JSON array of entry strings."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    client = ds_client()

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

    cache_p = os.path.join(here, "reports", "aggregate_entries.json")
    if os.path.exists(cache_p):
        aggs = json.load(open(cache_p, encoding="utf-8"))
        print(f"aggregate entries loaded from cache: {len(aggs)}", flush=True)
    else:
        aggs = []
        t0 = time.time()
        for si, (sid, (ts, turns)) in enumerate(order):
            convo = "\n".join(turns)[:6000]
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": AGG_SYSTEM},
                              {"role": "user", "content": convo}],
                    temperature=0, max_tokens=500)
                d, err = parse_decision(r.choices[0].message.content or "")
                lines = [str(x) for x in d if str(x).strip()] \
                    if not err and isinstance(d, list) else None
            except Exception:
                lines = None
            if lines:
                aggs.append({"sid": sid, "ts": ts, "entries": lines})
            if (si + 1) % 100 == 0:
                print(f"  aggregate {si+1}/{len(order)} ({time.time()-t0:.0f}s)",
                      flush=True)
        with open(cache_p, "w", encoding="utf-8") as f:
            json.dump(aggs, f, ensure_ascii=False)
        print(f"aggregate entries: {len(aggs)} sessions cached", flush=True)

    # ---- three-layer store: turns + consolidated + aggregate ----
    mem = load(os.path.join(here, "longmemeval_bench.pkl"), enable_hopfield=False)
    cons = json.load(open(os.path.join(here, "reports",
                                       "consolidated_entries.json"),
                          encoding="utf-8"))
    for c in cons:
        for text in c["entries"]:
            mem.remember_text(text, tags=["lme", c["sid"], "consolidated"],
                              source="model", timestamp=c["ts"], force_new=True)
    for c in aggs:
        for text in c["entries"]:
            mem.remember_text(text, tags=["lme", c["sid"], "aggregate"],
                              source="model", timestamp=c["ts"], force_new=True)
    print(f"three-layer library: {mem.size} entries", flush=True)

    rng = random.Random(args.seed)
    idx = rng.sample(range(len(data)), min(args.sample, len(data)))
    tally = {"correct": 0, "partial": 0, "wrong": 0, "judge_error": 0}
    per_type = {}
    traces = []
    t0 = time.time()
    for qi, i in enumerate(idx):
        q = data[i]
        hits = mem.recall(q["question"], top_k=args.topk)
        hits = sorted(hits, key=lambda h: h[0].timestamp or 0)
        lines = [f"- {m.text}" for m, _s, _e in hits]
        user = ("REMEMBERED FACTS (chronological; lists and totals included):\n"
                + "\n".join(lines)
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
        qt = q.get("question_type", "?")
        per_type.setdefault(qt, {})
        per_type[qt][verdict] = per_type[qt].get(verdict, 0) + 1
        traces.append({"idx": i, "question": q["question"], "answer": answer,
                       "gold": q.get("answer", ""), "verdict": verdict,
                       "type": qt})
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(idx)} ({time.time()-t0:.0f}s) {tally}", flush=True)

    n = len(idx)
    n_scored = sum(v for k, v in tally.items() if k != "judge_error")
    print(f"\n=== Three-layer (turns + consolidated + aggregate, n={n}, "
          f"top-{args.topk}) ===")
    for k in ("correct", "partial", "wrong", "judge_error"):
        print(f"  {k:12s} {tally.get(k,0):3d}")
    if n_scored:
        acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
        print(f"  score = {acc:.3f}  (strict = {tally.get('correct',0)/n_scored:.3f})")
        print("  registered baselines @50: two-layer (turns+cons) 0.380 strict | "
              "turn-only 0.320 | timeline overlay 0.440")
    print("\n=== per question_type ===")
    for qt, cc in sorted(per_type.items()):
        tot = sum(cc.values())
        print(f"  {qt:30s} n={tot:3d}  {dict(cc)}")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports",
                       f"aggregate_{args.sample}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "topk": args.topk, "model": MODEL,
                   "tally": tally, "per_type": per_type, "traces": traces},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
