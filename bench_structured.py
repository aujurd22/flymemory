"""P-2026-09-24-TL2: structured-timeline overlay experiment (Mushroom-Body Program).

Registered prediction (see research/RESEARCH.md, registered BEFORE this run):
  P1: structured timeline > prose timeline (50-question strict, seed 7)
  P2: temporal-reasoning error rate drops more than non-temporal types
  P3: overall strict >= turn-only baseline (32%)

Arm: overlay turn store with STRUCTURED entries ("SUBJECT: attr = value
(date)", numbers/names preserved verbatim) -- vs prose timeline overlay
(44% strict @50) and prose overlay (38% @50 / 41.6% @500) and turn-only
(32% @50 / 37.0% @500).

Run:
  DEEPSEEK_API_KEY=sk-... python bench_structured.py [--sample 500] [--skip-gen]
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter

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
STRUCT_SYSTEM = """Extract durable facts from this conversation as STRUCTURED
memory entries. One entry per line, exact format:
"SUBJECT: attribute = value (approx date)"
Rules: preserve ALL numbers, names, dates and quantities VERBATIM. One fact
per entry. Only facts present in the conversation. Reply with ONLY a JSON
array of these entry strings."""

ANSWER_SYSTEM = ("You are a personal assistant answering questions from your "
                 "long-term memory. Use ONLY the remembered entries below "
                 "(structured entries first). If they do not contain the "
                 "answer, say you don't know. Answer in one or two short "
                 "sentences.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--skip-gen", action="store_true",
                    help="reuse cached structured entries")
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

    cache_p = os.path.join(here, "reports", "structured_entries.json")
    if os.path.exists(cache_p):
        cons = json.load(open(cache_p, encoding="utf-8"))
        print(f"structured entries loaded from cache: {len(cons)}", flush=True)
    elif args.skip_gen:
        sys.exit("--skip-gen but no cache")
    else:
        cons = []
        t0 = time.time()
        for si, (sid, (ts, turns)) in enumerate(order):
            convo = "\n".join(turns)[:6000]
            base = time.strftime("%Y-%m", time.localtime(ts or time.time()))
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": STRUCT_SYSTEM},
                              {"role": "user", "content":
                                   f"Session date approx {base}.\n{convo}"}],
                    temperature=0, max_tokens=600)
                d, err = parse_decision(r.choices[0].message.content or "")
                lines = [str(x) for x in d if str(x).strip()] \
                    if not err and isinstance(d, list) else None
            except Exception:
                lines = None
            if lines:
                cons.append({"sid": sid, "ts": ts, "entries": lines})
            if (si + 1) % 100 == 0:
                print(f"  structured {si+1}/{len(order)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        with open(cache_p, "w", encoding="utf-8") as f:
            json.dump(cons, f, ensure_ascii=False)
        print(f"structured entries: {len(cons)} sessions cached "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- overlay: turn store + structured entries ----
    mem = load(os.path.join(here, "longmemeval_bench.pkl"), enable_hopfield=False)
    for c in cons:
        for text in c["entries"]:
            mem.remember_text(text, tags=["lme", c["sid"], "structured"],
                              source="model", timestamp=c["ts"], force_new=True)
    print(f"overlay library: {mem.size} entries", flush=True)

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
        user = ("REMEMBERED FACTS (structured entries, chronological):\n"
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
    print(f"\n=== Structured overlay (n={n}, top-{args.topk}) ===")
    for k in ("correct", "partial", "wrong", "judge_error"):
        print(f"  {k:12s} {tally.get(k,0):3d}")
    if n_scored:
        acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
        print(f"  score = {acc:.3f}  (strict = {tally.get('correct',0)/n_scored:.3f})")
        print("  registered baselines @50: turn-only 0.320 | prose overlay 0.380 "
              "| timeline overlay 0.440 (all strict)")
        print("  @500: turn-only 0.370 | prose overlay 0.416")
    print("\n=== per question_type ===")
    for qt, cc in sorted(per_type.items()):
        tot = sum(cc.values())
        print(f"  {qt:30s} n={tot:3d}  {dict(cc)}")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports",
                       f"structured_{args.sample}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "topk": args.topk, "model": MODEL,
                   "tally": tally, "per_type": per_type, "traces": traces},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
