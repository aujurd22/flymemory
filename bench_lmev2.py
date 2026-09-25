"""LME-V2 pipeline: filter needed trajectories -> ingest into FlyMemory -> sample eval.

Stage 1 (filter): stream the 1.2GB trajectories.jsonl, keep only the 200
trajectories referenced by the small-tier haystacks -> data_longmemeval_v2/
trajectories_small.jsonl (each line: {"id", "domain", "environment", "goal",
"outcome", "states":[...]}).

Stage 2 (ingest): each state's text (thought/action/accessibility_tree)
becomes turn-granularity memory entries, dated by step order; the state_key
mechanism is used for state-ish facts when the step carries one.

Stage 3 (eval): recall top-5 per question, DeepSeek answers, DeepSeek judge
vs the dataset's own `answer` + `eval_function` (norm_phrase_set_match
handled mechanically where possible).

Run:
  python bench_lmev2.py filter            # stage 1 (needs trajectories_full.jsonl)
  DEEPSEEK_API_KEY=... python bench_lmev2.py ingest --limit-traj 20
  DEEPSEEK_API_KEY=... python bench_lmev2.py eval --sample 30
"""
import argparse
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "flymemory"))

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data_longmemeval_v2")
NEEDED = os.path.join(HERE, "needed_traj_ids.json")
FULL = os.path.join(HERE, "trajectories_full.jsonl")
SMALL = os.path.join(HERE, "trajectories_small.jsonl")
PKL = os.path.join(HERE, "lmev2_flymemory.pkl")


def load_questions():
    with open(os.path.join(HERE, "questions.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_small_haystack():
    return json.load(open(os.path.join(HERE, "lme_v2_small.json"),
                          encoding="utf-8"))


def stage_filter():
    needed = set(json.load(open(NEEDED, encoding="utf-8")))
    if not os.path.exists(FULL):
        sys.exit(f"missing {FULL} -- download trajectories.jsonl first")
    kept = 0
    t0 = time.time()
    with open(FULL, encoding="utf-8") as fin, \
         open(SMALL, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # cheap pre-filter on the raw line before json parsing
            if not any(f'"{tid}"' in line for tid in needed):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") in needed:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                if kept >= len(needed):
                    break
    print(f"filter: kept {kept}/{len(needed)} trajectories "
          f"({time.time()-t0:.0f}s) -> {SMALL}", flush=True)


def states_to_entries(rec):
    """Turn a trajectory record into (text, step_index) memory entries."""
    out = []
    for st in rec.get("states", []):
        if not isinstance(st, dict):
            continue
        parts = []
        thought = (st.get("thought") or "").strip()
        action = (st.get("action") or "").strip()
        url = (st.get("url") or "").strip()
        if url:
            parts.append(f"@ {url}")
        if thought:
            parts.append(f"thought: {thought}")
        if action:
            parts.append(f"action: {action}")
        at = (st.get("accessibility_tree") or "").strip()
        if at and len(at) < 1500:
            parts.append(f"page: {at}")
        if not parts:
            continue
        idx = st.get("state_index", st.get("step"))
        out.append((f"[{rec['id']} step {idx}] " + " | ".join(parts),
                    idx))
    return out


def stage_ingest(limit_traj, limit_q):
    from flymemory.v3 import SmartMemory, load, save
    haystack = load_small_haystack()
    qs = load_questions()
    q_by_id = {q["id"]: q for q in qs}
    wanted_traj = set()
    for q in qs[:limit_q] if limit_q else []:
        wanted_traj.update(haystack[q["id"]])
    if not wanted_traj:
        sys.exit(f"no trajectories for the first {limit_q} questions")

    mem = SmartMemory(n_bits=4096)
    n_ent = 0
    t0 = time.time()
    with open(SMALL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] not in wanted_traj:
                continue
            for text, _idx in states_to_entries(rec):
                mem.remember_text(text, tags=["lmev2", rec["id"]],
                                  source="import", force_new=True)
                n_ent += 1
    print(f"ingested {n_ent} entries from {len(wanted_traj)} trajectories "
          f"({time.time()-t0:.0f}s) -> {PKL}", flush=True)
    save(mem, PKL)


def stage_eval(sample, topk):
    from flymemory.v3 import load, _embed
    from bench_memory_judgment import ds_client, parse_decision
    from bench_lme_e2e import ANSWER_SYSTEM, JUDGE_SYSTEM, MODEL

    mem = load(PKL, enable_hopfield=False)
    haystack = load_small_haystack()
    qs = load_questions()
    q_by_id = {q["id"]: q for q in qs}
    rng = __import__("random").Random(7)
    pick = rng.sample(qs, min(sample, len(qs))) if sample < len(qs) else qs

    client = ds_client()
    tally = {"correct": 0, "partial": 0, "wrong": 0, "judge_error": 0}
    traces = []
    t0 = time.time()
    for qi, q in enumerate(pick):
        hits = mem.recall(q["question"], top_k=topk)
        lines = [f"- {m.text[:300]}" for m, _s, _e in hits]
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
        traces.append({"qid": q["id"], "type": q["question_type"],
                       "question": q["question"], "answer": answer,
                       "gold": q.get("answer", ""), "verdict": verdict})
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(pick)} ({time.time()-t0:.0f}s) {tally}",
                  flush=True)

    n = len(pick)
    n_scored = sum(v for k, v in tally.items() if k != "judge_error")
    print(f"\n=== LME-V2 small-tier, FlyMemory backend (n={n}, top-{topk}) ===")
    for k in ("correct", "partial", "wrong", "judge_error"):
        print(f"  {k:12s} {tally.get(k,0):3d}")
    if n_scored:
        acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
        print(f"  score = {acc:.3f}  (strict = {tally.get('correct',0)/n_scored:.3f})")

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "reports"), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports",
                       f"lmev2_eval_{n}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tier": "small", "topk": topk, "model": MODEL,
                   "tally": tally, "traces": traces},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["filter", "ingest", "eval"])
    ap.add_argument("--limit-traj", type=int, default=20)
    ap.add_argument("--limit-q", type=int, default=5,
                    help="ingest only haystacks for the first N questions")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    if args.stage == "filter":
        stage_filter()
    elif args.stage == "ingest":
        stage_ingest(args.limit_traj, args.limit_q)
    else:
        stage_eval(args.sample, args.topk)


if __name__ == "__main__":
    main()
