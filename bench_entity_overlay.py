"""B2: Entity-State overlay experiment (v4 lever #2).

For the 20 state-update pairs from the fidelity audit, generate one
entity-state entry per pair that carries BOTH the superseded value and the
CURRENT value with explicit labels, then overlay onto the turn store.

Rationale: the wrong-attribution review showed aggregation/state questions
fail because no single entry states the derived or current value. An
entity-state entry makes "what is current" explicit at indexing time.

Baseline comparison (same 50 questions, seed 7):
  turn-only         32% strict / 35% weighted
  prose overlay     38% strict / 40% weighted
  timeline overlay  44% strict / 44% weighted

Run:
  DEEPSEEK_API_KEY=sk-... python bench_entity_overlay.py
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flymemory"))

from flymemory.v3 import SmartMemory, load  # noqa: E402
from bench_memory_judgment import ds_client, parse_decision  # noqa: E402
from bench_lme_e2e import ANSWER_SYSTEM, JUDGE_SYSTEM  # noqa: E402

MODEL = "deepseek-chat"

# (old fact, new fact, entity name) -- from the fidelity-audit pair set,
# extended with fresh themes
PAIRS = [
    ("The user's phone number is 138-0000-1111.", "The user's phone number is 139-9999-8888.", "user phone number"),
    ("The user's production server is 192.168.1.50.", "The user's production server is 192.168.1.99.", "production server IP"),
    ("The user lives at Oak Street 5.", "The user lives at Oak Street 8.", "home address"),
    ("The user's team is called Team Falcon.", "The user's team is called Team Kestrel.", "team name"),
    ("The user drives a gasoline hatchback.", "The user drives an electric car.", "car"),
    ("The user studies Japanese with a paper textbook.", "The user studies Japanese with a spaced-repetition app.", "Japanese study method"),
    ("The user's warehouse uses the XinDa label printer.", "The user's warehouse uses the HTW-109 label printer.", "warehouse label printer"),
    ("The user writes most backend code in Python.", "The user writes most backend code in Rust.", "backend language"),
    ("The user hosts their blog on a rented VPS.", "The user hosts their blog on a static hosting platform.", "blog hosting"),
    ("The user reports to Manager Zhang.", "The user reports to Manager Liu.", "manager"),
    ("The user prefers dark mode in every app.", "The user prefers light mode in every app.", "app theme preference"),
    ("The user's primary editor is PyCharm.", "The user's primary editor is Neovim.", "primary editor"),
    ("The user lives in Shenzhen.", "The user lives in Hangzhou.", "city of residence"),
    ("The user subscribes to the Pro tier.", "The user is on the free tier.", "subscription tier"),
    ("The user's laptop runs Windows 11.", "The user's laptop runs Fedora Linux.", "laptop OS"),
    ("The user's meeting is scheduled for 15:00.", "The user's meeting is scheduled for 16:00.", "meeting time"),
    ("The user works from the downtown office.", "The user works from the riverside office.", "work location"),
    ("The user deploys on AWS.", "The user deploys on Hetzner.", "deployment provider"),
    ("The user's cat is called Mochi.", "The user's cat is called Daifuku.", "cat's name"),
    ("The user's admin password hint is 'old one'.", "The user's admin password hint is 'blue fish'.", "admin password hint"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    client = ds_client()

    # ---- build the overlay store: turn store + entity-state entries ----
    mem = load(os.path.join(here, "longmemeval_bench.pkl"), enable_hopfield=False)
    for old, new, entity in PAIRS:
        entry_text = (f"ENTITY RECORD -- {entity}: previously \"{old}\" "
                      f"(SUPERSEDED); CURRENT value: \"{new}\".")
        mem.remember_text(entry_text, tags=["entity_state"], source="model",
                          force_new=True)
    print(f"overlay library: {mem.size} entries "
          f"(+{len(PAIRS)} entity-state records)", flush=True)

    with open(os.path.join(here, "data_longmemeval", "longmemeval_oracle.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    import random
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(data)), min(args.sample, len(data)))

    ANSWER_SYSTEM_ES = (ANSWER_SYSTEM + "\n\nThe remembered entries include "
                        "ENTITY RECORD entries that track state changes: they "
                        "state a previous value (SUPERSEDED) and the CURRENT "
                        "value. For 'what is X now' questions, always use the "
                        "CURRENT value from the entity record.")

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
                messages=[{"role": "system", "content": ANSWER_SYSTEM_ES},
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
    print(f"\n=== B2: entity-state overlay (n={n}, top-{args.topk}) ===")
    for k in ("correct", "partial", "wrong", "judge_error"):
        print(f"  {k:12s} {tally.get(k,0):3d}")
    if n_scored:
        acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
        print(f"  score = {acc:.3f}  (strict = {tally.get('correct',0)/n_scored:.3f})")
        print("  baselines (same 50 questions): turn-only 0.320 strict / "
              "0.350 weighted; prose overlay 0.380 strict / 0.400 weighted; "
              "timeline overlay 0.440 strict / 0.440 weighted")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports",
                       f"entity_overlay_{args.sample}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "topk": args.topk, "model": MODEL,
                   "tally": tally, "traces": traces},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
