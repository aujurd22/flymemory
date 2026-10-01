"""P-M5T: residual-registry transfer to agent memory (registered,
research/RESEARCH.md P-2026-09-30-M5T -- verdict PENDING there).

Cross-repo bridge: flyloop W9C's M5 residual registry (+0.787 E20,
bit-identical control) transplanted to agent-memory QA.

Phases (one independent store instance, 40 multi-session questions):
  1 base     : RRF top-5 -> answer -> judge            (baseline wrongs)
  2 register : one residual entry per wrong question
               ("[Correction note] Question asked: ... Correct answer:
                ... A previous attempt answered incorrectly.")
  3 retest   : (a) same questions re-answered (residuals compete in
               recall); (b) one LLM-paraphrased variant per wrong
               question (generalization test)
No separate control arm: temp 0 makes the no-residual rerun identical
to phase 1 by construction.

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_m5_transfer.py
"""
import json
import os
import pickle
import random
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from bench_memory_judgment import (  # noqa: E402
    ds_client, ds_chat, parse_decision)

QS = os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json")
PKL = os.path.join(_HERE, "longmemeval_bench.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_m5_snapshot.pkl")
N_Q = 40
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
PARAPHRASE_SYSTEM = """Rewrite the question with different wording but the
SAME factual target (do not change what is being asked, do not answer
it). Reply with ONLY the rewritten question."""


def main():
    t0 = time.time()
    shutil.copy(PKL, SNAP)
    from flymemory.v3 import load, _embed  # noqa: E402
    mem = load(SNAP, enable_hopfield=False)
    print(f"store loaded: {mem.size} entries ({time.time()-t0:.0f}s)",
          flush=True)

    qs = json.load(open(QS, encoding="utf-8"))
    ms = [q for q in qs if q["question_type"] == "multi-session"]
    subset = random.Random(7).sample(ms, 50)[:N_Q]
    client = ds_client()

    def answer(q, topk=5):
        hits = mem.recall(q["question"], top_k=topk)
        ctx = "\n".join(f"- {h[0].text[:200]}" for h in hits)
        raw = ds_chat(client, ANSWER_SYSTEM,
                      f"Remembered entries:\n{ctx}\n\n"
                      f"Question: {q['question']}\n\nAnswer now.")
        jr = ds_chat(client, JUDGE_SYSTEM,
                     f"Ground truth: {q['answer']}\n\n"
                     f"Assistant answer: {raw}\n\nGrade it.")
        dec, _ = parse_decision(jr)
        sc = dec.get("score", "wrong") if dec else "wrong"
        return sc, raw, [h for h in hits]

    # ---- phase 1: base ----
    wrongs = []
    for qi, q in enumerate(subset):
        sc, raw, hits = answer(q)
        if sc == "wrong":
            wrongs.append({"q": q, "raw": raw})
        if (qi + 1) % 10 == 0:
            print(f"  [phase1 {qi+1}/{N_Q}] wrong so far {len(wrongs)}",
                  flush=True)
    print(f"phase 1: {len(wrongs)}/{N_Q} wrong (correction targets)",
          flush=True)

    # ---- phase 2: residual registration ----
    plain = "--residual-plain" in sys.argv
    declarative = "--residual-declarative" in sys.argv
    for w in wrongs:
        if declarative:
            stmt = ds_chat(client,
                           "Turn a question and its correct answer into "
                           "ONE natural third-person factual statement "
                           "about the user, embedding the answer. No "
                           "question text, no meta wording. Reply with "
                           "only the statement.",
                           f"Question: {w['q']['question']}
"
                           f"Correct answer: {w['q']['answer']}")
            txt = stmt.strip() or f"{w['q']['answer']}."
        elif plain:
            txt = (f"[Correction note] Question asked: "
                   f"{w['q']['question']} "
                   f"Correct answer: {w['q']['answer']}. "
                   f"A previous attempt answered this incorrectly.")
        mem.remember_text(txt, source="model")
    print(f"phase 2: {len(wrongs)} residual entries registered",
          flush=True)

    # ---- phase 3a: same questions ----
    corr_same = 0
    resid_ranks = []
    for wi, w in enumerate(wrongs):
        hits = mem.recall(w["q"]["question"], top_k=8)
        # residual rank (position of the [Correction note] entry)
        for pos, h in enumerate(hits, start=1):
            if "[Correction note]" in h[0].text:
                resid_ranks.append(pos)
                break
        sc, raw, _ = answer(w["q"])
        corr_same += sc == "correct"
        if (wi + 1) % 10 == 0:
            print(f"  [phase3a {wi+1}/{len(wrongs)}] corrected so far "
                  f"{corr_same}", flush=True)
    n_w = max(len(wrongs), 1)
    rate_same = corr_same / n_w

    # ---- phase 3b: paraphrased variants ----
    corr_var = 0
    for wi, w in enumerate(wrongs):
        var = ds_chat(client, PARAPHRASE_SYSTEM,
                      f"Question: {w['q']['question']}\n\nRewrite it.")
        var = var.strip().strip('"') or w["q"]["question"]
        vq = dict(w["q"])
        vq["question"] = var
        sc, raw, _ = answer(vq)
        corr_var += sc == "correct"
        if (wi + 1) % 10 == 0:
            print(f"  [phase3b {wi+1}/{len(wrongs)}] variant corrected "
                  f"so far {corr_var}", flush=True)
    rate_var = corr_var / n_w

    mean_rank = float(np.mean(resid_ranks)) if resid_ranks else None
    in_pool = len(resid_ranks) / n_w

    if rate_same >= 0.60 and rate_var >= 0.25:
        verdict = "SUPPORTED -- residuals transfer past verbatim recognition"
    elif rate_same >= 0.60:
        verdict = "PARTIAL -- self-correction works, generalization weak"
    elif rate_same < 0.30:
        verdict = "NULL -- residual entries fail to compete or be used"
    else:
        verdict = "PARTIAL (bands boundary)"
    print(f"\nsame-question correction: {corr_same}/{n_w} = {rate_same:.1%}")
    print(f"variant correction:       {corr_var}/{n_w} = {rate_var:.1%}")
    print(f"residual in top-8: {in_pool:.1%} (mean rank {mean_rank})")
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"m5_transfer_{ts}.json")
    json.dump({"experiment": "P-M5T", "n_wrong": len(wrongs),
               "rate_same": round(rate_same, 4),
               "rate_variant": round(rate_var, 4),
               "residual_in_top8": round(in_pool, 4),
               "mean_rank": mean_rank, "verdict": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
