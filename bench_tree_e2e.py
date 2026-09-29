"""P-TREE stage 2: end-to-end answer check on the multi-session set.

Stage 1 (bench_tree_navigation.py) gave PARTIAL: session-hit +10.5pp with
no guard regression. Before any adoption decision, the answer-level check:
does TREE's wider collection (top-30 over navigated clusters) actually
improve STRICT answers vs production RRF top-5 -- or does the 6x longer
context dilute the answer?

50 questions sampled from the multi-session set (seed 7), both arms, same
answer prompt and judge as bench_lme_e2e.py (strict口径).

ADDITION LOCK CRITERIA (locked before this ran):
  ADOPT   : strict(TREE) >= strict(RRF) + 3pp on the 50 questions
            -> wire tree collection into the answer path for
            multi-session-class queries (classify_query routing)
  TIE     : |diff| <= 3pp -> do not adopt by default; keep as an optional
            collector behind a flag, revisit with an LLM-refined tree
  REJECT  : strict(TREE) < strict(RRF) - 3pp -> close the line, record

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_tree_e2e.py
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

from bench_memory_judgment import ds_client, parse_decision  # noqa: E402
from bench_tree_navigation import (  # noqa: E402
    spherical_kmeans, L1_K, L2_K, L1_TOP, L2_TOP, FINAL_K, SEED)

MODEL = "deepseek-chat"
N_Q = 50
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


def load_store_texts():
    from bench_tree_navigation import PKL, SNAP  # noqa: E402
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    active = [m for m in d["memories"] if m.get("superseded_by") is None]
    vecs, sess, texts = [], [], []
    for m in active:
        v = np.asarray(m["embedding"], dtype=np.float32)
        nrm = np.linalg.norm(v)
        if nrm == 0:
            continue
        vecs.append(v / nrm)
        sess.append(m["memory_id"])
        texts.append(m["text"])
    return np.stack(vecs), np.asarray(sess), texts


def main():
    E, ids, texts = load_store_texts()
    qs = json.load(open(os.path.join(_HERE, "data_longmemeval",
                                     "longmemeval_s_cleaned.json"),
                        encoding="utf-8"))
    ms = [q for q in qs if q["question_type"] == "multi-session"]
    rng = random.Random(7)
    subset = rng.sample(ms, N_Q)

    l1_assign, l1_cent = spherical_kmeans(E, L1_K, SEED)
    children, l2_cent = {}, {}
    for c in range(L1_K):
        idx = np.nonzero(l1_assign == c)[0]
        k2 = min(L2_K, max(1, len(idx) // 8))
        if len(idx) == 0:
            continue
        a2, c2 = spherical_kmeans(E[idx], k2, SEED + 100 + c)
        leaf_ids = c * 1000 + np.arange(k2)
        for j, g in enumerate(leaf_ids):
            l2_cent[int(g)] = c2[j]
            children[int(g)] = idx[a2 == j].tolist()

    def navigate(qv):
        l1 = np.argsort(-(l1_cent @ qv))[:L1_TOP]
        picked = []
        for c in l1:
            kids = sorted(g for g in children if g // 1000 == c)
            kids_arr = np.stack([l2_cent[g] for g in kids])
            for g in [kids[i] for i in np.argsort(-(kids_arr @ qv))[:L2_TOP]]:
                picked.extend(children.get(g, []))
        return picked

    from flymemory.v3 import _embed  # noqa: E402
    client = ds_client()
    tally = {a: {"correct": 0, "partial": 0, "wrong": 0} for a in ("RRF", "TREE")}
    traces = []
    for qi, q in enumerate(subset):
        qv = np.asarray(_embed(q["question"]), dtype=np.float32)
        qv /= (np.linalg.norm(qv) + 1e-9)
        rrf_idx = np.argsort(-(E @ qv))[:5]
        picked = navigate(qv)
        picked_arr = np.asarray(picked, dtype=np.int64)
        sims = E[picked_arr] @ qv
        tree_idx = picked_arr[np.argsort(-sims)[:FINAL_K]]
        for arm, idx in (("RRF", rrf_idx), ("TREE", tree_idx)):
            ctx = "\n".join(f"- {texts[i][:200]}" for i in idx)
            from bench_memory_judgment import ds_chat  # noqa: E402
            raw = ds_chat(
                client, ANSWER_SYSTEM,
                f"Remembered entries:\n{ctx}\n\nQuestion: {q['question']}\n\n"
                f"Answer now.")
            # judge
            jr = ds_chat(
                client, JUDGE_SYSTEM,
                f"Ground truth: {q['answer']}\n\nAssistant answer: {raw}\n\n"
                f"Grade it.")
            dec, _err = parse_decision(jr)
            score = dec.get("score", "wrong") if dec else "wrong"
            tally[arm][score] = tally[arm].get(score, 0) + 1
            traces.append({"qi": qi, "arm": arm, "score": score,
                           "answer": raw[:200]})
        strict_r = tally["RRF"]["correct"] / (qi + 1)
        strict_t = tally["TREE"]["correct"] / (qi + 1)
        print(f"[{qi+1}/{N_Q}] strict RRF {strict_r:.1%} | TREE {strict_t:.1%}",
              flush=True)

    n = len(subset)
    strict = {a: tally[a]["correct"] / n for a in tally}
    diff = strict["TREE"] - strict["RRF"]
    if diff >= 0.03:
        verdict = "ADOPT"
    elif diff > -0.03:
        verdict = "TIE"
    else:
        verdict = "REJECT"
    print(f"\nstrict: RRF {strict['RRF']:.1%} vs TREE {strict['TREE']:.1%} "
          f"({diff:+.1%}) -> {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"tree_e2e_{ts}.json")
    json.dump({"experiment": "P-TREE-stage2", "n": n, "strict": strict,
               "verdict": verdict, "traces": traces},
              open(out, "w"), indent=1)
    print(f"saved {out}", flush=True)


def ds_chat(client, system, user):
    from bench_memory_judgment import BASE_URL, MODEL  # noqa: E402
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0, max_tokens=300)
    return r.choices[0].message.content or ""


if __name__ == "__main__":
    main()
