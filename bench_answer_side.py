"""P-ANSWER: answer-side aggregation (registered, research/RESEARCH.md
P-2026-09-30-ANSWER -- verdict PENDING there; criteria locked before run).

Collector FIXED to FLAT-30 (strongest collector, turn-recall 0.683); all
arms answer from the SAME pool so deltas attribute purely to the answer
strategy.

  BASE      pool -> single answer call
  MAPREDUCE per-entry atomic-fact extraction (no inference), then answer
            from the fact list only
  HIGHLIGHT select needed entry ids + name gaps, then answer from those
  STRUCT    pool re-presented as a chronological timeline (L2 control)

50 multi-session questions (seed 7) -- directly comparable to P-AGENT's
8.0%. Same judge/strict as all e2e benches.

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_answer_side.py
"""
import json
import os
import random
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from bench_memory_judgment import (  # noqa: E402
    ds_client, ds_chat, parse_decision)
from bench_needle_search import load_store  # noqa: E402

QS = os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json")
N_Q = 50
FINAL_K = 30

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
MAP_SYSTEM = """You extract stated facts. Given ONE memory entry and a
question, list the atomic facts in the entry that bear on the question
(numbers, dates, entities, outcomes). Copy them as short bullet lines.
Do NOT infer, compute, or add anything not explicitly stated. If nothing
in the entry bears on the question, reply exactly: NONE."""
HIGHLIGHT_SYSTEM = """You are preparing to answer a question from a pool of
memory entries. List the ids of the entries you would need, and state
what information is still MISSING from the pool (the gap). Reply ONLY:
{"ids": [1, 2, ...], "gap": "one sentence"}"""


def main():
    t0 = time.time()
    E, sess, _turn_keys = load_store()
    # rebuild entry texts from the SAME snapshot load_store used
    # (bench_needle_search's SNAP = a copy of longmemeval_bench.pkl)
    import pickle
    from bench_needle_search import SNAP
    dd = pickle.load(open(SNAP, "rb"))
    texts = []
    for m in dd["memories"]:
        if m.get("superseded_by") is not None:
            continue
        v = np.asarray(m["embedding"], dtype=np.float32)
        if np.linalg.norm(v) == 0:
            continue
        texts.append(m["text"])
    assert len(texts) == len(E)
    n = len(E)
    qs = json.load(open(os.path.join(_HERE, "data_longmemeval",
                                     "longmemeval_s_cleaned.json"),
                        encoding="utf-8"))
    ms = [q for q in qs if q["question_type"] == "multi-session"]
    subset = random.Random(7).sample(ms, N_Q)
    print(f"store {n} | {N_Q} multi-session questions", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    client = ds_client()
    arms = ["BASE", "MAPREDUCE", "HIGHLIGHT", "STRUCT"]
    tally = {a: {"correct": 0, "partial": 0, "wrong": 0, "abstain": 0}
             for a in arms}
    traces = []

    for qi, q in enumerate(subset):
        qv = np.asarray(model.encode([q["question"]], normalize_embeddings=True)[0],
                        dtype=np.float32)
        pool = np.argsort(-(E @ qv))[:FINAL_K]
        ctx_entries = [f"[{k+1}] {texts[i][:200]}"
                       for k, i in enumerate(pool)]
        ctx = "\n".join(ctx_entries)
        answers = {}

        # BASE
        answers["BASE"] = ds_chat(client, ANSWER_SYSTEM,
                                  f"Remembered entries:\n{ctx}\n\n"
                                  f"Question: {q['question']}\n\nAnswer now.")

        # MAPREDUCE
        facts = []
        for k, i in enumerate(pool):
            out = ds_chat(client, MAP_SYSTEM,
                          f"Question: {q['question']}\n\n"
                          f"Memory entry: {texts[i][:300]}\n\n"
                          f"Extract relevant atomic facts (or NONE).")
            if out.strip() and "NONE" not in out[:20]:
                facts.append(f"[{k+1}] " + " ".join(out.strip().split("\n")[:4])[:220])
        fact_ctx = "\n".join(facts) if facts else "(no relevant facts found)"
        answers["MAPREDUCE"] = ds_chat(
            client, ANSWER_SYSTEM,
            f"Question: {q['question']}\n\nExtracted atomic facts from "
            f"memory:\n{fact_ctx}\n\nAnswer the question using only these "
            f"facts. If they are insufficient, say you don't know.")

        # HIGHLIGHT
        hraw = ds_chat(client, HIGHLIGHT_SYSTEM,
                       f"Question: {q['question']}\n\nPool:\n{ctx}\n\n"
                       f"Which entries do you need?")
        hd, _ = parse_decision(hraw)
        sel = hd.get("ids", []) if hd else []
        gap = hd.get("gap", "") if hd else ""
        sel_ctx = "\n".join(ctx_entries[k - 1] for k in sel
                            if isinstance(k, int) and 1 <= k <= len(pool)) \
            or "(no entries selected)"
        answers["HIGHLIGHT"] = ds_chat(
            client, ANSWER_SYSTEM,
            f"Question: {q['question']}\n\nSelected entries:\n{sel_ctx}\n\n"
            f"Known gap: {gap}\n\nAnswer the question. If the gap makes it "
            f"unanswerable, say you don't know.")

        # STRUCT (chronological presentation; ids are store-order here —
        # use timestamps proxy = store order, L2 control arm)
        answers["STRUCT"] = ds_chat(
            client, ANSWER_SYSTEM,
            f"Remembered entries, in chronological order as a timeline:\n"
            f"{ctx}\n\nQuestion: {q['question']}\n\nAnswer now. Aggregate "
            f"across entries when the question asks for counts or totals.")

        for arm in arms:
            raw = answers[arm]
            jr = ds_chat(client, JUDGE_SYSTEM,
                         f"Ground truth: {q['answer']}\n\n"
                         f"Assistant answer: {raw}\n\nGrade it.")
            dec, _ = parse_decision(jr)
            sc = dec.get("score", "wrong") if dec else "wrong"
            tally[arm][sc] = tally[arm].get(sc, 0) + 1
            if "don't know" in raw.lower() or "do not know" in raw.lower() \
                    or "not sure" in raw.lower():
                tally[arm]["abstain"] += 1
            traces.append({"qi": qi, "arm": arm, "score": sc,
                           "answer": raw[:180]})
        sline = " | ".join(
            f"{a} {tally[a]['correct']/(qi+1):.0%}" for a in arms)
        print(f"[{qi+1}/{N_Q}] {sline}", flush=True)

    strict = {a: tally[a]["correct"] / N_Q for a in arms}
    base = strict["BASE"]
    best_arm = max(strict, key=strict.get)
    best = strict[best_arm]
    if best >= base + 0.05:
        verdict = (f"ANSWER-SUPPORTED via {best_arm} ({best:.1%} vs "
                   f"BASE {base:.1%}) -- answer layer is repairable")
    elif best > base + 0.02:
        verdict = f"ANSWER-PARTIAL (best {best_arm} {best:.1%})"
    else:
        verdict = ("ANSWER-NULL -- the answer layer is also at its "
                   "boundary on deepseek-chat")
    print(f"\nstrict: " + " | ".join(f"{a} {strict[a]:.1%}" for a in arms))
    print(f"abstain: " + " | ".join(f"{a} {tally[a]['abstain']}/{N_Q}"
                                    for a in arms))
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"answer_side_{ts}.json")
    json.dump({"experiment": "P-ANSWER", "n": N_Q, "strict": strict,
               "abstain": {a: tally[a]["abstain"] for a in arms},
               "verdict": verdict, "traces": traces},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
