"""P-AGENT: query-side agentic collection (registered, research/RESEARCH.md
P-2026-09-29-AGENT -- verdict PENDING there; criteria locked before run).

Route decision: index-side PAUSED (P-TREE/P-NEEDLE closed at "same-topic
evidence discrimination"). This bench tests the remaining lever: a
multi-round self-rephrasing collector at TURN-level metrics on the
multi-session set.

Collector: round 1 = cosine top-5 (offline proxy for production RRF;
collection shape is the comparison). Rounds 2-3: DeepSeek sees the
question + a digest of the current pool and emits 2-3 rephrasings probing
different aspects; each rephrasing runs top-5; pool merges (dedup).
Arms: RRF-1 (baseline), FLAT-30 (bare width control), AGENT-3 (merged
pool -> top-30).

Gate: if COLLECT-NULL on the 133-question collection metric, the
end-to-end stage is skipped (budget discipline).

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_agentic_collect.py
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
    ds_client, ds_chat, parse_decision, MODEL)
from bench_needle_search import (  # noqa: E402
    load_store, needle_words, STOP)

QS = os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json")
N_Q = 50
FINAL_K = 30
ROUNDS = 3

REPHRASE_SYSTEM = """You are a retrieval planner for a personal-memory
search engine. Given a user's question and short previews of what has
already been retrieved, write 2-3 DIFFERENT search queries that would
probe MISSING aspects of the question (specific entities, paraphrases,
related activities, time windows). Do not repeat the original question.
Respond with ONLY a JSON array of strings, e.g.
["made a Negroni at home", "Negroni ingredients gin campari"]"""
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
    t0 = time.time()
    E, sess, turn_keys = load_store()
    n = len(E)
    qs = json.load(open(QS, encoding="utf-8"))
    ms = [q for q in qs if q["question_type"] == "multi-session"]
    print(f"store {n} | multi {len(ms)}", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    dd = pickle.load(open(SNAP, "rb"))
    act = [m for m in dd["memories"] if m.get("superseded_by") is None]
    texts, j = [None] * n, 0
    for m in act:
        v = np.asarray(m["embedding"], dtype=np.float32)
        if np.linalg.norm(v) == 0:
            continue
        texts[j] = m["text"]
        j += 1

    client = ds_client()

    def collect_agent(q):
        qv = np.asarray(model.encode([q["question"]], normalize_embeddings=True)[0],
                        dtype=np.float32)
        pool = {}                                   # idx -> best sim
        def add(idxs):
            for i in idx:
                s = float(E[i] @ qv)
                if i not in pool or s > pool[i]:
                    pool[i] = s
        add(np.argsort(-(E @ qv))[:5])
        for rnd in range(ROUNDS - 1):
            digest = "\n".join(f"- {texts[i][:110]}" for i in
                               sorted(pool, key=pool.get, reverse=True)[:10])
            raw = ds_chat(client, REPHRASE_SYSTEM,
                          f"Question: {q['question']}\n\nAlready retrieved:\n"
                          f"{digest}\n\nWrite 2-3 NEW search queries.")
            try:
                queries = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
            except Exception:
                queries = []
            if not isinstance(queries, list):
                queries = []
            for rq in queries[:3]:
                if not isinstance(rq, str) or not rq.strip():
                    continue
                rqv = np.asarray(model.encode([rq], normalize_embeddings=True)[0],
                                 dtype=np.float32)
                add(np.argsort(-(E @ rqv))[:5])
        return pool

    def metrics(subset, setname, run_agent):
        stats = {a: {"turn_hit": 0, "turn_recall": [], "n": 0}
                 for a in ("RRF-1", "FLAT-30", "AGENT-3")}
        for qi, q in enumerate(subset):
            qv = np.asarray(model.encode([q["question"]], normalize_embeddings=True)[0],
                            dtype=np.float32)
            ans_turns = set(q["answer_session_ids"])
            rrf1 = np.argsort(-(E @ qv))[:5]
            flat30 = np.argsort(-(E @ qv))[:FINAL_K]
            if run_agent:
                pool = collect_agent(q)
                idxs = np.asarray(sorted(pool.keys()), dtype=np.int64)
                sims = np.asarray([pool[int(i)] for i in idxs], dtype=np.float32)
                agent3 = idxs[np.argsort(-sims)[:FINAL_K]]
            else:
                agent3 = None
            for arm, idx in (("RRF-1", rrf1), ("FLAT-30", flat30),
                             ("AGENT-3", agent3)):
                if idx is None:
                    continue
                got = {turn_keys[i] for i in idx if turn_keys[i]}
                st = stats[arm]
                st["turn_hit"] += len(got & ans_turns) > 0
                st["turn_recall"].append(
                    len(got & ans_turns) / max(len(ans_turns), 1))
                st["n"] += 1
            if (qi + 1) % 25 == 0:
                print(f"  [{setname}] {qi+1}/{len(subset)}", flush=True)
        out = {}
        for arm, s in stats.items():
            if s["n"]:
                out[arm] = {"turn_hit": round(s["turn_hit"] / s["n"], 4),
                            "turn_recall": round(float(np.mean(s["turn_recall"])), 4),
                            "n": s["n"]}
        print(f"{setname}: " + " | ".join(
            f"{a} t-hit {v['turn_hit']:.1%} t-rec {v['turn_recall']:.3f}"
            for a, v in out.items()), flush=True)
        return out

    res = {"multi_session": metrics(ms, "multi-session", run_agent=True)}

    r = res["multi_session"]
    base = r["RRF-1"]["turn_recall"]
    ag = r["AGENT-3"]["turn_recall"]
    fl = r["FLAT-30"]["turn_recall"]
    d = ag - base
    if d >= 0.15 and ag > fl:
        collect_verdict = ("COLLECT-SUPPORTED (>=RRF+15pp and beats "
                           "bare width)")
    elif d >= 0.05:
        collect_verdict = f"COLLECT-PARTIAL (bare width at {fl:.3f})"
    else:
        collect_verdict = "COLLECT-NULL -- query side also exhausted"
    print(f"\ncollect: AGENT-3 {ag:.3f} vs RRF-1 {base:.3f} ({d:+.3f}) "
          f"| FLAT-30 {fl:.3f}\n{collect_verdict}", flush=True)

    # end-to-end gate (skipped on COLLECT-NULL)
    e2e = None
    if "NULL" not in collect_verdict:
        rng = random.Random(7)
        subset = rng.sample(ms, N_Q)
        tally = {a: {"correct": 0, "partial": 0, "wrong": 0}
                 for a in ("RRF-1", "AGENT-3")}
        for qi, q in enumerate(subset):
            qv = np.asarray(model.encode([q["question"]], normalize_embeddings=True)[0],
                            dtype=np.float32)
            pool = collect_agent(q)
            idxs = np.asarray(sorted(pool.keys()), dtype=np.int64)
            sims = np.asarray([pool[int(i)] for i in idxs], dtype=np.float32)
            agent3 = idxs[np.argsort(-sims)[:FINAL_K]]
            rrf1 = np.argsort(-(E @ qv))[:5]
            for arm, idx in (("RRF-1", rrf1), ("AGENT-3", agent3)):
                ctx = "\n".join(f"- {texts[i][:200]}" for i in idx)
                raw = ds_chat(client, ANSWER_SYSTEM,
                              f"Remembered entries:\n{ctx}\n\n"
                              f"Question: {q['question']}\n\nAnswer now.")
                jr = ds_chat(client, JUDGE_SYSTEM,
                             f"Ground truth: {q['answer']}\n\n"
                             f"Assistant answer: {raw}\n\nGrade it.")
                dec, _ = parse_decision(jr)
                sc = dec.get("score", "wrong") if dec else "wrong"
                tally[arm][sc] = tally[arm].get(sc, 0) + 1
            sr = tally["RRF-1"]["correct"] / (qi + 1)
            st = tally["AGENT-3"]["correct"] / (qi + 1)
            print(f"  [e2e {qi+1}/{N_Q}] strict RRF-1 {sr:.1%} | "
                  f"AGENT-3 {st:.1%}", flush=True)
        strict = {a: tally[a]["correct"] / N_Q for a in tally}
        diff = strict["AGENT-3"] - strict["RRF-1"]
        adopt = "ADOPT" if diff >= 0.03 else ("TIE" if diff > -0.03 else "REJECT")
        e2e = {"strict": strict, "diff": round(float(diff), 4),
               "adopt": adopt}
        print(f"\ne2e strict: RRF-1 {strict['RRF-1']:.1%} vs "
              f"AGENT-3 {strict['AGENT-3']:.1%} ({diff:+.1%}) -> {adopt}",
              flush=True)
    else:
        print("e2e skipped (COLLECT-NULL)", flush=True)

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"agentic_collect_{ts}.json")
    json.dump({"experiment": "P-AGENT", "results": res,
               "collect_verdict": collect_verdict, "e2e": e2e},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
