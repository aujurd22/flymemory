"""A/B: lexical-channel ranking inside RRF fusion, on LongMemEval-oracle.

Arms (evidence-session hit@3, same dense channel, same RRF k=60 pool=200):
  rrf_ratio : production _lex_scores (matched-IDF / total-query-IDF ratio)
  rrf_cum   : bench-style cumulative IDF sum (no normalization)
Isolates whether the ratio normalization costs recall; the 61% historical
number pre-dates RRF fusion so it cannot answer this.

Run: python bench_lex_ab.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    pass
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import load, split_chunks, _embed, _tokenize  # noqa: E402


def main():
    mem = load(os.path.join(_HERE, "longmemeval_bench.pkl"), enable_hopfield=False)
    E = mem._emb_matrix()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    n_docs = mem.size
    mid_to_idx = {m.memory_id: i for i, m in enumerate(mem.memories)}
    with open(os.path.join(_HERE, "data_longmemeval", "longmemeval_oracle.json"),
              encoding="utf-8") as f:
        data = json.load(f)

    POOL, RRF_K, K = 200, 60, 3
    hit = {"rrf_ratio": 0, "rrf_cum": 0}
    t0 = time.time()
    for qi, q in enumerate(data):
        q_text = q["question"]
        answer_sessions = set(q.get("answer_session_ids", []))
        chunks = split_chunks(q_text) or [q_text]
        Q = np.stack([_embed(c) for c in chunks])
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sims = (qn @ En.T).max(axis=0)
        dense_order = np.argsort(-sims)[:POOL]

        # cumulative IDF (bench style); keep score order -- mapping ids back to
        # indices in library order silently destroys the ranking (bug of the
        # first run, v1 reported a bogus 43.8% for this arm)
        scored = {}
        for c in chunks:
            for tok in _tokenize(c):
                d = mem._df.get(tok, 0)
                idf = np.log(1.0 + n_docs / d) if d else np.log(1.0 + n_docs)
                if d:
                    for mid in mem._lex_index.get(tok, ()):
                        scored[mid] = scored.get(mid, 0.0) + idf
        cum_order = [mid_to_idx[m] for m in sorted(scored, key=lambda m: -scored[m])[:POOL]]

        lex_ratio = mem._lex_scores(chunks)
        ratio_order = np.argsort(-lex_ratio)[:POOL]

        def fuse(order_a, order_b):
            fused = {}
            for rank, idx in enumerate(order_a):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
            for rank, idx in enumerate(order_b):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
            return sorted(fused, key=lambda i: -fused[i])[:K]

        for name, lex_order in (("rrf_ratio", ratio_order), ("rrf_cum", cum_order)):
            top = fuse(dense_order, lex_order)
            if any(set(mem.memories[int(i)].tags) & answer_sessions for i in top):
                hit[name] += 1
        if (qi + 1) % 100 == 0:
            print(f"  {qi+1}/{len(data)} ({time.time()-t0:.0f}s) {hit}", flush=True)

    n = len(data)
    for arm in ("rrf_ratio", "rrf_cum"):
        print(f"  {arm:9s} {hit[arm]:4d}/{n} = {hit[arm]/n*100:.1f}%")


if __name__ == "__main__":
    main()
