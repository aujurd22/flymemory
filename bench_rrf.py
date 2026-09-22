"""RRF rescue experiment: dense + BM25 rank fusion vs single channels.

On LongMemEval-S (199,509-turn library), measures evidence-session hit@3 for:
  dense : max-over-chunks cosine
  bm25  : IDF lexical
  rrf   : reciprocal rank fusion of the two (rrf_k=60, pool=200)

Run: python bench_rrf.py [--max-n ... ] (uses longmemeval_s_bench.pkl)
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import load, split_chunks, _embed, _tokenize  # noqa: E402


def main():
    mem = load(os.path.join(_HERE, "longmemeval_s_bench.pkl"), enable_hopfield=False)
    N = mem.size
    E = mem._emb_matrix()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    entry_tokens = [set(_tokenize(m.text)) for m in mem.memories]
    df = {}
    for toks in entry_tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n_docs = len(entry_tokens)
    print(f"library: {N} entries, df tokens: {len(df)}", flush=True)

    with open(os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json"),
              encoding="utf-8") as f:
        data = json.load(f)

    POOL = 200
    RRF_K = 60
    K = 3
    hit = {"dense": 0, "bm25": 0, "rrf": 0}
    t0 = time.time()
    for qi, q in enumerate(data):
        q_text = q["question"]
        answer_sessions = set(q.get("answer_session_ids", []))
        chunks = split_chunks(q_text) or [q_text]
        q_embs = [_embed(c) for c in chunks]
        Q = np.stack(q_embs)
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sims = (qn @ En.T).max(axis=0)
        dense_order = np.argsort(-sims)[:POOL]

        scored = {}
        total = 0.0
        for c in chunks:
            for tok in _tokenize(c):
                d = df.get(tok, 0)
                idf = np.log(1.0 + n_docs / d) if d else np.log(1.0 + n_docs)
                total += idf
                if d:
                    for i, toks in enumerate(entry_tokens):
                        if tok in toks:
                            scored[i] = scored.get(i, 0.0) + idf
        bm_order = sorted(scored, key=lambda i: -scored[i])[:POOL]

        fused = {}
        for rank, idx in enumerate(dense_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(bm_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        rrf_order = sorted(fused, key=lambda i: -fused[i])[:K]

        for name, order in (("dense", dense_order), ("bm25", bm_order), ("rrf", rrf_order)):
            if any(set(mem.memories[idx].tags) & answer_sessions for idx in order[:K]):
                hit[name] += 1
        if (qi + 1) % 100 == 0:
            print(f"  {qi+1}/500 ({time.time()-t0:.0f}s)", flush=True)

    print(f"=== evidence-hit@{K} (N={N}) ===")
    for arm in ("dense", "bm25", "rrf"):
        print(f"  {arm:6s} {hit[arm]:4d}/{N} = {hit[arm]/N*100:.0f}%")


if __name__ == "__main__":
    main()
