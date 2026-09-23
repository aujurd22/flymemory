"""Post-fusion state-aware rerank on LongMemEval-oracle (500 questions, 48s).

Arms (evidence-session hit@3):
  rrf       : RRF top-3 (k=60, pool 200) -- production order
  state10   : RRF top-10 reranked by eff = dw*src_w*(sim + 0.25*lex) -> top-3
  state20   : same over RRF top-20
Answers "should decay/source rerank AFTER fusion?" on a library whose
sessions span 2021-2024 (real dw spread, unlike the S-edition where every
entry is equally old).

Run: python bench_state_oracle.py
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
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import (load, split_chunks, _embed, LEX_WEIGHT,  # noqa: E402
                          SOURCE_WEIGHT)


def main():
    mem = load(os.path.join(_HERE, "longmemeval_bench.pkl"), enable_hopfield=False)
    E = mem._emb_matrix()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    now = time.time()
    last = np.array([m.last_accessed for m in mem.memories], dtype=np.float64)
    taus = np.array([mem._decay_tau_for(m) for m in mem.memories], dtype=np.float64)
    acc = np.array([m.access_count for m in mem.memories], dtype=np.float64)
    dw = np.minimum(((1.0 + (now - last) / taus) ** -0.5)
                    * (1.0 + 0.5 * np.log2(1.0 + acc)), 1.0)
    src_w = np.array([SOURCE_WEIGHT.get(m.source, 1.0) for m in mem.memories],
                     dtype=np.float32)
    print(f"library {mem.size}; dw p10/p50/p90 = "
          f"{np.percentile(dw,10):.2f}/{np.percentile(dw,50):.2f}/"
          f"{np.percentile(dw,90):.2f}", flush=True)

    with open(os.path.join(_HERE, "data_longmemeval", "longmemeval_oracle.json"),
              encoding="utf-8") as f:
        data = json.load(f)

    POOL, RRF_K, K = 200, 60, 3
    hit = {"rrf": 0, "state10": 0, "state20": 0}
    t0 = time.time()
    for qi, q in enumerate(data):
        q_text = q["question"]
        answer_sessions = set(q.get("answer_session_ids", []))
        chunks = split_chunks(q_text) or [q_text]
        Q = np.stack([_embed(c) for c in chunks])
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sims = (qn @ En.T).max(axis=0)
        dense_order = np.argsort(-sims)[:POOL]
        lex_vec = mem._lex_scores(chunks)
        bm_order = np.argsort(-lex_vec)[:POOL]
        fused = {}
        for rank, idx in enumerate(dense_order):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(bm_order):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
        rrf_order = sorted(fused, key=lambda i: -fused[i])
        eff_vec = dw * src_w * (sims + LEX_WEIGHT * lex_vec)

        hit["rrf"] += any(set(mem.memories[int(i)].tags) & answer_sessions
                          for i in rrf_order[:K])
        for pool_n, arm in ((10, "state10"), (20, "state20")):
            pool = rrf_order[:pool_n]
            state_order = sorted(pool, key=lambda i: -eff_vec[i])
            hit[arm] += any(set(mem.memories[int(i)].tags) & answer_sessions
                            for i in state_order[:K])
        if (qi + 1) % 100 == 0:
            print(f"  {qi+1}/{len(data)} ({time.time()-t0:.0f}s) {hit}", flush=True)

    n = len(data)
    for arm in ("rrf", "state10", "state20"):
        print(f"  {arm:8s} {hit[arm]:4d}/{n} = {hit[arm]/n*100:.1f}%")


if __name__ == "__main__":
    main()
