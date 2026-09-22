"""Full-scale rerank benchmark on LongMemEval-S (500 questions, 199,509 entries).

Arms (evidence-session hit@3):
  dense      : max-over-chunks cosine
  full       : production scoring, sims * decay_w * source_w (post timestamp repair)
  bm25       : IDF lexical
  rrf        : reciprocal rank fusion (rrf_k=60, pool=200) -- reproduces bench_rrf.py
  rerank     : cross-encoder rerank of rrf top-10 -> top-3
  oracle10   : answer present anywhere in rrf top-10 (rerank ceiling)

Run: python bench_rerank_full.py
Use CPU (CUDA_VISIBLE_DEVICES= python ...) when a training job holds the GPU:
encoder/CE default to CUDA and stall on first inference under VRAM contention
(2026-09-23, _XL24 training made the first run hang after model load).
"""
import json
import os
import sys
import time

# torch defaults to one thread per core; alongside a training job that
# livelocks the first CE predict (measured 2026-09-23: 220s+ vs 0.25s capped)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    pass

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

    now = time.time()
    last = np.array([m.last_accessed for m in mem.memories], dtype=np.float64)
    taus = np.array([mem._decay_tau_for(m) for m in mem.memories], dtype=np.float64)
    acc = np.array([m.access_count for m in mem.memories], dtype=np.float64)
    dw = np.minimum(((1.0 + (now - last) / taus) ** -0.5)
                    * (1.0 + 0.5 * np.log2(1.0 + acc)), 1.0)
    src_w = np.array([{"model": 1.15, "import": 1.15}.get(m.source, 1.0)
                      for m in mem.memories], dtype=np.float32)
    assert not np.isnan(dw).any(), "NaN decay weights -- timestamp repair missing"

    with open(os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json"),
              encoding="utf-8") as f:
        data = json.load(f)

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

    POOL = 200
    RRF_K = 60
    K = 3
    RERANK_POOL = 10
    hit = {"dense": 0, "full": 0, "bm25": 0, "rrf": 0, "rerank": 0, "oracle10": 0}
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
        for c in chunks:
            for tok in _tokenize(c):
                d = df.get(tok, 0)
                idf = np.log(1.0 + n_docs / d) if d else np.log(1.0 + n_docs)
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
        rrf_order = sorted(fused, key=lambda i: -fused[i])

        pool = rrf_order[:RERANK_POOL]
        pairs = [(q_text, mem.memories[int(i)].text) for i in pool]
        ce = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        rerank_order = [pool[i] for i in np.argsort(-np.asarray(ce))[:K]]

        tags_hit = lambda order, k: any(
            set(mem.memories[int(idx)].tags) & answer_sessions for idx in order[:k])
        hit["dense"] += tags_hit(dense_order, K)
        full_order = np.argsort(-(sims * dw * src_w))
        hit["full"] += tags_hit(full_order, K)
        hit["bm25"] += tags_hit(bm_order, K)
        hit["rrf"] += tags_hit(rrf_order, K)
        hit["rerank"] += tags_hit(rerank_order, K)
        hit["oracle10"] += tags_hit(rrf_order, RERANK_POOL)

        if (qi + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {qi+1}/{len(data)} ({el:.0f}s) "
                  + " ".join(f"{k}={v}" for k, v in hit.items()), flush=True)

    n = len(data)
    print(f"\n=== evidence-hit@{K} (N={n}, {N} entries, {time.time()-t0:.0f}s) ===")
    for arm in ("dense", "full", "bm25", "rrf", "rerank", "oracle10"):
        print(f"  {arm:9s} {hit[arm]:4d}/{n} = {hit[arm]/n*100:.1f}%")


if __name__ == "__main__":
    main()
