"""Cross-encoder reranking experiment: does reranking the top-10 improve QA scores?

For each of the 20 QA questions:
  1. Get top-10 from flymemory recall (RRF fusion path)
  2. Rerank with cross-encoder: score(query, candidate_text) for each candidate
  3. Take top-3 after reranking
  4. Check if the answer entry is in the reranked top-3

Compare: baseline (no rerank) vs reranked.

Run: python bench_rerank.py [--pkl longmemeval_bench.pkl] [--qa qa_sample.json]
(CUDA_VISIBLE_DEVICES= prefix to force CPU when a training job holds the GPU.)
"""
import argparse
import json
import os
import sys

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="longmemeval_s_bench.pkl")
    ap.add_argument("--qa", default="qa_sample.json")
    args = ap.parse_args()

    mem = load(os.path.join(_HERE, args.pkl), enable_hopfield=False)
    E = mem._emb_matrix()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    entry_tokens = [set(_tokenize(m.text)) for m in mem.memories]
    df = {}
    for toks in entry_tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n_docs = len(entry_tokens)

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

    with open(os.path.join(_HERE, args.qa), encoding="utf-8") as f:
        questions = json.load(f)

    K = 3
    base_pass = rerank_pass = oracle_pass = 0
    for qi, q in enumerate(questions):
        q_text = q.get("q") or q.get("question")
        gold = q.get("gold")
        answer_sessions = set(q.get("ans_sessions") or q.get("answer_session_ids", []))
        chunks = split_chunks(q_text) or [q_text]
        q_embs = [_embed(c) for c in chunks]
        Q = np.stack(q_embs)
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sims = (qn @ En.T).max(axis=0)

        # RRF fusion (dense + BM25)
        dense_order = np.argsort(-sims)[:200]
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
        bm_order = sorted(scored, key=lambda i: -scored[i])[:200]

        fused = {}
        for rank, idx in enumerate(dense_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(bm_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
        fused_order = sorted(fused, key=lambda i: -fused[i])

        # cross-encoder rerank: take top-10 from fused, rerank with cross-encoder
        rerank_pool = fused_order[:10]
        pairs = [(q_text, mem.memories[int(i)].text) for i in rerank_pool]
        scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
        rerank_order = [rerank_pool[i] for i in np.argsort(-np.array(scores))[:K]]

        base_hit = any(set(mem.memories[int(i)].tags) & answer_sessions
                       for i in fused_order[:K])
        oracle10 = any(set(mem.memories[int(i)].tags) & answer_sessions
                       for i in fused_order[:10])
        rerank_hit = any(set(mem.memories[int(i)].tags) & answer_sessions
                         for i in rerank_order[:K])
        base_pass += base_hit
        oracle_pass += oracle10
        rerank_pass += rerank_hit
        print(f"  Q{qi+1:2d}: base={base_hit} oracle10={oracle10} rerank={rerank_hit} "
              f"top1_fused={mem.memories[fused_order[0]].text[:35]}", flush=True)

    print(f"\nbaseline (RRF top-{K}):    {base_pass}/{len(questions)}")
    print(f"oracle   (RRF top-10):     {oracle_pass}/{len(questions)}")
    print(f"rerank  (cross-enc top-{K}): {rerank_pass}/{len(questions)}")


if __name__ == "__main__":
    main()
