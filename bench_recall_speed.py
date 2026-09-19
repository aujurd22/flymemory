"""Benchmark: vectorized recall vs per-entry Python loop, on the local library.

Run: python bench_recall_speed.py [--pkl PATH] [--queries 50]
If the library file is missing, falls back to a synthetic library of 2000
entries so the benchmark runs anywhere.
"""
import argparse
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys_dir = os.path.join(_HERE, "flymemory")
import sys  # noqa: E402
sys.path.insert(0, _HERE)
sys.path.insert(0, sys_dir)

from flymemory.v3 import SmartMemory, load, split_chunks, LEX_WEIGHT  # noqa: E402


def brute_force_recall(mem, query, top_k=5):
    q_texts = split_chunks(query)
    q_embs = [mem._encode(qt)[1] for qt in q_texts]
    now = time.time()
    scored = []
    for m in mem.memories:
        sim = max(mem._semantic_similarity(qe, m.embedding) for qe in q_embs)
        eff = sim * mem._decay_weight(m)
        if m.superseded_by is None:
            scored.append((eff, m.memory_id))
    scored.sort(reverse=True)
    return [mid for _, mid in scored[:top_k]], now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(_HERE, "flymemory", "flymemory_v3.pkl"))
    ap.add_argument("--queries", type=int, default=50)
    args = ap.parse_args()

    if os.path.exists(args.pkl):
        mem = load(args.pkl)
        print(f"library: {args.pkl} ({mem.size} entries)")
    else:
        print("library file not found; building 2000 synthetic entries")
        mem = SmartMemory(n_bits=4096, enable_hopfield=False)
        rng = np.random.default_rng(0)
        topics = ["果蝇视觉系统", "打印机色带库存", "跨境电商物流", "GPU 训练任务",
                  "MCP 服务运维", "数据库备份策略", "前端界面设计", "音频降噪算法"]
        for i in range(2000):
            t = f"{topics[i % len(topics)]}条目{i}：包含一些细节描述与编号 REF{i}。"
            mem.remember(t, source="bench")
    n = mem.size

    queries = [f"第{i}个测试查询：关于各主题的现状与进展如何" for i in range(args.queries)]

    # warm-up (model load, matrix build, index build)
    mem.recall(queries[0], top_k=5)

    # --- scoring-only comparison: encode all queries once, outside the timer ---
    from flymemory.v3 import split_chunks as sc
    encoded = []
    for q in queries:
        q_texts = sc(q)
        t0 = time.perf_counter()
        q_embs = [mem._encode(qt)[1] for qt in q_texts]
        encoded.append((q_texts, q_embs, time.perf_counter() - t0))
    enc_ms = sum(t for _, _, t in encoded) * 1000 / len(queries)

    def score_loop(q_embs):
        now = time.time()
        best = {}
        for m in mem.memories:
            sim = max(float(qe @ m.embedding) /
                      (np.linalg.norm(qe) * np.linalg.norm(m.embedding) + 1e-8)
                      for qe in q_embs)
            if m.superseded_by is None:
                best[m.memory_id] = sim * mem._decay_weight(m)
        return sorted(best.items(), key=lambda x: -x[1])[:5]

    t0 = time.perf_counter()
    for _, q_embs, _ in encoded:
        score_loop(q_embs)
    loop_ms = (time.perf_counter() - t0) * 1000 / len(queries)

    t0 = time.perf_counter()
    for q_texts, q_embs, _ in encoded:
        M = mem._emb_matrix()
        Q = np.stack(q_embs)
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        sim_vec = (qn @ mn.T).max(axis=0)
        now = time.time()
        last = np.array([m.last_accessed for m in mem.memories], dtype=np.float64)
        acc = np.array([m.access_count for m in mem.memories], dtype=np.float64)
        dw = np.minimum(((1.0 + (now - last) / mem.decay_tau) ** -0.5)
                        * (1.0 + 0.5 * np.log2(1.0 + acc)), 1.0)
        lex = mem._lex_scores(q_texts)
        _ = dw * (sim_vec + LEX_WEIGHT * lex)
    vec_ms = (time.perf_counter() - t0) * 1000 / len(queries)

    # end-to-end (encode + score + lex + sort), the number users actually feel
    t0 = time.perf_counter()
    for q in queries:
        mem.recall(q, top_k=5)
    e2e_ms = (time.perf_counter() - t0) * 1000 / len(queries)

    print(f"N={n}  queries={len(queries)}")
    print(f"query encoding (model, CPU) : {enc_ms:8.1f} ms/query  <- dominates at small N")
    print(f"scoring: per-entry loop     : {loop_ms:8.1f} ms/query")
    print(f"scoring: vectorized matmul  : {vec_ms:8.1f} ms/query  -> {loop_ms / max(vec_ms, 0.01):.0f}x on scoring")
    print(f"end-to-end recall()         : {e2e_ms:8.1f} ms/query")
    print(f"projected scoring loop at 100k entries: {loop_ms * 100000 / n / 1000:.1f} s/query "
          f"(vectorized: {vec_ms * 100000 / n / 1000:.2f} s)")


if __name__ == "__main__":
    main()
