"""Two-stage prefilter scale study: fidelity and latency vs library size.

Random-hyperplane codes are angle-preserving (SimHash property), so the
prefilter premise is expected to hold at scale. Entries are constructed
directly (bypassing dedup — this is a scale benchmark, not a dedup test):
for N in {1k, 2k, 5k, 10k, 20k} synthetic entries:
  - fidelity: |Hamming-top-100 ∩ dense-top-100| / 100
  - latency:  dense full scan vs two-stage (per query, incl. query encoding)

Run: python bench_two_stage_scale.py [--max-n 20000] [--queries 30]
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import SmartMemory, split_chunks, _embed  # noqa: E402


def popcount(x):
    pc = getattr(np, "bitwise_count", None)
    if pc is not None:
        return pc(x).sum(axis=-1, dtype=np.int32)
    return np.unpackbits(x, axis=-1).sum(axis=-1, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=30)
    args = ap.parse_args()

    mem = SmartMemory(n_bits=4096, two_stage=True, code_keep=0.5)
    topics = ["果蝇视觉系统", "打印机色带库存", "跨境电商物流", "GPU 训练任务",
              "MCP 服务运维", "数据库备份策略", "前端界面设计", "音频降噪算法",
              "知识产权法律", "金融风控模型", "农业灌溉系统", "航天材料工程"]
    queries = [f"查询{i}：关于各主题的具体细节与进展" for i in range(args.queries)]

    def add_entry(i):
        """Direct construction: bypasses dedup by design (scale benchmark)."""
        t = topics[i % len(topics)]
        text = f"{t}条目{i}：包含字段 REF{i} 与若干细节描述，用于规模测试。"
        emb = _embed(text)
        from flymemory.v3 import MemoryEntry
        e = MemoryEntry(text=text, response="", embedding=emb,
                        timestamp=time.time(), last_accessed=time.time(),
                        access_count=0, tags=["bench"], memory_id=mem._next_id)
        mem._next_id += 1
        mem.memories.append(e)
        mem._index_entry(e)
        mem._mat_dirty = True
        mem._codes_dirty = True

    for target in [1000, 2000, 5000, 10000, 20000]:
        if args.max_n < target:
            break
        t_store = time.perf_counter()
        while mem.size < target:
            add_entry(mem.size)
        store_s = time.perf_counter() - t_store
        N = mem.size

        # dense full-scan latency (incl. query encoding, like production)
        t0 = time.perf_counter()
        for q in queries:
            q_embs = [_embed(c) for c in split_chunks(q)]
            Q = np.stack(q_embs)
            qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
            M = mem._emb_matrix()
            mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
            sim = (qn @ mn.T).max(axis=0)
            dense_set = set(np.argsort(-sim)[:100].tolist())
        dense_ms = (time.perf_counter() - t0) * 1000 / len(queries)

        # two-stage: fidelity + latency
        fid = []
        t0 = time.perf_counter()
        for q in queries:
            q_embs = [_embed(c) for c in split_chunks(q)]
            Q = np.stack(q_embs)
            qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
            M = mem._emb_matrix()
            mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
            sim = (qn @ mn.T).max(axis=0)
            dense_set = set(np.argsort(-sim)[:100].tolist())
            q_packed = mem._query_code(q_embs)
            ham = popcount(np.bitwise_xor(mem._packed_code_matrix(), q_packed))
            ham_set = set(np.argsort(ham, kind="stable")[:100].tolist())
            fid.append(len(dense_set & ham_set) / 100.0)
        two_ms = (time.perf_counter() - t0) * 1000 / len(queries)

        print(f"N={N:6d}  fidelity@100={np.mean(fid):.4f}  "
              f"dense={dense_ms:7.2f} ms  two-stage={two_ms:7.2f} ms  "
              f"(store {store_s:.0f}s cumulative)", flush=True)


if __name__ == "__main__":
    main()
