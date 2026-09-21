"""Two-stage prefilter scale study, v2: diverse synthetic entries.

v1 used templated near-identical texts, which made the fidelity@100 metric
meaningless at large N (any 100 of thousands of same-topic near-duplicates is
"correct"). v2 generates semantically DIVERSE entries by combinatorial
composition from word lists (topic x aspect x entity x detail), so the dense
top-100 at scale is well-defined and the prefilter fidelity is a real number.

Measured per scale point:
  - prefilter fidelity@100: |Hamming-top100 ∩ dense-top100| / 100
  - stage-1 Hamming scan latency (XOR + popcount, no rerank)
  - dense full-scan latency (encode + matmul + argsort)

Run: python bench_two_stage_scale.py [--max-n 20000] [--queries 20]
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


TOPICS = ["果蝇神经回路", "打印机耗材", "跨境电商物流", "GPU 训练调度",
          "MCP 服务运维", "数据库备份策略", "前端界面设计", "音频降噪算法",
          "知识产权合规", "金融风控模型", "农业灌溉系统", "航天材料工程",
          "古建筑修缮", "海洋生态调查", "咖啡烘焙曲线", "地铁隧道监测"]
ASPECTS = ["成本结构", "延迟瓶颈", "容错机制", "数据口径", "验收标准",
           "故障复盘", "容量规划", "灰度方案", "供应商对比", "参数标定"]
ENTITIES = ["华东节点", "备用集群", "老陈的工位", "三号机房", "华东二区",
            "采购部台账", "夜间值班组", "样机 B 台架", "海外仓 W2", "测试车厢"]
DETAILS = ["上季度复盘时被点名", "与供应商签约前需二次核对", "凌晨三点的告警里出现过",
           "在白板推演中被画了圈", "昨天的例会只讨论了一半", "归档前必须找老周签字",
           "和去年那起事故同源", "新的值班表还没排进来", "账上显示已报废但实物还在",
           "接口文档里写的是另一套口径"]


def make_text(i, rng):
    t = TOPICS[i % len(TOPICS)]
    a = ASPECTS[rng.integers(0, len(ASPECTS))]
    e = ENTITIES[rng.integers(0, len(ENTITIES))]
    d = DETAILS[rng.integers(0, len(DETAILS))]
    return f"{t}的{a}：涉及{e}，{d}。记录编号 F{i:06d}。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=20)
    args = ap.parse_args()

    mem = SmartMemory(n_bits=4096, two_stage=True, code_keep=0.5)
    rng = np.random.default_rng(7)
    queries = [f"{TOPICS[rng.integers(0, len(TOPICS))]}的{ASPECTS[rng.integers(0, len(ASPECTS))]}现状如何"
               for _ in range(args.queries)]

    for target in [1000, 2000, 5000, 10000, 20000]:
        if args.max_n < target:
            break
        t_store = time.perf_counter()
        # direct construction: a scale benchmark must bypass dedup (templated
        # texts are mutually >0.92 similar and would merge forever)
        from flymemory.v3 import MemoryEntry
        now = time.time()
        while mem.size < target:
            i = mem.size
            text = make_text(i, rng)
            emb = _embed(text)
            e = MemoryEntry(text=text, response="", embedding=emb,
                            timestamp=now, last_accessed=now, access_count=0,
                            tags=["bench"], memory_id=mem._next_id)
            mem._next_id += 1
            mem.memories.append(e)
            mem._index_entry(e)
            mem._mat_dirty = True
            mem._codes_dirty = True
        store_s = time.perf_counter() - t_store
        N = mem.size

        fid_list, ham_total, dense_total = [], 0.0, 0.0
        for q in queries:
            chunks = split_chunks(q) or [q]
            q_embs = [_embed(c) for c in chunks]
            Q = np.stack(q_embs)
            qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
            M = mem._emb_matrix()
            mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

            t0 = time.perf_counter()
            sim = (qn @ mn.T).max(axis=0)
            dense_set = set(np.argsort(-sim)[:100].tolist())
            dense_total += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            q_packed = mem._query_code(q_embs)
            ham = popcount(np.bitwise_xor(mem._packed_code_matrix(), q_packed))
            ham_set = set(np.argsort(ham, kind="stable")[:100].tolist())
            ham_total += (time.perf_counter() - t0) * 1000

            fid_list.append(len(dense_set & ham_set) / 100.0)

        nq = max(len(queries), 1)
        print(f"N={N:6d}  fidelity@100={np.mean(fid_list):.4f}  "
              f"hamming={ham_total / nq:6.2f} ms  "
              f"dense={dense_total / nq:6.2f} ms  "
              f"(store {store_s:.0f}s cumulative)", flush=True)


if __name__ == "__main__":
    main()
