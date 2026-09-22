"""D2: connectome-grounded parameterization study.

BANC real parameters (extracted 2026-09-22, connectome_params.json):
  KC=1335, MBON=53, KC->MBON edges=808, convergence 22.4 KC/MBON (median 8),
  contact sparsity 1.14%.

Maps to FlyMemory prefilter codes: n_bits ~= KC population (test 1024/2048/4096;
1335 rounds up to 2048), keep fraction swept around the measured operating
points. Decision rule (pre-registered): the smallest memory footprint (n_bits)
whose mean fidelity@100 is within 0.02 of the best-in-grid at every scale
point.

Run: python bench_connectome_params.py [--max-n 10000]
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


def codes_with_params(E, q_embs_list, n_bits, keep, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    proj = rng.standard_normal((n_bits, E.shape[1])).astype(np.float32) / np.sqrt(E.shape[1])
    code = E @ proj.T
    k = max(int(n_bits * keep), 1)
    thresh = np.partition(code, -k, axis=1)[:, -k][:, None]
    packed = np.packbits((code >= thresh).astype(np.uint8), axis=1)
    q_packed = []
    for q_embs in q_embs_list:
        union = np.zeros(n_bits, dtype=bool)
        for qe in q_embs:
            qc = proj @ qe
            t = np.partition(qc, -k)[-k]
            union |= qc >= t
        q_packed.append(np.packbits(union.astype(np.uint8)))
    return packed, q_packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(_HERE, "flymemory", "flymemory_v3.pkl"))
    ap.add_argument("--max-n", type=int, default=10000)
    ap.add_argument("--queries", type=int, default=20)
    args = ap.parse_args()

    mem = SmartMemory(n_bits=4096)  # embedder path only; grids build their own codes
    rng = np.random.default_rng(7)
    topics = ["果蝇神经回路", "打印机耗材", "跨境电商物流", "GPU 训练调度",
              "MCP 服务运维", "数据库备份策略", "前端界面设计", "音频降噪算法",
              "知识产权合规", "金融风控模型", "农业灌溉系统", "航天材料工程",
              "古建筑修缮", "海洋生态调查", "咖啡烘焙曲线", "地铁隧道监测"]
    aspects = ["成本结构", "延迟瓶颈", "容错机制", "数据口径", "验收标准",
               "故障复盘", "容量规划", "灰度方案", "供应商对比", "参数标定"]
    entities = ["华东节点", "备用集群", "老陈的工位", "三号机房", "华东二区",
                "采购部台账", "夜间值班组", "样机 B 台架", "海外仓 W2", "测试车厢"]
    details = ["上季度复盘时被点名", "与供应商签约前需二次核对", "凌晨三点的告警里出现过",
               "在白板推演中被画了圈", "昨天的例会只讨论了一半", "归档前必须找老周签字",
               "和去年那起事故同源", "新的值班表还没排进来", "账上显示已报废但实物还在",
               "接口文档里写的是另一套口径"]

    # store with the production path (dedup on), diverse texts
    for target in [1000, 2000, 5000, min(args.max_n, 10000)]:
        while mem.size < target:
            t = topics[mem.size % len(topics)]
            a = aspects[mem.size % len(aspects)]
            e = entities[mem.size % len(entities)]
            d = details[mem.size % len(details)]
            mem.remember_text(f"{t}的{a}：涉及{e}，{d}。记录编号 C{mem.size:06d}。",
                              source="bench")
    N = mem.size
    E = mem._emb_matrix()
    queries = [f"{topics[rng.integers(0, len(topics))]}的{aspects[rng.integers(0, len(aspects))]}现状如何"
               for _ in range(args.queries)]
    q_embs_list = [[_embed(c) for c in (split_chunks(q) or [q])] for q in queries]
    print(f"library N={N}, queries={len(queries)}\n")

    grid = [(n_bits, keep)
            for n_bits in (1024, 2048, 4096)
            for keep in (0.05, 0.25, 0.50)]

    print(f"{'n_bits':>7s} {'keep':>6s} {'bytes/entry':>12s} {'fidelity@100':>13s}")
    results = {}
    for n_bits, keep in grid:
        packed, q_packed = codes_with_params(E, q_embs_list, n_bits, keep)
        fids = []
        for q_embs, qb in zip(q_embs_list, q_packed):
            Q = np.stack(q_embs)
            qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
            mn = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
            sim = (qn @ mn.T).max(axis=0)
            dense_top = set(np.argsort(-sim)[:100].tolist())
            ham = popcount(np.bitwise_xor(packed, qb))
            ham_top = set(np.argsort(ham, kind="stable")[:100].tolist())
            fids.append(len(dense_top & ham_top) / 100.0)
        results[(n_bits, keep)] = float(np.mean(fids))
        print(f"{n_bits:7d} {keep*100:5.0f}% {n_bits//8:12d} {np.mean(fids):13.4f}", flush=True)

    best = max(results.values())
    print("\nconnectome-grounded reading (KC=1335 -> n_bits=2048):")
    for (nb, kp), f in sorted(results.items()):
        note = " <-- connectome-scale" if nb == 2048 else ""
        print(f"  n_bits={nb} keep={kp:.2f}: {f:.4f}{note}")
    best_key = max(results, key=results.get)
    print(f"best in grid: n_bits={best_key[0]} keep={best_key[1]} -> {best:.4f}")


if __name__ == "__main__":
    main()
