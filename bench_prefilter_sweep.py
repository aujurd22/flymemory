"""Prefilter-fidelity sparsity sweep (flymemory-native version of FlyPoet's U-curve).

FlyPoet (PAPER.md §3.2) found the k-WTA sweet spot at 25% keep — NOT the 5-10%
suggested by biology alone. FlyMemory's Hamming prefilter codes currently use
5% k-WTA. This benchmark sweeps the keep fraction and measures how well the
Hamming top-C prefilter preserves the dense-cosine top-C (the only job of the
prefilter), on the real library with realistic queries.

Decision rule (pre-registered): adopt the smallest keep fraction whose mean
fidelity is within 0.005 of the best; if 5% is already within 0.005 of the
best, keep 5% and record the negative result.

Run: python bench_prefilter_sweep.py [--pkl PATH] [--queries N]
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

from flymemory.v3 import SmartMemory, load, split_chunks, _embed  # noqa: E402

QUERIES = [
    "小说《固定的比特》写到哪里了",
    "伏笔总账里有哪些伏笔",
    "色带耗材库存预警怎么样了",
    "果蝇视觉系统的假设是什么",
    "GSO1S615F00NT83 这个货号关联了几件",
    "temperature scaling calibration 进展",
    "flypoet k-WTA 的结论有哪些",
    "HKT 登录流程的最新状态",
    "压缩恢复包是怎么工作的",
    "两段式检索的使用前提",
    "ERP 系统的部署注意事项",
    "打印机的库存和耗材",
    "supersede 和 forget 的区别",
    "数据库备份策略讨论",
    "用户偏好什么样的界面设计",
    "PowerShell 脚本编码问题",
    "果蝇蘑菇体的稀疏编码机制",
    "每周例会的时间安排",
    "服务器硬盘空间告警",
    "模型微调的数据泄漏问题",
    "Which OS does the user run now",
    "What is the latest FlyPoet conclusion",
    "How does the Hamming prefilter work",
    "版本号和发布流程",
    "日志文件在哪里",
    "API 接口的鉴权方式",
    "内存占用的优化方法",
    "定时任务的配置方法",
    "翻译一下这段英文文档",
    "把刚才的结论存入记忆",
]


def codes_for(mem, keep_fraction):
    E = mem._emb_matrix()
    code = E @ mem.proj.T
    k = max(min(int(mem.n_bits * keep_fraction), code.shape[1]), 1)
    thresh = np.partition(code, -k, axis=1)[:, -k][:, None]
    bits = code >= thresh
    return bits, np.packbits(bits.astype(np.uint8), axis=1)


def query_code(mem, query, keep_fraction):
    chunks = split_chunks(query) or [query]
    union = np.zeros(mem.n_bits, dtype=bool)
    for c in chunks:
        emb = _embed(c)
        code = mem.proj @ emb
        k = max(min(int(mem.n_bits * keep_fraction), len(code)), 1)
        thresh = np.partition(code, -k)[-k]
        union |= code >= thresh
    return np.packbits(union.astype(np.uint8))


def popcount(x):
    pc = getattr(np, "bitwise_count", None)
    if pc is not None:
        return pc(x).sum(axis=-1, dtype=np.int32)
    return np.unpackbits(x, axis=-1).sum(axis=-1, dtype=np.int32)


def dense_topk(mem, q_embs, k):
    M = mem._emb_matrix()
    qn = np.stack(q_embs) / (np.linalg.norm(np.stack(q_embs), axis=1, keepdims=True) + 1e-8)
    mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    sim = (qn @ mn.T).max(axis=0)
    order = np.argsort(-sim)[:k]
    return list(order.tolist()), sim  # ordered ids (best first), full sim vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(_HERE, "flymemory", "flymemory_v3.pkl"))
    ap.add_argument("--candidates", type=int, default=100)
    args = ap.parse_args()

    if not os.path.exists(args.pkl):
        print(f"library not found: {args.pkl}")
        sys.exit(1)
    mem = load(args.pkl)
    n = mem.size
    print(f"library: {args.pkl} ({n} entries), candidates C={args.candidates}")

    # pre-encode all queries
    q_data = []
    for q in QUERIES:
        chunks = split_chunks(q) or [q]
        q_data.append((q, [_embed(c) for c in chunks]))

    print(f"\n{'keep':>6s} {'fidelity@C':>11s} {'top10-surv':>11s} {'latency ms':>11s}")
    results = {}
    for keep in (0.02, 0.05, 0.10, 0.25, 0.50):
        bits, packed = codes_for(mem, keep)
        qbits = [query_code(mem, q, keep) for q, _ in q_data]
        fid, surv = [], []
        t0 = time.time()
        for (q, q_embs), qb in zip(q_data, qbits):
            dense_ids, _ = dense_topk(mem, q_embs, args.candidates)
            dense_set = set(dense_ids)
            dense_top10 = set(dense_ids[:10])
            ham = popcount(np.bitwise_xor(packed, qb))
            ham_set = set(np.argsort(ham, kind="stable")[:args.candidates].tolist())
            fid.append(len(dense_set & ham_set) / args.candidates)
            surv.append(len(dense_top10 & ham_set) / 10.0)
        ms = (time.time() - t0) * 1000 / len(q_data)
        f, s = float(np.mean(fid)), float(np.mean(surv))
        results[keep] = (f, s)
        print(f"{keep*100:5.0f}% {f:11.4f} {s:11.3f} {ms:11.1f}")

    best = max(results, key=lambda k: results[k][0])
    print(f"\nbest keep = {best*100:.0f}% (fidelity {results[best][0]:.4f})")
    for keep in (0.02, 0.05, 0.10, 0.25, 0.50):
        gap = results[best][0] - results[keep][0]
        verdict = "ADOPT-OK" if gap <= 0.005 else "WORSE"
        print(f"  keep {keep*100:2.0f}%: gap {gap:+.4f} -> {verdict}")


if __name__ == "__main__":
    main()
