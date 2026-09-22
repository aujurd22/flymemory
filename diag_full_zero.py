"""Diagnose why the production-scoring arm (eff = sims * dw * src_w) hit 0/500
on LongMemEval-S while pure dense hit 27%: who occupies the top-3?

Prints, for a few representative questions, the full-arm top-8 with
sim / dw / src_w / eff / text length / timestamp / hit flag.

Run: python diag_full_zero.py
(CUDA_VISIBLE_DEVICES= prefix to force CPU when a training job holds the GPU.)
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import load, split_chunks, _embed  # noqa: E402


def main():
    mem = load(os.path.join(_HERE, "longmemeval_s_bench.pkl"), enable_hopfield=False)
    E = mem._emb_matrix()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    now = time.time()
    last = np.array([m.last_accessed for m in mem.memories], dtype=np.float64)
    taus = np.array([mem._decay_tau_for(m) for m in mem.memories], dtype=np.float64)
    acc = np.array([m.access_count for m in mem.memories], dtype=np.float64)
    dw = np.minimum(((1.0 + (now - last) / taus) ** -0.5)
                    * (1.0 + 0.5 * np.log2(1.0 + acc)), 1.0)
    src_w = np.array([{"model": 1.15, "import": 1.15}.get(m.source, 1.0)
                      for m in mem.memories], dtype=np.float32)
    lens = np.array([len(m.text) for m in mem.memories])

    with open(os.path.join(_HERE, "qa_sample.json"), encoding="utf-8") as f:
        questions = json.load(f)

    for qi in (0, 1, 4, 14, 15):
        q = questions[qi]
        q_text = q["q"]
        answer_sessions = set(q["ans_sessions"])
        chunks = split_chunks(q_text) or [q_text]
        Q = np.stack([_embed(c) for c in chunks])
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sims = (qn @ En.T).max(axis=0)
        eff = sims * dw * src_w

        dense_rank = np.argsort(-sims)
        full_rank = np.argsort(-eff)
        dense_hit3 = any(set(mem.memories[int(i)].tags) & answer_sessions
                         for i in dense_rank[:3])

        print(f"\n=== Q{qi+1} [{q['type']}] {q_text[:70]}")
        print(f"    dense_hit@3={dense_hit3}  answer_sessions={sorted(answer_sessions)[:3]}")
        print(f"    {'rk':>2} {'sim':>6} {'dw':>6} {'eff':>6} {'len':>5} {'date':>11} hit text")
        for rank, idx in enumerate(full_rank[:8], 1):
            m = mem.memories[int(idx)]
            hit = "YES" if set(m.tags) & answer_sessions else "."
            ts = datetime.fromtimestamp(m.last_accessed).strftime("%Y-%m-%d") \
                if m.last_accessed and m.last_accessed > 0 else "?"
            print(f"    {rank:2d} {sims[idx]:6.3f} {dw[idx]:6.3f} {eff[idx]:6.3f} "
                  f"{lens[idx]:5d} {ts:>11} {hit:3s} {m.text[:60]!r}")
        # distribution context
        print(f"    dw range: p10={np.percentile(dw,10):.3f} p50={np.percentile(dw,50):.3f} "
              f"p90={np.percentile(dw,90):.3f}   len p50={np.percentile(lens,50):.0f}")


if __name__ == "__main__":
    main()
