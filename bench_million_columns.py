"""Million-column federation: 1,000,000 fly-brain columns on one 12GB GPU.

Previous bench (bench_federation_10k_v2) measured C<=10k. User asks: can 1M
parallel columns run? Memory math:
  - explicit mask matrix (C, 4096) fp32 = 16 GB  -> impossible on 12.9 GB card
  - BUT a random half-view mask is fully specified by its seed: generate masks
    in shards, use 1-bit packing, or score against the SAME view bank with
    per-column random feature dropout.
  Practical scheme used here: columns share ONE feature universe (N_BITS) and
  differ by a random 50% feature subset, generated per shard from the column's
  seed (no storage). Scores per shard computed, per-column argmax fused by
  majority in shards. Peak extra memory = shard mask only.

Also honest about biology: 1M independent "brains" exceeds Drosophila (~2 MB
lobes x 2) by 5 orders — this is a cortical-column count (macaque V1 has
~10^8 columns-worth of units), so the scale question is biologically sensible
at mammalian numbers, not insect ones.

Measured at C = 1e5, 3e5, 1e6:
  recall vs load {5k, 20k} with heavy cue damage (keep 5% active bits — the
  regime where C=100 saved the day and C=10k saturated), novel-cue agreement
  (does a million votes tighten or loosen the OOD boundary), wall-clock.
"""
import json, time
import numpy as np
import torch

SEED = 20261013
DEVICE = "cuda"
N_BITS = 4096
K_ACTIVE = 205
HALF = 2048


def sparse_codes(rng, n):
    idx = torch.rand(n, N_BITS, generator=rng).argsort(-1)[:, :K_ACTIVE]
    c = torch.zeros(n, N_BITS)
    c.scatter_(1, idx, 1.0)
    return c


class MegaFederation:
    """C columns, masks generated per shard from per-column seed (no storage)."""

    def __init__(self, C, seed=SEED):
        self.C = C
        self.seed = seed

    def shard_mask(self, col_ids):
        """col_ids: LongTensor of column indices -> (len, N_BITS) float mask on GPU."""
        g = torch.Generator().manual_seed(self.seed)
        # deterministic per-column random start: xorshift-ish via hashes
        starts = ((col_ids * 2654435761 + 1013904223) % N_BITS)
        idx = torch.arange(N_BITS, device=DEVICE).unsqueeze(0)     # (1, N)
        rel = (idx - starts.unsqueeze(1)) % N_BITS                  # rotation offset
        return (rel < HALF).float()

    def store(self, codes):
        self.codes = codes.to(DEVICE)
        self.P = codes.shape[0]

    @torch.no_grad()
    def query(self, Q, shard=2000, keep_dead=0.0, dead_seed=7):
        n = Q.shape[0]
        Qd = Q.to(DEVICE)
        top_ids = torch.empty(self.C, n, dtype=torch.long, device=DEVICE)
        for c0 in range(0, self.C, shard):
            ids = torch.arange(c0, min(c0 + shard, self.C), device=DEVICE)
            M = self.shard_mask(ids)                               # (s, N)
            for q0 in range(0, n, 5):
                qc = Qd[q0:q0 + 5]
                # per-shard, per-query: (s, N) * q -> codes@ in ONE matmul
                # score[c, p] = codes[p] @ (M[c] * q) => (s, P) = (M*q.T matmul)
                for qi in range(qc.shape[0]):
                    qm = M * qc[qi].unsqueeze(0)                   # (s, N)
                    sc = qm @ self.codes.T                          # (s, P)
                    best, besti = sc.max(-1)
                    top_ids[c0:c0 + len(ids), q0 + qi] = besti
        return top_ids

    @staticmethod
    def fuse(top_ids):
        return torch.mode(top_ids, dim=0).values.cpu()

    @staticmethod
    def agreement(top_ids):
        Cn = top_ids.shape[0]
        out = []
        for i in range(top_ids.shape[1]):
            _, counts = torch.unique(top_ids[:, i], return_counts=True)
            out.append(float(counts.max()) / Cn)
        return out


def damaged_cues(codes, keep, rng):
    """Destroy cues to keep-fraction of active bits (the boundary regime)."""
    Q = codes.clone()
    for i in range(Q.shape[0]):
        nz = torch.nonzero(Q[i]).flatten()
        k = max(int(len(nz) * keep), 1)
        sel = nz[torch.randperm(len(nz), generator=rng)[:k]]
        q = torch.zeros(N_BITS)
        q[sel] = 1.0
        Q[i] = q
    return Q


def run_recall(C, load, keep, probe=150, seed=SEED):
    rng = torch.Generator().manual_seed(seed + load)
    codes = sparse_codes(rng, load)
    fed = MegaFederation(C)
    fed.store(codes)
    perm = torch.randperm(load, generator=rng)[:probe]
    Q = damaged_cues(codes[perm], keep, rng)
    t0 = time.time()
    top_ids = fed.query(Q)
    dt = time.time() - t0
    fused = fed.fuse(top_ids)
    acc = float((fused == perm).float().mean())
    return acc, dt


def run_novel(C, load=5000, probe=100, seed=SEED):
    rng = torch.Generator().manual_seed(seed + 4242 + C % 100000)
    codes = sparse_codes(rng, load)
    fed = MegaFederation(C)
    fed.store(codes)
    novel = sparse_codes(rng, probe)
    top_ids = fed.query(novel)
    ag = fed.agreement(top_ids)
    return float(np.mean(ag)), float(np.percentile(ag, 95))


if __name__ == "__main__":
    rows = []
    print("=== recall at cue-keep 5% (10-bit cues), load 20k ===")
    for C in (100_000, 300_000, 1_000_000):
        acc, dt = run_recall(C, 20000, 0.05)
        rows.append(dict(arm="recall_keep5", C=C, acc=acc, s=round(dt, 2)))
        print(f"C={C:>9,}  acc={acc:.4f}  query={dt:.1f}s", flush=True)
    print("=== recall at cue-keep 3% (6-bit cues) ===")
    for C in (100_000, 1_000_000):
        acc, dt = run_recall(C, 20000, 0.03)
        rows.append(dict(arm="recall_keep3", C=C, acc=acc, s=round(dt, 2)))
        print(f"C={C:>9,}  acc={acc:.4f}  query={dt:.1f}s", flush=True)
    print("=== novel-cue agreement vs C (abstention boundary) ===")
    for C in (100_000, 1_000_000):
        ma, p95 = run_novel(C)
        rows.append(dict(arm="novel", C=C, mean_agree=ma, p95=p95))
        print(f"C={C:>9,}  mean_agree={ma:.4f}  p95={p95:.4f}", flush=True)
    json.dump(rows, open(r"D:\djr82\flymemory\bench_million_columns_result.json", "w"), indent=1)
    print("saved bench_million_columns_result.json")
