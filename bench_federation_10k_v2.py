"""Federation scale test v2: 10 -> 10,000 columns, sparse {0,1} codes (MBON form).

v1 post-mortem: ±1 dense codes have a narrow similarity dynamic range
(all pairwise sims ~0.80±0.004), so the capacity arm measured code-space
geometry, not federation, and novel-cue agreement was a flat artifact.
v2 uses sparse {0,1} codes (N=4096, K=205 active, 5%) with overlap-count
scores — the configuration where one-step linear readout is the biologically
correct MBON analogue and similarity has real dynamic range (self 102 vs
other ~5).

Measured at C in {10, 100, 1000, 10000} columns (each column sees a random
HALF=2048-coordinate view, starts rotated):
  B1 recall vs load (1k/5k/20k) with 25% coordinate dropout on cues
  B2 recall with 20% DEAD columns (federation robustness)
  B3 novel-cue per-query top-agreement (abstention signal) vs C
  P1 wall-clock per query batch vs C (the bandwidth bill)

Fusion: majority over per-column argmax identities.
Deterministic. GPU required for C=10000 arms.
"""
import json, time
import numpy as np
import torch

SEED = 20261013
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BITS = 4096
K_ACTIVE = 205
HALF = 2048


def sparse_codes(rng, n):
    idx = torch.rand(n, N_BITS, generator=rng).argsort(-1)[:, :K_ACTIVE]
    c = torch.zeros(n, N_BITS)
    c.scatter_(1, idx, 1.0)
    return c


class Federation:
    def __init__(self, C, seed):
        g = torch.Generator().manual_seed(seed)
        # v1 post-mortem: (j*HALF) % N_BITS cycles through only TWO distinct
        # views (start 0 and 2048) no matter how large C is. Random starts give
        # every column a genuinely different half-view.
        starts = torch.randint(0, N_BITS, (C,), generator=g).tolist()
        masks = torch.zeros(C, N_BITS)
        for j, s in enumerate(starts):
            s = int(s)
            end = min(s + HALF, N_BITS)
            masks[j, s:end] = 1.0
            if end - s < HALF:                      # wrap around
                masks[j, :HALF - (end - s)] = 1.0
        self.masks = masks.to(DEVICE)
        self.C = C

    def store(self, codes):
        self.codes = codes.to(DEVICE)

    @torch.no_grad()
    def query(self, Q, col_chunk=250, q_chunk=50, dead_frac=0.0, seed=0):
        n = Q.shape[0]
        Q = Q.to(DEVICE)
        top_ids = torch.empty(self.C, n, dtype=torch.long, device=DEVICE)
        for c0 in range(0, self.C, col_chunk):
            mc = self.masks[c0:c0 + col_chunk]
            for q0 in range(0, n, q_chunk):
                qm = (Q[q0:q0 + q_chunk].unsqueeze(1) * mc.unsqueeze(0)).permute(1, 0, 2)
                sc = torch.einsum("cnm,pm->cnp", qm, self.codes)
                best, ids = sc.max(-1)
                top_ids[c0:c0 + col_chunk, q0:q0 + q_chunk] = ids
        if dead_frac:
            g = torch.Generator().manual_seed(seed)
            dead = torch.rand(self.C, generator=g).to(DEVICE) < dead_frac
            top_ids[dead] = -1
        return top_ids

    @torch.no_grad()
    def fuse(self, top_ids):
        """Majority with per-query modal id (ties -> smallest id; measured tie rate separately)."""
        fused = torch.mode(top_ids, dim=0).values
        return fused.cpu()


def run_recall(C, load, probe=200, dropout=0.25, dead=0.0, seed=SEED):
    rng = torch.Generator().manual_seed(seed + load)
    codes = sparse_codes(rng, load)
    fed = Federation(C, seed + C)
    fed.store(codes)
    perm = torch.randperm(load, generator=rng)[:probe]
    Q = codes[perm].clone()
    Q[torch.rand_like(Q) < dropout] = 0.0
    t0 = time.time()
    top_ids = fed.query(Q, dead_frac=dead, seed=seed)
    dt = time.time() - t0
    fused = fed.fuse(top_ids)
    acc = float((fused == perm).float().mean())
    return acc, dt


def run_novel_agreement(C, load=5000, probe=200, seed=SEED):
    rng = torch.Generator().manual_seed(seed + 999 + C)
    codes = sparse_codes(rng, load)
    fed = Federation(C, seed + C + 1)
    fed.store(codes)
    novel = sparse_codes(rng, probe)
    top_ids = fed.query(novel)
    agrees = []
    for i in range(probe):
        _, counts = torch.unique(top_ids[:, i], return_counts=True)
        agrees.append(float(counts.max()) / C)
    return float(np.mean(agrees)), float(np.percentile(agrees, 95))


if __name__ == "__main__":
    rows = []
    print("=== B1 recall vs load (25% coord dropout), majority fusion ===")
    for C in (10, 100, 1000, 10000):
        for load in (1000, 5000, 20000):
            acc, dt = run_recall(C, load)
            rows.append(dict(arm="recall", C=C, load=load, acc=acc, s=round(dt, 3)))
            print(f"C={C:6d} load={load:6d}  acc={acc:.3f}  {dt:.2f}s", flush=True)
    print("=== B2 dead columns 20% (C=1000, load=5000) ===")
    acc, dt = run_recall(1000, 5000, dead=0.20)
    rows.append(dict(arm="dead20", C=1000, load=5000, acc=acc))
    print(f"acc={acc:.3f}", flush=True)
    print("=== B3 novel-cue top-agreement vs C (abstention signal) ===")
    for C in (10, 100, 1000, 10000):
        ma, p95 = run_novel_agreement(C)
        rows.append(dict(arm="novel", C=C, mean_agree=ma, p95=p95))
        print(f"C={C:6d}  novel mean_top_agree={ma:.4f} p95={p95:.4f}", flush=True)
    json.dump(rows, open(r"D:\djr82\flymemory\bench_federation_10k_result.json", "w"), indent=1)
    print("saved bench_federation_10k_result.json  device=", DEVICE)
