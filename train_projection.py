"""Learn-to-hash projection training (BRE-style, Kulis & Darrell 2009).

Learns a 384 -> n_bits projection P so that Hamming distance between binary
codes (top `keep` fraction of P@x) approximates the dense cosine distance of
the source embeddings. This makes the two-stage Hamming prefilter order-safe:
Hamming ranking approximates dense ranking.

Trained on the real library embeddings plus jittered copies for local coverage.
Pure CPU torch; minutes to train.

Run:  python train_projection.py [--pkl PATH] [--out PATH] [--epochs 30]
Pre-registered adoption rule: adopt only if prefilter fidelity (Hamming top-100
preserving dense top-100) improves by > +0.03 over the random projection at the
same keep, measured by bench_prefilter_sweep.py logic on held-out queries.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(_HERE, "flymemory", "flymemory_v3.pkl"))
    ap.add_argument("--out", default=os.path.join(_HERE, "flymemory", "learned_projection.npz"))
    ap.add_argument("--n-bits", type=int, default=4096)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pairs-per-epoch", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    import pickle
    import torch

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    E = np.stack([m["embedding"] for m in data["memories"]]).astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    print(f"base embeddings: {E.shape}")

    # augmentation: jittered copies teach the local geometry around each point
    rng = np.random.default_rng(0)
    jitters = []
    for eps in (0.02, 0.08, 0.20):
        j = E + rng.standard_normal(E.shape).astype(np.float32) * eps
        jitters.append(j / (np.linalg.norm(j, axis=1, keepdims=True) + 1e-8))
    X = np.concatenate([E] + jitters, axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    n = X.shape[0]
    print(f"training points (with jitter): {n}")

    dev = "cpu"
    Xt = torch.tensor(X, device=dev)

    def cos_t(a, b):
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)

    # model: linear projection; code = top-`keep` fraction by projection value
    torch.manual_seed(0)
    P = torch.randn(384, args.n_bits, device=dev) * (1.0 / np.sqrt(384))
    P.requires_grad_(True)
    k = max(int(args.n_bits * args.keep), 1)

    opt = torch.optim.Adam([P], lr=args.lr)

    def codes_for_rows(rows):
        """Differentiable-ish code similarity: soft binary via sigmoid scaling.
        We use a straight-through trick: hard threshold in forward via surrogate,
        gradient flows through the pre-activation values."""
        vals = rows @ P                                  # (B, n_bits)
        if args.keep >= 1.0:
            keep_mask = torch.ones_like(vals)
        else:
            thresh = vals.kthvalue(int(vals.shape[1] * (1.0 - args.keep)) + 1, dim=1).values
            keep_mask = (vals >= thresh.unsqueeze(1)).float()
        bits = torch.sigmoid((vals - vals.mean(dim=1, keepdim=True)) * 4.0)
        bits = bits * keep_mask + (1 - keep_mask) * 0.0
        bits = bits * 2.0 - 1.0                          # [-1, 1], sign-like
        return bits

    def hamming_frac(a_codes, b_codes):
        return (1.0 - (a_codes * b_codes).mean(dim=-1)) / 2.0   # in [0,1]

    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        n_pairs = min(args.pairs_per_epoch, n * 2)
        ii = perm[torch.randperm(n)[:n_pairs]]
        jj = perm[torch.randperm(n)[:n_pairs]]
        xa, xb = Xt[ii], Xt[jj]

        ca = codes_for_rows(xa)
        cb = codes_for_rows(xb)
        ham_pred = hamming_frac(ca, cb)

        with torch.no_grad():
            dense_dist = (1.0 - cos_t(xa, xb)) / 2.0     # target Hamming fraction
        loss = torch.mean((ham_pred - dense_dist) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}/{args.epochs}  BRE loss {loss.item():.5f}")

    Pn = P.detach().cpu().numpy().astype(np.float32)
    np.savez(args.out, P=Pn, keep=np.float32(args.keep), n_bits=np.int64(args.n_bits),
             trained="2026-09-22 BRE (jitter-augmented library embeddings)")
    print(f"saved projection -> {args.out}  shape={Pn.shape}")


if __name__ == "__main__":
    main()
