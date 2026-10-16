"""Scale 100k x learning-rule sweep (user: 10万规模 + 尽可能多的学习方式).

Extends bench_scale_learning (C<=1024, readout-only learning). Changes:
  Scale  : C in {1, 8, 64, 512} columns; task enlarged so C=512 sees 100k+
           total training patterns (50 classes x 2048 = 102,400 patterns).
  Rules  : five learning modes, all same architecture per column
           (frozen-or-trainable projection D->512, k-WTA 10%, linear readout):
     R1 readout    : projection frozen, readout trained (MB-accurate; prev bench)
     R2 full       : projection + readout both trained end-to-end
     R3 lowpass    : projection trained with 5x smaller LR (bio: slow structural
                     plasticity vs fast synaptic plasticity)
     R4 twostage   : readout first (fast), then unfreeze projection briefly
                     (bio: MBON first, calyx tuning later)
     R5 frozen     : no training at all (retrieval control; prototype scoring)

Task stays 50-class, D=32 input (bayes ~10%) but train per class x40 -> x2048,
so the data-poor excuse disappears: with 100k patterns, R2/R3/R4 get their fair
shot. Question: which rule wins at small C vs large C — does the ranking
itself change with scale (that would be a scale effect on learning mode)?

Also measured: fused vs single accuracy for each rule at each C.

GPU required for C=512 x R2 (512 columns x end-to-end training on 100k
patterns, epochs=3 — budgeted).
"""
import json, time
import numpy as np
import torch
import torch.nn as nn

SEED = 20261013
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES = 50
D = 32
N_TRAIN, N_TEST = 2048, 128       # per class -> 102,400 train patterns total
PROTO_SCALE = 0.6
VIEW_NOISE = 0.9
EXP_W = 512
SPARSITY = 0.10
EPOCHS_R1 = 6
EPOCHS_R2 = 3
EPOCHS_R4B = 1


def make_protos(g):
    return (torch.randn(N_CLASSES, D, generator=g) * PROTO_SCALE).to(DEVICE)


def make_split(g, protos, n_per):
    cls = torch.arange(N_CLASSES).repeat_interleave(n_per)
    noise = torch.randn(len(cls), D, generator=g)
    x = protos.cpu()[cls] + noise * VIEW_NOISE
    return x.to(DEVICE), cls.to(DEVICE)


class MBColumn(nn.Module):
    def __init__(self, seed, trainable_proj=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        proj = (torch.randn(D, EXP_W, generator=g) / np.sqrt(D)).to(DEVICE)
        self.proj = nn.Parameter(proj, requires_grad=trainable_proj)
        self.k = max(int(EXP_W * SPARSITY), 1)
        self.readout = nn.Linear(EXP_W, N_CLASSES).to(DEVICE)

    def trunk(self, x):
        h = x @ self.proj
        thresh = torch.topk(h.abs(), self.k, dim=-1).values[..., -1:]
        return h * (h.abs() >= thresh)

    def forward(self, x):
        return self.readout(self.trunk(x))


def train_col(col, x, ys, epochs, lr_proj, lr_ro, bs=1024, seed=0):
    torch.manual_seed(seed)
    opts = []
    if lr_proj:
        opts.append({"params": [col.proj], "lr": lr_proj})
    opts.append({"params": col.readout.parameters(), "lr": lr_ro})
    opt = torch.optim.Adam(opts)
    lossf = nn.CrossEntropyLoss()
    n = len(ys)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(col(x[idx]), ys[idx])
            loss.backward()
            opt.step()
    return col


def train_rule(col, x, ys, rule, seed):
    if rule == "R1_readout":
        return train_col(col, x, ys, EPOCHS_R1, 0.0, 3e-3, seed=seed)
    if rule == "R2_full":
        return train_col(col, x, ys, EPOCHS_R2, 3e-3, 3e-3, seed=seed)
    if rule == "R3_lowpass":
        return train_col(col, x, ys, EPOCHS_R2, 6e-4, 3e-3, seed=seed)
    if rule == "R4_twostage":
        train_col(col, x, ys, EPOCHS_R1, 0.0, 3e-3, seed=seed)
        return train_col(col, x, ys, EPOCHS_R4B, 3e-3, 1e-3, seed=seed + 1)
    raise ValueError(rule)


def frozen_score(col, x, xs, ys):
    with torch.no_grad():
        H = col.trunk(xs)
        protos = torch.stack([H[ys == c].mean(0) for c in range(N_CLASSES)])
        hn = col.trunk(x)
        hn = hn / (hn.norm(dim=-1, keepdim=True) + 1e-9)
        pn = protos / (protos.norm(dim=-1, keepdim=True) + 1e-9)
        return hn @ pn.T


@torch.no_grad()
def acc_of(logits, ys):
    return float((logits.argmax(-1) == ys).float().mean())


def run_arm(C, rule, gseed):
    g = torch.Generator().manual_seed(gseed)
    protos = make_protos(g)
    xtr, ytr = make_split(g, protos, N_TRAIN)
    xte, yte = make_split(g, protos, N_TEST)
    t0 = time.time()

    if rule == "R5_frozen":
        cols = [MBColumn(gseed * 1000 + j) for j in range(C)]
        for c in cols:
            for p in c.parameters():
                p.requires_grad_(False)
        P = torch.stack([frozen_score(c, xte, xtr, ytr) for c in cols]).mean(0)
        fused = acc_of(P, yte)
        singles = [acc_of(frozen_score(c, xte, xtr, ytr), yte) for c in cols]
        return fused, float(np.mean(singles)), time.time() - t0

    cols = []
    for j in range(C):
        col = MBColumn(gseed * 1000 + j, trainable_proj=(rule in ("R2_full", "R3_lowpass", "R4_twostage")))
        train_rule(col, xtr, ytr, rule, gseed * 1000 + j)
        cols.append(col)
    P = torch.stack([torch.softmax(col(xte), -1) for col in cols]).mean(0)
    fused = acc_of(P, yte)
    singles = [acc_of(col(xte), yte) for col in cols]
    return fused, float(np.mean(singles)), time.time() - t0


if __name__ == "__main__":
    rows = []
    rules = ["R1_readout", "R2_full", "R3_lowpass", "R4_twostage", "R5_frozen"]
    print(f"train patterns total = {N_CLASSES * N_TRAIN:,}")
    print("C     " + "  ".join(f"{r:>12s}" for r in rules) + "   (fused; single in parens)")
    for C in (1, 8, 64, 512):
        cells = []
        for rule in rules:
            f, s, dt = run_arm(C, rule, SEED + C)
            rows.append(dict(C=C, rule=rule, fused=f, single=s, seconds=round(dt, 1)))
            cells.append(f"{f:.3f}({s:.3f})")
        print(f"{C:5d}  " + "  ".join(f"{c:>12s}" for c in cells), flush=True)
        json.dump(rows, open(r"D:\djr82\flymemory\bench_scale100k_rules_result.json", "w"), indent=1)
    print("saved bench_scale100k_rules_result.json  device=", DEVICE)
