"""Two-column minimal experiment v2: prototype coding (v1 post-mortem: the
slice-offset coding put only 10 coordinates of signal against 128-d unit noise
-> chance-level for every arm). v2 uses class-prototype views with independent
per-view noise, plus a distractor trap injected ONLY into view A.

Claim under test (flymemory #15777 blueprint): two MB-style columns each seeing
ONE view + score vote >= single column on the trapped subset (voting defeats a
local illusion), with per-view agreement doubling as an OOD abstention signal.
"""
import json
import numpy as np
import torch
import torch.nn as nn

SEED = 20261013
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES = 12
D_A, D_B = 128, 128
N_TRAIN, N_TEST = 60, 30          # per class
N_OOD = 4
PROTO_SCALE = 0.6
VIEW_NOISE = 0.5
TRAP_FRAC = 0.30
TRAP_STRENGTH = 0.9               # wrong-prototype bump relative to true signal

def make_protos(g):
    return torch.randn(N_CLASSES, D_A, generator=g) * PROTO_SCALE, torch.randn(N_CLASSES, D_B, generator=g) * PROTO_SCALE

def make_dataset(g, protos_a, protos_b, n_per, classes, trap):
    xa, xb, ys, tm = [], [], [], []
    for cls in classes:
        a = protos_a[cls].repeat(n_per, 1) + torch.randn(n_per, D_A, generator=g) * VIEW_NOISE
        b = protos_b[cls].repeat(n_per, 1) + torch.randn(n_per, D_B, generator=g) * VIEW_NOISE
        mask = torch.rand(n_per, generator=g) < TRAP_FRAC if trap else torch.zeros(n_per, dtype=torch.bool)
        if trap:
            wrong = (cls + 1 + torch.randint(0, N_CLASSES - 2, (n_per,), generator=g)) % N_CLASSES
            for i in torch.nonzero(mask).flatten().tolist():
                a[i] += (protos_a[int(wrong[i])] - protos_a[cls]) * (TRAP_STRENGTH / (1 + TRAP_STRENGTH))
        xa.append(a); xb.append(b)
        ys.append(torch.full((n_per,), cls))
        tm.append(mask)
    return torch.cat(xa), torch.cat(xb), torch.cat(ys), torch.cat(tm)

class MBColumn(nn.Module):
    """calyx expansion -> k-WTA -> plastic readout."""
    def __init__(self, d_in, n_exp=512, sparsity=0.10, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = (torch.randn(d_in, n_exp, generator=g) / np.sqrt(d_in)).to(DEVICE)
        self.sparsity = sparsity
        self.n_exp = n_exp
        self.readout = nn.Linear(n_exp, N_CLASSES).to(DEVICE)

    def forward(self, x):
        h = x.to(DEVICE) @ self.proj
        k = max(int(self.n_exp * self.sparsity), 1)
        thresh = torch.topk(h.abs(), k, dim=-1).values[..., -1:]
        h = h * (h.abs() >= thresh)
        return self.readout(h)

def train_column(col, x, ys, epochs=80, lr=3e-3, bs=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(col.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(ys)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(col(x[idx]), ys[idx].to(DEVICE))
            loss.backward()
            opt.step()
    return col

@torch.no_grad()
def softmax_of(col, x):
    return torch.softmax(col(x), -1).cpu()

def main():
    g = torch.Generator().manual_seed(SEED)
    pa, pb = make_protos(g)
    xa_tr, xb_tr, ys_tr, _ = make_dataset(torch.Generator().manual_seed(SEED), pa, pb, N_TRAIN, list(range(N_CLASSES)), True)
    xa_te, xb_te, ys_te, trap_te = make_dataset(torch.Generator().manual_seed(SEED + 1), pa, pb, N_TEST, list(range(N_CLASSES)), True)
    ood_classes = list(range(N_CLASSES, N_CLASSES + N_OOD))
    go = torch.Generator().manual_seed(SEED + 2)
    ood_pa = torch.randn(N_OOD, D_A, generator=go) * PROTO_SCALE
    ood_pb = torch.randn(N_OOD, D_B, generator=go) * PROTO_SCALE
    ood_a, ood_b, _, _ = make_dataset(go, ood_pa, ood_pb, 20, list(range(N_OOD)), False)

    colA = train_column(MBColumn(D_A, seed=1), xa_tr, ys_tr, seed=1)
    colB = train_column(MBColumn(D_B, seed=2), xb_tr, ys_tr, seed=2)

    sa = softmax_of(colA, xa_te)
    sb = softmax_of(colB, xb_te)
    pred_a, pred_b = sa.argmax(-1), sb.argmax(-1)
    fused = (sa + sb).argmax(-1)

    res = {
        "single_A_acc": float((pred_a == ys_te).float().mean()),
        "single_B_acc": float((pred_b == ys_te).float().mean()),
        "vote_acc": float((fused == ys_te).float().mean()),
        "trap_single_A": float((pred_a[trap_te] == ys_te[trap_te]).float().mean()),
        "trap_single_B": float((pred_b[trap_te] == ys_te[trap_te]).float().mean()),
        "trap_vote": float((fused[trap_te] == ys_te[trap_te]).float().mean()),
        "known_agreement": float((pred_a == pred_b).float().mean()),
    }
    # OOD abstention: columns never trained on OOD classes; their argmaxes should disagree more
    oa = softmax_of(colA, ood_a).argmax(-1)
    ob = softmax_of(colB, ood_b).argmax(-1)
    res["ood_agreement"] = float((oa == ob).float().mean())
    # OOD margin profile vs known margin profile
    res["known_mean_top_margin"] = float(torch.topk(sa + sb, 2, dim=-1).values[:, 0].sub(torch.topk(sa + sb, 2, dim=-1).values[:, 1]).mean())
    res["ood_mean_top_margin"] = float(torch.topk(softmax_of(colA, ood_a) + softmax_of(colB, ood_b), 2, dim=-1).values[:, 0].sub(torch.topk(softmax_of(colA, ood_a) + softmax_of(colB, ood_b), 2, dim=-1).values[:, 1]).mean())

    print(json.dumps(res, indent=1))
    json.dump(res, open(r"D:\djr82\flymemory\bench_two_column_result.json", "w"), indent=1)
    print("saved. device=", DEVICE)

if __name__ == "__main__":
    main()
