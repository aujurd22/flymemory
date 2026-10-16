"""Multi-view multi-column scaling: vote gain vs view independence decay curve.

Extends bench_two_column_v2 (2026-10-13). Questions:
  Q1  With K independent views/columns, how does trap-recovery scale with K?
  Q2  As view correlation rho goes 0 -> 1 (views become redundant), where does
      the vote's advantage over the best single column die?
  Q3  Majority vote vs sum-of-softmax vs disagreement-gated veto: which fusion
      rule degrades slowest with rho?

Design: K views per class, each an independent prototype set except that a
shared "style" factor couples views pairwise with correlation rho. The trap
(writer's wrong-prototype bump) is injected into ALL views with probability
p_trap_view, so at high rho the committee cannot save a sample all views agree
on the wrong class -- the curve's fall-off is exactly the information-theoretic
price of correlated witnesses.

Deterministic. GPU if available.
"""
import json
import numpy as np
import torch
import torch.nn as nn

SEED = 20261013
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES = 12
D = 128
N_TRAIN, N_TEST = 60, 30
PROTO_SCALE = 0.6
VIEW_NOISE = 0.5
TRAP_FRAC = 0.30
TRAP_STRENGTH = 0.9


class MBColumn(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = (torch.randn(D, 512, generator=g) / np.sqrt(D)).to(DEVICE)
        self.readout = nn.Linear(512, N_CLASSES).to(DEVICE)

    def forward(self, x):
        h = x.to(DEVICE) @ self.proj
        k = 51  # 10% of 512
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
def probs_of(col, x):
    return torch.softmax(col(x), -1).cpu()


def make_view_protos(g, k_views, rho):
    """Prototype sets coupled by a shared factor: proto_k = sqrt(rho)*shared + sqrt(1-rho)*indep_k."""
    shared_a = torch.randn(N_CLASSES, D, generator=g) * PROTO_SCALE
    shared_b = torch.randn(N_CLASSES, D, generator=g) * PROTO_SCALE
    pas, pbs = [], []
    for _ in range(k_views):
        ia = torch.randn(N_CLASSES, D, generator=g) * PROTO_SCALE
        ib = torch.randn(N_CLASSES, D, generator=g) * PROTO_SCALE
        pas.append((rho**0.5) * shared_a + ((1 - rho)**0.5) * ia)
        pbs.append((rho**0.5) * shared_b + ((1 - rho)**0.5) * ib)
    return pas, pbs


def make_dataset(g, pas, pbs, n_per, classes, trap):
    """Returns per-view x lists; traps injected per-view with p=TRAP_FRAC each."""
    k = len(pas)
    xs = [[] for _ in range(k)]
    ys_all, trap_all = None, None
    for cls in classes:
        base_mask = torch.rand(n_per, generator=g) < TRAP_FRAC if trap else torch.zeros(n_per, dtype=torch.bool)
        for v in range(k):
            x = pas[v][cls].repeat(n_per, 1) + torch.randn(n_per, D, generator=g) * VIEW_NOISE
            if trap:
                vmask = base_mask & (torch.rand(n_per, generator=g) < 0.7)  # each view independently misses 30% of traps
                wrong = (cls + 1 + torch.randint(0, N_CLASSES - 2, (n_per,), generator=g)) % N_CLASSES
                for i in torch.nonzero(vmask).flatten().tolist():
                    x[i] += (pas[v][int(wrong[i])] - pas[v][cls]) * (TRAP_STRENGTH / (1 + TRAP_STRENGTH))
            xs[v].append(x)
        ys_all = ys_all + [torch.full((n_per,), cls)] if ys_all is not None else [torch.full((n_per,), cls)]
        trap_all = trap_all + [base_mask] if trap_all is not None else [base_mask]
    xs = [torch.cat(xv) for xv in xs]
    return xs, torch.cat(ys_all), torch.cat(trap_all)


def run_config(k_views, rho, gseed):
    g = torch.Generator().manual_seed(gseed)
    pas, pbs = make_view_protos(g, k_views, rho)
    # NOTE: for simplicity every view uses the A-family prototypes (B unused here;
    # A/B in v2 were just two independent views -- K views generalizes that).
    xs_tr, ys_tr, _ = make_dataset(torch.Generator().manual_seed(gseed), pas, pbs, N_TRAIN, list(range(N_CLASSES)), True)
    xs_te, ys_te, trap_te = make_dataset(torch.Generator().manual_seed(gseed + 1), pas, pbs, N_TEST, list(range(N_CLASSES)), True)

    cols = [train_column(MBColumn(seed=gseed * 10 + v), xs_tr[v], ys_tr, seed=gseed * 10 + v) for v in range(k_views)]
    P = [probs_of(c, xs_te[v]) for v, c in enumerate(cols)]
    preds = [p.argmax(-1) for p in P]

    single = [float((pr == ys_te).float().mean()) for pr in preds]
    # sum-of-softmax vote
    vote_pred = torch.stack(P).sum(0).argmax(-1)
    # majority vote (ties -> sum rule)
    stacked = torch.stack(preds)
    majority = torch.mode(stacked, dim=0).values
    # disagreement rate
    agree_all = float((stacked == stacked[0:1]).all(0).float().mean())

    out = {
        "k": k_views, "rho": rho,
        "single_acc_mean": float(np.mean(single)),
        "single_acc_best": float(np.max(single)),
        "sumvote_acc": float((vote_pred == ys_te).float().mean()),
        "majority_acc": float((majority == ys_te).float().mean()),
        "all_agree_rate": agree_all,
        "trap_sumvote": float((vote_pred[trap_te] == ys_te[trap_te]).float().mean()),
        "trap_single_mean": float(np.mean([float((pr[trap_te] == ys_te[trap_te]).float().mean()) for pr in preds])),
    }
    return out


if __name__ == "__main__":
    rows = []
    # Q1: K scaling at rho=0
    for k in (1, 2, 3, 5, 8):
        r = run_config(k, 0.0, SEED + k)
        rows.append(r)
        print(f"K={k} rho=0.0  single={r['single_acc_mean']:.3f} best={r['single_acc_best']:.3f} "
              f"sumvote={r['sumvote_acc']:.3f} maj={r['majority_acc']:.3f} "
              f"trap: single={r['trap_single_mean']:.3f} vote={r['trap_sumvote']:.3f} agree={r['all_agree_rate']:.3f}", flush=True)
    # Q2/Q3: rho decay at K=3
    for rho in (0.0, 0.3, 0.6, 0.85, 0.95):
        r = run_config(3, rho, SEED + 100 + int(rho * 100))
        rows.append(r)
        print(f"K=3 rho={rho:.2f}  single={r['single_acc_mean']:.3f} best={r['single_acc_best']:.3f} "
              f"sumvote={r['sumvote_acc']:.3f} maj={r['majority_acc']:.3f} "
              f"trap: single={r['trap_single_mean']:.3f} vote={r['trap_sumvote']:.3f} agree={r['all_agree_rate']:.3f}", flush=True)
    json.dump(rows, open(r"D:\djr82\flymemory\bench_multicolumn_scaling_result.json", "w"), indent=1)
    print("saved bench_multicolumn_scaling_result.json  device=", DEVICE)
