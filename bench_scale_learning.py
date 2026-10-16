"""Scale x Learning: does LEARNING (not just retrieval) improve with column count?

All previous benches (committee/two-column/multicolumn/federation) measured
RETRIEVAL with frozen random projections and nearest-neighbor readout — no
learning anywhere. This bench adds the learning variable and scans scale:

  Each column: frozen random expansion (D -> 512, the calyx analogue) -> k-WTA
  10% -> TRAINABLE linear readout (the MBON analogue). Trained jointly?

  MB-accurate protocol: columns train INDEPENDENTLY (no inter-column gradients,
  like real MBONs), then fuse at inference by mean softmax. The only thing
  scale changes is the number of independent learners + the diversity of their
  frozen random views.

Arms over C in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}:
  L1 trained-ensemble : C independently trained columns, mean-softmax fusion
  L2 frozen-ensemble  : C untrained columns scoring by prototype similarity
                        (what previous benches did) -- the control
  L3 single-wide      : ONE column whose expansion width = C*512 (same total
                        parameters as L1) -- is scale better spent wide or many?

Task: harder than v2 — 50 classes, view noise up, so single-column learning
leaves headroom for ensembles to close. Accuracy vs C on test set + the
learning curve (train epochs) at each C.

Question answered: does per-column learning accuracy saturate while FUSED
accuracy keeps climbing (scale -> capability via committee learning), and does
the gain beat the same-parameter single-wide column (the "many small learners
vs one big learner" comparison — the fly's actual evolutionary choice).

Deterministic, GPU.
"""
import json, time
import numpy as np
import torch
import torch.nn as nn

SEED = 20261013
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES = 50
D = 32                            # tight input space: real bayes headroom (~10% at noise 0.9)
N_TRAIN, N_TEST = 40, 20          # per class
PROTO_SCALE = 0.6
VIEW_NOISE = 0.9                  # calibrated: nearest-prototype error ~10%
EXP_W = 512                       # per-column expansion width
SPARSITY = 0.10


def make_protos(g):
    return (torch.randn(N_CLASSES, D, generator=g) * PROTO_SCALE).to(DEVICE)


def make_split(g, protos, n_per):
    """All randomness drawn on CPU (generator lives there), then moved to GPU."""
    cls = torch.arange(N_CLASSES).repeat_interleave(n_per)
    noise = torch.randn(len(cls), D, generator=g)
    x = (protos.cpu()[cls] + noise * VIEW_NOISE)
    return x.to(DEVICE), cls.to(DEVICE)


class MBColumn(nn.Module):
    def __init__(self, seed, width=EXP_W, trainable=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = (torch.randn(D, width, generator=g) / np.sqrt(D)).to(DEVICE)
        k = max(int(width * SPARSITY), 1)
        self.k = k
        if trainable:
            self.readout = nn.Linear(width, N_CLASSES).to(DEVICE)
        else:
            self.readout = None
            self.proto_bank = None

    def forward(self, x):
        """Trainable mode: full circuit h = kWTA(expand(x)); logits = readout(h).
        Frozen mode: returns k-WTA features (score_frozen does its own scoring)."""
        h = x @ self.proj
        thresh = torch.topk(h.abs(), self.k, dim=-1).values[..., -1:]
        h = h * (h.abs() >= thresh)
        if self.readout is not None:
            return self.readout(h)
        return h

    # frozen-mode scoring: nearest prototype in k-WTA space
    def init_frozen(self, xs, ys):
        with torch.no_grad():
            H = self.forward(xs)
            protos = torch.stack([H[ys == c].mean(0) for c in range(N_CLASSES)])
            self.proto_bank = nn.Parameter(protos, requires_grad=False)

    @torch.no_grad()
    def score_frozen(self, x):
        h = self.forward(x)
        hn = h / (h.norm(dim=-1, keepdim=True) + 1e-9)
        pn = self.proto_bank / (self.proto_bank.norm(dim=-1, keepdim=True) + 1e-9)
        return hn @ pn.T


def train_column(col, x, ys, epochs=40, lr=3e-3, bs=128, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(col.parameters(), lr=lr)
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


@torch.no_grad()
def eval_logits(col, x):
    return col(x)


def run_arm(C, gseed, mode):
    g = torch.Generator().manual_seed(gseed)
    protos = make_protos(g)
    xtr, ytr = make_split(g, protos, N_TRAIN)
    xte, yte = make_split(g, protos, N_TEST)

    if mode == "single-wide":
        col = MBColumn(gseed, width=C * EXP_W)
        train_column(col, xtr, ytr, seed=gseed)
        acc = float((eval_logits(col, xte).argmax(-1) == yte).float().mean())
        return acc, None
    if mode == "frozen":
        cols = []
        for j in range(C):
            c = MBColumn(gseed * 1000 + j, trainable=False)
            c.init_frozen(xtr, ytr)
            cols.append(c)
        P = torch.stack([c.score_frozen(xte) for c in cols]).mean(0)
        acc = float((P.argmax(-1) == yte).float().mean())
        return acc, None
    # trained ensemble: independent training, mean-softmax fusion
    cols = []
    for j in range(C):
        c = MBColumn(gseed * 1000 + j)
        train_column(c, xtr, ytr, seed=gseed * 1000 + j)
        cols.append(c)
    P = torch.stack([torch.softmax(eval_logits(c, xte), -1) for c in cols]).mean(0)
    acc = float((P.argmax(-1) == yte).float().mean())
    singles = [float((eval_logits(c, xte).argmax(-1) == yte).float().mean()) for c in cols]
    return acc, float(np.mean(singles))


if __name__ == "__main__":
    rows = []
    print("=== accuracy vs column count (50 classes, noisy views) ===")
    print("C      trained_fused  trained_single  frozen_ens  single_wide(same params)")
    for C in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        t0 = time.time()
        tf, ts = run_arm(C, SEED + C, "trained")
        fr, _ = run_arm(C, SEED + C, "frozen")
        sw, _ = run_arm(C, SEED + C, "single-wide") if C <= 256 else (float("nan"), None)
        rows.append(dict(C=C, trained_fused=tf, trained_single=ts, frozen=fr, single_wide=sw))
        print(f"{C:6d}  {tf:.3f}          {ts:.3f}          {fr:.3f}      {sw:.3f}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(r"D:\djr82\flymemory\bench_scale_learning_result.json", "w"), indent=1)
    print("saved bench_scale_learning_result.json  device=", DEVICE)
