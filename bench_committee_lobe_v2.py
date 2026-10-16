"""Committee-of-lobe experiment v2: one-step linear readout (MBON-style).

v1 post-mortem (recorded 2026-10-12): the iterative Hopfield recall in
flymemory/hopfield.py collapses sparse {+1 heavy, -1 light} bipolar codes onto
the all-+1 attractor (sign(W@s) iterated => majority-state fixation; bias is
the classic Hopfield imbalance failure). Neither centered Hebbian nor the
projection rule rescues sparse codes under sign-iteration. One-step linear
readout + nearest-neighbor scoring — functionally what MBONs do (fixed sparse
random synaptic weights, no recurrent loop) — gives 200/200 corrupted-cue and
200/200 partial-cue recall at P=50, N=2048, K=102 (5% active).

This bench therefore asks the committee question with the correct readout:

  single    : one readout matrix holding all P patterns
  routed    : production-style hash routing to 8 compartments (each query
              scored only against its routed compartment)  [fly-accurate]
  committee : K lobes, each with an INDEPENDENT RANDOM PROJECTION of every
              pattern (lobes see different features, like parallel MBON
              compartments with different connectivity), decision by overlap
              vote; disagreement reported as abstention signal

Key claim under test: does per-lobe random feature diversity (committee)
beat a single full-feature readout under correlated interference, and does
cross-lobe disagreement detect out-of-distribution cues (novel patterns)?

Deterministic, CPU-only, no network.
"""
import sys, json
sys.path.insert(0, r"D:\djr82\flymemory")
import numpy as np

N_BITS = 2048
K_ACTIVE = 102          # 5% active bits per code
K_LOBES = 5
LOADS = [50, 150, 300, 450, 600, 800, 1200]
N_PROBES = 150
SEED = 20261012


def sparse_code(rng):
    code = np.full(N_BITS, -1.0, dtype=np.float32)
    code[rng.choice(N_BITS, size=K_ACTIVE, replace=False)] = 1.0
    return code


def corrupt(code, rng, frac):
    c = code.copy()
    c[rng.choice(N_BITS, size=int(N_BITS * frac), replace=False)] *= -1.0
    return c


def partial(code, rng, keep_frac):
    c = -np.ones(N_BITS, dtype=np.float32)
    keep = rng.choice(N_BITS, size=int(N_BITS * keep_frac), replace=False)
    c[keep] = code[keep]
    return c


def scores(X, q):
    return (X @ q) / (np.linalg.norm(X, axis=1) + 1e-9)


class SingleLobe:
    def __init__(self, rng):
        self.X = []

    def store(self, code):
        self.X.append(code)

    def query(self, cue):
        if not self.X:
            return -1, -1.0
        s = scores(np.array(self.X), cue)
        return int(np.argmax(s)), float(s.max())


class RoutedFly:
    """Production CompartmentalMemory analogue: hash-route to 8 compartments."""
    def __init__(self, rng, k=8):
        self.comps = [[] for _ in range(k)]

    def store(self, code):
        cid = hash(code.tobytes()) % len(self.comps)
        self.comps[cid].append(code)

    def query(self, cue):
        best, best_ov = -1, -1.0
        for lst in self.comps:
            if not lst:
                continue
            s = scores(np.array(lst), cue)
            i = int(np.argmax(s))
            if s[i] > best_ov:
                best_ov, best = float(s[i]), i
        return best, best_ov


class Committee:
    """K lobes, each with its own random feature projection phi_k of patterns."""
    def __init__(self, rng, k=K_LOBES):
        self.projs = [rng.standard_normal((N_BITS // 2, N_BITS)).astype(np.float32) / np.sqrt(N_BITS)
                      for _ in range(k)]
        self.banks = [[] for _ in range(k)]

    def store(self, code):
        for k in range(K_LOBES):
            self.banks[k].append(self.projs[k] @ code)

    def query(self, cue):
        per_lobe_best = []
        for k in range(K_LOBES):
            B = np.array(self.banks[k])
            qs = self.projs[k] @ cue
            s = (B @ qs) / (np.linalg.norm(B, axis=1) + 1e-9)
            per_lobe_best.append((int(np.argmax(s)), float(s.max())))
        ids = [i for i, _ in per_lobe_best]
        vals, counts = np.unique(ids, return_counts=True)
        top = int(vals[np.argmax(counts)])
        agreement = float(counts.max() / len(ids))
        return top, float(np.mean([o for _, o in per_lobe_best])), agreement


def run_arm(arm_cls, load, rng, committee=False):
    arm = arm_cls(np.random.default_rng(SEED))
    codes = [sparse_code(rng) for _ in range(load)]
    for c in codes:
        arm.store(c)

    hit = dict(corrupt=0, partial=0, novel_false=0)
    novel_agree = []
    for _ in range(N_PROBES):
        tgt_i = int(rng.integers(load))
        if committee:
            idx, _, _ = arm.query(corrupt(codes[tgt_i], rng, 0.25))
        else:
            idx, _ = arm.query(corrupt(codes[tgt_i], rng, 0.25))
        hit["corrupt"] += (idx == tgt_i)
        tgt_i = int(rng.integers(load))
        if committee:
            idx, _, _ = arm.query(partial(codes[tgt_i], rng, 0.20))
        else:
            idx, _ = arm.query(partial(codes[tgt_i], rng, 0.20))
        hit["partial"] += (idx == tgt_i)
        novel = sparse_code(rng)
        if committee:
            _, _, ag = arm.query(novel)
            novel_agree.append(ag)
        hit["novel_false"] += 0  # with linear readout, use margin-based check below
    out = dict(n=load,
               corrupt_acc=hit["corrupt"] / N_PROBES,
               partial_acc=hit["partial"] / N_PROBES)
    if committee:
        # abstention quality: on novel cues, how often does the committee fail
        # to agree? high agreement on OOD cues would be bad.
        out["mean_agreement_novel"] = float(np.mean(novel_agree))
        out["disagreement_novel"] = float(1.0 - np.mean(novel_agree))
    return out


def margin_false_alarm(arm_cls, load, rng, committee=False):
    """False-alarm rate via margin: query a novel code; 'recognized' if best
    normalized score exceeds the 5th percentile of within-library best scores."""
    arm = arm_cls(np.random.default_rng(SEED + 1))
    codes = [sparse_code(rng) for _ in range(load)]
    for c in codes:
        arm.store(c)
    if committee:
        known = [arm.query(c)[2] if False else None for c in codes]
    # known-cue score distribution
    known_best = []
    for c in codes:
        if committee:
            _, ov, _ = arm.query(corrupt(c, np.random.default_rng(0), 0.25))
        else:
            _, ov = arm.query(corrupt(c, np.random.default_rng(0), 0.25))
        known_best.append(ov)
    thr = float(np.percentile(known_best, 5))
    fa = 0
    for _ in range(N_PROBES):
        novel = sparse_code(rng)
        if committee:
            _, ov, _ = arm.query(novel)
        else:
            _, ov = arm.query(novel)
        fa += (ov > thr)
    return fa / N_PROBES


if __name__ == "__main__":
    rows = []
    for load in LOADS:
        r1 = run_arm(SingleLobe, load, np.random.default_rng(SEED + load))
        r2 = run_arm(RoutedFly, load, np.random.default_rng(SEED + load))
        r3 = run_arm(Committee, load, np.random.default_rng(SEED + load), committee=True)
        fa1 = margin_false_alarm(SingleLobe, load, np.random.default_rng(SEED + load))
        fa2 = margin_false_alarm(RoutedFly, load, np.random.default_rng(SEED + load))
        fa3 = margin_false_alarm(Committee, load, np.random.default_rng(SEED + load), committee=True)
        rows.append(dict(load=load, single={**r1, "false_alarm": fa1},
                         routed={**r2, "false_alarm": fa2},
                         committee={**r3, "false_alarm": fa3}))
        print(f"load={load:5d}  single corr={r1['corrupt_acc']:.3f} part={r1['partial_acc']:.3f} fa={fa1:.3f} | "
              f"routed {r2['corrupt_acc']:.3f}/{r2['partial_acc']:.3f}/{fa2:.3f} | "
              f"committee {r3['corrupt_acc']:.3f}/{r3['partial_acc']:.3f}/{fa3:.3f} "
              f"disagree_novel={r3.get('disagreement_novel', -1):.3f}", flush=True)
    json.dump(rows, open(r"D:\djr82\flymemory\bench_committee_lobe_result.json", "w"), indent=1)
    print("saved bench_committee_lobe_result.json")
