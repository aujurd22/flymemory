"""Committee-of-lobe experiment: K independent Hopfield lobes vs one big lobe.

Biological grounding: the fly mushroom body has parallel lobes and ~34 MBON
compartments. FlyMemory's CompartmentalMemory routes each memory to ONE
compartment by hash (divide-and-conquer to prevent crosstalk). This bench asks
the complementary question: what does a CONCURRENT committee (all lobes see
every memory; decisions by overlap vote) buy over a single lobe of equal total
capacity, and over the production router?

Arms (same total capacity C = n_bits * ... patterns budget):
  single  : one HopfieldMemory with capacity budget C
  routed  : production CompartmentalMemory (hash route, no vote) -- fly-accurate
  committee: K lobes each storing every pattern (K x C/k each? no -- each lobe
             stores all patterns; total weight memory is K x single), vote by
             mean overlap; also reports disagreement (abstention signal)

Metrics per arm at each load N:
  top1 accuracy with corrupted cues (25% bit flip on stored code)
  top1 accuracy with partial cues (20% bits kept -- pattern completion)
  false-recall rate on novel cues (does it "recognize" something never stored?)
  committee disagreement on novel cues (abstention signal quality)

Deterministic seeds. No network, CPU only.
"""
import sys, json, hashlib
sys.path.insert(0, r"D:\djr82\flymemory")
import numpy as np
from flymemory.hopfield import HopfieldMemory

N_BITS = 2048
SPARSITY = 0.05
K_KEEP = max(int(N_BITS * SPARSITY), 1)
K_LOBES = 5
LOADS = [50, 150, 300, 450, 600, 800]
N_PROBES = 120
SEED = 20261012


def sparse_code(rng):
    """±1 dense code, as Hopfield dynamics require (sign(W@s) needs bipolar states).

    NOTE found en route: TextEncoder.encode emits {0,1} sparse codes where any
    text with < k_keep nonzero bits maps to the ALL-ONES vector (k-WTA threshold
    falls on 0.0), so every short memory shares one code and Hopfield recall in
    the v1/v2 path is degenerate. v3 production (embed+RRF) has Hopfield OFF by
    default and is unaffected. This bench uses correct bipolar codes."""
    code = np.full(N_BITS, -1.0, dtype=np.float32)
    idx = rng.choice(N_BITS, size=K_KEEP, replace=False)
    code[idx] = 1.0
    return code


def corrupt(code, rng, frac):
    c = code.copy()
    flips = rng.choice(N_BITS, size=int(N_BITS * frac), replace=False)
    c[flips] = -c[flips]
    return c


def partial(code, rng, keep_frac):
    c = -np.ones(N_BITS, dtype=np.float32)
    keep = rng.choice(N_BITS, size=int(N_BITS * keep_frac), replace=False)
    c[keep] = code[keep]
    return c


def overlap(a, b):
    return float(np.mean(a == b))


class SingleLobe:
    def __init__(self):
        self.mem = HopfieldMemory(N_BITS, SPARSITY)

    def store(self, code):
        self.mem.store(code)

    def query(self, cue):
        conv, _ = self.mem.recall(cue)
        idx, ov = self.mem.find_nearest(conv)
        return (idx if ov >= 0 else -1), ov


class RoutedFly:
    """Production CompartmentalMemory: hash-route, recall scans all compartments."""
    def __init__(self, k=8):
        self.comps = [HopfieldMemory(N_BITS, SPARSITY) for _ in range(k)]

    def store(self, code):
        cid = hash(code.tobytes()) % len(self.comps)
        self.comps[cid].store(code)

    def query(self, cue):
        best, best_ov = -1, -1.0
        for comp in self.comps:
            conv, _ = comp.recall(cue)
            idx, ov = comp.find_nearest(conv)
            if idx >= 0 and ov > best_ov:
                best_ov, best = ov, idx
        return best, best_ov


class Committee:
    """Every lobe stores every pattern; per-lobe recall; vote by overlap."""
    def __init__(self, k=K_LOBES):
        self.lobes = [HopfieldMemory(N_BITS, SPARSITY) for _ in range(k)]

    def store(self, code):
        for lobe in self.lobes:
            lobe.store(code)

    def query(self, cue):
        votes, ovs = [], []
        for lobe in self.lobes:
            conv, _ = lobe.recall(cue)
            idx, ov = lobe.find_nearest(conv)
            votes.append(idx)
            ovs.append(ov)
        # majority over converged identities + mean overlap
        vals, counts = np.unique(votes, return_counts=True)
        top = vals[np.argmax(counts)]
        agreement = counts.max() / len(votes)
        return int(top), float(np.mean(ovs)), float(agreement)


def run_arm(arm_cls, load, rng, committee=False):
    arm = arm_cls()
    codes = [sparse_code(rng) for _ in range(load)]
    for c in codes:
        arm.store(c)

    res = dict(n=load, corrupt=0, partial=0, novel_false=0, novel_agree=[])
    for i in range(N_PROBES):
        # corrupted-cue recognition
        tgt_i = int(rng.integers(load))
        q = corrupt(codes[tgt_i], rng, 0.25)
        if committee:
            idx, ov, ag = arm.query(q)
        else:
            idx, ov = arm.query(q)
        res["corrupt"] += 1 if idx == tgt_i else 0
        # partial-cue completion
        tgt_i = int(rng.integers(load))
        q = partial(codes[tgt_i], rng, 0.20)
        if committee:
            idx, ov, ag = arm.query(q)
        else:
            idx, ov = arm.query(q)
        res["partial"] += 1 if idx == tgt_i else 0
        # novel cue: false recognition + committee agreement
        novel = sparse_code(rng)
        if committee:
            idx, ov, ag = arm.query(novel)
            res["novel_agree"].append(ag)
            if ov > 0.90:
                res["novel_false"] += 1
        else:
            idx, ov = arm.query(novel)
            if ov > 0.90:
                res["novel_false"] += 1
    n_probe = N_PROBES
    out = dict(n=load,
               corrupt_acc=res["corrupt"] / n_probe,
               partial_acc=res["partial"] / n_probe,
               false_recall=res["novel_false"] / n_probe)
    if committee:
        out["mean_agreement_novel"] = float(np.mean(res["novel_agree"]))
    return out


def codes_index_probe_fix():
    pass


if __name__ == "__main__":
    rows = []
    for load in LOADS:
        r_single = run_arm(SingleLobe, load, np.random.default_rng(SEED + load))
        r_routed = run_arm(RoutedFly, load, np.random.default_rng(SEED + load))
        r_comm = run_arm(Committee, load, np.random.default_rng(SEED + load), committee=True)
        rows.append(dict(load=load, single=r_single, routed=r_routed, committee=r_comm))
        print(f"load={load:4d}  single corr={r_single['corrupt_acc']:.2f} part={r_single['partial_acc']:.2f} false={r_single['false_recall']:.2f} | "
              f"routed {r_routed['corrupt_acc']:.2f}/{r_routed['partial_acc']:.2f}/{r_routed['false_recall']:.2f} | "
              f"committee {r_comm['corrupt_acc']:.2f}/{r_comm['partial_acc']:.2f}/{r_comm['false_recall']:.2f} "
              f"agree_novel={r_comm.get('mean_agreement_novel', -1):.2f}", flush=True)
    json.dump(rows, open(r"D:\djr82\flymemory\bench_committee_lobe_result.json", "w"), indent=1)
    print("saved bench_committee_lobe_result.json")
