"""P-CLEANUP pre-study: is the production FlyMemory store two-scale
(dense-core + sparse-fringe), and does core-first eviction beat LRU on it?

Borrowed from intuition-mechanism G6 / P72 (MEMORY_GEOMETRY.md Law 6):
"evict redundant cores, keep the fringe (streaming Hart)" -- core-first
eviction preserved 100% recognition vs LRU 92.4% on two-scale synthetic
corpora, all policies tie on thin-coverage (single-scale) ones. FlyMemory's
decay_cleanup is currently LRU-family (last_accessed). This pre-study
decides whether P-CLEANUP (a production cleanup-policy experiment) is
worth registering.

Read-only on a snapshot copy of the live pkl (the service keeps running).

Part 1 -- geometry of the store:
  - 1-NN cosine distribution (bimodality = core/fringe signature)
  - coverage counts at cos thresholds {0.60, 0.70, 0.80} (redundancy layers)
  - connected-component sizes at cos >= 0.70 (topic-cluster structure)

Part 2 -- eviction simulation (budget 15% of active entries, holdout 500
anchors excluded from eviction):
  policies: LRU (oldest last_accessed first -- current production rule),
  core-first (highest coverage count first), random (seed 7),
  fringe-first (lowest coverage first -- worst-case control)
  metrics:
    evicted-recoverability: fraction of evicted entries whose top-1
      nearest surviving entry still has cos >= 0.75 (the information is
      carried by a remaining entry)
    holdout top-5 keep: fraction of each anchor's original top-5
      neighbours that remain in its post-eviction top-5

PRE-REGISTERED decision bands:
  - core-first recoverability >= LRU + 10pp  -> production redundancy
    structure is exploitable: REGISTER P-CLEANUP
  - |core-first - LRU| < 5pp -> thin-coverage degeneracy (single-scale):
    do NOT register; G6 recorded as not-applicable to this geometry
  - otherwise PARTIAL: inspect the coverage histogram before deciding

Run:  py -3.13 bench_cleanup_geometry.py
"""
import copy
import json
import os
import pickle
import shutil
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(_HERE, "flymemory", "flymemory_v3.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_cleanup_snapshot.pkl")
THRESHOLDS = [0.60, 0.70, 0.80]
BUDGET = 0.15
HOLDOUT_N = 500
RECOVER_TAU = 0.75
SEED = 7


def load_matrix():
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    mems = d["memories"]
    active = [m for m in mems if m.get("superseded_by") is None]
    idx = {}
    vecs, last_acc, cov_key = [], [], []
    for i, m in enumerate(active):
        v = np.asarray(m["embedding"], dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        idx[len(vecs)] = i
        vecs.append(v / n)
        last_acc.append(m.get("last_accessed") or m["timestamp"])
        cov_key.append(i)
    E = np.stack(vecs)
    return E, np.asarray(last_acc), active, [active[i] for i in cov_key]


def topk_sims(E, k, block=1024):
    """Top-k cosine neighbours per row (excluding self), blockwise."""
    n = E.shape[0]
    sims = np.zeros((n, k), dtype=np.float32)
    idxs = np.zeros((n, k), dtype=np.int32)
    for s in range(0, n, block):
        blk = E[s:s + block] @ E.T
        rows = np.arange(s, min(s + block, n))
        blk[rows - s, rows] = -2.0          # remove self
        part = np.argpartition(-blk, k, axis=1)[:, :k]
        vals = np.take_along_axis(blk, part, axis=1)
        order = np.argsort(-vals, axis=1)
        sims[s:s + block] = np.take_along_axis(vals, order, axis=1)
        idxs[s:s + block] = np.take_along_axis(part, order, axis=1)
    return sims, idxs


def main():
    t0 = time.time()
    E, last_acc, active, _ = load_matrix()
    n = E.shape[0]
    print(f"active entries: {n} (snapshot {time.strftime('%H:%M:%S')}, "
          f"{time.time()-t0:.0f}s)", flush=True)

    # ---------- Part 1: geometry ----------
    sims1, idxs1 = topk_sims(E, 5)
    nn1 = sims1[:, 0]
    print(f"\n1-NN cosine: mean {nn1.mean():.3f} median {np.median(nn1):.3f} "
          f"p10 {np.percentile(nn1, 10):.3f} p90 {np.percentile(nn1, 90):.3f}")
    hist, edges = np.histogram(nn1, bins=16, range=(0.3, 1.0))
    print("1-NN histogram (0.3-1.0, 16 bins):")
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"  {e0:.2f}-{e1:.2f}: {'#' * int(60 * h / max(hist.max(), 1))} {h}")

    cov_counts = {}
    for th in THRESHOLDS:
        c = (sims1[:, :5] >= th).sum(1)  # cheap approx via top5 only
        # exact: count over all pairs is O(n^2) but fine once
        full = (E @ E.T >= th)
        np.fill_diagonal(full, False)
        c = full.sum(1)
        cov_counts[th] = c
        print(f"coverage @cos>={th}: mean {c.mean():.1f} median "
              f"{np.median(c):.0f} p90 {np.percentile(c, 90):.0f} "
              f"max {c.max()} | zero-coverage {np.mean(c == 0):.1%}")

    # connected components at 0.70
    adj = E @ E.T >= 0.70
    np.fill_diagonal(adj, False)
    seen = np.zeros(n, dtype=bool)
    comp_sizes = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], 0
        seen[i] = True
        while stack:
            j = stack.pop()
            comp += 1
            for nxt in np.nonzero(adj[j] & ~seen)[0]:
                seen[nxt] = True
                stack.append(nxt)
        comp_sizes.append(comp)
    comp_sizes = np.sort(np.asarray(comp_sizes))[::-1]
    print(f"components @0.70: {len(comp_sizes)} | top10 sizes "
          f"{comp_sizes[:10].tolist()} | singleton "
          f"{np.mean(comp_sizes == 1):.1%}")

    # ---------- Part 2: eviction simulation ----------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    holdout, pool = perm[:HOLDOUT_N], perm[HOLDOUT_N:]
    budget = int(len(pool) * BUDGET)

    full_sim = E @ E.T
    np.fill_diagonal(full_sim, -2.0)
    # anchor original top-5 (over pool ∪ others, excluding self)
    anchor_top5 = np.argsort(-full_sim[holdout], axis=1)[:, :5]

    cov70 = cov_counts[0.70]
    strategies = {
        "LRU": pool[np.argsort(last_acc[pool])[:budget]],
        "core-first": pool[np.argsort(-cov70[pool])[:budget]],
        "random": rng.choice(pool, size=budget, replace=False),
        "fringe-first": pool[np.argsort(cov70[pool])[:budget]],
    }

    print(f"\neviction budget {budget} of {len(pool)} "
          f"({BUDGET:.0%}), holdout {HOLDOUT_N}")
    rows = []
    for name, ev in strategies.items():
        keep = np.setdiff1d(np.arange(n), ev, assume_unique=False)
        # recoverability of evicted entries
        rec = full_sim[np.ix_(ev, keep)].max(axis=1) >= RECOVER_TAU
        # holdout top-5 keep
        post = full_sim[np.ix_(holdout, keep)]
        new_top5 = np.argsort(-post, axis=1)[:, :5]
        kept_top5 = keep[new_top5]
        keep_rate = np.mean([
            len(set(anchor_top5[r]) & set(kept_top5[r])) / 5
            for r in range(len(holdout))])
        rows.append({"policy": name, "recoverability": round(float(rec.mean()), 4),
                     "holdout_top5_keep": round(float(keep_rate), 4)})
        print(f"{name:13s} recoverability {rec.mean():.1%} | "
              f"holdout top-5 keep {keep_rate:.1%}", flush=True)

    # decision
    cf = next(r for r in rows if r["policy"] == "core-first")["recoverability"]
    lr = next(r for r in rows if r["policy"] == "LRU")["recoverability"]
    diff = cf - lr
    if diff >= 0.10:
        verdict = "REGISTER P-CLEANUP (redundancy structure exploitable)"
    elif abs(diff) < 0.05:
        verdict = ("DO NOT REGISTER (thin-coverage degeneracy: "
                   "policies tie on this geometry)")
    else:
        verdict = "PARTIAL -- inspect coverage histogram before deciding"
    print(f"\ncore-first - LRU recoverability: {diff:+.1%}")
    print(f"DECISION: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"cleanup_geometry_{ts}.json")
    json.dump({"n_active": int(n), "nn1": {"mean": float(nn1.mean()),
               "median": float(np.median(nn1))},
               "coverage": {str(t): {"mean": float(c.mean()),
                                     "zero_pct": float(np.mean(c == 0))}
                            for t, c in cov_counts.items()},
               "components_070": int(len(comp_sizes)),
               "eviction": rows, "diff_cf_minus_lru": float(diff),
               "decision": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
