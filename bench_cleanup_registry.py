"""P-CLEANUP (registered, research/RESEARCH.md P-2026-09-29-CLEANUP --
verdict PENDING there; criteria locked before this ran).

Reshaped eviction-policy experiment on a production-store snapshot.
Same protocol as the pre-study (bench_cleanup_geometry.py): budget 15%,
500 holdout anchors, seed 7. New primary arm:

  backup-first -- eviction candidates are entries with a twin at
  cos >= 0.85; within candidates evict the older last_accessed first
  (keeping the newer of each pair); budget topped up by cov70 desc,
  top-up count recorded. Twin-backed entries are recoverable by
  construction; evicting pairs instead of hubs should spare the
  retrieval order.

Dual metrics (BOTH load-bearing):
  evicted-recoverability  -- top-1 surviving neighbour cos >= 0.75
  holdout top-5 keep      -- order stability on 500 anchors

Run:  py -3.13 bench_cleanup_registry.py
"""
import json
import os
import pickle
import shutil
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(_HERE, "flymemory", "flymemory_v3.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_cleanup_snapshot.pkl")
BUDGET = 0.15
HOLDOUT_N = 500
RECOVER_TAU = 0.75
TWIN_TAU = 0.85
SEED = 7


def load_matrix():
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    active = [m for m in d["memories"] if m.get("superseded_by") is None]
    vecs, last_acc = [], []
    for m in active:
        v = np.asarray(m["embedding"], dtype=np.float32)
        nrm = np.linalg.norm(v)
        if nrm == 0:
            continue
        vecs.append(v / nrm)
        last_acc.append(m.get("last_accessed") or m["timestamp"])
    return np.stack(vecs), np.asarray(last_acc)


def main():
    t0 = time.time()
    E, last_acc = load_matrix()
    n = E.shape[0]
    print(f"active entries: {n}", flush=True)

    full = E @ E.T
    np.fill_diagonal(full, -2.0)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    holdout, pool = perm[:HOLDOUT_N], perm[HOLDOUT_N:]
    budget = int(len(pool) * BUDGET)

    anchor_top5 = np.argsort(-full[holdout], axis=1)[:, :5]

    # degree at 0.70 (row counts; symmetric matrix) and twin presence
    deg70 = (full >= 0.70).sum(1)
    has_twin = (full >= TWIN_TAU).sum(1) > 0

    strategies = {}
    strategies["LRU"] = pool[np.argsort(last_acc[pool])[:budget]]
    strategies["core-first"] = pool[np.argsort(-deg70[pool])[:budget]]
    strategies["random"] = rng.choice(pool, size=budget, replace=False)
    strategies["fringe-first"] = pool[np.argsort(deg70[pool])[:budget]]

    # backup-first: twin candidates, older first; top-up by deg70 desc
    twins = pool[has_twin[pool]]
    order = twins[np.argsort(last_acc[twins])]
    sel = set(order[:budget].tolist())
    topup = 0
    if len(sel) < budget:
        rest = pool[~np.isin(pool, list(sel))]
        for cand in rest[np.argsort(-deg70[rest])]:
            if len(sel) >= budget:
                break
            sel.add(int(cand))
            topup += 1
    strategies["backup-first"] = np.asarray(sorted(sel), dtype=np.int64)
    print(f"backup-first: {budget - topup} twin evictions + {topup} deg70 "
          f"top-up", flush=True)

    rows = []
    for name, ev in strategies.items():
        keep = np.setdiff1d(np.arange(n), ev, assume_unique=False)
        rec = full[np.ix_(ev, keep)].max(axis=1) >= RECOVER_TAU
        post = full[np.ix_(holdout, keep)]
        new_top5 = keep[np.argsort(-post, axis=1)[:, :5]]
        keep_rate = np.mean([
            len(set(anchor_top5[r]) & set(new_top5[r])) / 5
            for r in range(len(holdout))])
        rows.append({"policy": name,
                     "recoverability": round(float(rec.mean()), 4),
                     "top5_keep": round(float(keep_rate), 4)})
        print(f"{name:13s} recoverability {rec.mean():.1%} | "
              f"top-5 keep {keep_rate:.1%}", flush=True)

    def g(policy, metric):
        return next(r[metric] for r in rows if r["policy"] == policy)

    lr_rec, lr_keep = g("LRU", "recoverability"), g("LRU", "top5_keep")
    bf_rec, bf_keep = (g("backup-first", "recoverability"),
                       g("backup-first", "top5_keep"))
    d_rec, d_keep = bf_rec - lr_rec, bf_keep - lr_keep
    if d_rec >= 0.10 and d_keep >= -0.02:
        verdict = ("SUPPORTED -- hub-sparing twin eviction dominates LRU "
                   "on both axes")
    elif d_rec >= 0.10:
        verdict = "PARTIAL -- mirror persists (twin definition needs revision)"
    elif d_rec < 0.05 or d_keep < -0.10:
        verdict = "NOT SUPPORTED -- LRU already reasonable on this geometry"
    else:
        verdict = "PARTIAL (bands boundary)"
    print(f"\nbackup-first vs LRU: recoverability {d_rec:+.1%}, "
          f"top-5 keep {d_keep:+.1%}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"cleanup_registry_{ts}.json")
    json.dump({"experiment": "P-CLEANUP", "n_active": int(n),
               "budget": budget, "rows": rows,
               "backup_first": {"twin_evictions": budget - topup,
                                "topup": topup},
               "deltas": {"recoverability": round(d_rec, 4),
                          "top5_keep": round(d_keep, 4)},
               "verdict": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
