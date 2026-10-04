"""P73-PROBE: run the Memory Geometry Controller against the production
FlyMemory store (read-only snapshot).

Measures the controller's 8 inputs on the production geometry, then calls
flyloop's registered controller (flyloop/controller.py, zero free
parameters) and reports the policy it prescribes vs FlyMemory's current
defaults (flat exemplar store, LRU-family eviction, no verification gate,
no novelty screening, rehearsal = refresh-on-recall).

Caliber notes (declared, not hidden):
  - the store has no labeled classes; connected components at cos>=0.70
    with >=5 members stand in for "classes" (isolates -- 55% of the
    store -- cannot have class geometry);
  - D=384 has NO registered cap*(r, D) curve (controller accepts 6/12
    only): R8's substrate-feasibility gate is expected to fire -- that is
    a finding about the controller's registered coverage, not a bug;
  - eps (write/read noise) is unmeasured on this substrate: reported as
    UNMEASURED, controller run with the harness default.

Run:  py -3.13 p73_controller_probe.py
"""
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, r"D:/djr82/flyloop")          # controller source

from flyloop.controller import controller  # noqa: E402  (pure, no I/O)

PKL = os.path.join(_HERE, "longmemeval_bench.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_p73_snapshot.pkl")
THRESH_FRAC = 0.70     # component edges
MIN_CLASS = 5          # min members for a "class"


def load_matrix():
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    mems = [m for m in d["memories"] if m.get("superseded_by") is None]
    E = np.stack([np.asarray(m["embedding"], dtype=np.float32)
                  / (np.linalg.norm(m["embedding"]) + 1e-9) for m in mems])
    return E


def components(adj):
    seen = np.zeros(len(adj), dtype=bool)
    comps = []
    for i in range(len(adj)):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            for nxt in np.nonzero(adj[j] & ~seen)[0]:
                seen[nxt] = True
                stack.append(nxt)
        comps.append(comp)
    return comps


def main():
    t0 = time.time()
    E = load_matrix()
    n = len(E)
    sim = E @ E.T
    np.fill_diagonal(sim, -2.0)
    print(f"store: {n} entries ({time.time()-t0:.0f}s)", flush=True)

    # ---- class structure proxy: connected components at cos>=0.70 ----
    adj = sim >= THRESH_FRAC
    np.fill_diagonal(adj, False)
    comps = components(adj)
    classes = [c for c in comps if len(c) >= MIN_CLASS]
    print(f"components: {len(comps)} | classes (>={MIN_CLASS} members): "
          f"{len(classes)} | covered by classes: "
          f"{sum(len(c) for c in classes)/n:.1%}", flush=True)

    # ---- L1/L2 geometry: R, nn, mind over real classes ----
    Rs, NNs = [], []
    cents = []
    for c in classes:
        Ec = E[c]
        cent = Ec.mean(0)
        cent /= (np.linalg.norm(cent) + 1e-9)
        cents.append(cent)
        Rs.append(float((1 - Ec @ cent).mean()))
        sub = sim[np.ix_(c, c)]
        np.fill_diagonal(sub, -2.0)
        NNs.append(float(sub.max(1).mean()))
    cents = np.stack(cents)
    Csim = cents @ cents.T
    np.fill_diagonal(Csim, 2.0)
    mind = float(1 - Csim.min())          # min inter-class centroid distance
    R = float(np.mean(Rs))
    nn_mean = float(np.mean(NNs))
    r_ratio = R / mind
    nn_ratio = nn_mean / mind

    # two-scale: within-class NN-distance spread (p90/p50 > 1.8)
    ts_flags = []
    for c in classes:
        sub = sim[np.ix_(c, c)]
        np.fill_diagonal(sub, -2.0)
        nnd = sub.max(1)
        p50, p90 = np.percentile(nnd, 50), np.percentile(nnd, 90)
        ts_flags.append(p50 > 0 and p90 / max(p50, 1e-9) > 1.8)
    ts = any(ts_flags)

    # ---- L3 coverage: reachable fraction at THRESH within own class ----
    covs = []
    for c in classes:
        sub = sim[np.ix_(c, c)]
        np.fill_diagonal(sub, -2.0)
        covs.append(float((sub.max(1) >= THRESH_FRAC).mean()))
    cov = float(np.mean(covs))

    # ---- L4 clearance: held-out probes vs the union of stored coverage ----
    probe_idx = []
    for c in classes:
        probe_idx.append(c[int(len(c) * 0.2)])     # one probe per class
    clr_ratios = []
    stored_mask = np.ones(n, dtype=bool)
    stored_mask[probe_idx] = False
    for pi in probe_idx:
        dmin = 1 - float(sim[pi][stored_mask].max())
        clr_ratios.append(dmin / max(mind / 2, 1e-9))
    clr = float(np.mean(clr_ratios))

    # ---- L5 query length (P70-b knee 6-12 tokens; typical query 10-20) --
    lq_low, lq_high = 10 / 12, 20 / 12

    # ---- L6 fringe fraction: zero neighbors at THRESH ----
    deg0 = float(np.mean(((sim >= THRESH_FRAC).sum(1)) == 0))
    ff = deg0

    print(f"\nmeasured geometry (caliber: components@0.70 as classes, "
          f"n_classes={len(classes)}):")
    print(f"  R = {R:.3f}   mind = {mind:.3f}   r = R/mind = {r_ratio:.3f}")
    print(f"  NN = {nn_mean:.3f}   nn/mind = {nn_ratio:.3f}")
    print(f"  ts (two-scale) = {ts} | cov = {cov:.3f} | clr = {clr:.2f} "
          f"| ff = {ff:.3f} | lq in [{lq_low:.2f},{lq_high:.2f}] | "
          f"eps = UNMEASURED")

    # ---- controller runs ----
    results = {}
    for D in (6, 12):
        for lq in (lq_low, lq_high):
            try:
                pol = controller(r=r_ratio, nn=nn_ratio, ts=ts, cov=cov,
                                 clr=clr, lq=lq, ff=ff, eps=0.10, D=D)
                results[f"D{D}_lq{lq:.2f}"] = pol
                print(f"\ncontroller(D={D}, lq={lq:.2f}): "
                      f"type={pol.get('type')} capacity={pol.get('capacity')} "
                      f"eviction={pol.get('eviction')} "
                      f"verification={pol.get('verification')} "
                      f"query_gate={pol.get('query_gate')} "
                      f"novelty={pol.get('novelty_screening')}")
                for note in (pol.get("notes") or [])[:4]:
                    print(f"    note: {note[:110]}")
            except Exception as e:
                results[f"D{D}_lq{lq:.2f}"] = {"error": str(e)}
                print(f"\ncontroller(D={D}, lq={lq:.2f}) RAISED: {e}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"p73_probe_{ts}.json")
    json.dump({"n": n, "r": r_ratio, "mind": mind, "nn_ratio": nn_ratio,
               "ts": ts, "cov": cov, "clr": clr, "ff": ff,
               "n_classes": len(classes), "controller": {
                   k: (v if isinstance(v, dict) and "error" in v
                       else {kk: str(vv)[:80] for kk, vv in v.items()})
                   for k, v in results.items()}},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
