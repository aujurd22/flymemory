"""P-CLUSTER validation: reproduction gate on the CURRENT production
snapshot (registered in research/RESEARCH.md P-2026-10-16-CLUSTER --
criteria locked before this ran).

Gate 1 (clustering): threshold scan 0.5/0.55/0.6/0.65 on 200 sampled
multi-member lineages; SUPPORTED iff recovery >= 99% AND cross-link
<= 1% at th in {0.55, 0.6}.

Gate 2 (middle-version trap, supersede marks hidden): oldest-member
text as query; A1 = cosine top-1; CLUSTER = current_for(top-1).
SUPPORTED iff stale(A1) - stale(CLUSTER) >= 30pp.

Run:  py -3.13 bench_cluster_layer.py
"""
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.clusters import ClusterLayer, _norm_rows  # noqa: E402

PKL = os.path.join(_HERE, "flymemory", "flymemory_v3.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_cluster_snapshot.pkl")
THRESHOLDS = [0.5, 0.55, 0.6, 0.65]
SEED = 7


def main():
    t0 = time.time()
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    mems = d["memories"]
    by_id = {m["memory_id"]: m for m in mems}
    print(f"store: {len(mems)} entries ({time.time()-t0:.0f}s)", flush=True)

    # ---- lineage assembly (supersede edges as ground truth) ----
    def lineage_members(mid):
        seen = {mid}
        frontier = [mid]
        while frontier:
            cur = frontier.pop()
            m = by_id.get(cur)
            if not m:
                continue
            sb = m.get("superseded_by")
            if sb and sb != "None" and sb not in seen:
                seen.add(sb)
                frontier.append(sb)
        for other in mems:
            if other.get("superseded_by") == mid and \
                    other["memory_id"] not in seen:
                seen.add(other["memory_id"])
                frontier.append(other["memory_id"])
        return seen

    lineages = {}
    for m in mems:
        mid = m["memory_id"]
        if mid not in lineages:
            lin = lineage_members(mid)
            for x in lin:
                lineages[x] = lin
    multi = [l for l in set(map(frozenset, lineages.values()))
             if len(l) > 1]
    print(f"multi-member lineages: {len(multi)}", flush=True)
    rng = random.Random(SEED)
    sample = rng.sample(multi, min(200, len(multi)))

    E = np.asarray([m["embedding"] for m in mems], dtype=np.float32)
    E = _norm_rows(E)
    idx_of = {m["memory_id"]: i for i, m in enumerate(mems)}

    # ---- gate 1: threshold scan ----
    results = {}
    for th in THRESHOLDS:
        # lineage recovery: all lineage pairs directly >= th (H5 protocol)
        recover = partial = 0
        for lin in sample:
            mids = [m for m in lin if m in idx_of]
            ok = True
            for i in range(len(mids)):
                for j in range(i + 1, len(mids)):
                    if float(E[idx_of[mids[i]]] @ E[idx_of[mids[j]]]) < th:
                        ok = False
                        break
                if not ok:
                    break
            recover += ok
            partial += (not ok)
        # cross-link: random cross-lineage pairs >= th
        act = rng.sample([m["memory_id"] for m in mems], min(400, len(mems)))
        cross = pairs = 0
        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                a, b = act[i], act[j]
                if lineages.get(a) is not None and \
                        frozenset(lineages[a]) == frozenset(lineages.get(b, ())) \
                        and len(lineages[a]) > 1:
                    continue
                pairs += 1
                if pairs > 800:
                    break
                if float(E[idx_of[a]] @ E[idx_of[b]]) >= th:
                    cross += 1
            if pairs > 800:
                break
        n = recover + partial
        results[str(th)] = {
            "recover_rate": round(recover / max(1, n), 4),
            "cross_link_rate": round(cross / max(1, pairs), 4)}
        print(f"  th={th}: recovery {recover}/{n}="
              f"{recover/max(1,n):.1%} | cross-link {cross}/{pairs}="
              f"{cross/max(1,pairs):.1%}", flush=True)

    # ---- gate 2: middle-version trap (supersede marks hidden) ----
    # build layer once at th=0.55 with ONLY active-view entries (the way
    # production would see them after supersede exclusion -- but we need
    # lineage members present for the trap demo, so use the full set and
    # ask cosine-top1 over NON-superseded only for A1)
    layer = ClusterLayer(mems, link_threshold=0.55)
    active = [m for m in mems if not m.get("superseded_by")
              or m.get("superseded_by") == "None"]
    act_idx = np.asarray([idx_of[m["memory_id"]] for m in active])
    act_ids = [m["memory_id"] for m in active]
    trap = rng.sample([l for l in sample if len(l) >= 2], 100)
    stale_a1 = stale_cl = 0
    n_trap = 0
    for lin in trap:
        members = sorted(lin, key=lambda m: by_id[m].get("timestamp") or 0)
        oldest, newest = members[0], members[-1]
        q = np.asarray(by_id[oldest]["embedding"], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = E[act_idx] @ q
        top1 = act_ids[int(np.argmax(sims))]
        n_trap += 1
        if top1 != newest:
            stale_a1 += 1
        resolved = layer.current_for(top1)
        if resolved != newest:
            stale_cl += 1
    d1 = (stale_a1 - stale_cl) / max(n_trap, 1)
    print(f"\n  trap demo (n={n_trap}): A1 stale {stale_a1} -> CLUSTER "
          f"stale {stale_cl} | delta {d1:+.1%}", flush=True)

    gate1 = any(results[str(t)]["recover_rate"] >= 0.99 and
                results[str(t)]["cross_link_rate"] <= 0.01
                for t in (0.55, 0.6))
    gate2 = d1 >= 0.30
    verdict = ("SUPPORTED" if gate1 and gate2 else
               "PARTIAL" if gate1 or gate2 else "NULL")
    print(f"\ngate1 (recovery>=99% & cross<=1% @ 0.55/0.6): {gate1}")
    print(f"gate2 (stale delta >= 30pp): {gate2}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"cluster_layer_{ts}.json")
    json.dump({"experiment": "P-CLUSTER", "n_mems": len(mems),
               "n_lineages": len(multi), "scan": results,
               "trap": {"n": n_trap, "stale_a1": stale_a1,
                        "stale_cluster": stale_cl, "delta": round(d1, 4)},
               "gate1": gate1, "gate2": gate2, "verdict": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
