"""P-CLUSTER validation v3: the UNMARKED-world pool test.

v1/v2 findings (both recorded in RESEARCH.md):
- v1: whole-store connected components PERCOLATE (98% of 16,588 entries
  in one giant component at th=0.55 -- dense-store property; H5's
  random-pair metric structurally cannot see it).
- v2: pooling from ACTIVE entries only is the wrong test -- the I1
  filter already excludes marked old versions, so that pool measures
  retrieval misses, not staleness.

The honest production question: when supersede marks are MISSING (the
caller forgot -- the actual residual risk class), do old+new versions
both stay active, and can a cluster layer resolve to the newest? Test:
pool = cosine top-10 over ALL entries (marked or not -- simulating the
unmarked world); A1 = pool top-1; CLUSTER = local cluster resolution
on the pool (th=0.55/0.6), >=2 top-3 hits in one cluster -> newest
member wins. Ground truth = newest lineage member.

Run:  py -3.13 bench_cluster_layer_v3.py
"""
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.clusters import _norm_rows  # noqa: E402

PKL = os.path.join(_HERE, "flymemory", "flymemory_v3.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_cluster3_snapshot.pkl")
SEED = 7
TOPK = 10


def make_query(text):
    return re.sub(r"\s+", "", text)[:18]


def pool_cluster_resolve(cand_ids, cand_sims, E, idx_of, by_id, th):
    scored = sorted(zip(cand_sims, cand_ids), key=lambda x: -x[0])
    top3 = [m for _, m in scored[:3]]
    n = len(cand_ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if float(E[idx_of[cand_ids[i]]] @ E[idx_of[cand_ids[j]]]) >= th:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    order = sorted(clusters.values(),
                   key=lambda idxs: max(cand_sims[i] for i in idxs),
                   reverse=True)
    for idxs in order:
        ctexts = [cand_ids[i] for i in idxs]
        hits = [m for m in top3 if m in ctexts]
        if len(hits) >= 2:
            return max(hits, key=lambda m: by_id[m].get("timestamp") or 0.0)
    return scored[0][1]


def main():
    t0 = time.time()
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    mems = d["memories"]
    by_id = {m["memory_id"]: m for m in mems}
    E = _norm_rows(np.asarray([m["embedding"] for m in mems],
                              dtype=np.float32))
    idx_of = {m["memory_id"]: i for i, m in enumerate(mems)}
    print(f"store: {len(mems)} entries ({time.time()-t0:.0f}s)", flush=True)

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
    multi = [sorted(l, key=lambda x: by_id[x].get("timestamp") or 0)
             for l in set(map(frozenset, lineages.values())) if len(l) >= 2]
    rng = random.Random(SEED)
    sample = rng.sample(multi, min(150, len(multi)))
    print(f"multi lineages: {len(multi)} | sample: {len(sample)}", flush=True)

    from flymemory.v3 import _embed  # noqa: E402

    results = {}
    for th in (0.5, 0.55, 0.6):
        stale_a1 = stale_cl = in_pool = 0
        n_ok = 0
        for lin in sample:
            oldest, newest = lin[0], lin[-1]
            q_text = make_query(by_id[oldest]["text"])
            qe = np.asarray(_embed(q_text), dtype=np.float32)
            qe /= (np.linalg.norm(qe) + 1e-9)
            sims = E @ qe                      # ALL entries (unmarked world)
            top = np.argsort(-sims)[:TOPK]
            cand_ids = [int(top[i]) and mems[int(top[i])]["memory_id"]
                        for i in range(TOPK)]
            cand_sims = [float(sims[i]) for i in range(TOPK)]
            if cand_ids[0] != newest:
                stale_a1 += 1
            if newest in cand_ids:
                in_pool += 1
            resolved = pool_cluster_resolve(cand_ids, cand_sims, E, idx_of,
                                            by_id, th)
            if resolved != newest:
                stale_cl += 1
            n_ok += 1
        results[str(th)] = {
            "n": n_ok, "stale_a1": stale_a1, "stale_cluster": stale_cl,
            "newest_in_top10": round(in_pool / n_ok, 3),
            "delta_pp": round((stale_a1 - stale_cl) / n_ok * 100, 1)}
        print(f"  th={th}: A1 stale {stale_a1}/{n_ok} "
              f"({stale_a1/n_ok:.0%}) | CLUSTER stale {stale_cl}/{n_ok} "
              f"({stale_cl/n_ok:.0%}) | newest-in-top10 "
              f"{in_pool/n_ok:.0%} | delta "
              f"{(stale_a1-stale_cl)/n_ok*100:+.1f}pp", flush=True)

    best = max(results.values(), key=lambda r: r["delta_pp"])
    gate2 = best["delta_pp"] >= 30
    verdict = "SUPPORTED" if gate2 else \
              ("PARTIAL" if best["delta_pp"] >= 10 else "NULL")
    print(f"\ngate2 (stale delta >= 30pp, unmarked-world pool): {gate2}")
    print(f"PRE-REGISTERED VERDICT (gate2): {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"cluster_layer_v3_{ts}.json")
    json.dump({"experiment": "P-CLUSTER-v3",
               "protocol": "unmarked-world pool (all entries)",
               "n_mems": len(mems), "results": results,
               "gate2": gate2, "verdict": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
