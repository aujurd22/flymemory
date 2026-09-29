"""P-TREE (registered, research/RESEARCH.md P-2026-09-29-TREE -- verdict
PENDING there; criteria locked before this ran).

Question: does hierarchical topic navigation (PageIndex-style, ZERO-LLM
version -- spherical k-means tree, embedding-centroid routing) lift
evidence collection on multi-session questions, where single-shot RRF
top-5 structurally under-collects (the L5 aggregation-miss family)?

Store: the 9,729-entry shared LongMemEval-S oracle turn store
(longmemeval_bench.pkl). Every entry's tags carry its source session
("answer_<sid>_<turn>"), so evidence localization is exact.

Tree: two-level spherical k-means (L1: 32 clusters; L2: <=16 children per
L1), seed 42, 25 iterations, zero LLM. Navigation: query -> top-2 L1
clusters -> top-2 L2 children within each -> collect ALL entries of the
selected children -> exact rerank inside the collected set, top-30.

Arms:
  RRF  : production recall top-5 (current path)
  TREE : navigation + collect + rerank top-30 (wider by construction --
         the hypothesis is that TOPIC-GUIDED widening beats flat top-5)

Metrics:
  session-hit      : fraction of questions whose collected set contains
                     >=1 entry from any answer_session
  session-coverage : mean fraction of the answer sessions covered
                     (multi-session completeness)

Question sets: PRIMARY = multi-session (133); GUARD = single-session-*
(156) -- navigation must not hurt lookups.

PRE-REGISTERED CRITERIA:
  SUPPORTED      : multi-session session-hit(TREE) >= RRF + 15pp AND
                   guard pass (single-session TREE >= RRF - 3pp)
                   -> proceed to end-to-end answer verification
  PARTIAL        : +5pp to +15pp
  NOT SUPPORTED  : < +5pp (topic trees add nothing on the shared turn
                   store; L5's aggregation remedy stays the TOOL3
                   multi-round route)
  Any TREE regression > 3pp on the guard set caps the verdict at PARTIAL.

Run:  py -3.13 bench_tree_navigation.py
"""
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

PKL = os.path.join(_HERE, "longmemeval_bench.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_tree_snapshot.pkl")
QS = os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json")
SEED = 42
L1_K, L2_K = 32, 16
L1_TOP, L2_TOP = 2, 2
FINAL_K = 30
SESSION_TAG_MINLEN = 8


def load_store():
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    mems = [m for m in d["memories"] if m.get("superseded_by") is None]
    E, sess, mems_ok = [], [], []
    for m in mems:
        v = np.asarray(m["embedding"], dtype=np.float32)
        nrm = np.linalg.norm(v)
        if nrm == 0:
            continue
        sid = None
        for t in m.get("tags") or []:
            t = str(t)
            if t not in ("lme", "consolidated", "chunk") and \
                    len(t) >= SESSION_TAG_MINLEN:
                sid = t.rsplit("_", 1)[0] if t.rsplit("_", 1)[-1].isdigit() \
                    else t
                break
        E.append(v / nrm)
        sess.append(sid)
        mems_ok.append(m)
    return np.stack(E), sess, mems_ok


def spherical_kmeans(E, k, seed, iters=25):
    rng = np.random.default_rng(seed)
    cent = E[rng.choice(len(E), size=k, replace=False)].copy()
    assign = np.zeros(len(E), dtype=np.int64)
    for _ in range(iters):
        sims = E @ cent.T
        new_assign = np.argmax(sims, axis=1)
        if np.array_equal(new_assign, assign) and _ > 0:
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                cent[c] = E[mask].mean(0)
            cent[c] /= (np.linalg.norm(cent[c]) + 1e-9)
    return assign, cent


def embed_query(model, text):
    v = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32)


def main():
    t0 = time.time()
    E, sess, _ = load_store()
    n = len(E)
    qs = json.load(open(QS, encoding="utf-8"))
    sess_map = {q["question_id"]: q for q in qs}
    print(f"store: {n} entries | questions: {len(qs)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # tree
    l1_assign, l1_cent = spherical_kmeans(E, L1_K, SEED)
    l2_assign = np.zeros(n, dtype=np.int64)
    l2_cent = np.zeros((L1_K * L2_K, E.shape[1]), dtype=np.float32)
    for c in range(L1_K):
        idx = np.nonzero(l1_assign == c)[0]
        k2 = min(L2_K, max(1, len(idx) // 8))
        if len(idx) == 0:
            continue
        a2, c2 = spherical_kmeans(E[idx], k2, SEED + 100 + c)
        l2_assign[idx] = c + 1000 + a2
        for j, gi in enumerate(range(c * L2_K, c * L2_K + k2)):
            l2_cent[gi] = c2[j]
    # child cluster membership lists
    children = {}
    for i in range(n):
        children.setdefault(l2_assign[i], []).append(i)
    print(f"tree: {L1_K} L1 clusters, {len(children)} L2 leaves "
          f"({time.time()-t0:.0f}s)", flush=True)

    # embedder for queries
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def navigate(qv):
        l1 = np.argsort(-(l1_cent @ qv))[:L1_TOP]
        picked = []
        for c in l1:
            kids = [g for g in children if g // L2_K == c]
            if not kids:
                picked.extend(children.get(c, [])[:])
                continue
            kids_arr = np.stack([l2_cent[g] for g in kids])
            for g in [kids[i] for i in np.argsort(-(kids_arr @ qv))[:L2_TOP]]:
                picked.extend(children.get(g, []))
        return picked

    results = {}
    for setname, pred in [
        ("multi-session", lambda t: t == "multi-session"),
        ("guard-single", lambda t: t.startswith("single-session")),
        ("all", lambda t: True),
    ]:
        subset = [q for q in qs if pred(q["question_type"])]
        stats = {arm: {"hit": 0, "cov": [], "n": 0}
                 for arm in ("RRF", "TREE")}
        for qi, q in enumerate(subset):
            qv = embed_query(model, q["question"])
            ans_sessions = set(q["answer_session_ids"])
            # RRF arm: exact cosine top-5 over the same store (production
            # RRF's lexical channel is off in this offline harness; the
            # comparison is collection-shape, not reranker quality)
            top5 = np.argsort(-(E @ qv))[:5]
            rrf_sess = {sess[i] for i in top5}
            # TREE arm
            picked = navigate(qv)
            picked_arr = np.asarray(picked)
            sims = E[picked_arr] @ qv
            tree_idx = picked_arr[np.argsort(-sims)[:FINAL_K]]
            tree_sess = {sess[i] for i in tree_idx}
            for arm, s in (("RRF", rrf_sess), ("TREE", tree_sess)):
                hit = bool(s & ans_sessions)
                stats[arm]["hit"] += hit
                stats[arm]["cov"].append(len(s & ans_sessions)
                                         / max(len(ans_sessions), 1))
                stats[arm]["n"] += 1
            if (qi + 1) % 50 == 0:
                print(f"  [{setname}] {qi+1}/{len(subset)}", flush=True)
        for arm in ("RRF", "TREE"):
            s = stats[arm]
            results.setdefault(setname, {})[arm] = {
                "session_hit": round(s["hit"] / s["n"], 4),
                "session_coverage": round(float(np.mean(s["cov"])), 4),
                "n": s["n"]}
        r = results[setname]
        print(f"{setname:14s} RRF hit {r['RRF']['session_hit']:.1%} "
              f"cov {r['RRF']['session_coverage']:.3f} | "
              f"TREE hit {r['TREE']['session_hit']:.1%} "
              f"cov {r['TREE']['session_coverage']:.3f} "
              f"(n={r['RRF']['n']})", flush=True)

    d_ms = (results["multi-session"]["TREE"]["session_hit"]
            - results["multi-session"]["RRF"]["session_hit"])
    guard_ok = (results["guard-single"]["TREE"]["session_hit"]
                >= results["guard-single"]["RRF"]["session_hit"] - 0.03)
    if d_ms >= 0.15 and guard_ok:
        verdict = "SUPPORTED -- proceed to end-to-end answer verification"
    elif d_ms >= 0.05:
        verdict = "PARTIAL"
    else:
        verdict = ("NOT SUPPORTED -- L5 remedy stays the TOOL3 multi-round "
                   "route")
    if not guard_ok:
        verdict = "PARTIAL (guard regression caps it)"
    print(f"\nmulti-session delta: {d_ms:+.1%} | guard pass: {guard_ok}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"tree_navigation_{ts}.json")
    json.dump({"experiment": "P-TREE", "results": results,
               "delta_multi_session": round(float(d_ms), 4),
               "guard_pass": bool(guard_ok), "verdict": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
