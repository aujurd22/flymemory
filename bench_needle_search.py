"""P-NEEDLE: coarse->needle two-stage evidence collection (registered,
research/RESEARCH.md P-2026-09-29-NEEDLE -- verdict PENDING there).

Stage A: tree navigation (as P-TREE), collect the WHOLE region (no top-30
cut). Stage B: lexical needle scoring inside the region -- distinct query
content words present in the entry text, top-30 by score (cosine
tiebreak). Metrics at TURN level via the dataset's answer_xxx_N
annotations matching store tags exactly.

Run:  py -3.13 bench_needle_search.py
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
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from bench_tree_navigation import (  # noqa: E402
    PKL, SNAP, QS, spherical_kmeans, embed_query,
    L1_K, L2_K, L1_TOP, L2_TOP, FINAL_K, SEED)

STOP = set("""the a an and or of to in on for with is are was were be been
i my me you your it its this that these those do does did have has had
how many much more most often long much what which who when where why
than from at as by""".split())


def load_store():
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    active = [m for m in d["memories"] if m.get("superseded_by") is None]
    E, sess, turn_keys = [], [], []
    for m in active:
        v = np.asarray(m["embedding"], dtype=np.float32)
        nrm = np.linalg.norm(v)
        if nrm == 0:
            continue
        E.append(v / nrm)
        sid, tk = None, None
        for t in m.get("tags") or []:
            t = str(t)
            if t not in ("lme", "consolidated", "chunk") and len(t) >= 8:
                base, _, num = t.rpartition("_")
                if num.isdigit():
                    sid, tk = base, t
                else:
                    sid = t
                break
        sess.append(sid)
        turn_keys.append(tk)
    return np.stack(E), sess, turn_keys


def needle_words(text):
    return {w for w in text.lower().split() if w.isalpha() and w not in STOP}


def main():
    t0 = time.time()
    E, sess, turn_keys = load_store()
    n = len(E)
    qs = json.load(open(QS, encoding="utf-8"))
    ms = [q for q in qs if q["question_type"] == "multi-session"]
    guard = [q for q in qs if q["question_type"].startswith("single-session")]
    print(f"store {n} | multi {len(ms)} guard {len(guard)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    l1_assign, l1_cent = spherical_kmeans(E, L1_K, SEED)
    children, l2_cent = {}, {}
    for c in range(L1_K):
        idx = np.nonzero(l1_assign == c)[0]
        k2 = min(L2_K, max(1, len(idx) // 8))
        if len(idx) == 0:
            continue
        a2, c2 = spherical_kmeans(E[idx], k2, SEED + 100 + c)
        leaf_ids = c * 1000 + np.arange(k2)
        for j, g in enumerate(leaf_ids):
            l2_cent[int(g)] = c2[j]
            children[int(g)] = idx[a2 == j].tolist()

    def navigate(qv):
        l1 = np.argsort(-(l1_cent @ qv))[:L1_TOP]
        picked = []
        for c in l1:
            kids = sorted(g for g in children if g // 1000 == c)
            kids_arr = np.stack([l2_cent[g] for g in kids])
            for g in [kids[i] for i in np.argsort(-(kids_arr @ qv))[:L2_TOP]]:
                picked.extend(children.get(g, []))
        return picked

    def evaluate(subset, setname):
        stats = {a: {"turn_hit": 0, "turn_recall": [], "sess_hit": 0, "n": 0}
                 for a in ("RRF", "TREE", "NEEDLE")}
        for qi, q in enumerate(subset):
            qv = embed_query(model, q["question"])
            ans_turns = set(q["answer_session_ids"])      # turn-level ids
            ans_sess = {t.rsplit("_", 1)[0] if t.rsplit("_", 1)[-1].isdigit()
                        else t for t in ans_turns}
            rrf_idx = np.argsort(-(E @ qv))[:5]
            picked = navigate(qv)
            picked_arr = np.asarray(picked, dtype=np.int64)
            sims = E[picked_arr] @ qv
            tree_idx = picked_arr[np.argsort(-sims)[:FINAL_K]]
            # needle: lexical hits inside the region, cosine tiebreak
            nw = needle_words(q["question"])
            nscore = np.asarray([
                len(nw & needle_words(texts[i])) if nw else 0.0
                for i in picked], dtype=np.float32)
            rank = np.lexsort((-sims, -nscore))          # needle desc, sim tie
            needle_idx = picked_arr[rank[:FINAL_K]]

            for arm, idx in (("RRF", rrf_idx), ("TREE", tree_idx),
                             ("NEEDLE", needle_idx)):
                got_turns = {turn_keys[i] for i in idx if turn_keys[i]}
                hit = len(got_turns & ans_turns) > 0
                rec = len(got_turns & ans_turns) / max(len(ans_turns), 1)
                got_sess = {sess[i] for i in idx}
                s_hit = len(got_sess & ans_sess) > 0
                stats[arm]["turn_hit"] += hit
                stats[arm]["turn_recall"].append(rec)
                stats[arm]["sess_hit"] += s_hit
                stats[arm]["n"] += 1
            if (qi + 1) % 50 == 0:
                print(f"  [{setname}] {qi+1}/{len(subset)}", flush=True)
        out = {}
        for arm, s in stats.items():
            out[arm] = {"turn_hit": round(s["turn_hit"] / s["n"], 4),
                        "turn_recall": round(float(np.mean(s["turn_recall"])), 4),
                        "session_hit": round(s["sess_hit"] / s["n"], 4),
                        "n": s["n"]}
        print(f"{setname:14s} " + " | ".join(
            f"{a} turn-hit {out[a]['turn_hit']:.1%} "
            f"t-recall {out[a]['turn_recall']:.3f}" for a in out), flush=True)
        return out

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # texts for needle scoring
    texts = [None] * n
    dd = pickle.load(open(SNAP, "rb"))
    act = [m for m in dd["memories"] if m.get("superseded_by") is None]
    j = 0
    for m in act:
        v = np.asarray(m["embedding"], dtype=np.float32)
        if np.linalg.norm(v) == 0:
            continue
        texts[j] = m["text"]
        j += 1

    results = {"multi_session": evaluate(ms, "multi-session"),
               "guard_single": evaluate(guard, "guard-single")}

    r = results["multi_session"]
    d_th = r["NEEDLE"]["turn_hit"] - r["RRF"]["turn_hit"]
    tr_beats_tree = r["NEEDLE"]["turn_recall"] > r["TREE"]["turn_recall"]
    if d_th >= 0.15 and tr_beats_tree:
        verdict = ("COLLECT-SUPPORTED -- proceed to the 50-question "
                   "end-to-end check")
    elif d_th >= 0.05:
        verdict = "COLLECT-PARTIAL"
    else:
        verdict = "COLLECT-NULL -- line stays closed"
    g = results["guard_single"]
    print(f"\nmulti turn-hit delta (NEEDLE-RRF): {d_th:+.1%} | "
          f"t-recall beats TREE: {tr_beats_tree}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"needle_search_{ts}.json")
    json.dump({"experiment": "P-NEEDLE", "results": results,
               "delta_turn_hit": round(float(d_th), 4), "verdict": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
