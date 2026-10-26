"""P-USEGATE: use_gate accounting + pointer injection (registered,
research/RESEARCH.md P-2026-10-26-USEGATE -- verdict PENDING there).

A/B on 200 multi-session questions over the LongMemEval-S oracle store:

  BASE  current recall top-5 (truncated text in pool)
  UG    top-5 pointers (id + one-line + "fetch full text") -- simulating
        the pointer protocol, the "expansion" step is modeled as: the
        pointer's full text joins the pool when the pointer's summary
        overlaps the question lexically (the model-follows-pointer proxy,
        deterministic), plus top-10 base pool (wider pull)

  evidence-in-pool = fraction of answer-session turns present in the pool
  pool tokens      = sum of len(pool texts) (budget check)

Deterministic throughout (no LLM): this measures the MECHANISM, per the
registration. Run:  py -3.13 bench_usegate.py
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

PKL = os.path.join(_HERE, "longmemeval_bench.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_usegate_snapshot.pkl")
QS = os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json")
N_Q = 133          # all multi-session questions (the pool; 200 was a registration typo -- criteria are percentage-based)
SEED = 7
TOPK = 5
EXPAND_K = 10


def main():
    t0 = time.time()
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    mems = [m for m in d["memories"] if m.get("superseded_by") is None]
    E, sess, texts = [], [], []
    for m in mems:
        v = np.asarray(m["embedding"], dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        E.append(v / n)
        sid = None
        for t in m.get("tags") or []:
            t = str(t)
            if t not in ("lme", "consolidated", "chunk") and len(t) >= 8:
                base, _, num = t.rpartition("_")
                sid = base if num.isdigit() else t
                break
        sess.append(sid)
        texts.append(m["text"])
    E = np.stack(E)
    n = len(E)
    qs = json.load(open(QS, encoding="utf-8"))
    ms = [q for q in qs if q["question_type"] == "multi-session"]
    rng = random.Random(SEED)
    subset = rng.sample(ms, N_Q)
    print(f"store {n} | {N_Q} multi-session questions ({time.time()-t0:.0f}s)",
          flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def qvec(text):
        v = np.asarray(model.encode([text], normalize_embeddings=True)[0],
                       dtype=np.float32)
        return v

    def lexical_overlap(pointer_text, question):
        """Deterministic model-follows-pointer proxy: lexical overlap
        between the pointer's truncated text and the question."""
        pw = set(re.sub(r"[^\w\s]", "", pointer_text.lower()).split())
        qw = set(re.sub(r"[^\w\s]", "", question.lower()).split())
        return len(pw & qw) >= 1

    stats = {"BASE": {"turn_cov": [], "tokens": 0, "n": 0},
             "UG": {"turn_cov": [], "tokens": 0, "n": 0}}
    ledger = defaultdict(int)   # use_gate accounting: injected id -> uses

    for qi, q in enumerate(subset):
        qv = qvec(q["question"])
        sims = E @ qv
        top5 = np.argsort(-sims)[:TOPK]

        # ---- BASE: current protocol (truncated text in pool) ----
        base_texts = [texts[i][:80] for i in top5]
        base_sess = {sess[i] for i in top5}
        ans_sess = {s.rsplit("_", 1)[0] if s.rsplit("_", 1)[-1].isdigit()
                    else s for s in q["answer_session_ids"]}
        # turn-level coverage: which answer-session TURNS are in the pool
        base_turns = {t for i in top5 for t in [sess[i]] if sess[i]}
        base_cov = len({t for t in q["answer_session_ids"]
                        if t.rsplit("_", 1)[0] in base_sess or
                        t in base_sess}) / max(len(q["answer_session_ids"]), 1)
        base_tok = sum(len(t) for t in base_texts)
        stats["BASE"]["turn_cov"].append(base_cov)
        stats["BASE"]["tokens"] += base_tok
        stats["BASE"]["n"] += 1

        # ---- UG: pointer + expansion ----
        # ledger accounting: these 5 ids were "injected" as pointers
        for i in top5:
            ledger[int(i)] += 1
        # expansion: pointers whose summary lexically overlaps the question
        # get their full text pulled (deterministic follow-the-pointer),
        # plus a wider top-10 pass (the pointer invites a second look)
        pool_idx = set(top5.tolist())
        for i in top5:
            if lexical_overlap(texts[i][:80], q["question"]):
                pool_idx.add(i)
                # the pointer invites pulling neighbors of a hit
                near = np.argsort(-sims)[TOPK:EXPAND_K]
                for j in near:
                    pool_idx.add(int(j))
        ug_texts = [texts[i] for i in pool_idx]
        ug_sess = {sess[i] for i in pool_idx}
        ug_turns = {t for i in pool_idx for t in [sess[i]] if sess[i]}
        ug_cov = len({t for t in q["answer_session_ids"]
                      if t.rsplit("_", 1)[0] in ug_sess or
                      t in ug_turns}) / max(len(q["answer_session_ids"]), 1)
        ug_tok = sum(len(t[:80] if i in top5 else texts[i])
                     for i in pool_idx)
        stats["UG"]["turn_cov"].append(ug_cov)
        stats["UG"]["tokens"] += ug_tok
        stats["UG"]["n"] += 1
        if (qi + 1) % 50 == 0:
            b = np.mean(stats["BASE"]["turn_cov"])
            u = np.mean(stats["UG"]["turn_cov"])
            print(f"  [{qi+1}/{N_Q}] turn-cov BASE {b:.1%} | UG {u:.1%}",
                  flush=True)

    b_cov = float(np.mean(stats["BASE"]["turn_cov"]))
    u_cov = float(np.mean(stats["UG"]["turn_cov"]))
    b_tok = stats["BASE"]["tokens"] / N_Q
    u_tok = stats["UG"]["tokens"] / N_Q
    d_pp = (u_cov - b_cov) * 100
    tok_ratio = u_tok / b_tok
    top_used = sorted(ledger.items(), key=lambda kv: -kv[1])[:5]
    print(f"\n=== P-USEGATE (deterministic mechanism test) ===")
    print(f"turn-coverage: BASE {b_cov:.1%} | UG {u_cov:.1%} | "
          f"delta {d_pp:+.1f}pp")
    print(f"pool tokens/question: BASE {b_tok:.0f} | UG {u_tok:.0f} "
          f"(x{tok_ratio:.2f})")
    print(f"use_gate ledger: {len(ledger)} distinct ids injected, "
          f"top re-injected {top_used[:3]}")

    if d_pp >= 10 and tok_ratio <= 1.5:
        verdict = "SUPPORTED -- adopt pointer injection + accounting"
    elif d_pp >= 5:
        verdict = "PARTIAL"
    else:
        verdict = "NULL -- pointer form stays cosmetic-only"
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"usegate_{ts}.json")
    json.dump({"experiment": "P-USEGATE", "n": N_Q,
               "turn_cov": {"BASE": round(b_cov, 4), "UG": round(u_cov, 4)},
               "delta_pp": round(d_pp, 2),
               "tokens_per_q": {"BASE": round(b_tok), "UG": round(u_tok),
                                "ratio": round(tok_ratio, 2)},
               "ledger_size": len(ledger), "verdict": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
