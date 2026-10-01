"""P-AUDIT: map-reduce unsupported-claims gate (registered, RESEARCH.md
P-2026-09-30-AUDIT -- verdict PENDING there; criteria locked before run).

Replays the P-ANSWER MAPREDUCE path on the same 50 multi-session
questions (seed 7, temperature 0), persisting extracted fact lists AND
final answers, then audits every answer for claims unsupported by its
own fact list.

Gate: unsupported-claim rate < 5% -> pass (protocol may be wired into
classify_query aggregation routing, caller-side).

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_mapreduce_audit.py
"""
import json
import os
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from bench_memory_judgment import (  # noqa: E402
    ds_client, ds_chat, parse_decision)
from bench_needle_search import load_store  # noqa: E402
from bench_answer_side import MAP_SYSTEM, ANSWER_SYSTEM  # noqa: E402

QS = os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json")
N_Q = 50
FINAL_K = 30
AUDIT_SYSTEM = """You are a strict fact-support auditor. Given a QUESTION, the
EXTRACTED FACTS (the only permitted evidence), and a FINAL ANSWER, check
whether the answer contains any SPECIFIC claim (number, date, entity
name, outcome) that NO extracted fact supports. Ignore hedging and
"don't know" phrasing. Reply ONLY:
{"unsupported": true/false, "claims": ["..."]}
- unsupported=true only if at least one specific claim lacks support."""
MODEL = "deepseek-chat"


def main():
    t0 = time.time()
    E, sess, turn_keys = load_store()
    texts = []
    import pickle
    from bench_needle_search import SNAP
    dd = pickle.load(open(SNAP, "rb"))
    for m in dd["memories"]:
        if m.get("superseded_by") is not None:
            continue
        import numpy as _np
        v = _np.asarray(m["embedding"], dtype=_np.float32)
        if _np.linalg.norm(v) == 0:
            continue
        texts.append(m["text"])
    assert len(texts) == len(E), f"{len(texts)} != {len(E)}"
    qs = json.load(open(QS, encoding="utf-8"))
    ms = [q for q in qs if q["question_type"] == "multi-session"]
    subset = random.Random(7).sample(ms, N_Q)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = ds_client()

    rows, flagged = [], 0
    for qi, q in enumerate(subset):
        qv = np.asarray(model.encode([q["question"]], normalize_embeddings=True)[0],
                        dtype=np.float32)
        pool = np.argsort(-(E @ qv))[:FINAL_K]
        facts = []
        for k, i in enumerate(pool):
            out = ds_chat(client, MAP_SYSTEM,
                          f"Question: {q['question']}\n\n"
                          f"Memory entry: {texts[i][:300]}\n\n"
                          f"Extract relevant atomic facts (or NONE).")
            if out.strip() and "NONE" not in out[:20]:
                facts.append(f"[{k+1}] " + " ".join(out.strip().split("\n")[:4])[:220])
        fact_ctx = "\n".join(facts) if facts else "(no relevant facts found)"
        answer = ds_chat(client, ANSWER_SYSTEM,
                         f"Question: {q['question']}\n\nExtracted atomic "
                         f"facts from memory:\n{fact_ctx}\n\nAnswer the "
                         f"question using only these facts. If they are "
                         f"insufficient, say you don't know.")
        araw = ds_chat(client, AUDIT_SYSTEM,
                       f"QUESTION: {q['question']}\n\nEXTRACTED FACTS:\n"
                       f"{fact_ctx}\n\nFINAL ANSWER: {answer}\n\nAudit it.")
        dec, _ = parse_decision(araw)
        unsupported = bool(dec.get("unsupported")) if dec else None
        claims = dec.get("claims", []) if dec else []
        if unsupported:
            flagged += 1
        rows.append({"qi": qi, "question": q["question"][:120],
                     "facts": facts, "answer": answer,
                     "unsupported": unsupported, "claims": claims})
        print(f"[{qi+1}/{N_Q}] unsupported={unsupported} "
              f"{claims[:1] if claims else ''}", flush=True)

    rate = flagged / N_Q
    verdict = ("GATE PASSED -- protocol may be wired into aggregation "
               "routing (caller-side)" if rate < 0.05 else
               "GATE FAILED -- tighten MAP prompt and re-audit once")
    print(f"\nunsupported-claim rate: {flagged}/{N_Q} = {rate:.1%}")
    print(f"PRE-REGISTERED VERDICT: {verdict} (rate {rate:.1%})")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"mapreduce_audit_{ts}.json")
    json.dump({"experiment": "P-AUDIT", "n": N_Q, "flagged": flagged,
               "rate": round(rate, 4), "verdict": verdict, "rows": rows},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
