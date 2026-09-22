"""LongMemEval (oracle edition) adapter: retrieval-effectiveness benchmark.

Ingests all unique evidence-haystack sessions of LongMemEval-oracle into one
shared SmartMemory (turn granularity, session-date timestamps, session_id
tags), then answers all 500 questions by recall. Metric: evidence-hit — an
ANSWER-SESSION chunk entering top-k (answer generation + LLM judging is a
separate downstream layer). Baselines on the same store: recency-only,
BM25-only.

Run: python bench_longmemeval.py [--data PATH] [--max-turns N]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import SmartMemory, MemoryEntry, _embed, _tokenize  # noqa: E402


def parse_lme_date(s):
    """'2023/05/21 (Sun) 21:10' -> epoch seconds (None on parse failure)."""
    try:
        return datetime.strptime(s.split("(")[0].strip(), "%Y/%m/%d %H:%M").timestamp()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "data_longmemeval", "longmemeval_oracle.json"))
    ap.add_argument("--max-turns", type=int, default=10_000_000)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    K = args.topk

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    print(f"questions: {len(data)}", flush=True)

    sessions = {}
    for q in data:
        for sid, ds, sess in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
            if sid not in sessions:
                ts = parse_lme_date(ds)
                turns = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in sess]
                sessions[sid] = (ts, turns)
    order = sorted(sessions.items(), key=lambda kv: (kv[1][0] is None, kv[1][0]))
    total_turns = sum(len(t) for _, t in order)
    print(f"unique sessions: {len(order)}, total turns: {total_turns}", flush=True)

    mem = SmartMemory(n_bits=4096, code_keep=0.5)
    accepted = []          # normalized embeddings of accepted turns (greedy near-dup gate)
    done = 0
    t_ing = time.time()
    for sid, (ts, turns) in order:
        ts = ts or 0.0
        for turn in turns:
            if done >= args.max_turns:
                break
            emb = _embed(turn)
            en = emb / (np.linalg.norm(emb) + 1e-8)
            dup = False
            if accepted:
                acc = np.stack(accepted[-4000:])
                sims = acc @ en
                if float(sims.max()) > 0.985:
                    dup = True
            if dup:
                done += 1
                continue
            accepted.append(en)
            e = MemoryEntry(text=turn, response="", embedding=en,
                            timestamp=ts, last_accessed=ts, access_count=0,
                            tags=["lme", sid], memory_id=mem._next_id)
            mem._next_id += 1
            mem.memories.append(e)
            mem._index_entry(e)
            mem._mat_dirty = True
            mem._codes_dirty = True
            done += 1
    print(f"ingested turns: {done} in {time.time()-t_ing:.0f}s; library: {mem.size}", flush=True)

    # ---- QA scoring structures ----
    E = mem._emb_matrix()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    now = time.time()
    last = np.array([m.last_accessed for m in mem.memories], dtype=np.float64)
    taus = np.array([mem._decay_tau_for(m) for m in mem.memories], dtype=np.float64)
    acc_arr = np.array([m.access_count for m in mem.memories], dtype=np.float64)
    dw = np.minimum(((1.0 + (now - last) / taus) ** -0.5)
                    * (1.0 + 0.5 * np.log2(1.0 + acc_arr)), 1.0)
    src_w = np.array([{"model": 1.15, "import": 1.15}.get(m.source, 1.0)
                      for m in mem.memories], dtype=np.float32)
    entry_tokens = [_tokenize(m.text) for m in mem.memories]
    df = {}
    for toks in entry_tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n_docs = len(entry_tokens)

    def bm25_top(q_text, k):
        scored = {}
        total = 0.0
        for c in split_chunks(q_text) or [q_text]:
            for tok in _tokenize(c):
                d = df.get(tok, 0)
                idf = np.log(1.0 + n_docs / d) if d else idf_max
                total += idf
                if d:
                    for i, toks in enumerate(entry_tokens):
                        if tok in toks:
                            scored[i] = scored.get(i, 0.0) + idf
        return sorted(scored, key=lambda i: -scored[i])[:k]

    def hit_at(order, k, answer_sessions):
        for rank, idx in enumerate(order[:k], 1):
            if answer_sessions & set(mem.memories[idx].tags):
                return True
        return False

    # ---- QA over all questions ----
    arms = {"full": [], "recency": [], "bm25": []}
    per_type = {}
    t_qa = time.time()
    for q in data:
        q_text = q["question"]
        answer_sessions = set(q.get("answer_session_ids", []))
        q_embs = [_embed(c) for c in (split_chunks(q_text) or [q_text])]
        Q = np.stack(q_embs)
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sims = (qn @ En.T).max(axis=0)

        order_full = np.argsort(-(sims * dw * src_w))
        order_rec = np.argsort(-last)
        order_bm = bm25_top(q_text, 20)

        answer_idx = set()
        for i, m in enumerate(mem.memories):
            if answer_sessions & set(m.tags):
                answer_idx.add(i)

        qtype = q.get("question_type", "?")
        for arm, order in (("full", order_full), ("recency", order_rec), ("bm25", order_bm)):
            hit = any(idx in answer_idx for idx in order[:K])
            arms[arm].append(hit)
        per_type.setdefault(qtype, []).append(
            any(idx in answer_idx for idx in order_full[:K]))

    n = len(data)
    print(f"\n=== evidence-hit@{K} (N={n}) ===")
    for arm in ("full", "recency", "bm25"):
        hits = sum(arms[arm])
        print(f"  {arm:10s} {hits:4d}/{n} = {hits/n*100:.0f}%")
    print("\n=== per question_type (full pipeline) ===")
    for qt, res in sorted(per_type.items()):
        h = sum(res)
        print(f"  {qt:28s} {h:3d}/{len(res):3d} = {h/len(res)*100:.0f}%")


if __name__ == "__main__":
    main()
