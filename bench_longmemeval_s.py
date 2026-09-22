"""LongMemEval-S (cleaned) full-scale benchmark: 500 questions over ~200k turns.

Pipeline: bulk turn-granularity ingest (batch-encode, direct construction) →
production recall for all 500 questions → evidence-session hit@k, per ability
type, plus recency/BM25 baselines on the same store.

Turn granularity (not sentence-chunked) keeps the library at ~200k entries;
production hook splits sentences, so S-edition turn-level numbers are the
conservative floor for retrieval quality.

Run: python bench_longmemeval_s.py [--max-turns 210000] [--topk 3]
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import SmartMemory, MemoryEntry, _embed, _tokenize, split_chunks  # noqa: E402


def parse_lme_date(s):
    """'2023/05/20 (Sat) 02:21' / '2023/05/21 21:10' -> epoch seconds.

    The S-edition haystack_dates put the weekday paren MIDWAY, so the old
    split('(')[0] stripped the time too and every parse silently returned
    None -> 199,509 entries with last_accessed=None -> dw all-NaN -> the
    production-scoring arm scored a bogus 0/500 (found 2026-09-23 via
    diag_full_zero.py). Regex extraction + a hard failure now."""
    m = re.search(r"(\d{4}/\d{2}/\d{2})(?:\s*\([^)]*\))?\s*(\d{1,2}:\d{2})?", s)
    if not m:
        raise ValueError(f"unparseable LME date: {s!r}")
    date, hm = m.group(1), m.group(2) or "00:00"
    return datetime.strptime(f"{date} {hm}", "%Y/%m/%d %H:%M").timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json"))
    ap.add_argument("--max-turns", type=int, default=210_000)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    K = args.topk

    t0 = time.time()
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    print(f"parse: {len(data)} questions in {time.time()-t0:.0f}s", flush=True)

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

    # ---- bulk ingest: batch encode, direct construction ----
    from flymemory.v3 import _get_model
    model = _get_model()
    mem = SmartMemory(n_bits=4096, code_keep=0.5)
    rng = np.random.default_rng(11)

    turns_flat = []
    for sid, (ts, turns) in order:
        for turn in turns:
            turns_flat.append((ts, sid, turn))
    turns_flat = turns_flat[: args.max_turns]

    B = 4000
    t_enc = time.time()
    for b0 in range(0, len(turns_flat), B):
        batch = turns_flat[b0 : b0 + B]
        texts = [x[2] for x in batch]
        embs = model.encode(texts, batch_size=64, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)
        now = time.time()
        for (ts, sid, turn), emb in zip(batch, embs):
            e = MemoryEntry(text=turn, response="", embedding=emb,
                            timestamp=ts, last_accessed=ts, access_count=0,
                            tags=["lme", sid], memory_id=mem._next_id)
            mem._next_id += 1
            mem.memories.append(e)
            mem._index_entry(e)
        mem._mat_dirty = True
        mem._codes_dirty = True
        el = time.time() - t_enc
        print(f"  encoded {b0+len(batch)}/{len(turns_flat)} turns "
              f"({el:.0f}s, {len(batch)/max(el,1):.0f}/s)", flush=True)
    print(f"library: {mem.size} entries; encode total {time.time()-t_enc:.0f}s", flush=True)

    # ---- caches ----
    E = mem._emb_matrix()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    now = time.time()
    last = np.array([m.last_accessed for m in mem.memories], dtype=np.float64)
    taus = np.array([mem._decay_tau_for(m) for m in mem.memories], dtype=np.float64)
    acc = np.array([m.access_count for m in mem.memories], dtype=np.float64)
    dw = np.minimum(((1.0 + (now - last) / taus) ** -0.5)
                    * (1.0 + 0.5 * np.log2(1.0 + acc)), 1.0)
    src_w = np.array([{"model": 1.15, "import": 1.15}.get(m.source, 1.0)
                      for m in mem.memories], dtype=np.float32)
    entry_tokens = [set(_tokenize(m.text)) for m in mem.memories]
    df = {}
    for toks in entry_tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n_docs = len(entry_tokens)
    print(f"caches ready; df tokens: {len(df)}", flush=True)
    n = len(data)
    from flymemory.v3 import save
    save(mem, os.path.join(_HERE, "longmemeval_s_bench.pkl"))
    print(f"benchmark library saved ({mem.size} entries)", flush=True)

    # ---- QA: 500 questions, evidence-hit@k ----
    per_type = {}
    arm_hits = {"full": 0, "recency": 0, "bm25": 0}
    arm10 = {"full": 0}
    t_qa = time.time()
    for qi, q in enumerate(data):
        q_text = q["question"]
        answer_sessions = set(q.get("answer_session_ids", []))
        chunks = split_chunks(q_text) or [q_text]
        q_embs = [_embed(c) for c in chunks]
        Q = np.stack(q_embs)
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sims = (qn @ En.T).max(axis=0)
        eff = sims * dw * src_w
        order = np.argsort(-eff)

        hit3 = hit10 = False
        for rank, idx in enumerate(order[:10], 1):
            if answer_sessions & set(mem.memories[idx].tags):
                if rank <= K:
                    hit3 = True
                hit10 = True
                break
        arm_hits["full"] += hit3
        arm10["full"] += hit10
        qt = q.get("question_type", "?")
        per_type.setdefault(qt, [0, 0])
        per_type[qt][0] += hit3
        per_type[qt][1] += 1

        # recency baseline: newest turns first
        order_rec = np.argsort(-last)
        for idx in order_rec[:K]:
            if answer_sessions & set(mem.memories[idx].tags):
                arm_hits["recency"] += 1
                break

        # bm25 baseline
        scored = {}
        total = 0.0
        for c in chunks:
            for tok in _tokenize(c):
                d = df.get(tok, 0)
                idf = np.log(1.0 + n_docs / d) if d else np.log(1.0 + n_docs)
                total += idf
                if d:
                    for i, toks in enumerate(entry_tokens):
                        if tok in toks:
                            scored[i] = scored.get(i, 0.0) + idf
        for rank, idx in enumerate(sorted(scored, key=lambda i: -scored[i])[:K], 1):
            if answer_sessions & set(mem.memories[idx].tags):
                arm_hits["bm25"] += 1
                break

        if (qi + 1) % 100 == 0:
            print(f"  QA {qi+1}/500 done ({time.time()-t_qa:.0f}s)", flush=True)

    print(f"\nQA total {time.time()-t_qa:.0f}s")
    print(f"\n=== evidence-hit@{K} (N={n}, {mem.size} entries) ===")
    for arm in ("full", "recency", "bm25"):
        print(f"  {arm:10s} {arm_hits[arm]:4d}/{n} = {arm_hits[arm]/n*100:.0f}%")
    print("\n=== per question_type (full) ===")
    for qt, (h, tot) in sorted(per_type.items()):
        print(f"  {qt:28s} {h:3d}/{tot:3d} = {h/max(tot,1)*100:.0f}%")


if __name__ == "__main__":
    main()
