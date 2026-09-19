"""Contradiction-resolution benchmark: is memory state-aware retrieval better
than similarity retrieval when facts change over time?

Each scenario stores a sequence of temporal states ("user uses X" -> "user
switched to Y") plus stable distractor facts. Supersede markings are provided
per scenario — in production these markings are made by the calling model,
which is exactly the architecture under test (judgment lives in the caller,
mechanical resolution lives here).

Policies compared (same embedder, same chunking, differ ONLY in how the
current-state winner is chosen):
  dense         : top-1 cosine (classic RAG, no time awareness)
  dense+recency : top-3 cosine, newest timestamp wins (the common "cheap fix")
  bm25          : top-1 IDF lexical match
  bm25+recency  : top-3 lexical, newest wins
  flymemory     : full pipeline — chunked dedup store, power-law decay,
                  lexical boost, superseded entries excluded from recall

Adversarial traps built into the scenarios:
  - old state paraphrases the query better than the new state
  - a NEWER off-hand mention of the OLD state ("misses X", "the old X machine")
    that is not a state change — fools recency heuristics
  - new state that lacks the attribute keyword the query uses

Also measured: history recovery — with include_superseded=True, superseded
facts must remain retrievable (nothing is silently lost).

Run: python bench_contradiction.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

import numpy as np  # noqa: E402

from flymemory.v3 import SmartMemory, split_chunks, _embed, _tokenize  # noqa: E402

DAY = 86400.0

# --------------------------------------------------------------------------
# Scenarios. facts: (text, days_ago). supersedes: (old_idx, new_idx).
# answer: index of the fact that is the CURRENT truth. history_* optional.
# --------------------------------------------------------------------------
SCENARIOS = [
    {
        "facts": [
            ("The user runs Windows 11 on their main laptop.", 400),
            ("The user switched their laptop to Fedora Linux last week.", 5),
            ("The user maintains an e-commerce logistics dashboard.", 200),
            ("The user complained that their old Windows VM in the cloud is slow.", 2),
        ],
        "supersedes": [(0, 1)],
        "query": "Which operating system does the user currently run?",
        "answer": 1,
        "history_query": "Has the user ever run Windows?",
        "history_answer": 0,
    },
    {
        "facts": [
            ("The user writes most backend code in Python.", 500),
            ("The user migrated their backend services to Rust in March.", 10),
            ("The user drinks too much coffee while coding.", 300),
            ("The user keeps an old Python 2 script archive around for reference.", 1),
        ],
        "supersedes": [(0, 1)],
        "query": "What programming language does the user currently use for backend work?",
        "answer": 1,
        "history_query": "Did the user ever write backend code in Python?",
        "history_answer": 0,
    },
    {
        "facts": [
            ("The user lives in Shenzhen.", 600),
            ("The user moved to Hangzhou for a new job two months ago.", 60),
            ("The user likes hiking on weekends.", 250),
            ("The user's Shenzhen apartment lease expired last year.", 3),
        ],
        "supersedes": [(0, 1)],
        "query": "What city does the user live in now?",
        "answer": 1,
        "history_query": "Did the user ever live in Shenzhen?",
        "history_answer": 0,
    },
    {
        "facts": [
            ("The user's primary editor is PyCharm.", 350),
            ("The user moved all daily editing to Neovim recently.", 7),
            ("The user plays badminton twice a week.", 150),
            ("The user uninstalled three JetBrains plugins they no longer needed.", 2),
        ],
        "supersedes": [(0, 1)],
        "query": "What editor or IDE does the user mainly use?",
        "answer": 1,
    },
    {
        "facts": [
            ("The user's phone number is 138-0000-1111.", 450),
            ("The user updated their contact number to 139-9999-8888 in June.", 90),
            ("The user prefers email for formal communication.", 200),
            ("The old 138 number still appears on some printed business cards.", 4),
        ],
        "supersedes": [(0, 1)],
        "query": "What is the user's current phone number?",
        "answer": 1,
        "history_query": "What was the user's old phone number?",
        "history_answer": 0,
    },
    {
        "facts": [
            ("The user is building a ticket-scraping side project.", 300),
            ("The ticket-scraping side project is finished and archived.", 20),
            ("The user now spends evenings on a connectome analysis toolkit.", 15),
            ("The user thinks side projects are fun.", 100),
        ],
        "supersedes": [(0, 1)],
        "query": "What is the user currently working on in the evenings?",
        "answer": 2,
    },
    {
        "facts": [
            ("The user subscribes to the Pro tier of a design tool.", 380),
            ("The user cancelled the Pro subscription in February.", 200),
            ("The user uses the free tier with limited exports now.", 195),
            ("The user once wrote a glowing review of the Pro tier.", 2),
        ],
        "supersedes": [(0, 1), (1, 2)],
        "query": "What is the user's current subscription status for the design tool?",
        "answer": 2,
    },
    {
        "facts": [
            ("The user drives an old gasoline hatchback.", 700),
            ("The user sold the hatchback and bought an electric car in May.", 120),
            ("The user complains about petrol prices out of habit.", 30),
            ("The user installs charging apps on their phone.", 3),
        ],
        "supersedes": [(0, 1)],
        "query": "What kind of car does the user drive?",
        "answer": 1,
    },
    {
        "facts": [
            ("The user reports to Manager Zhang.", 500),
            ("The user's new manager is Manager Liu since the reorg.", 45),
            ("The user attends the weekly team sync every Monday.", 250),
            ("Manager Zhang still appears in some old meeting notes.", 1),
        ],
        "supersedes": [(0, 1)],
        "query": "Who is the user's current manager?",
        "answer": 1,
        "history_query": "Who was the user's manager before the reorg?",
        "history_answer": 0,
    },
    {
        "facts": [
            ("The user's warehouse uses the XinDa label printer.", 400),
            ("The warehouse switched its label printer to the HTW-109 in August.", 35),
            ("The user fixes shipping labels daily.", 200),
            ("An old XinDa ribbon cartridge sits in the supply drawer.", 2),
        ],
        "supersedes": [(0, 1)],
        "query": "Which label printer does the warehouse use?",
        "answer": 1,
    },
    {
        "facts": [
            ("The user studies Japanese with a paper textbook.", 550),
            ("The user switched to a spaced-repetition app for Japanese in April.", 140),
            ("The user watches Japanese dramas without subtitles sometimes.", 90),
            ("The paper textbook still sits on the desk.", 3),
        ],
        "supersedes": [(0, 1)],
        "query": "How does the user study Japanese these days?",
        "answer": 1,
    },
    {
        "facts": [
            ("The user hosts their blog on a rented VPS.", 480),
            ("The user migrated the blog to a static hosting platform in July.", 55),
            ("The user writes two posts per month.", 220),
            ("An old VPS backup archive still exists on their disk.", 1),
        ],
        "supersedes": [(0, 1)],
        "query": "Where is the user's blog hosted?",
        "answer": 1,
    },
    {
        "facts": [
            ("The user prefers dark mode in every app.", 600),
            ("The user switched to light mode after eye surgery last month.", 30),
            ("The user reads on a tablet in the evening.", 200),
            ("Their old screenshots all show dark mode themes.", 2),
        ],
        "supersedes": [(0, 1)],
        "query": "Does the user prefer dark mode or light mode?",
        "answer": 1,
    },
    {
        "facts": [
            ("The user's team is called Team Falcon.", 520),
            ("The team was renamed to Team Kestrel in September.", 12),
            ("The user designs circuit boards at work.", 250),
            ("A Falcon logo is still printed on their mug.", 2),
        ],
        "supersedes": [(0, 1)],
        "query": "What is the user's team called?",
        "answer": 1,
        "history_query": "What was the team's old name?",
        "history_answer": 0,
    },
]


def embed_query(q):
    """Production-style: max cosine over query chunks."""
    return [_embed(c) for c in split_chunks(q)]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-8) / (np.linalg.norm(b) + 1e-8))


def dense_topk(mem_emb, q_embs, k):
    sims = np.array([max(cos(qe, e) for qe in q_embs) for e in mem_emb])
    return np.argsort(-sims)[:k], sims


def bm25_scores(tokenized_docs, q_tokens, df, n_docs):
    import math
    total = 0.0
    acc = np.zeros(len(tokenized_docs))
    for tok in q_tokens:
        d = df.get(tok, 0)
        if not d:
            continue
        idf = math.log(1.0 + n_docs / d)
        total += idf
        for i, doc in enumerate(tokenized_docs):
            if tok in doc:
                acc[i] += idf
    return acc / (total + 1e-9)


def run_policy(name, scenario_data):
    """Returns (current@1 hits, current@3 hits, history@2 hits, stale-top1)."""
    cur1 = cur3 = hist_hit = stale = 0
    policies = name.split("+")  # e.g. "dense", "dense+recency", "bm25+recency", "flymemory"
    for sc in scenario_data:
        if name == "flymemory":
            mem = SmartMemory(n_bits=4096)  # decay_tau default 30d
            id_of = {}
            for i, (text, days_ago) in enumerate(sc["facts"]):
                r = mem.remember(text, tags=[f"ev{i}"], source="import",
                                 timestamp=time.time() - days_ago * DAY)
                id_of[i] = r["memory_id"]
            for old, new in sc["supersedes"]:
                mem.supersede(id_of[old], id_of[new])
            hits = mem.recall(sc["query"], top_k=3)
            got = [next(int(t[2:]) for t in h[0].tags if t.startswith("ev"))
                   for h in hits]
            cur1 += bool(got and got[0] == sc["answer"])
            cur3 += (sc["answer"] in got)
            if "history_query" in sc:
                hh = mem.recall(sc["history_query"], top_k=2, include_superseded=True)
                got_hist = {int(t[2:]) for h in hh for t in h[0].tags if t.startswith("ev")}
                hist_hit += (sc["history_answer"] in got_hist)
            continue

        # shared-setup baselines: plain lists, no supersede knowledge
        texts = [f[0] for f in sc["facts"]]
        times = np.array([time.time() - f[1] * DAY for f in sc["facts"]])
        doc_embs = [_embed(t) for t in texts]
        q_embs = embed_query(sc["query"])
        top_idx, sims = dense_topk(doc_embs, q_embs, 3)
        if "recency" in policies:
            pick = top_idx[np.argsort(-times[top_idx])][0]
        else:
            pick = top_idx[0]
        if name.startswith("dense"):
            cur1 += (int(pick) == sc["answer"])
            cur3 += (sc["answer"] in [int(i) for i in top_idx])
            stale += int(int(pick) in [o for o, _ in sc["supersedes"]])
        else:  # bm25
            docs = [_tokenize(t) for t in texts]
            df = {}
            for d in docs:
                for tok in d:
                    df[tok] = df.get(tok, 0) + 1
            scores = bm25_scores(docs, _tokenize(sc["query"]), df, len(docs))
            top = np.argsort(-scores)[:3]
            pick = top[np.argsort(-times[top])][0] if "recency" in policies else top[0]
            cur1 += (int(pick) == sc["answer"])
            cur3 += (sc["answer"] in [int(i) for i in top])
            stale += int(int(pick) in [o for o, _ in sc["supersedes"]])
        if "history_query" in sc and name.startswith(("dense", "bm25")):
            qh = embed_query(sc["history_query"])
            tidx, _ = dense_topk(doc_embs, qh, 2) if name.startswith("dense") else (None, None)
            if tidx is not None:
                hist_hit += (sc["history_answer"] in [int(i) for i in tidx])
            else:
                docs = [_tokenize(t) for t in texts]
                df = {}
                for d in docs:
                    for tok in d:
                        df[tok] = df.get(tok, 0) + 1
                s = bm25_scores(docs, _tokenize(sc["history_query"]), df, len(docs))
                hist_hit += (sc["history_answer"] in [int(i) for i in np.argsort(-s)[:2]])
    return cur1, cur3, hist_hit, stale


def main():
    n = len(SCENARIOS)
    n_hist = sum(1 for s in SCENARIOS if "history_query" in s)
    print(f"scenarios: {n} (history sub-queries: {n_hist}); "
          f"policies share one embedder and one chunker; supersede markings "
          f"simulate the calling model's judgment\n")

    rows = []
    for name in ["dense", "dense+recency", "bm25", "bm25+recency", "flymemory"]:
        c1, c3, hist, stale = run_policy(name, SCENARIOS)
        rows.append((name, c1, c3, hist, stale))

    print(f"{'policy':14s} {'current@1':>10s} {'current@3':>10s} {'history@2':>10s} {'stale top1':>11s}")
    for name, c1, c3, hist, stale in rows:
        h_str = f"{hist}/{n_hist}" if n_hist else "-"
        print(f"{name:14s} {c1:>4d}/{n:<4d} {c3:>4d}/{n:<4d} {h_str:>10s} {stale:>5d}/{n:<5d}")

    print("\ncurrent@1: correct CURRENT fact ranked first mechanically; "
          "current@3: it appears in top-3 (the calling model resolves from the "
          "hook-injected top-3 with age/provenance stamps); "
          "stale top1 = a superseded fact ranked first; "
          "history@2 = superseded fact recoverable (flymemory uses include_superseded).")


if __name__ == "__main__":
    main()
