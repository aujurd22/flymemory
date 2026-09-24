"""GitHub survey: 1000+ repos across memory / AI-memory / bio-inspired AI.

Multi-query GitHub Search API sweep -> dedup -> score -> raw JSON dump.
Queries are grouped in buckets so the final report can slice by theme.

Run: python github_survey.py [--token-env GITHUB_TOKEN] [--max-per-query 200]
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

QUERIES = {
    # --- bucket: agent/LLM memory systems ---
    "agent_memory": [
        "LLM memory management",
        "agent memory framework",
        "long-term memory LLM",
        "conversational memory chatbot",
        "memory augmented language model",
        "AI assistant memory persistence",
        "chatbot long-term memory vector",
        "memory for AI agents",
    ],
    # --- bucket: named systems & prior art ---
    "named_systems": [
        "mem0",
        "memgpt",
        "letta agent",
        "zep graph memory",
        "memory rag store",
        "hipporag",
        "a-mem agentic memory",
        "generative agents memory",
    ],
    # --- bucket: memory benchmarks & research ---
    "benchmarks": [
        "longmemeval",
        "locomo benchmark memory",
        "memory benchmark evaluation LLM",
        "temporal reasoning benchmark",
        "episodic memory benchmark",
        "memory forgetting benchmark LLM",
    ],
    # --- bucket: temporal / state / knowledge-graph memory ---
    "temporal_state": [
        "temporal knowledge graph",
        "knowledge graph memory agent",
        "entity memory state tracking",
        "graph memory LLM",
        "temporal validity facts",
        "bi-temporal knowledge base",
    ],
    # --- bucket: memory consolidation / forgetting ---
    "consolidation": [
        "memory consolidation neural network",
        "catastrophic forgetting continual learning",
        "experience replay agent",
        "active forgetting machine learning",
        "elastic weight consolidation",
        "generative replay",
        "sleep consolidation AI",
    ],
    # --- bucket: bio-inspired (fly/mushroom body) ---
    "bio_fly": [
        "fruit fly mushroom body",
        "drosophila connectome computational",
        "mushroom body model learning",
        "kenyon cell sparse coding",
        "drosophila hemisphere lateralization model",
        "insect inspired navigation model",
        "feedback neurons dilution fly",
    ],
    # --- bucket: bio-inspired (general) ---
    "bio_general": [
        "hopfield network modern",
        "dense associative memory",
        "winner take all neural network",
        "k-winner-take-all sparse",
        "neuromorphic memory architecture",
        "olfactory circuit model spiking",
        "synaptic pruning algorithm",
        "hebbian learning implementation",
        "brain inspired continual learning",
        "sparse distributed memory kanerva",
    ],
    # --- bucket: memory formats & stores ---
    "formats": [
        "vector database memory agent",
        "semantic memory store",
        "episodic semantic procedural memory",
        "hierarchical memory network",
        "retrieval augmented generation memory",
        "memory compression language model",
        "lifelong learning language model",
    ],
}


def gh_get(url, token=None, retries=3):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "flymemory-survey",
    })
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                print(f"    FAIL {url[:80]}: {e}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-env", default="GITHUB_TOKEN")
    ap.add_argument("--max-per-query", type=int, default=200)
    ap.add_argument("--out", default="reports/github_survey_raw.json")
    args = ap.parse_args()
    token = os.environ.get(args.token_env) or None

    all_repos = {}
    bucket_counts = {}
    for bucket, queries in QUERIES.items():
        for qi, q in enumerate(queries):
            gq = urllib.parse.quote(q)
            got = 0
            for page in (1, 2):
                url = (f"https://api.github.com/search/repositories?"
                       f"q={gq}&per_page=100&page={page}&sort=stars&order=desc")
                r = gh_get(url, token)
                if not r or "items" not in r:
                    break
                for item in r["items"]:
                    full = item["full_name"]
                    if full in all_repos:
                        all_repos[full]["buckets"].add(bucket)
                        continue
                    all_repos[full] = {
                        "full_name": full,
                        "url": item["html_url"],
                        "description": (item.get("description") or "")[:300],
                        "stars": item["stargazers_count"],
                        "language": item.get("language"),
                        "topics": item.get("topics", []),
                        "pushed_at": item.get("pushed_at"),
                        "created_at": item.get("created_at"),
                        "archived": item.get("archived", False),
                        "buckets": {bucket},
                    }
                    got += 1
                if len(r["items"]) < 100:
                    break
                time.sleep(2.2)  # search api: 30 req/min authenticated
            print(f"[{bucket}] ({qi+1}/{len(queries)}) {q!r}: +{got} "
                  f"(total {len(all_repos)})", flush=True)
        bucket_counts[bucket] = sum(1 for r in all_repos.values()
                                    if bucket in r["buckets"])

    print(f"\nTOTAL unique repos: {len(all_repos)}")
    for b, n in bucket_counts.items():
        print(f"  {b:16s} {n}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    serial = {k: {**v, "buckets": sorted(v["buckets"])} for k, v in all_repos.items()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"total": len(all_repos), "repos": serial}, f,
                  ensure_ascii=False)
    print(f"raw dump: {args.out}")


if __name__ == "__main__":
    main()
