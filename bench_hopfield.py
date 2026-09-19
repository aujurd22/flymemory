"""Benchmark: does the Hopfield associative-expansion layer (C) beat pure
vector recall (A) at surfacing *related* memories?

Ground truth for "related": seed-import chunks are grouped by their source
file tag; a query quoting one chunk should ideally surface sibling chunks of
the same file. Metrics: Recall@5 and MRR over the sibling set.

Decision rule: keep the Hopfield layer only if C beats A beyond noise; else
the 64 MiB W matrix and its maintenance cost are not justified.

Run: python bench_hopfield.py [--pkl PATH] [--queries 100]
"""
import argparse
import os
import random
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import load  # noqa: E402


def recall_a(mem, query, top_k=5):
    return [m.memory_id for m, _, _ in mem.recall(query, top_k=top_k)]


def recall_c(mem, query, top_k=5, boost=0.15, hops=2):
    """A + associative expansion: boost candidates that co-activate on W with
    the top-1 hit."""
    base = mem.recall(query, top_k=top_k)
    if not base or mem.W is None:
        return [m.memory_id for m, _, _ in base]
    scored = {}
    for m, sim, eff in base:
        scored[m.memory_id] = eff
    assoc = mem.associate(base[0][0], top_k=10, hops=hops)
    for m, overlap in assoc:
        scored[m.memory_id] = scored.get(m.memory_id, 0.0) + boost * overlap
    ranked = sorted(scored.items(), key=lambda x: -x[1])
    return [mid for mid, _ in ranked[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(_HERE, "flymemory", "flymemory_v3.pkl"))
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not os.path.exists(args.pkl):
        print(f"library not found: {args.pkl}\nThis benchmark needs the real seed-imported library.")
        sys.exit(1)
    mem = load(args.pkl, enable_hopfield=True)
    print(f"library: {mem.size} entries, W={'on' if mem.W is not None else 'off'}")

    # group seed-import chunks by source file tag
    groups = defaultdict(list)
    for m in mem.memories:
        if "zcode-memory" in m.tags or "import" in m.tags:
            file_tag = next((t for t in m.tags if t not in ("zcode-memory", "import", "chunk")), None)
            if file_tag:
                groups[file_tag].append(m)
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if not groups:
        print("no seed-import groups with >=3 chunks; cannot benchmark")
        sys.exit(1)

    rng = random.Random(args.seed)
    file_names = sorted(groups)
    rng.shuffle(file_names)
    samples = file_names[: args.queries]

    hits_a = hits_c = 0
    mrr_a = mrr_c = 0.0
    n = 0
    for fname in samples:
        chunks = groups[fname]
        query_chunk = rng.choice(chunks)
        relevant = {m.memory_id for m in chunks if m.memory_id != query_chunk.memory_id}
        qa = recall_a(mem, query_chunk.text[:100], top_k=5)
        qc = recall_c(mem, query_chunk.text[:100], top_k=5)
        n += 1
        hits_a += len(set(qa) & relevant) / min(5, len(relevant))
        hits_c += len(set(qc) & relevant) / min(5, len(relevant))
        for i, mid in enumerate(qa, 1):
            if mid in relevant:
                mrr_a += 1.0 / i
                break
        for i, mid in enumerate(qc, 1):
            if mid in relevant:
                mrr_c += 1.0 / i
                break

    print(f"queries={n} (one per source file), sibling chunks as ground truth")
    print(f"A pure vector  : Recall@5={hits_a / n:.3f}  MRR={mrr_a / n:.3f}")
    print(f"C vector+assoc : Recall@5={hits_c / n:.3f}  MRR={mrr_c / n:.3f}")
    delta = (hits_c - hits_a) / n
    print(f"delta Recall@5 : {delta:+.3f}")
    print("verdict:", "KEEP Hopfield layer (C wins beyond noise)"
          if delta > 0.02 else "DELETE/skip Hopfield layer (C does not beat A)")


if __name__ == "__main__":
    main()
