"""FlyMemory Demo — pattern completion on real text memories.

Demonstrates:
1. Store conversation memories
2. Recall from partial cues (pattern completion)
3. Multi-compartment prevents interference
4. Capacity test

Run: python demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flymemory import FlyMemoryStore
import numpy as np
import time

print("=" * 60)
print("FlyMemory Demo — Hopfield associative memory for AI conversations")
print("Based on Drosophila mushroom body architecture")
print("=" * 60)

store = FlyMemoryStore(n_compartments=8, n_bits=4096)

# ===== Step 1: Store memories =====
print("\n--- Step 1: Storing memories ---")
conversations = [
    ("User asked about Python async programming",
     "Explained asyncio, await/async syntax, event loops, and aiohttp for web scraping"),
    ("User asked about machine learning basics",
     "Covered supervised vs unsupervised learning, linear regression, decision trees, and neural networks"),
    ("User asked about Docker containers",
     "Explained containerization, images vs containers, Dockerfile, docker-compose, and Kubernetes orchestration"),
    ("User asked about database design",
     "Discussed normalization, ACID properties, indexing strategies, NoSQL vs SQL tradeoffs"),
    ("User asked about web security",
     "Covered OWASP top 10, XSS prevention, CSRF tokens, SQL injection, and HTTPS/TLS"),
    ("User asked about React hooks",
     "Explained useState, useEffect, useMemo, useCallback, and custom hooks patterns"),
    ("User asked about system design",
     "Discussed load balancing, caching strategies, database sharding, and microservices"),
    ("User asked about git workflows",
     "Explained rebase vs merge, git flow, feature branches, and CI/CD pipelines"),
    ("User asked about API design",
     "Covered REST principles, GraphQL, rate limiting, versioning, and authentication"),
    ("User asked about Rust programming",
     "Explained ownership, borrowing, lifetimes, pattern matching, and cargo"),
]

for ctx, resp in conversations:
    mid = store.remember(ctx, resp)
    print(f"  stored memory {mid}: {ctx[:50]}...")

print(f"\n  Total memories stored: {store.size}")

# ===== Step 2: Recall from partial cues =====
print("\n--- Step 2: Pattern completion (partial cue -> full memory) ---")
queries = [
    ("something about async python", "Python async programming"),
    ("how to secure my website", "web security"),
    ("what database should I use", "database design"),
    ("rust language features", "Rust programming"),
    ("design a scalable system", "system design"),
]

for query, expected_topic in queries:
    results = store.recall(query, top_k=1)
    if results:
        mem, ov = results[0]
        match = "[OK]" if expected_topic.lower()[:10] in mem.context.lower() else "[?]"
        print(f"  '{query}' -> '{mem.context[:50]}...' (overlap {ov:.2f}) {match}")
    else:
        print(f"  '{query}' -> no result")

# ===== Step 3: Interference test =====
print("\n--- Step 3: No catastrophic forgetting ---")
print("Storing 50 more memories to test interference...")
for i in range(50):
    store.remember(f"Generic context number {i}", f"Generic response {i}")

print(f"Total memories now: {store.size}")
print("\nRe-test original memories:")
for query, expected_topic in queries[:3]:
    results = store.recall(query, top_k=1)
    if results:
        mem, ov = results[0]
        print(f"  '{query}' -> '{mem.context[:50]}...' (overlap {ov:.2f})")
    else:
        print(f"  '{query}' -> no result")

# ===== Step 4: Capacity =====
print(f"\n--- Capacity ---")
print(f"Current: {store.size} memories in {store.hopfield.n_compartments} compartments")
print(f"Each compartment: 4096 bits, ~500 pattern capacity")
print(f"Total capacity: ~{store.hopfield.n_compartments * 500} memories before degradation")
print(f"(Compartmentalization prevents interference between compartments)")

print("\n=== Demo complete ===")
