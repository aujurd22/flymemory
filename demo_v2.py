"""FlyMemory v2.0 Demo — semantic encoding + dopamine gating + pattern completion."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flymemory.v2 import FlyMemoryV2
import numpy as np

print("=" * 60)
print("FlyMemory v2.0 — Semantic + Dopamine + Pattern Completion")
print("=" * 60)

mem = FlyMemoryV2(n_bits=4096, dopamine_threshold=0.3)

# Step 1: Store memories with dopamine gating
print("\n--- Step 1: 多巴胺门控存储 ---")
inputs = [
    "User asked about Python async programming and event loops",
    "User asked about machine learning and neural networks",
    "User asked about Docker and Kubernetes deployment",
    "User asked about Python async again with more detail",  # 应被拒绝（不够新颖）
    "User asked about Rust ownership and borrowing",
    "User asked about database ACID properties",
    "Same Python async question repeated",  # 应被拒绝
    "User asked about web security and OWASP",
]
for text in inputs:
    result = mem.remember(text)
    status = "✓ stored" if result["stored"] else "✗ rejected"
    reason = result.get("reason", "")
    print(f"  [{status}] {text[:55]}...")
    if reason:
        print(f"           {reason}")

print(f"\n  存储了 {mem.stats['stored']}/{mem.stats['total_inputs']} (多巴胺门控拒绝了 {mem.stats['total_inputs']-mem.stats['stored']} 个重复)")

# Step 2: Pattern completion
print("\n--- Step 2: Pattern completion (partial cue → full recall) ---")
queries = [
    ("something about async programming", "Python async programming"),
    ("how to deploy containers", "Docker and Kubernetes deployment"),
    ("what is ownership in Rust", "Rust ownership and borrowing"),
    ("neural network basics", "machine learning and neural networks"),
]
for query, expected in queries:
    results = mem.recall(query, top_k=1)
    if results:
        text, ov = results[0]
        match = "✓" if expected.lower()[:10] in text.lower() else "?"
        print(f"  '{query}' → overlap {ov:.2f} {match}")
    else:
        print(f"  '{query}' → no result")

# Step 3: Stats
print(f"\n--- Stats ---")
print(f"  Total memories: {mem.size}")
print(f"  Stored: {mem.stats['stored']}, Rejected (not novel): {mem.stats['rejected_novelty']}")

print("\n=== Demo complete ===")
