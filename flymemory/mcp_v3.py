"""FlyMemory v3 MCP Server — smart memory with auto-dedup, semantic search, and decay.

Tools:
  flymemory_remember(text, tags) → store (auto-dedup via semantic similarity)
  flymemory_recall(query, top_k) → semantic search with decay weighting
  flymemory_stats() → statistics including decay info
  flymemory_cleanup() → remove memories that have decayed below threshold
"""
import sys, os, json, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "flymemory_v3.pkl")

_mem = None

def get_memory():
    global _mem
    if _mem is None:
        from flymemory.v3 import SmartMemory, load
        if os.path.exists(DB_PATH):
            _mem = load(DB_PATH)
        else:
            _mem = SmartMemory(n_bits=4096, decay_half_life=3600.0)
    return _mem

def save_memory():
    if _mem is None: return
    from flymemory.v3 import save
    save(_mem, DB_PATH)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("flymemory", instructions="""
FlyMemory v3: Smart associative memory inspired by Drosophila mushroom body.
Features: auto-dedup (semantic similarity), semantic search (MiniLM cosine),
memory decay (Ebbinghaus forgetting curve), dopamine gating (novelty-based storage).

Use flymemory_remember to store important findings.
Use flymemory_recall to retrieve relevant memories via semantic search.
Use flymemory_cleanup to remove decayed memories.
""")

@mcp.tool()
def flymemory_remember(text: str, tags: str = "") -> str:
    """Store an important finding or decision.

    Auto-dedup: if semantically similar to an existing memory,
    strengthens/merges instead of creating a duplicate.
    Args:
        text: The text to remember
        tags: Optional comma-separated tags
    """
    mem = get_memory()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    result = mem.remember(text, tags=tag_list)
    save_memory()
    action = result["action"]
    nov = result.get("novelty", 0)
    mid = result.get("memory_id", "?")
    if action == "new":
        return f"[NEW #{mid}] {text[:80]}"
    elif action == "merged":
        return f"[MERGED #{mid}] {text[:80]}"
    elif action == "strengthened":
        return f"[STRENGTHENED #{mid}] {text[:80]}"
    else:
        return f"[REJECTED] too similar (novelty={nov:.2f})"

@mcp.tool()
def flymemory_recall(query: str, top_k: int = 5) -> str:
    """Recall relevant memories via semantic search + Hopfield pattern completion.

    Args:
        query: The query text (can be partial/incomplete)
        top_k: Number of memories to return
    Returns:
        Relevant memories with scores, ranked by semantic similarity × decay weight
    """
    mem = get_memory()
    results = mem.recall(query, top_k=top_k)
    if not results:
        return "No relevant memories found."
    output = []
    for entry, sim, eff in results:
        decay_pct = f"decay={mem_decay_pct(entry, mem):.0f}%"
        output.append(f"[sim={sim:.2f} {decay_pct}] {entry.text[:80]}")
    return "\n".join(output)

def mem_decay_pct(entry, mem):
    dw = mem._decay_weight(entry)
    return dw * 100

@mcp.tool()
def flymemory_stats() -> str:
    """Get memory system statistics."""
    mem = get_memory()
    if mem.size == 0:
        return "Memory empty."
    now = time.time()
    ages = [(now - m.timestamp) / 60 for m in mem.memories]  # minutes
    access_counts = [m.access_count for m in mem.memories]
    avg_decay = np.mean([mem._decay_weight(m) for m in mem.memories])
    return (f"Memories: {mem.size}, "
            f"Oldest: {max(ages):.0f}min, Newest: {min(ages):.0f}min, "
            f"Avg access count: {np.mean(access_counts):.1f}, "
            f"Avg decay weight: {avg_decay:.2f}")

@mcp.tool()
def flymemory_cleanup(min_retention: float = 0.1) -> str:
    """Remove memories that have decayed below the retention threshold.

    Args:
        min_retention: Minimum decay weight to keep (default 0.1)
    """
    mem = get_memory()
    removed = mem.decay_cleanup(min_retention)
    if removed > 0:
        save_memory()
    return f"Removed {removed} decayed memories. Remaining: {mem.size}"

@mcp.tool()
def flymemory_auto(context: str, response: str = "") -> str:
    """Automatic memory management — call this every conversation turn.

    Does two things in one call:
    1. RECALL: searches for memories relevant to the current context
    2. STORE: stores this interaction if it's novel enough (dopamine gate)

    This is the "always-on" memory tool. Call it at the start of every
    conversation turn with the user's message as context, and optionally
    the AI's response.

    Args:
        context: The user's message or current conversation context
        response: Optional — the AI's response to store
    Returns:
        Combined recall results + storage confirmation
    """
    mem = get_memory()
    output_parts = []

    # ===== RECALL: find relevant memories =====
    if mem.size > 0:
        results = mem.recall(context, top_k=3)
        if results:
            recall_parts = []
            for entry, sim, eff in results:
                if sim > 0.4:  # only report meaningful matches
                    recall_parts.append(f"  [{sim:.2f}] {entry.text[:80]}")
            if recall_parts:
                output_parts.append("RECALLED MEMORIES:")
                output_parts.extend(recall_parts)
            else:
                output_parts.append("RECALL: no relevant memories for this topic.")
        else:
            output_parts.append("RECALL: memory empty.")
    else:
        output_parts.append("RECALL: memory empty (first use).")

    # ===== STORE: store this interaction if novel =====
    combined_text = context
    if response:
        combined_text = f"{context} ||| {response}"
    result = mem.remember(combined_text, tags=["auto"])
    save_memory()

    action = result["action"]
    if action == "new":
        output_parts.append(f"STORED: [NEW #{result.get('memory_id','?')}] {context[:60]}")
    elif action == "merged":
        output_parts.append(f"STORED: [MERGED] {context[:60]}")
    elif action == "strengthened":
        output_parts.append(f"STORED: [STRENGTHENED] {context[:60]}")
    else:
        output_parts.append(f"STORED: [SKIPPED] (not novel enough)")

    return "\n".join(output_parts)

if __name__ == "__main__":
    mcp.run(transport="stdio")
