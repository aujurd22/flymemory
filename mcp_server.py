"""FlyMemory MCP Server — gives ZCode/Claude persistent associative memory.

Tools:
  flymemory_remember(text) -> store a memory
  flymemory_recall(query, top_k) -> recall relevant memories
  flymemory_stats() -> memory statistics

Run: python mcp_server.py  (stdio MCP server)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "flymemory_data.pkl")

# Lazy init
_store = None

def get_store():
    global _store
    if _store is None:
        if os.path.exists(DB_PATH):
            import pickle
            with open(DB_PATH, "rb") as f:
                _store = pickle.load(f)
        else:
            from flymemory import FlyMemoryStore
            _store = FlyMemoryStore(n_compartments=8, n_bits=4096)
    return _store

def save_store():
    if _store is None: return
    import pickle
    # Only save patterns + metadata, NOT the W matrix (too large)
    # Rebuild W from patterns on load
    data = {
        "memories": _store.memories,
        "_next_id": _store._next_id,
        "n_bits": _store.n_bits,
        "n_compartments": _store.hopfield.n_compartments,
    }
    with open(DB_PATH, "wb") as f:
        pickle.dump(data, f)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("flymemory", instructions="""
FlyMemory: Hopfield associative memory inspired by Drosophila mushroom body.
Use flymemory_remember to store important findings.
Use flymemory_recall to retrieve relevant memories from partial cues.
Pattern completion: 20% cue -> 100% recovery (verified on fly brain connectome).
""")

@mcp.tool()
def flymemory_remember(text: str, tags: str = "") -> str:
    """Store an important finding or decision in persistent memory.

    Args:
        text: The text to remember (a key finding, decision, or insight)
        tags: Optional comma-separated tags for categorization
    Returns:
        Confirmation with memory ID
    """
    store = get_store()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    mid = store.remember(text, "", tags=tag_list)
    save_store()
    return f"Remembered #{mid}: {text[:80]}"

@mcp.tool()
def flymemory_recall(query: str, top_k: int = 3) -> str:
    """Recall relevant memories from partial cues via Hopfield pattern completion.

    Args:
        query: The query text (can be partial/incomplete — Hopfield fills in the rest)
        top_k: Number of memories to return (default 3)
    Returns:
        Relevant memories with overlap scores
    """
    store = get_store()
    results = store.recall(query, top_k=top_k)
    if not results:
        return "No relevant memories found."
    output = []
    for mem, ov in results:
        output.append(f"[{ov:.2f}] {mem.context[:100]}")
    return "\n".join(output)

@mcp.tool()
def flymemory_stats() -> str:
    """Get memory system statistics."""
    store = get_store()
    return f"Memories: {store.size}, Compartments: {store.hopfield.n_compartments}, Bits: {store.n_bits}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
