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
    from flymemory import FlyMemoryStore
    if _store is None:
        if os.path.exists(DB_PATH):
            import pickle
            with open(DB_PATH, "rb") as f:
                data = pickle.load(f)
            # 存档里只存了 patterns + 元数据（不含 W 矩阵），这里要重建出
            # 一个真正的 FlyMemoryStore 对象，并重放 codes 把 W 矩阵攒回来。
            # 原来的实现直接把 dict 当 store 用，重启后 remember/recall/stats
            # 全会因为 dict 没有 .memories/.hopfield/.size 而崩。
            store = FlyMemoryStore(
                n_compartments=data["n_compartments"], n_bits=data["n_bits"]
            )
            store.memories = data["memories"]
            store._next_id = data["_next_id"]
            for mem in store.memories.values():
                store.hopfield.compartments[mem.compartment].store(mem.code)
            _store = store
        else:
            _store = FlyMemoryStore(n_compartments=8, n_bits=4096)
    return _store

def save_store():
    if _store is None: return
    import pickle
    # 只持久化 patterns + 元数据（不存 W 矩阵，省磁盘；启动时由 get_store 重放重建）
    data = {
        "memories": _store.memories,
        "_next_id": _store._next_id,
        "n_bits": _store.n_bits,
        "n_compartments": _store.hopfield.n_compartments,
    }
    # 先写临时文件再原子替换，避免两个 MCP 客户端同时写导致文件损坏
    tmp = DB_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, DB_PATH)

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
