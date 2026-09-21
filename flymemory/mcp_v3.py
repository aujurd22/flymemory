"""FlyMemory v3 MCP Server — smart memory with auto-dedup, semantic search, and decay.

Tools:
  flymemory_remember(text, tags) → store (auto-dedup via semantic similarity)
  flymemory_recall(query, top_k) → semantic search with decay weighting
  flymemory_stats() → statistics including decay info
  flymemory_cleanup() → remove memories that have decayed below threshold

两种运行模式：
  python mcp_v3.py            → stdio（默认，行为与旧版一致）
  python mcp_v3.py --http     → streamable-http 常驻服务（127.0.0.1:8765/mcp），
                                所有会话共享一份模型与记忆，连接毫秒级；
                                端口已被占用时直接退出（幂等，适合开机自启）。
"""
import sys, os, json, pickle, time, socket, argparse, threading
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
# 包目录与其父目录都入 path：父目录保证 `import flymemory` 成立（不依赖 PYTHONPATH），
# 包目录保证直接以脚本方式运行时也能退化为同目录导入。
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

# --http 常驻模式：尽早把日志落文件（pythonw 下 stderr 本为 None），
# 必须赶在 sentence_transformers/transformers 导入之前，否则它们的 logger 绑定到旧流。
if "--http" in sys.argv:
    try:
        _logf = open(os.path.join(_here, "server.log"), "a", buffering=1, encoding="utf-8", errors="replace")
    except PermissionError:  # 主日志被残留句柄锁住时退化为带 PID 的文件名
        _logf = open(os.path.join(_here, f"server.{os.getpid()}.log"), "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stderr = _logf
    sys.stdout = _logf

# 模型已缓存则强制离线：HF_HUB_OFFLINE 是 huggingface_hub 导入时读取的，
# 事后再改 os.environ 无效，须在重导入前设置并同步改 constants。
try:
    from huggingface_hub import constants as _hf_constants
    if (not os.environ.get("HF_HUB_OFFLINE")
            and os.path.isdir(os.path.join(_hf_constants.HF_HUB_CACHE,
                                           "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"))):
        os.environ["HF_HUB_OFFLINE"] = "1"
        _hf_constants.HF_HUB_OFFLINE = True
except Exception:
    pass

# 关键：必须在主线程启动时就导入 sentence_transformers / torch 以及 flymemory.v3，
# 否则 FastMCP 会在工作线程里首次 import 触发 OpenBLAS/torch 死锁（v1 踩过的坑）。
# numpy 已在上方顶层 import，OpenBLAS 已在主线程初始化；这里补齐 torch 一侧。
import sentence_transformers  # noqa: F401  强制主线程导入 torch / transformers
try:
    from flymemory.v3 import SmartMemory, load, save, _get_model  # noqa: F401
except ImportError:  # 无 PYTHONPATH、以脚本方式直接运行时
    from v3 import SmartMemory, load, save, _get_model  # noqa: F401

DB_PATH = os.path.join(os.path.dirname(__file__), "flymemory_v3.pkl")

HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8765

# streamable-http 模式下多个请求并发进入工具线程，记忆与模型加载必须串行化
_mem_lock = threading.Lock()

_mem = None

def get_memory():
    global _mem
    if _mem is None:
        if os.path.exists(DB_PATH):
            _mem = load(DB_PATH)
        else:
            _mem = SmartMemory(n_bits=4096, decay_tau=2592000.0)  # 30 天特征时间，见 v3.SmartMemory 注释
    return _mem

def save_memory():
    if _mem is None: return
    save(_mem, DB_PATH)

from mcp.server.fastmcp import FastMCP

# stateless_http=True：不跟踪会话，每个请求自包含——服务重启不会使已连接的
# ZCode 会话失效（否则报 Session not found 且客户端不会自动重连）。
mcp = FastMCP("flymemory", host=HTTP_HOST, port=HTTP_PORT, stateless_http=True, instructions="""
FlyMemory v3: long-term memory for AI agents — chunked storage, hybrid
recall (multilingual embeddings + IDF lexical boost for exact identifiers),
power-law decay, semantic dedup, and model-driven supersede (stale states
are marked by the calling model and leave the default recall).
A Hopfield associative-expansion layer is experimental (bench_hopfield.py).

Use flymemory_remember to store important findings (source=model).
Use flymemory_recall to retrieve; pass include_superseded for history queries.
Use flymemory_supersede to mark an outdated memory as replaced by a newer one.
Use flymemory_cleanup to remove fully decayed memories.
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
    with _mem_lock:
        mem = get_memory()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        result = mem.remember_text(text, tags=tag_list, source="model")
        save_memory()
    action = result["action"]
    counts = result.get("counts") or {}
    id_str = ",".join(f"#{i}" for i in (result.get("memory_ids") or [])[:8]) or "?"
    if action == "rejected":
        return f"[REJECTED] too similar (novelty={result.get('novelty', 0):.2f})"
    kinds = [k for k in ("new", "merged", "strengthened") if counts.get(k)]
    label = " ".join(f"{k.upper()}x{counts[k]}" if counts[k] > 1 else k.upper() for k in kinds) or action.upper()
    return f"[{label} {id_str}] {text[:80]}"

@mcp.tool()
def flymemory_recall(query: str, top_k: int = 5, include_superseded: bool = False) -> str:
    """Hybrid recall: semantic similarity + IDF lexical boost (exact identifiers),
    ranked by similarity × time decay. Superseded (outdated) entries are excluded
    unless include_superseded=True — use it for history questions like
    "did I ever use X?".

    Args:
        query: The query text (can be partial/incomplete)
        top_k: Number of memories to return
        include_superseded: include entries marked as superseded
    Returns:
        Relevant memories with scores, age and provenance stamps
    """
    with _mem_lock:
        mem = get_memory()
        results = mem.recall(query, top_k=top_k, include_superseded=include_superseded)
        if not results:
            return "No relevant memories found."
        output = []
        for entry, sim, eff in results:
            decay_pct = f"decay={mem_decay_pct(entry, mem):.0f}%"
            output.append(f"[sim={sim:.2f} | {_age_str(entry.timestamp)} | "
                          f"src={entry.source} | {decay_pct}] {entry.text[:80]}")
        return "\n".join(output)

def mem_decay_pct(entry, mem):
    dw = mem._decay_weight(entry)
    return dw * 100

def _age_str(ts: float) -> str:
    """记忆条目的相对年龄标注——模型据此把旧状态当'可能已过期'处理，而非当前事实。"""
    dt = time.time() - ts
    if dt < 90:
        return "刚刚"
    if dt < 3600:
        return f"{int(dt // 60)}分钟前"
    if dt < 86400:
        return f"{int(dt // 3600)}小时前"
    return f"{int(dt // 86400)}天前"

@mcp.tool()
def flymemory_stats() -> str:
    """Get memory system statistics."""
    with _mem_lock:
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
def flymemory_cleanup(min_retention: float = 0.1, protect_model_stored: bool = True) -> str:
    """Directed forgetting: remove memories whose retention decayed below the
    threshold. With protect_model_stored=True (default), model/import-stored
    entries are only pruned below min_retention/3 — forgetting targets unjudged
    hook chatter, not judged knowledge.

    Args:
        min_retention: Minimum decay weight to keep (default 0.1)
        protect_model_stored: shield model/import-stored entries (3x lower prune threshold)
    """
    with _mem_lock:
        mem = get_memory()
        removed = mem.decay_cleanup(min_retention, protect_model_stored=protect_model_stored)
        if removed > 0:
            save_memory()
    return f"Removed {removed} decayed memories. Remaining: {mem.size}"

@mcp.tool()
def flymemory_forget(memory_id: int) -> str:
    """Targeted forgetting: hard-delete one memory by id.

    For entries that are WRONG (not merely outdated — use flymemory_supersede
    for replaced-but-true history). The judgment is made by the calling model.
    Args:
        memory_id: id of the memory to delete (from recall results)
    """
    with _mem_lock:
        mem = get_memory()
        text = mem.forget(memory_id)
        if text is not None:
            save_memory()
    if text is not None:
        return f"[FORGOT #{memory_id}] {text[:80]}"
    return f"[NOT FOUND] memory #{memory_id} does not exist"

@mcp.tool()
def flymemory_supersede(old_memory_id: int, new_memory_id: int) -> str:
    """Mark an outdated memory as superseded by a newer one.

    判断由调用方模型做出：当新结论取代了召回列表中的某条旧状态时调用本工具，
    被取代的旧条目此后不再出现在默认召回中。

    Args:
        old_memory_id: 被取代的旧条目 id（从召回结果的 [#id] 取）
        new_memory_id: 取代它的新条目 id（通常是刚 remember 返回的 id）
    """
    with _mem_lock:
        mem = get_memory()
        ok, reason = mem.supersede(old_memory_id, new_memory_id)
        if ok:
            save_memory()
    if ok:
        return f"[SUPERSEDED] #{old_memory_id} -> #{new_memory_id} (excluded from default recall)"
    return f"[REJECTED] {reason}"

@mcp.tool()
def flymemory_session_pack(minutes: float = 180) -> str:
    """Compression-recovery pack: the recent working trail plus the newest
    model-stored conclusions, chronological. Injected by the SessionStart
    (compact) hook right after the host compresses a conversation; also useful
    manually after returning to a session.

    Args:
        minutes: how far back the trail reaches (default 180)
    """
    with _mem_lock:
        mem = get_memory()
        pack = mem.session_pack(minutes=minutes)
    return pack if pack else "Nothing to recover: no entries in the requested window."

@mcp.tool()
def flymemory_consolidate(memory_ids: list, conclusion: str) -> str:
    """Consolidate several related memories into ONE higher-order conclusion.

    The judgment (which ids belong together, what the conclusion says) belongs
    to the calling model. Mechanics: creates a new model-stored entry carrying
    evidence_ids linking back to the raw entries — raw entries are KEPT as
    evidence (never deleted), so consolidation adds abstraction without
    destroying history.

    Args:
        memory_ids: ids of the related memories being consolidated (>= 2)
        conclusion: the higher-order conclusion text
    """
    with _mem_lock:
        mem = get_memory()
        ids = [int(i) for i in memory_ids]
        if len(set(ids)) < 2:
            return "[REJECTED] consolidation needs at least 2 distinct memories"
        by_id = {m.memory_id: m for m in mem.memories}
        missing = [i for i in ids if i not in by_id]
        if missing:
            return f"[REJECTED] missing memory ids: {missing}"
        if _contains_credential(conclusion):
            return "[REJECTED] credential-like content in conclusion"
        r = mem.remember_text(conclusion, tags=["consolidation"], source="model")
        if not r.get("stored"):
            return "[REJECTED] conclusion not stored (dedup rejected)"
        eid = r["memory_id"]
        entry = next(m for m in mem.memories if m.memory_id == eid)
        entry.evidence_ids = sorted(set(ids) - {eid})
        save_memory()
    return (f"[CONSOLIDATED -> #{eid}] evidence: {entry.evidence_ids} | "
            f"{conclusion[:80]}")

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
    with _mem_lock:
        mem = get_memory()
        output_parts = []

        recalled_ids = set()
        # ===== RECALL: find relevant memories =====
        if mem.size > 0:
            results = mem.recall(context, top_k=3)
            if results:
                recall_parts = []
                for entry, sim, eff in results:
                    if sim > 0.4:  # only report meaningful matches
                        recalled_ids.add(entry.memory_id)
                        recall_parts.append(f"  [{sim:.2f} | {_age_str(entry.timestamp)}] {entry.text[:80]}")
                if recall_parts:
                    output_parts.append("RECALLED MEMORIES:")
                    output_parts.extend(recall_parts)
                else:
                    output_parts.append("RECALL: no relevant memories for this topic.")
            else:
                output_parts.append("RECALL: memory empty.")
        else:
            output_parts.append("RECALL: memory empty (first use).")

        # ===== RECENT channel: compression protection for the working thread =====
        # Pure recency, independent of similarity — deictic references ("that
        # thing from just now") survive compaction through this block.
        recent = [m for m in mem.recent(minutes=90, limit=6)
                  if m.memory_id not in recalled_ids]
        if recent:
            output_parts.append("RECENT CONTEXT (last ~90 min):")
            output_parts.extend(f"  [{_age_str(m.timestamp)}] {m.text[:70]}"
                                for m in recent)

        # ===== STORE: store this interaction if novel =====
        combined_text = context
        if response:
            combined_text = f"{context} ||| {response}"
        result = mem.remember_text(combined_text, tags=["auto"], source="hook")
        save_memory()

    action = result["action"]
    if action == "rejected":
        output_parts.append("STORED: [SKIPPED] (not novel enough)")
    else:
        counts = result.get("counts") or {}
        kinds = [k for k in ("new", "merged", "strengthened") if counts.get(k)]
        label = "+".join(f"{k.upper()}x{counts[k]}" if counts[k] > 1 else k.upper() for k in kinds) or action.upper()
        id_str = ",".join(f"#{i}" for i in (result.get("memory_ids") or [])[:8])
        output_parts.append(f"STORED: [{label} {id_str}] {context[:60]}")

    return "\n".join(output_parts)

def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def main():
    parser = argparse.ArgumentParser(description="FlyMemory v3 MCP server")
    parser.add_argument("--http", action="store_true",
                        help="run as persistent streamable-http server instead of stdio")
    parser.add_argument("--port", type=int, default=HTTP_PORT)
    args = parser.parse_args()

    if args.http:
        if port_in_use(HTTP_HOST, args.port):
            # 幂等：常驻实例已在运行（如开机自启 + 手动启动撞车）就直接退出
            if sys.stderr:
                sys.stderr.write(f"[flymemory] {HTTP_HOST}:{args.port} already in use, exiting.\n")
            sys.exit(0)
        # 日志重定向与离线判定已在模块顶部完成（须先于重导入）

        def _warmup():
            with _mem_lock:
                get_memory()
                try:
                    _get_model()
                except Exception as e:  # 离线等情况下跳过，首轮调用时再尝试
                    sys.stderr.write(f"[flymemory] model warmup skipped: {e}\n")

        # 常驻模式：先起服务（连接毫秒级可用），模型在后台线程预热，
        # 首个工具调用若模型未就绪会在 _mem_lock 上等待预热线程完成。
        threading.Thread(target=_warmup, daemon=True).start()
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        # stdio 模式保持旧行为：主线程预加载 MiniLM 模型（首次会下载 all-MiniLM-L6-v2，约 80MB），
        # 避免首轮 flymemory_auto 在工作线程里下载模型导致卡顿 / 超时。
        try:
            _get_model()
        except Exception as e:  # 离线等情况下跳过，首轮调用时再尝试
            sys.stderr.write(f"[flymemory] model preload skipped: {e}\n")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
