"""FlyMemory v3.0 — Smart memory with auto-dedup, semantic search, and decay.

Three upgrades over v2:
1. Auto-dedup: semantic similarity check (MiniLM cosine), no manual threshold.
   Duplicate content is automatically merged (strengthens existing memory).
2. Semantic search: recall uses MiniLM cosine distance (not binary match),
   giving graded relevance scores that reflect meaning, not just bit overlap.
3. Memory decay: older memories gradually lose weight following a
   power-law forgetting curve (Ebbinghaus: R = t^(-0.5)).
   Frequently accessed memories resist decay (rehearsal effect).
"""

import numpy as np
import re
import time
import os
import pickle
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

# ===== Lazy model loading =====
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # 默认 CPU：MiniLM 单句编码仅 ~10-30ms，而 GPU 常被 ComfyUI 等重负载打满
        # （显存 11.9/12.3GB、util 100% 时 cuda 排队可致工具调用 120s 超时，2026-09-18 实测）。
        # 需要 GPU 时设 FLYMEMORY_DEVICE=cuda。
        device = os.environ.get("FLYMEMORY_DEVICE", "cpu")
        # 多语言模型：all-MiniLM-L6-v2 对中文语义近乎噪声（结构平行的无关短句实测 0.983、
        # 相关句仅 0.50，是"记忆混乱"的总根源），L12-v2 支持 50+ 语言且同为 384 维，
        # 换模型后须全库重嵌入。可用 FLYMEMORY_MODEL 覆盖。
        model_name = os.environ.get("FLYMEMORY_MODEL",
                                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        _model = SentenceTransformer(model_name, device=device)
    return _model


# ===== Memory Entry =====
@dataclass
class MemoryEntry:
    text: str
    response: str
    embedding: np.ndarray          # MiniLM 384-dim
    timestamp: float               # creation time
    last_accessed: float           # last recall time
    access_count: int              # how many times recalled
    tags: list
    memory_id: int
    superseded_by: Optional[int] = None   # 被更新的条目取代后指向新条目，默认召回跳过


# ===== Hopfield with semantic search + decay =====
class SmartMemory:
    """Hopfield associative memory with semantic search and Ebbinghaus decay."""

    def __init__(self, n_bits: int = 4096, sparsity: float = 0.05,
                 decay_half_life: float = 2592000.0):  # 30 days: 承载长期参考知识，1h 半衰期会让不常召回的知识 4 天内衰减到清理阈值
        self.n_bits = n_bits
        self.sparsity = sparsity
        self.k_keep = max(int(n_bits * sparsity), 1)
        rng = np.random.default_rng(42)
        self.proj = rng.standard_normal((n_bits, 384)).astype(np.float32) / np.sqrt(384)
        self.W = np.zeros((n_bits, n_bits), dtype=np.float32)
        self.memories: List[MemoryEntry] = []
        self._next_id = 0
        self.decay_half_life = decay_half_life  # seconds

    def _encode(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """text → (binary_code, raw_embedding)."""
        model = _get_model()
        emb = model.encode(text, convert_to_numpy=True).astype(np.float32)
        code = self.proj @ emb
        k = min(self.k_keep, len(code))
        thresh = np.partition(code, -k)[-k]
        binary = np.where(code >= thresh, 1.0, -1.0)
        return binary, emb

    def _semantic_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two MiniLM embeddings (0-1, higher = more similar)."""
        norm1 = np.linalg.norm(emb1) + 1e-8
        norm2 = np.linalg.norm(emb2) + 1e-8
        return float(max(0.0, np.dot(emb1, emb2) / (norm1 * norm2)))

    def _decay_weight(self, mem: MemoryEntry) -> float:
        """Ebbinghaus forgetting curve: R = t^(-0.5), modulated by access count.

        R (retention) decreases with time since last access,
        but each access (rehearsal) resets and strengthens the trace.
        """
        now = time.time()
        dt = now - mem.last_accessed
        # Power-law decay: R = (1 + dt/τ)^(-0.5), τ = half_life
        tau = self.decay_half_life
        base_retention = (1.0 + dt / tau) ** (-0.5)
        # Rehearsal effect: each access boosts retention
        rehearsal_factor = 1.0 + 0.5 * np.log2(1 + mem.access_count)
        return min(base_retention * rehearsal_factor, 1.0)

    def _hopfield_store(self, binary: np.ndarray):
        self.W += np.outer(binary, binary)

    def _hopfield_recall(self, binary: np.ndarray, max_iter: int = 10) -> np.ndarray:
        s = binary.copy()
        for _ in range(max_iter):
            new_s = np.sign(self.W @ s)
            new_s[new_s == 0] = 1
            if np.array_equal(new_s, s):
                break
            s = new_s
        return s

    def remember_text(self, text: str, response: str = "",
                      tags: Optional[list] = None) -> Dict:
        """分块存储入口：长文本按句切成语义块，逐块走去重/合并，返回聚合结果。

        整条消息存成一条会稀释向量相似度（字面词重叠压过主题相关），分块后
        每块主题单一，召回精度显著提升。短消息经 split_chunks 的短块合并后
        自然回落为单块，行为不变。

        Returns dict with:
          action: "new" / "merged" / "strengthened" / "rejected" / "mixed"
          counts: {"new": n, "merged": n, "strengthened": n}
          memory_ids: 涉及的全部条目 id；chunks: 实际处理的块数
        """
        chunks = split_chunks(text)
        counts = {"new": 0, "merged": 0, "strengthened": 0, "rejected": 0}
        ids = []
        last = None
        for c in chunks:
            r = self.remember(c, response=response, tags=tags)
            counts[r["action"]] = counts.get(r["action"], 0) + 1
            if r.get("memory_id") is not None:
                ids.append(r["memory_id"])
            last = r
        if last is None:
            return {"stored": False, "action": "rejected", "novelty": 0.0,
                    "memory_id": None, "counts": counts, "memory_ids": [], "chunks": 0}
        stored_kinds = [k for k in ("new", "merged", "strengthened") if counts.get(k)]
        if sum(counts.values()) == 1:
            action = last["action"]
        elif len(stored_kinds) == 1:
            action = stored_kinds[0]
        else:
            action = "mixed"
        return {"stored": True, "action": action, "novelty": last.get("novelty", 0.0),
                "memory_id": ids[-1] if ids else None, "counts": counts,
                "memory_ids": ids, "chunks": len(chunks)}

    def remember(self, text: str, response: str = "",
                 tags: Optional[list] = None) -> Dict:
        """Store a memory with auto-dedup via semantic similarity.

        Returns dict with:
          stored: True/False
          action: "new" / "merged" / "rejected"
          novelty: 0-1
          memory_id: int
        """
        binary, emb = self._encode(text)

        # ===== Auto-dedup: semantic similarity check =====
        best_match = None
        best_sim = 0.0
        for mem in self.memories:
            sim = self._semantic_similarity(emb, mem.embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = mem

        # 短文本阈值高于长文本：改用多语言模型后结构噪声已消失（无关句 0.2-0.3），
        # 但短句改写（如日期不同的相似句 0.93）仍应各自成条，故阈值略收紧。
        if len(text) < 30:
            dup_t, merge_t = 0.95, 0.85
        else:
            dup_t, merge_t = 0.92, 0.75

        # Threshold: >dup_t = duplicate, merge_t~dup_t = merge, <merge_t = new
        if best_sim > dup_t:
            # Duplicate: strengthen existing memory (rehearsal)
            best_match.access_count += 1
            best_match.last_accessed = time.time()
            return {"stored": True, "action": "strengthened",
                    "novelty": 1.0 - best_sim, "memory_id": best_match.memory_id}

        if best_sim > merge_t:
            # Partially new: merge (update text if new one is longer)
            if len(text) > len(best_match.text):
                new_binary, new_emb = self._encode(text)
                best_match.text = text
                best_match.embedding = new_emb
            best_match.access_count += 1
            best_match.last_accessed = time.time()
            return {"stored": True, "action": "merged",
                    "novelty": 1.0 - best_sim, "memory_id": best_match.memory_id}

        # Truly new memory
        now = time.time()
        entry = MemoryEntry(
            text=text, response=response,
            embedding=emb, timestamp=now, last_accessed=now,
            access_count=0, tags=tags or [], memory_id=self._next_id
        )
        self.memories.append(entry)
        self._hopfield_store(binary)
        self._next_id += 1
        return {"stored": True, "action": "new",
                "novelty": 1.0 - best_sim, "memory_id": entry.memory_id}

    def supersede(self, old_id: int, new_id: int) -> bool:
        """标记旧条目被新条目取代：默认召回不再返回旧条目。

        判断由调用方模型做出（它理解语义），这里只做机械标记。
        Returns True if the old entry was found and marked.
        """
        for mem in self.memories:
            if mem.memory_id == old_id:
                mem.superseded_by = new_id
                return True
        return False

    def recall(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float, float]]:
        """Semantic recall with decay weighting.

        Score = semantic_similarity × decay_weight
        多句查询按块拆分、取各块相似度的最大值——整句混编会让查询向量
        被多主题稀释，与分块存储正好互补。
        Returns top_k (memory, semantic_score, effective_score).
        """
        model = _get_model()
        q_texts = split_chunks(query) or [query]
        q_embs = [model.encode(q, convert_to_numpy=True).astype(np.float32)
                  for q in q_texts]

        scored = []
        for mem in self.memories:
            if mem.superseded_by is not None:
                continue  # 被取代的旧状态不再进入默认召回
            # Semantic similarity (MiniLM cosine), 多块查询取最大
            sim = max(self._semantic_similarity(qe, mem.embedding) for qe in q_embs)
            # Decay weight (Ebbinghaus)
            dw = self._decay_weight(mem)
            # Effective score
            effective = sim * dw
            scored.append((mem, sim, effective))
            # Update access tracking (rehearsal effect)
            if sim > 0.5:
                mem.last_accessed = time.time()
                mem.access_count += 1

        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]

    def decay_cleanup(self, min_retention: float = 0.1) -> int:
        """Remove memories that have decayed below threshold."""
        removed = 0
        surviving = []
        for mem in self.memories:
            dw = self._decay_weight(mem)
            if dw >= min_retention:
                surviving.append(mem)
            else:
                removed += 1
        self.memories = surviving
        return removed

    @property
    def size(self):
        return len(self.memories)


# ===== Chunking: 每句独立成块，避免整条向量被多主题稀释 =====
def split_chunks(text: str, min_len: int = 10, max_len: int = 120) -> List[str]:
    """按句终止符切，每句一个块；超长句按逗号细分；过短碎片并入前块。

    注意不要把多个句子缓冲进同一块——中文信息密度高，两句常是两个主题，
    攒到 max_len 再分组会让短消息永远切不开（2026-09-19 实测踩过）。
    """
    chunks: List[str] = []
    for p in (p.strip() for p in re.split(r"(?<=[。！？；!?\n])\s*", text) if p.strip()):
        if len(p) > max_len:
            b = ""
            for s in re.split(r"(?<=[，,、：:])\s*", p):
                if b and len(b) + len(s) > max_len:
                    chunks.append(b)
                    b = s
                else:
                    b = f"{b}{s}" if b else s
            if b:
                chunks.append(b)
        else:
            chunks.append(p)
    merged: List[str] = []
    for c in chunks:
        if merged and len(c) < min_len:
            merged[-1] = f"{merged[-1]}{c}"
        else:
            merged.append(c)
    return [c.strip() for c in merged if c.strip()]


# ===== Persistence =====
def save(memory: SmartMemory, path: str):
    """Save memory to disk (embeddings + metadata, rebuild W on load)."""
    data = {
        "memories": [
            {
                "text": m.text, "response": m.response,
                "embedding": m.embedding.tolist(),
                "timestamp": m.timestamp, "last_accessed": m.last_accessed,
                "access_count": m.access_count, "tags": m.tags,
                "memory_id": m.memory_id,
                "superseded_by": m.superseded_by,
            } for m in memory.memories
        ],
        "_next_id": memory._next_id,
        "n_bits": memory.n_bits,
        "decay_half_life": memory.decay_half_life,
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def load(path: str) -> SmartMemory:
    """Load memory from disk. Rebuilds W from embeddings via encode."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    mem = SmartMemory(n_bits=data["n_bits"], decay_half_life=data["decay_half_life"])
    mem._next_id = data.get("_next_id", len(data["memories"]))
    for md in data["memories"]:
        entry = MemoryEntry(
            text=md["text"], response=md["response"],
            embedding=np.array(md["embedding"], dtype=np.float32),
            timestamp=md["timestamp"], last_accessed=md["last_accessed"],
            access_count=md["access_count"], tags=md["tags"],
            memory_id=md["memory_id"],
            superseded_by=md.get("superseded_by"),
        )
        mem.memories.append(entry)
        # Rebuild W from binary codes
        code = mem.proj @ entry.embedding
        k = min(mem.k_keep, len(code))
        thresh = np.partition(code, -k)[-k]
        binary = np.where(code >= thresh, 1.0, -1.0)
        mem._hopfield_store(binary)
    return mem
