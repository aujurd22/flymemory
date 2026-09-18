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
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
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

        # Threshold: >0.92 = duplicate, 0.75-0.92 = merge, <0.75 = new
        if best_sim > 0.92:
            # Duplicate: strengthen existing memory (rehearsal)
            best_match.access_count += 1
            best_match.last_accessed = time.time()
            return {"stored": True, "action": "strengthened",
                    "novelty": 1.0 - best_sim, "memory_id": best_match.memory_id}

        if best_sim > 0.75:
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

    def recall(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float, float]]:
        """Semantic recall with decay weighting.

        Score = semantic_similarity × decay_weight
        Returns top_k (memory, semantic_score, effective_score).
        """
        model = _get_model()
        query_emb = model.encode(query, convert_to_numpy=True).astype(np.float32)

        scored = []
        for mem in self.memories:
            # Semantic similarity (MiniLM cosine)
            sim = self._semantic_similarity(query_emb, mem.embedding)
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
        )
        mem.memories.append(entry)
        # Rebuild W from binary codes
        code = mem.proj @ entry.embedding
        k = min(mem.k_keep, len(code))
        thresh = np.partition(code, -k)[-k]
        binary = np.where(code >= thresh, 1.0, -1.0)
        mem._hopfield_store(binary)
    return mem
