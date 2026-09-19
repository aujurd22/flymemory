"""FlyMemory v3.x — chunked semantic+lexical memory with decay and model-driven supersede.

Production recall path:
  text → multilingual MiniLM embedding → per-sentence chunks →
  vectorized cosine (max over query chunks) × power-law decay
  + lexical (IDF) boost for exact identifiers → top-k

Memory management:
  - auto-dedup with length-tiered thresholds (strengthen / merge / new)
  - supersede: the calling model marks stale states; superseded entries leave
    the default recall but stay queryable (include_superseded=True)
  - source provenance: "hook" (mechanical capture, unjudged) /
    "model" (model judged worth storing) / "import" (seed import)

Experimental (see bench_hopfield.py — keep or delete based on measurement):
  4096-d sparse binary codes + Hopfield matrix W for associative expansion.

Decay note: retention R(t) = (1 + t/tau)^-0.5 (power law, heavy tail).
This is NOT an exponential half-life: R(tau) = 1/sqrt(2) ≈ 0.707 and the
true half-life (R = 0.5) is 3*tau. The parameter is therefore named decay_tau.
"""

import numpy as np
import re
import math
import time
import os
import pickle
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Iterable, Set
from dataclasses import dataclass

# ===== Lazy model loading =====
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Default to CPU: MiniLM encodes one sentence in ~10-30ms on CPU, while the
        # GPU is often saturated by training / ComfyUI (queueing there caused 120s
        # tool timeouts, measured 2026-09-18). Set FLYMEMORY_DEVICE=cuda to override.
        device = os.environ.get("FLYMEMORY_DEVICE", "cpu")
        # Multilingual embedder: all-MiniLM-L6-v2 is near-noise on Chinese
        # (structure-parallel unrelated sentences scored 0.983, truly related 0.50),
        # which was the root cause of "stale memory confusion" reports. L12-v2 covers
        # 50+ languages and is also 384-dim (same projection matrix). Switching the
        # model requires re-embedding the whole library. Override: FLYMEMORY_MODEL.
        model_name = os.environ.get("FLYMEMORY_MODEL",
                                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        _model = SentenceTransformer(model_name, device=device)
    return _model


def _embed(text: str) -> np.ndarray:
    return _get_model().encode(text, convert_to_numpy=True).astype(np.float32)


# Lexical-channel weight: exact identifier matches (part numbers, paths, IDs)
# must be able to outrank mere semantic similarity. 0.25 keeps semantic ranking
# primary while letting a rare-token exact match win. See bench_recall_speed.py.
LEX_WEIGHT = 0.25


def _tokenize(text: str) -> Set[str]:
    """ASCII word tokens (len>=2, lowercased) + CJK character bigrams."""
    tokens: Set[str] = set()
    for w in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
        if len(w) >= 2:
            tokens.add(w)
    chars = text
    for i in range(len(chars) - 1):
        a, b = chars[i], chars[i + 1]
        if "\u4e00" <= a <= "\u9fff" and "\u4e00" <= b <= "\u9fff":
            tokens.add(a + b)
    return tokens


# ===== Memory Entry =====
@dataclass
class MemoryEntry:
    text: str
    response: str
    embedding: np.ndarray          # 384-dim multilingual embedding
    timestamp: float               # creation time
    last_accessed: float           # last recall time
    access_count: int              # how many times recalled
    tags: list
    memory_id: int
    superseded_by: Optional[int] = None   # replaced by a newer entry; excluded from default recall
    source: str = "hook"                  # hook=mechanical capture / model=model-judged / import / auto


# ===== Chunking: one block per sentence so multi-topic messages stay separable =====
def split_chunks(text: str, min_len: int = 10, max_len: int = 120) -> List[str]:
    """Split at sentence terminators; one sentence per chunk; over-long sentences
    split at commas; a short fragment merges into the previous chunk ONLY when
    that chunk does not end with sentence-final punctuation (i.e. they are parts
    of the same sentence). A short but complete sentence ("好。") stands alone.

    Do NOT buffer multiple sentences into one chunk: Chinese is dense and two
    sentences are usually two topics — buffering up to max_len made short
    messages un-splittable (measured 2026-09-19).
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

    def _ends_sentence(s: str) -> bool:
        return bool(s) and s[-1] in "。！？；!?…"

    merged: List[str] = []
    for c in chunks:
        if merged and len(c) < min_len and not _ends_sentence(merged[-1]):
            merged[-1] = f"{merged[-1]}{c}"
        else:
            merged.append(c)
    return [c.strip() for c in merged if c.strip()]


# ===== SmartMemory =====
class SmartMemory:
    """Semantic+lexical associative memory with Ebbinghaus-style decay.

    Power-law decay R(t) = (1 + t/tau)^-0.5 modulated by rehearsal:
    R is ~0.707 at t=tau and 0.5 at t=3*tau (the true half-life).
    """

    def __init__(self, n_bits: int = 4096, sparsity: float = 0.05,
                 decay_tau: float = 2592000.0,          # 30 days: long-lived reference knowledge
                 decay_half_life: Optional[float] = None,  # legacy kwarg alias for decay_tau
                 enable_hopfield: bool = False):
        """enable_hopfield=False by DEFAULT: bench_hopfield.py measured the
        associative expansion HURTING retrieval (Recall@5 0.189 → 0.043 on a
        1370-entry library — the 4096-bit W matrix is saturated far beyond its
        capacity, so crosstalk dominates). The 64 MiB matrix, per-store outer
        product and load-time rebuild are only paid when explicitly enabled."""
        if decay_half_life is not None:
            decay_tau = decay_half_life
        self.n_bits = n_bits
        self.sparsity = sparsity
        self.k_keep = max(int(n_bits * sparsity), 1)
        rng = np.random.default_rng(42)
        self.proj = rng.standard_normal((n_bits, 384)).astype(np.float32) / np.sqrt(384)
        self.enable_hopfield = enable_hopfield
        self.W = np.zeros((n_bits, n_bits), dtype=np.float32) if enable_hopfield else None
        self.memories: List[MemoryEntry] = []
        self._next_id = 0
        self.decay_tau = float(decay_tau)
        # vectorized-recall cache
        self._mat: Optional[np.ndarray] = None
        self._mat_dirty = True
        # lexical inverted index: token -> {memory_id}, df: token -> doc count
        self._lex_index: Dict[str, Set[int]] = defaultdict(set)
        self._df: Counter = Counter()
        self._entry_tokens: Dict[int, Set[str]] = {}

    # ----- encoding -----
    def _binary_from_emb(self, emb: np.ndarray) -> np.ndarray:
        code = self.proj @ emb
        k = min(self.k_keep, len(code))
        thresh = np.partition(code, -k)[-k]
        return np.where(code >= thresh, 1.0, -1.0)

    def _encode(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        emb = _embed(text)
        return self._binary_from_emb(emb), emb

    def _semantic_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        norm1 = np.linalg.norm(emb1) + 1e-8
        norm2 = np.linalg.norm(emb2) + 1e-8
        return float(max(0.0, np.dot(emb1, emb2) / (norm1 * norm2)))

    def _decay_weight(self, mem: MemoryEntry) -> float:
        dt = time.time() - mem.last_accessed
        base = (1.0 + dt / self.decay_tau) ** (-0.5)
        rehearsal = 1.0 + 0.5 * np.log2(1 + mem.access_count)
        return min(base * rehearsal, 1.0)

    # ----- Hopfield (experimental; bench_hopfield.py decides keep/delete) -----
    def _hopfield_store(self, binary: np.ndarray):
        if self.W is not None:
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

    def associate(self, memory: MemoryEntry, top_k: int = 5, hops: int = 2) -> List[Tuple[MemoryEntry, float]]:
        """Associative expansion: from a retrieved entry, evolve its sparse code on W
        and rank other entries by code overlap — surfaces entries *related* to the
        cue (co-stored context) rather than semantically *similar* ones.
        Only meaningful when enable_hopfield=True."""
        if self.W is None:
            return []
        s = self._binary_from_emb(memory.embedding)
        for _ in range(hops):
            s = np.sign(self.W @ s)
            s[s == 0] = 1
        scored = []
        for m in self.memories:
            if m.memory_id == memory.memory_id or m.superseded_by is not None:
                continue
            overlap = float(self._binary_from_emb(m.embedding) @ s) / self.k_keep
            if overlap > 0:
                scored.append((m, overlap))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # ----- lexical index -----
    def _index_entry(self, entry: MemoryEntry):
        toks = _tokenize(entry.text)
        self._entry_tokens[entry.memory_id] = toks
        for t in toks:
            self._lex_index[t].add(entry.memory_id)
            self._df[t] += 1

    def _unindex(self, memory_id: int):
        toks = self._entry_tokens.pop(memory_id, None)
        if not toks:
            return
        for t in toks:
            posting = self._lex_index.get(t)
            if posting:
                posting.discard(memory_id)
                if not posting:
                    del self._lex_index[t]
            if self._df[t] > 0:
                self._df[t] -= 1

    def _reindex_all(self):
        self._lex_index = defaultdict(set)
        self._df = Counter()
        self._entry_tokens = {}
        for m in self.memories:
            self._index_entry(m)

    def _lex_scores(self, q_texts: Iterable[str]) -> np.ndarray:
        """IDF-weighted lexical match ratio per memory, in [0, 1]."""
        n = len(self.memories)
        scores = np.zeros(n, dtype=np.float32)
        if n == 0 or not self._df:
            return scores
        n_docs = max(n, 1)
        matched: Dict[int, float] = {}
        total_idf = 0.0
        for qt in q_texts:
            for tok in _tokenize(qt):
                df = self._df.get(tok, 0)
                if not df:
                    continue  # token absent from corpus cannot match anything
                idf = math.log(1.0 + n_docs / df)
                total_idf += idf
                for mid in self._lex_index.get(tok, ()):
                    matched[mid] = matched.get(mid, 0.0) + idf
        if total_idf <= 0:
            return scores
        for i, m in enumerate(self.memories):
            scores[i] = min(matched.get(m.memory_id, 0.0) / total_idf, 1.0)
        return scores

    # ----- vectorized embedding matrix cache -----
    def _emb_matrix(self) -> np.ndarray:
        if self._mat is None or self._mat_dirty:
            if self.memories:
                self._mat = np.stack([m.embedding for m in self.memories]).astype(np.float32)
            else:
                self._mat = np.zeros((0, 384), dtype=np.float32)
            self._mat_dirty = False
        return self._mat

    # ----- store -----
    def remember_text(self, text: str, response: str = "",
                      tags: Optional[list] = None, source: str = "hook") -> Dict:
        """Chunked store entry point: long text is split per sentence and each chunk
        goes through dedup/merge; short messages fall back to a single chunk via
        split_chunks' fragment merging.

        Returns dict with:
          action: "new" / "merged" / "strengthened" / "rejected" / "mixed"
          counts: {"new": n, "merged": n, "strengthened": n}
          memory_ids: all touched entry ids; chunks: chunks processed
        """
        chunks = split_chunks(text)
        counts = {"new": 0, "merged": 0, "strengthened": 0, "rejected": 0}
        ids = []
        last = None
        for c in chunks:
            r = self.remember(c, response=response, tags=tags, source=source)
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
                 tags: Optional[list] = None, source: str = "hook") -> Dict:
        """Store one chunk with auto-dedup via semantic similarity.

        Returns dict with: stored, action ("new"/"merged"/"strengthened"/"rejected"),
        novelty, memory_id.
        """
        binary, emb = self._encode(text)

        best_match = None
        best_sim = 0.0
        for mem in self.memories:
            sim = self._semantic_similarity(emb, mem.embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = mem

        # Tiered thresholds: short rewritten sentences (e.g. differing only in a date)
        # measured 0.93 on the multilingual model — similar but distinct facts should
        # stay separate entries; corrections are handled by supersede, not merging.
        if len(text) < 30:
            dup_t, merge_t = 0.95, 0.85
        else:
            dup_t, merge_t = 0.92, 0.75

        if best_sim > dup_t:
            best_match.access_count += 1
            best_match.last_accessed = time.time()
            return {"stored": True, "action": "strengthened",
                    "novelty": 1.0 - best_sim, "memory_id": best_match.memory_id}

        if best_sim > merge_t:
            if len(text) > len(best_match.text):
                new_binary, new_emb = self._encode(text)
                self._unindex(best_match.memory_id)
                best_match.text = text
                best_match.embedding = new_emb
                best_match.source = source
                self._index_entry(best_match)
                self._mat_dirty = True
            best_match.access_count += 1
            best_match.last_accessed = time.time()
            return {"stored": True, "action": "merged",
                    "novelty": 1.0 - best_sim, "memory_id": best_match.memory_id}

        now = time.time()
        entry = MemoryEntry(
            text=text, response=response,
            embedding=emb, timestamp=now, last_accessed=now,
            access_count=0, tags=tags or [], memory_id=self._next_id,
            source=source,
        )
        self.memories.append(entry)
        self._index_entry(entry)
        self._mat_dirty = True
        if self.enable_hopfield:
            self._hopfield_store(binary)
        self._next_id += 1
        return {"stored": True, "action": "new",
                "novelty": 1.0 - best_sim, "memory_id": entry.memory_id}

    # ----- supersede -----
    def supersede(self, old_id: int, new_id: int) -> bool:
        """Mark an entry as superseded by a newer one (excluded from default recall).

        The judgment is made by the calling model (it understands semantics);
        this is the mechanical marker only.
        """
        for mem in self.memories:
            if mem.memory_id == old_id:
                mem.superseded_by = new_id
                return True
        return False

    # ----- recall -----
    def recall(self, query: str, top_k: int = 5,
               include_superseded: bool = False) -> List[Tuple[MemoryEntry, float, float]]:
        """Hybrid recall: vectorized semantic similarity (max over query chunks)
        × power-law decay + IDF lexical boost (exact identifiers), superseded
        entries excluded unless include_superseded=True.

        Returns top_k of (memory, semantic_sim, effective_score).
        """
        if not self.memories:
            return []
        q_texts = split_chunks(query) or [query]
        Q = np.stack([_embed(qt) for qt in q_texts])
        M = self._emb_matrix()
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        sim_vec = (qn @ mn.T).max(axis=0)                      # (N,)

        now = time.time()
        last = np.array([m.last_accessed for m in self.memories], dtype=np.float64)
        acc = np.array([m.access_count for m in self.memories], dtype=np.float64)
        dw = np.minimum(((1.0 + (now - last) / self.decay_tau) ** -0.5)
                        * (1.0 + 0.5 * np.log2(1.0 + acc)), 1.0)

        lex_vec = self._lex_scores(q_texts)
        eff = dw * (sim_vec + LEX_WEIGHT * lex_vec)

        order = np.argsort(-eff)
        results = []
        for idx in order:
            m = self.memories[int(idx)]
            if m.superseded_by is not None and not include_superseded:
                continue
            results.append((m, float(sim_vec[idx]), float(eff[idx])))
            if len(results) >= top_k:
                break

        # rehearsal side effect (matches pre-vectorization semantics: every
        # entry with sim>0.5 gets its last_accessed/access_count refreshed)
        hot = np.where(sim_vec > 0.5)[0]
        for idx in hot:
            m = self.memories[int(idx)]
            if m.superseded_by is not None and not include_superseded:
                continue
            m.last_accessed = now
            m.access_count += 1
        return results

    def decay_cleanup(self, min_retention: float = 0.1) -> int:
        removed = 0
        surviving = []
        for mem in self.memories:
            if self._decay_weight(mem) >= min_retention:
                surviving.append(mem)
            else:
                removed += 1
                self._unindex(mem.memory_id)
        if removed:
            self.memories = surviving
            self._mat_dirty = True
        return removed

    @property
    def size(self):
        return len(self.memories)


# ===== Persistence =====
def save(memory: SmartMemory, path: str):
    """Save to disk (embeddings + metadata; W is rebuilt on load)."""
    data = {
        "schema": "3.1",
        "memories": [
            {
                "text": m.text, "response": m.response,
                "embedding": m.embedding.tolist(),
                "timestamp": m.timestamp, "last_accessed": m.last_accessed,
                "access_count": m.access_count, "tags": m.tags,
                "memory_id": m.memory_id,
                "superseded_by": m.superseded_by,
                "source": m.source,
            } for m in memory.memories
        ],
        "_next_id": memory._next_id,
        "n_bits": memory.n_bits,
        "decay_tau": memory.decay_tau,
        # legacy alias so an older code version can still load this file
        "decay_half_life": memory.decay_tau,
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load(path: str, enable_hopfield: bool = False) -> SmartMemory:
    """Load from disk. Accepts both the current schema (decay_tau, source,
    superseded_by) and the legacy one (decay_half_life only)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    tau = data.get("decay_tau") or data.get("decay_half_life") or 2592000.0
    mem = SmartMemory(n_bits=data["n_bits"], decay_tau=tau, enable_hopfield=enable_hopfield)
    mem._next_id = data.get("_next_id", len(data["memories"]))
    for md in data["memories"]:
        entry = MemoryEntry(
            text=md["text"], response=md["response"],
            embedding=np.array(md["embedding"], dtype=np.float32),
            timestamp=md["timestamp"], last_accessed=md["last_accessed"],
            access_count=md["access_count"], tags=md["tags"],
            memory_id=md["memory_id"],
            superseded_by=md.get("superseded_by"),
            source=md.get("source", "auto"),
        )
        mem.memories.append(entry)
        if mem.enable_hopfield:
            mem._hopfield_store(mem._binary_from_emb(entry.embedding))
    mem._mat_dirty = True
    mem._reindex_all()
    return mem
