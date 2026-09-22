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
import sys
import pickle
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Iterable, Set
from dataclasses import dataclass, field

# ===== Lazy model loading =====
_model = None

def _model_name() -> str:
    return os.environ.get("FLYMEMORY_MODEL",
                          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


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
        _model = SentenceTransformer(_model_name(), device=device)
    return _model


def _embed(text: str) -> np.ndarray:
    return _get_model().encode(text, convert_to_numpy=True).astype(np.float32)


# Lexical-channel weight: exact identifier matches (part numbers, paths, IDs)
# must be able to outrank mere semantic similarity. 0.25 keeps semantic ranking
# primary while letting a rare-token exact match win. See bench_recall_speed.py.
LEX_WEIGHT = 0.25

# Source weight = cheap importance proxy (the "importance" dimension without a
# judge model): model-precision-stored and curated-import entries were judged
# worth keeping by the calling agent; hook captures are unjudged transcript —
# including self-referential complaints that would otherwise outrank the very
# content they complain about (measured 2026-09-20: hook echo 0.719 beat the
# novel-progress entry at 0.71 on the same topic).
SOURCE_WEIGHT = {"model": 1.15, "import": 1.15, "model-supersede": 1.15,
                 "hook": 0.9, "auto": 1.0}


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


def _is_junk_chunk(text: str) -> bool:
    """Reject chunks that are mostly symbols/table-borders/markdown debris.

    ASCII-art lines ("┌────┐", "│ ● │") and markdown leftovers ("---", "**x**")
    embed into vectors that spuriously match unrelated queries (measured 0.65
    cosine against a Chinese prose query, 2026-09-20) and outrank real content
    via lexical ties. Contentful = letters / digits / CJK.
    """
    if not text:
        return True
    contentful = sum(1 for ch in text if ch.isascii() and ch.isalnum()
                     or "\u4e00" <= ch <= "\u9fff")
    return contentful / len(text) < 0.45 or contentful < 4


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
    evidence_ids: List[int] = field(default_factory=list)  # for consolidated entries: the raw entries kept as evidence


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


_CRED_PATTERNS = ("ghp_", "github_pat_", "pypi-AgEI", "sk-ant-", "sk-proj-",
                  "AKIA", "-----BEGIN", "xoxb-", "xoxp-")


def _contains_credential(text: str) -> bool:
    """Server-side sanitizer: applies to EVERY write path (hook, model, import),
    not just the hook -- the hook filter alone is a single point of trust."""
    return any(p in text for p in _CRED_PATTERNS)


# merge provenance rule: a restatement must never demote the provenance of the
# entry it merges into (a hook retelling of a model-judged conclusion stays
# "model" -- SOURCE_WEIGHT depends on it)
_SOURCE_RANK = {"auto": 0, "hook": 1, "import": 2, "model": 3}


# ===== SmartMemory =====
class SmartMemory:
    """Semantic+lexical associative memory with Ebbinghaus-style decay.

    Power-law decay R(t) = (1 + t/tau)^-0.5 modulated by rehearsal:
    R is ~0.707 at t=tau and 0.5 at t=3*tau (the true half-life).
    """

    def __init__(self, n_bits: int = 4096, sparsity: float = 0.05,
                 decay_tau: float = 2592000.0,          # 30 days: long-lived reference knowledge
                 decay_half_life: Optional[float] = None,  # legacy kwarg alias for decay_tau
                 enable_hopfield: bool = False,
                 two_stage: bool = False,
                 hamming_candidates: int = 100,
                 code_keep: float = 0.5):
        """enable_hopfield=False by DEFAULT: bench_hopfield.py measured the
        associative expansion HURTING retrieval (Recall@5 0.189 → 0.043 on a
        1370-entry library — the 4096-bit W matrix is saturated far beyond its
        capacity, so crosstalk dominates). The 64 MiB matrix, per-store outer
        product and load-time rebuild are only paid when explicitly enabled.

        two_stage: Hamming prefilter (packed sparse codes) + dense rerank.
        OFF by default — see README "Two-stage retrieval" for the usage
        preconditions. Enable only for large libraries (N >= ~5000)."""
        if decay_half_life is not None:
            decay_tau = decay_half_life
        self.n_bits = n_bits
        self.sparsity = sparsity
        self.k_keep = max(int(n_bits * sparsity), 1)
        rng = np.random.default_rng(42)
        self.proj = rng.standard_normal((n_bits, 384)).astype(np.float32) / np.sqrt(384)
        self.enable_hopfield = enable_hopfield
        self.W = np.zeros((n_bits, n_bits), dtype=np.float32) if enable_hopfield else None
        self.two_stage = two_stage
        self.hamming_candidates = max(int(hamming_candidates), 1)
        # keep fraction for the PREFILTER binary codes. Measured sweep
        # (bench_prefilter_sweep.py, N=1370, C=100): fidelity@C 0.691 @5%,
        # 0.849 @25%, 0.861 @50% -- the biology-derived 5% is too sparse for
        # retrieval prefiltering (FlyPoet's U-curve transfers here).
        self.code_keep = float(code_keep)
        self.memories: List[MemoryEntry] = []
        self._next_id = 0
        self.decay_tau = float(decay_tau)
        # vectorized-recall cache
        self._mat: Optional[np.ndarray] = None
        self._mat_dirty = True
        self._codes_dirty = True
        # packed-code cache for the Hamming prefilter (built lazily)
        self._codes: Optional[np.ndarray] = None
        self._codes_dirty = True
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

    def _decay_tau_for(self, mem: MemoryEntry) -> float:
        """Dopamine-gated decay (the DA-neuron analogy): model-stored entries
        were JUDGED worth keeping, so their effective tau doubles. Unjudged
        hook chatter decays at the base rate."""
        return self.decay_tau * (2.0 if mem.source == "model" else 1.0)

    def _decay_weight(self, mem: MemoryEntry) -> float:
        dt = time.time() - mem.last_accessed
        tau = self._decay_tau_for(mem)
        base = (1.0 + dt / tau) ** (-0.5)
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

    # ----- packed-code cache (Hamming prefilter) -----
    _popcount = getattr(np, "bitwise_count", None)

    def _pack_bits(self, bits: np.ndarray) -> np.ndarray:
        return np.packbits(bits.astype(np.uint8), axis=-1)

    def _query_code(self, q_embs: List[np.ndarray]) -> np.ndarray:
        """Query binary code: per-chunk k-WTA code, OR-unioned over chunks
        (a prefilter must widen, not narrow). Returns packed (n_bytes,) uint8."""
        union = np.zeros(self.n_bits, dtype=bool)
        for qe in q_embs:
            code = self.proj @ qe
            k = max(min(int(self.n_bits * self.code_keep), len(code)), 1)
            thresh = np.partition(code, -k)[-k]
            union |= code >= thresh
        return np.packbits(union.astype(np.uint8))

    def _packed_code_matrix(self) -> np.ndarray:
        """(N, n_bits/8) uint8 packed codes, rebuilt lazily from embeddings."""
        if self._codes is None or self._codes_dirty:
            if self.memories:
                E = np.stack([m.embedding for m in self.memories]).astype(np.float32)
                code = E @ self.proj.T
                k = max(min(int(self.n_bits * self.code_keep), code.shape[1]), 1)
                thresh = np.partition(code, -k, axis=1)[:, -k][:, None]
                self._codes = np.packbits((code >= thresh).astype(np.uint8), axis=1)
            else:
                self._codes = np.zeros((0, self.n_bits // 8), dtype=np.uint8)
            self._codes_dirty = False
        return self._codes

    def _hamming(self, q_packed: np.ndarray) -> np.ndarray:
        codes = self._packed_code_matrix()
        xor = np.bitwise_xor(codes, q_packed)
        pc = self._popcount
        if pc is not None:
            return pc(xor).sum(axis=-1, dtype=np.int32)
        return np.unpackbits(xor, axis=-1).sum(axis=-1, dtype=np.int32)

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
        """IDF-weighted lexical match ratio per memory, in [0, 1].

        Query tokens ABSENT from the corpus count toward the denominator at max
        IDF — otherwise a query full of unstemmed content words ("study" when
        only "studies" is stored) inflates the ratio for whatever matches the
        remaining function words (measured 2026-09-19 in bench_contradiction).
        """
        n = len(self.memories)
        scores = np.zeros(n, dtype=np.float32)
        if n == 0 or not self._df:
            return scores
        n_docs = max(n, 1)
        idf_max = math.log(1.0 + n_docs)
        matched: Dict[int, float] = {}
        total_idf = 0.0
        for qt in q_texts:
            for tok in _tokenize(qt):
                df = self._df.get(tok, 0)
                idf = math.log(1.0 + n_docs / df) if df else idf_max
                total_idf += idf
                if df:
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
                      tags: Optional[list] = None, source: str = "hook",
                      timestamp: Optional[float] = None) -> Dict:
        """Chunked store entry point: long text is split per sentence and each chunk
        goes through dedup/merge; short messages fall back to a single chunk via
        split_chunks' fragment merging.

        timestamp: optional backdated creation time (epoch seconds) for importing
        historical records — affects decay ordering.

        Returns dict with:
          action: "new" / "merged" / "strengthened" / "rejected" / "mixed"
          counts: {"new": n, "merged": n, "strengthened": n}
          memory_ids: all touched entry ids; chunks: chunks processed
        """
        if _contains_credential(text):
            # server-side sanitizer: credentials never enter the library,
            # regardless of which write path delivered them
            return {"stored": False, "action": "rejected", "novelty": 0.0,
                    "memory_id": None, "reason": "credential-like content"}
        chunks = [c for c in split_chunks(text) if not _is_junk_chunk(c)]
        counts = {"new": 0, "merged": 0, "strengthened": 0, "rejected": 0}
        ids = []
        last = None
        for c in chunks:
            r = self.remember(c, response=response, tags=tags, source=source,
                              timestamp=timestamp)
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
                 tags: Optional[list] = None, source: str = "hook",
                 timestamp: Optional[float] = None) -> Dict:
        """Store one chunk with auto-dedup via semantic similarity.

        timestamp: optional backdated creation time (epoch seconds).
        Junk chunks (symbol/table-border debris) are rejected.

        Returns dict with: stored, action ("new"/"merged"/"strengthened"/"rejected"),
        novelty, memory_id.
        """
        if _is_junk_chunk(text):
            return {"stored": False, "action": "rejected", "novelty": 0.0,
                    "memory_id": None}
        binary, emb = self._encode(text)

        if _contains_credential(text):
            # server-side sanitizer: credentials never enter the library,
            # regardless of which write path delivered them
            return {"stored": False, "action": "rejected", "novelty": 0.0,
                    "memory_id": None, "reason": "credential-like content"}

        # dedup scan, vectorized (max cosine over the cached matrix); superseded
        # entries excluded -- P0: the history layer is never a dedup target, new
        # facts must not be written into nodes default recall cannot see
        active_idx = [i for i, m in enumerate(self.memories)
                      if m.superseded_by is None]
        best_match = None
        best_sim = 0.0
        if active_idx:
            M = self._emb_matrix()[active_idx]
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
            e = emb / (np.linalg.norm(emb) + 1e-8)
            sims = M @ e
            j = int(np.argmax(sims))
            best_sim = float(sims[j])
            best_match = self.memories[active_idx[j]]

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
                self._index_entry(best_match)
                self._mat_dirty = True
                self._codes_dirty = True
            # merge semantics (explicit): text/embedding may refresh from a
            # longer restatement; source NEVER downgrades (rank order: model >
            # import > hook > auto); creation timestamp and tags are preserved
            # (the age stamp reflects when the fact was first learned); response
            # refreshes only when the restatement provides one.
            if _SOURCE_RANK.get(source, 0) >= _SOURCE_RANK.get(best_match.source, 0):
                best_match.source = source
            if response:
                best_match.response = response
            best_match.access_count += 1
            best_match.last_accessed = time.time()
            return {"stored": True, "action": "merged",
                    "novelty": 1.0 - best_sim, "memory_id": best_match.memory_id}

        now = timestamp if timestamp is not None else time.time()
        entry = MemoryEntry(
            text=text, response=response,
            embedding=emb, timestamp=now, last_accessed=now,
            access_count=0, tags=tags or [], memory_id=self._next_id,
            source=source,
        )
        self.memories.append(entry)
        self._index_entry(entry)
        self._mat_dirty = True
        self._codes_dirty = True
        if self.enable_hopfield:
            self._hopfield_store(binary)
        self._next_id += 1
        return {"stored": True, "action": "new",
                "novelty": 1.0 - best_sim, "memory_id": entry.memory_id}

    # ----- supersede -----
    def supersede(self, old_id: int, new_id: int) -> Tuple[bool, str]:
        """Mark an entry as superseded by a newer one (excluded from default recall).

        The judgment is made by the calling model (it understands semantics);
        this is the mechanical marker only. Validates both ids, rejects
        self-supersede and cycles (a must never end up superseded by its own
        chain), so superseded_by forms a proper lineage, not arbitrary ints.

        Returns (ok, reason).
        """
        if old_id == new_id:
            return False, "old_id == new_id"
        by_id = {m.memory_id: m for m in self.memories}
        if old_id not in by_id:
            return False, f"old #{old_id} does not exist"
        if new_id not in by_id:
            return False, f"new #{new_id} does not exist"
        cur, seen = new_id, set()
        while cur is not None and cur not in seen:
            if cur == old_id:
                return False, "cycle: new already (transitively) supersedes old"
            seen.add(cur)
            nxt = by_id.get(cur)
            cur = nxt.superseded_by if nxt else None
        by_id[old_id].superseded_by = new_id
        return True, "ok"

    # ----- recall -----
    def recall(self, query: str, top_k: int = 5,
               include_superseded: bool = False,
               two_stage: Optional[bool] = None) -> List[Tuple[MemoryEntry, float, float]]:
        """Hybrid recall: vectorized semantic similarity (max over query chunks)
        × power-law decay + IDF lexical boost (exact identifiers), superseded
        entries excluded unless include_superseded=True.

        two_stage: Hamming prefilter over packed sparse codes, then dense rerank
        on the top-C candidates (constructor: two_stage=True, hamming_candidates).
        Usage preconditions in README "Two-stage retrieval" — enable only for
        large libraries (N >= ~5000); below that the dense path is faster.

        Returns top_k of (memory, semantic_sim, effective_score).
        """
        if not self.memories:
            return []
        q_texts = split_chunks(query) or [query]
        Q = np.stack([_embed(qt) for qt in q_texts])
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        now = time.time()
        use_2s = self.two_stage if two_stage is None else two_stage

        if use_2s:
            # stage 1: Hamming prefilter over packed codes (query code = OR-union
            # of per-chunk codes — a prefilter must widen, not narrow)
            q_packed = self._query_code(list(Q))
            ham = self._hamming(q_packed)
            c = min(self.hamming_candidates, len(self.memories))
            cand = np.argsort(ham, kind="stable")[:c]
            # stage 2: dense rerank on candidates only
            Mc = self._emb_matrix()[cand]
            mnc = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-8)
            sim_cand = (qn @ mnc.T).max(axis=0)
            mems = [self.memories[int(i)] for i in cand]
            last = np.array([m.last_accessed for m in mems], dtype=np.float64)
            acc = np.array([m.access_count for m in mems], dtype=np.float64)
            dw_c = np.minimum(((1.0 + (now - last) / self.decay_tau) ** -0.5)
                              * (1.0 + 0.5 * np.log2(1.0 + acc)), 1.0)
            lex_cand = self._lex_scores(q_texts)[cand]
            sw_c = np.array([SOURCE_WEIGHT.get(m.source, 1.0) for m in mems],
                            dtype=np.float32)
            eff_c = dw_c * sw_c * (sim_cand + LEX_WEIGHT * lex_cand)
            order = cand[np.argsort(-eff_c)]
            results = []
            for idx in order:
                m = self.memories[int(idx)]
                if m.superseded_by is not None and not include_superseded:
                    continue
                pos = int(np.where(cand == idx)[0][0])
                results.append((m, float(sim_cand[pos]), float(eff_c[pos])))
                if len(results) >= top_k:
                    break
            # rehearsal side effect on returned candidates only (prefilter mode:
            # non-candidates were never scored)
            for m, _, _ in results:
                m.last_accessed = now
                m.access_count += 1
            return results

        M = self._emb_matrix()
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        sim_vec = (qn @ mn.T).max(axis=0)                      # (N,)

        now = time.time()
        last = np.array([m.last_accessed for m in self.memories], dtype=np.float64)
        acc = np.array([m.access_count for m in self.memories], dtype=np.float64)
        taus = np.array([self._decay_tau_for(m) for m in self.memories], dtype=np.float64)
        dw = np.minimum(((1.0 + (now - last) / taus) ** -0.5)
                        * (1.0 + 0.5 * np.log2(1.0 + acc)), 1.0)
        if include_superseded:
            # history mode: an explicit lookup of what was true back then.
            # Entries are retrieved for their content, immune to decay —
            # archived, not forgotten.
            dw = np.ones_like(dw)

        lex_vec = self._lex_scores(q_texts)
        src_w = np.array([SOURCE_WEIGHT.get(m.source, 1.0) for m in self.memories],
                         dtype=np.float32)
        eff = dw * src_w * (sim_vec + LEX_WEIGHT * lex_vec)

        # lexical channel as a separate ranked list for RRF fusion with the
        # semantic ranking (each channel covers what the other misses)
        lex_order = np.argsort(-lex_vec)[:200]
        dense_order = np.argsort(-sim_vec)[:200]
        fused = {}
        for rank, idx in enumerate(dense_order):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(lex_order):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank)
        order = np.array(sorted(fused, key=lambda i: -fused[i]))

        results = []
        for idx in order:
            m = self.memories[int(idx)]
            if m.superseded_by is not None and not include_superseded:
                continue
            results.append((m, float(sim_vec[idx]), float(eff[idx])))
            if len(results) >= top_k:
                break

        # rehearsal = reactivation: only entries actually INJECTED into the
        # caller's context get refreshed. The earlier wide rule (refresh every
        # entry with sim>0.5) made ALL entries immortal -- measured 2026-09-20:
        # 100% pinned at retention 1.0, decay inert, cleanup never fires
        # (bench_rehearsal_sim.py). Rehearsal must stay scarce to mean anything.
        for m, _, _ in results:
            m.last_accessed = now
            m.access_count += 1
        return results

    def recent(self, minutes: float = 90, limit: int = 12) -> List[MemoryEntry]:
        """Working-memory trail: entries captured within the last `minutes`,
        chronological, superseded excluded. This is the compression-protection
        channel — context compaction destroys the recent narrative thread, and
        no semantic query can recover a deictic reference like "that thing from
        just now"; the time-ordered trail can."""
        cutoff = time.time() - minutes * 60
        trail = [m for m in self.memories
                 if m.timestamp >= cutoff and m.superseded_by is None]
        trail.sort(key=lambda m: m.timestamp)
        return trail[-limit:]

    @staticmethod
    def _cut(text: str, n: int) -> str:
        """Truncate at a sentence boundary when possible (a hard cut can turn
        'don't use A, use B' into 'don't use A' -- measured concern, 2026-09-21)."""
        if len(text) <= n:
            return text
        cut = text[:n]
        for p in ("。", "！", "？", ".", "!", "?", "；", ";"):
            i = cut.rfind(p)
            if i >= n * 0.5:
                return cut[:i + 1]
        return cut

    def session_pack(self, minutes: float = 180,
                     trail_limit: int = 15, conclusions_limit: int = 5) -> str:
        """Compression-recovery pack: the recent session trail plus the active
        long-term conclusions. Injected by the SessionStart(compact) hook right
        after the host compresses a conversation. Lines carry memory ids so the
        model can supersede/forget from the pack. Empty string when there is
        nothing to recover."""
        now = time.time()
        trail = self.recent(minutes=minutes, limit=trail_limit)
        conclusions = sorted((m for m in self.memories
                              if m.source == "model" and m.superseded_by is None),
                             key=lambda m: -m.timestamp)[:conclusions_limit]
        parts = []
        if trail:
            lines = [f"  [#{m.memory_id} | "
                     f"{time.strftime('%H:%M', time.localtime(m.timestamp))} | "
                     f"{m.source}] {self._cut(m.text, 120)}" for m in trail]
            parts.append("RECENT SESSION TRAIL (oldest → newest):\n" + "\n".join(lines))
        if conclusions:
            lines = [f"  [#{m.memory_id} | "
                     f"{time.strftime('%m-%d %H:%M', time.localtime(m.timestamp))} | "
                     f"{m.source}] {self._cut(m.text, 140)}" for m in conclusions]
            parts.append("ACTIVE LONG-TERM CONCLUSIONS (newest first):\n" + "\n".join(lines))
        return "\n".join(parts)

    def decay_cleanup(self, min_retention: float = 0.1,
                      protect_model_stored: bool = False) -> int:
        """Directed forgetting (cf. the directed-forgetting shower, REPORT_MEM §③):
        prune entries whose retention fell below min_retention.

        protect_model_stored=True implements the asymmetry measured in
        forgetting_shower.py — forgetting must target fine detail, not backbone:
        model/import-stored entries (judged worth keeping) are only pruned when
        retention falls below min_retention/3, while unjudged hook chatter is
        pruned at the full threshold.
        """
        removed = 0
        surviving = []
        for mem in self.memories:
            dw = self._decay_weight(mem)
            threshold = min_retention
            if protect_model_stored and mem.source in ("model", "import"):
                threshold = min_retention / 3.0
            if dw >= threshold:
                surviving.append(mem)
            else:
                removed += 1
                self._unindex(mem.memory_id)
        if removed:
            self.memories = surviving
            self._mat_dirty = True
        self._codes_dirty = True
        return removed

    def forget(self, memory_id: int) -> Optional[str]:
        """Targeted forgetting: hard-delete one entry by id (Berry et al. 2018:
        active forgetting is a function, not a failure). Returns the deleted
        text, or None if the id does not exist."""
        for i, mem in enumerate(self.memories):
            if mem.memory_id == memory_id:
                self._unindex(memory_id)
                text = mem.text
                self.memories.pop(i)
                self._mat_dirty = True
                self._codes_dirty = True
                return text
        return None

    @property
    def size(self):
        return len(self.memories)


# ===== Persistence =====
def save(memory: SmartMemory, path: str):
    """Save to disk (embeddings + metadata; W is rebuilt on load)."""
    data = {
        "schema": "3.2",
        "embedder": _model_name(),          # identity of the embedding model
        "config": {                          # runtime flags, honored on load
            "enable_hopfield": memory.enable_hopfield,
            "two_stage": memory.two_stage,
            "hamming_candidates": memory.hamming_candidates,
        },
        "memories": [
            {
                "text": m.text, "response": m.response,
                "embedding": m.embedding.tolist(),
                "timestamp": m.timestamp, "last_accessed": m.last_accessed,
                "access_count": m.access_count, "tags": m.tags,
                "memory_id": m.memory_id,
                "superseded_by": m.superseded_by,
                "source": m.source,
                "evidence_ids": list(m.evidence_ids),
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


def load(path: str, enable_hopfield: Optional[bool] = None) -> SmartMemory:
    """Load from disk. Accepts both the current schema (decay_tau, source,
    superseded_by, config, embedder) and the legacy one (decay_half_life only).
    enable_hopfield: None = honor the config stored in the file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    tau = data.get("decay_tau") or data.get("decay_half_life") or 2592000.0
    cfg = data.get("config") or {}
    hop = enable_hopfield if enable_hopfield is not None else cfg.get("enable_hopfield", False)
    mem = SmartMemory(n_bits=data["n_bits"], decay_tau=tau, enable_hopfield=hop,
                      two_stage=cfg.get("two_stage", False),
                      hamming_candidates=cfg.get("hamming_candidates", 100))
    stored_embedder = data.get("embedder")
    if stored_embedder and stored_embedder != _model_name():
        # embeddings were produced by a different model: retrieval quality is
        # degraded until the library is re-embedded -- fail loud in the log
        sys.stderr.write(f"[flymemory] WARNING: library embedded with '{stored_embedder}', "
                         f"current model is '{_model_name()}' -- re-embed recommended\n")
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
            evidence_ids=list(md.get("evidence_ids", [])),
        )
        mem.memories.append(entry)
        if mem.enable_hopfield:
            mem._hopfield_store(mem._binary_from_emb(entry.embedding))
    mem._mat_dirty = True
    mem._reindex_all()
    return mem
