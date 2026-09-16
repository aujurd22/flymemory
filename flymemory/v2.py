"""FlyMemory v2.0 — Complete system with semantic encoding, dopamine gating, and MCP server.

Three integrated components:
1. SemanticEncoder: sentence-transformers → 384-dim → sparse binary (k-WTA)
2. DopamineGate: novelty-based storage (only remember surprising content)
3. HopfieldEngine: associative memory with pattern completion
"""

import numpy as np
import json
import os
import time
from typing import List, Tuple, Optional, Dict

# ===== Semantic Encoder =====
class SemanticEncoder:
    """sentence-transformers → sparse binary via random projection + k-WTA."""

    def __init__(self, n_bits: int = 4096, sparsity: float = 0.05, seed: int = 42):
        self.n_bits = n_bits
        self.sparsity = sparsity
        self.k_keep = max(int(n_bits * sparsity), 1)
        rng = np.random.default_rng(seed)
        # Random projection matrix: 384 → n_bits (JL lemma: preserves geometry)
        self.proj = rng.standard_normal((n_bits, 384)).astype(np.float32) / np.sqrt(384)
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def encode(self, text: str) -> np.ndarray:
        """text → 384-dim embedding → sparse binary code."""
        model = self._get_model()
        emb = model.encode(text, convert_to_numpy=True).astype(np.float32)
        # Random projection to n_bits
        code = self.proj @ emb
        # k-WTA binarization: top k_keep → +1, rest → -1
        k = min(self.k_keep, len(code))
        thresh = np.partition(code, -k)[-k]
        return np.where(code >= thresh, 1.0, -1.0)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        codes = embs @ self.proj
        k = self.k_keep
        for i in range(codes.shape[0]):
            row = codes[i]
            thresh = np.partition(row, -k)[-k]
            codes[i] = np.where(row >= thresh, 1.0, -1.0)
        return codes


# ===== Dopamine Gate =====
class DopamineGate:
    """Novelty-based storage gating: only remember SURPRISING content.

    prediction_error = 1 - max_similarity(new, stored)
    Store only if prediction_error > threshold (default 0.3).
    This is the Rescorla-Wagner model: δ = r - V(prediction).
    If fully predicted (δ≈0), no learning occurs (dopamine doesn't fire).
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def check(self, new_code: np.ndarray, stored_codes: List[np.ndarray]) -> Tuple[bool, float]:
        """Returns (should_store, novelty_score)."""
        if not stored_codes:
            return True, 1.0  # First memory: always store
        # Compute max overlap with stored codes
        max_match = 0.0
        for sc in stored_codes:
            match = np.mean(new_code == sc)
            max_match = max(max_match, match)
        novelty = 1.0 - max_match
        return novelty > self.threshold, novelty


# ===== Hopfield Engine =====
class HopfieldEngine:
    """Hopfield associative memory with pattern completion."""

    def __init__(self, n_bits: int = 4096):
        self.n_bits = n_bits
        self.W = np.zeros((n_bits, n_bits), dtype=np.float32)
        self.stored = []

    def store(self, code: np.ndarray):
        self.W += np.outer(code, code)
        self.stored.append(code.copy())

    def recall(self, cue: np.ndarray, max_iter: int = 10) -> np.ndarray:
        s = cue.copy()
        for _ in range(max_iter):
            new_s = np.sign(self.W @ s)
            new_s[new_s == 0] = 1
            if np.array_equal(new_s, s):
                break
            s = new_s
        return s

    def nearest(self, state: np.ndarray) -> Tuple[int, float]:
        best_idx, best_ov = -1, -1.0
        for i, st in enumerate(self.stored):
            ov = np.mean(state == st)
            if ov > best_ov:
                best_ov, best_idx = ov, i
        return best_idx, best_ov


# ===== FlyMemory v2: Complete System =====
class FlyMemoryV2:
    """Complete fly-brain-inspired memory system.

    Combines: semantic encoding + dopamine gating + Hopfield pattern completion.
    """

    def __init__(self, n_bits: int = 4096, dopamine_threshold: float = 0.1,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SemanticEncoder(n_bits, 0.05)
        self.hopfield = HopfieldEngine(n_bits)
        self.gate = DopamineGate(dopamine_threshold)
        self.memories = []  # list of (text, code)
        self.stats = {"stored": 0, "rejected_novelty": 0, "total_inputs": 0}

    def remember(self, text: str, tags: Optional[List[str]] = None) -> Dict:
        """Try to store a memory with dopamine novelty gating."""
        code = self.encoder.encode(text)
        emb = self.encoder._get_model().encode(text, convert_to_numpy=True)
        self.stats["total_inputs"] += 1
        novelty = 1.0

        if self.memories:
            # Cosine distance on MiniLM embeddings = semantic novelty
            max_sim = max(
                float(np.dot(emb, m["emb"]) /
                      (np.linalg.norm(emb) * np.linalg.norm(m["emb"]) + 1e-8))
                for m in self.memories
            )
            novelty = 1.0 - max_sim

        if novelty < 0.1:
            self.stats["rejected_novelty"] += 1
            return {"stored": False, "novelty": novelty,
                    "reason": "not surprising (too similar to existing)"}

        self.hopfield.store(code)
        self.memories.append({"text": text, "code": code,
                              "emb": emb, "tags": tags or []})
        self.stats["stored"] += 1
        return {"stored": True, "novelty": novelty, "memory_id": len(self.memories)-1}

    def recall(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Recall memories relevant to the query via pattern completion."""
        code = self.encoder.encode(query)
        converged = self.hopfield.recall(code)
        results = []
        for m in self.memories:
            ov = np.mean(converged == m["code"])
            results.append((m["text"], ov))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    @property
    def size(self):
        return len(self.memories)
