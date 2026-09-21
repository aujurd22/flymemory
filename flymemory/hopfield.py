"""FlyMemory Core — Hopfield Associative Memory Engine (v1/v2 experimental).

STATUS (2026-09-21): NOT part of the production v3 recall path. Kept for the
v1/v2 experiments and small-N research. Theory notes for future work:

1. Capacity: `0.138 * n_bits` is the classic DENSE Hopfield bound (Hopfield
   1982 / Amit-Gutfreund-Sompolinsky). With 5% sparse k-WTA codes the constant
   is different (sparse codes increase capacity per Amir et al. 1983-style
   analysis, but the exact factor for this projection+clipping pipeline was
   never derived here). bench_hopfield.py measured crosstalk already dominating
   at N=1370 -- far below 0.138*4096 ≈ 565 -- so treat 500/compartment as an
   UNVERIFIED folk number.
2. The Hebbian accumulation keeps the diagonal W[i,i] = |s|^2 terms (standard
   Hopfield formulations usually zero it); harmless at this scale, nonstandard.
3. Synchronous update can enter period-2 oscillations; the `prev` comparison
   detects one 2-cycle shape but is not a convergence guarantee. Asynchronous
   (random-order) update would have the energy-monotonicity guarantee.

Pattern completion verified: 20% cue -> 100% recovery (T94, small-N regime).
"""

import numpy as np
from typing import List, Tuple


class HopfieldMemory:
    """Hopfield associative memory with sparse binary coding.

    Capacity: ~0.138 * n_bits patterns (theoretical limit).
    With n_bits=4096 and 5% sparsity: practical capacity ~500 patterns.
    """

    def __init__(self, n_bits: int = 4096, sparsity: float = 0.05):
        self.n_bits = n_bits
        self.sparsity = sparsity
        self.k_keep = max(int(n_bits * sparsity), 1)
        self.W = np.zeros((n_bits, n_bits), dtype=np.float32)
        self.n_stored = 0
        self.stored_codes = []

    def encode_features(self, features: np.ndarray, proj: np.ndarray) -> np.ndarray:
        """Convert real-valued features to sparse binary code via k-WTA + random projection."""
        code = (proj @ features.astype(np.float32))
        k = min(self.k_keep, len(code))
        thresh = np.partition(code, -k)[-k]
        return np.where(code >= thresh, 1.0, -1.0)

    def store(self, code: np.ndarray) -> bool:
        """Store a pattern. Returns True if new (not duplicate)."""
        for existing in self.stored_codes:
            if np.array_equal(existing, code):
                return False
        self.W += np.outer(code, code)
        self.n_stored += 1
        self.stored_codes.append(code.copy())
        return True

    def recall(self, cue: np.ndarray, max_iter: int = 20) -> Tuple[np.ndarray, int]:
        """Hopfield dynamics: converge to nearest stored pattern."""
        s = cue.copy()
        prev = None
        it = 0
        for it in range(max_iter):
            new_s = np.sign(self.W @ s)
            new_s[new_s == 0] = 1
            if prev is not None and np.array_equal(new_s, prev):
                break
            prev = s.copy()
            s = new_s
        return s, it + 1

    def overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(a == b))

    def find_nearest(self, converged: np.ndarray) -> Tuple[int, float]:
        best_idx, best_ov = -1, -1.0
        for i, code in enumerate(self.stored_codes):
            ov = self.overlap(converged, code)
            if ov > best_ov:
                best_ov, best_idx = ov, i
        return best_idx, best_ov

    @property
    def capacity_estimate(self) -> int:
        return int(0.138 * self.n_bits)


class CompartmentalMemory:
    """Multi-compartment memory (like 34 fly MBONs, prevents catastrophic forgetting).

    Each compartment is an independent HopfieldMemory. Memories are routed
    by hash to prevent interference between unrelated memories.
    """

    def __init__(self, n_compartments: int = 8, n_bits: int = 4096, sparsity: float = 0.05):
        self.compartments = [HopfieldMemory(n_bits, sparsity) for _ in range(n_compartments)]
        self.n_compartments = n_compartments

    def route(self, code: np.ndarray) -> int:
        return hash(code.tobytes()) % self.n_compartments

    def store(self, code: np.ndarray) -> Tuple[int, bool]:
        cid = self.route(code)
        is_new = self.compartments[cid].store(code)
        return cid, is_new

    def recall_all(self, cue: np.ndarray, top_k: int = 3) -> List[Tuple[int, np.ndarray, float]]:
        """Recall from all compartments, return top_k (cid, code, overlap)."""
        results = []
        for cid, comp in enumerate(self.compartments):
            converged, _ = comp.recall(cue)
            idx, ov = comp.find_nearest(converged)
            if idx >= 0:
                results.append((cid, comp.stored_codes[idx], ov))
        results.sort(key=lambda x: -x[2])
        return results[:top_k]
