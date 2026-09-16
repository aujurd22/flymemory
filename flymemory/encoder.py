"""Text encoder: text → sparse binary vector for Hopfield memory.

Uses hashing-based encoding (no ML model dependency).
Each word is hashed to multiple bit positions. k-WTA keeps top 5%.
"""

import hashlib
import numpy as np
from typing import Union


class TextEncoder:
    """Hashes text into sparse binary vectors for Hopfield memory."""

    def __init__(self, n_bits: int = 4096, sparsity: float = 0.05,
                 n_hashes_per_word: int = 3):
        self.n_bits = n_bits
        self.sparsity = sparsity
        self.k_keep = max(int(n_bits * sparsity), 1)
        self.n_hashes = n_hashes_per_word

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a sparse binary vector.

        Each word is hashed to n_hashes_per_word bit positions.
        The result is k-WTA sparsified to top 5%.
        """
        code = np.zeros(self.n_bits, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            for h in range(self.n_hashes):
                hsh = int(hashlib.md5(f"{word}_{h}".encode()).hexdigest(), 16)
                idx = hsh % self.n_bits
                code[idx] += 1.0
        # k-WTA: keep top k_keep activations, zero the rest
        k = min(self.k_keep, len(code))
        if k > 0 and code.max() > 0:
            thresh = np.partition(code, -k)[-k]
            code = np.where(code >= thresh, 1.0, 0.0)
        return code

    def encode_pair(self, context: str, response: str) -> np.ndarray:
        """Encode a context-response pair."""
        combined = f"{context} ||| {response}"
        return self.encode(combined)
