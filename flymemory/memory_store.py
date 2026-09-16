"""FlyMemory Store — high-level memory management with Hopfield recall."""

import time
import numpy as np
from dataclasses import dataclass
from .hopfield import CompartmentalMemory
from .encoder import TextEncoder


@dataclass
class Memory:
    context: str
    response: str
    timestamp: float
    tags: list
    compartment: int
    code: np.ndarray
    memory_id: int


class FlyMemoryStore:
    def __init__(self, n_compartments=8, n_bits=4096, sparsity=0.05):
        self.encoder = TextEncoder(n_bits, sparsity)
        self.hopfield = CompartmentalMemory(n_compartments, n_bits, sparsity)
        self.memories = {}
        self._next_id = 0
        self.n_bits = n_bits

    def remember(self, context, response, tags=None):
        if tags is None:
            tags = []
        code = self.encoder.encode(f"{context} ||| {response}")
        cid, is_new = self.hopfield.store(code)
        mid = self._next_id
        self._next_id += 1
        self.memories[mid] = Memory(
            context=context, response=response,
            timestamp=time.time(), tags=tags,
            compartment=cid, code=code, memory_id=mid
        )
        return mid

    def recall(self, query, top_k=3):
        code = self.encoder.encode(query)
        all_matches = []
        for cid, comp in enumerate(self.hopfield.compartments):
            converged, _ = comp.recall(code)
            idx, ov = comp.find_nearest(converged)
            if idx >= 0:
                all_matches.append((cid, comp.stored_codes[idx], ov))
        all_matches.sort(key=lambda x: -x[2])
        results = []
        for cid, code_f, ov in all_matches[:top_k]:
            for mem in self.memories.values():
                if np.array_equal(mem.code, code_f):
                    results.append((mem, ov))
                    break
        return results

    @property
    def size(self):
        return len(self.memories)
