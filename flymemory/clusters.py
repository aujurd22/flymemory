"""Unsupervised entity-version cluster layer (experimental, P-CLUSTER).

Mechanism (validated by the memory_boundary project, H4b/H5/H6, seven
pre-registered experiments): entity-version pairs ("3 岁" -> "4 岁")
form dense cosine regions, so connected components over NORMALIZED
embedding edges (cos >= link_threshold) recover supersede lineages
without any naming rule. Within a cluster, the newest member by
timestamp is the current state -- adding that hierarchy to recall
removed the largest failure class of pure-semantic retrieval (attribute
stale 46% -> 0% in the controlled run; e2e 14% -> 89%).

Deliberately NOT wired into SmartMemory.recall: this module is the
git-clean, unit-tested carrier of the mechanism so it is available,
auditable and re-runnable; adoption into the default path is a separate
registered decision (CONVERGENCE discipline).

Usage:
    cl = ClusterLayer(memories, link_threshold=0.55)
    cl.cluster_of(memory_id)          -> cluster id
    cl.newest_in_cluster(memory_id)   -> newest member entry (by timestamp)
    cl.current_for(query_entry)       -> the entry a "current state" query
                                         should resolve to

CRITICAL: embeddings in the production pkl are UNNORMALIZED (norms
3-4). Every cosine comparison here normalizes first -- the
normalize-before-compare trap (memory_boundary README).
"""
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np


def _norm_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-9)


class ClusterLayer:
    """Connected-component clustering over embedding cosine edges with
    newest-by-timestamp resolution inside each cluster.

    entries: list of dicts with 'memory_id', 'embedding', 'timestamp'
    (the MemoryEntry __dict__ or the pkl dict works for both).
    """

    def __init__(self, entries: Sequence, link_threshold: float = 0.55):
        self.link_threshold = float(link_threshold)
        self.ids: List[int] = [e["memory_id"] for e in entries]
        self.ts: Dict[int, float] = {
            e["memory_id"]: float(e.get("timestamp") or 0.0)
            for e in entries}
        E = np.asarray([e["embedding"] for e in entries], dtype=np.float32)
        E = _norm_rows(E)                      # normalize-first (trap)
        n = len(E)
        # union-find over cos >= threshold edges (numpy blockwise)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        block = 512
        for s in range(0, n, block):
            sims = E[s:s + block] @ E.T         # (b, n)
            for i_local in range(sims.shape[0]):
                row = sims[i_local]
                js = np.nonzero(row >= self.link_threshold)[0]
                for j in js:
                    a, b = find(s + i_local), find(int(j))
                    if a != b:
                        parent[a] = b
        # cluster id = union-find root index; map to member ids
        self.members: Dict[int, List[int]] = defaultdict(list)
        self.cluster_id: Dict[int, int] = {}
        for i in range(n):
            root = find(i)
            mid = self.ids[i]
            self.members[root].append(mid)
            self.cluster_id[mid] = root

    # ---- queries -----------------------------------------------------
    def cluster_of(self, memory_id: int) -> Optional[int]:
        return self.cluster_id.get(memory_id)

    def newest_in_cluster(self, memory_id: int) -> Optional[int]:
        root = self.cluster_id.get(memory_id)
        if root is None:
            return None
        return max(self.members[root], key=lambda m: self.ts.get(m, 0.0))

    def current_for(self, memory_id: int) -> int:
        """The entry a 'current state' query should resolve to: the newest
        member of the entry's cluster (singletons resolve to themselves)."""
        n = self.newest_in_cluster(memory_id)
        return n if n is not None else memory_id

    def n_clusters(self) -> int:
        return len(self.members)

    def multi_clusters(self) -> List[List[int]]:
        return [m for m in self.members.values() if len(m) > 1]
