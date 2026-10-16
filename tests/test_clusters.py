"""Unit tests for the cluster layer (P-CLUSTER safety property).

The safety property that matters most: SINGLETON clusters must resolve
to themselves -- the vast majority of entries are singletons and must
see zero behavior change. Plus the normalize-first invariant (the pkl
stores unnormalized embeddings).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "flymemory"))

from flymemory.clusters import ClusterLayer  # noqa: E402


def mk(mid, vec, ts):
    return {"memory_id": mid, "embedding": vec, "timestamp": ts}


def test_singleton_resolves_to_self():
    rng = np.random.default_rng(0)
    entries = [mk(i, rng.standard_normal(32).astype(np.float32), i)
               for i in range(10)]
    layer = ClusterLayer(entries, link_threshold=0.9)
    for e in entries:
        assert layer.current_for(e["memory_id"]) == e["memory_id"]


def test_unormalized_inputs_still_cluster():
    """The normalize-first invariant: same geometry with/norm scaling
    must give the same clustering."""
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
    v2 /= np.linalg.norm(v2)
    entries_n = [mk(1, v1, 1), mk(2, v2, 2), mk(3, -v1, 3)]
    entries_u = [mk(1, v1 * 3.7, 1), mk(2, v2 * 2.9, 2),
                 mk(3, -v1 * 4.1, 3)]      # unnormalized, norms 3-4
    ln = ClusterLayer(entries_n, link_threshold=0.55)
    lu = ClusterLayer(entries_u, link_threshold=0.55)
    assert ln.cluster_of(1) == ln.cluster_of(2)
    assert lu.cluster_of(1) == lu.cluster_of(2)
    assert ln.cluster_of(1) != ln.cluster_of(3)
    assert lu.cluster_of(1) != lu.cluster_of(3)


def test_newest_wins_in_cluster():
    v1 = np.array([1.0, 0.02, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.04, 0.0], dtype=np.float32)
    entries = [mk(10, v1, 100.0), mk(11, v2, 200.0),
               mk(12, np.array([0.0, 0.0, 1.0], dtype=np.float32), 150.0)]
    layer = ClusterLayer(entries, link_threshold=0.9)
    assert layer.cluster_of(10) == layer.cluster_of(11)
    assert layer.current_for(10) == 11          # newer wins
    assert layer.current_for(11) == 11
    assert layer.current_for(12) == 12          # singleton untouched
