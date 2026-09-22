import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("FLYMEMORY_DEVICE", "cpu")
# full-core torch threads livelock against concurrent training jobs
# (measured 2026-09-23); benches and tests cap at 4
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from flymemory.v3 import SmartMemory, split_chunks, load, save  # noqa: E402


@pytest.fixture()
def mem():
    """Fresh instance; decay_tau=1h so decay tests can use small time offsets."""
    return SmartMemory(n_bits=4096, decay_tau=3600.0)


@pytest.fixture(scope="session")
def warm_model():
    """Force the embedding model to load once for the whole session."""
    from flymemory.v3 import _get_model
    return _get_model()


_CE_HUB_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                           "models--cross-encoder--ms-marco-MiniLM-L-6-v2")


@pytest.fixture(scope="session")
def reranker_ready():
    """Gate for cross-encoder tests: skip where the model is not already in
    the local HF cache (CI has no network and must stay green)."""
    if not os.path.isdir(_CE_HUB_DIR):
        pytest.skip("cross-encoder model not in local HF cache")
    try:
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 4))
    except ImportError:
        pass
    return True
