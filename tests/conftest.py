import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("FLYMEMORY_DEVICE", "cpu")

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
