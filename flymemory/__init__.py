"""FlyMemory: Hopfield associative memory for AI conversations."""
from .memory_store import FlyMemoryStore, Memory
from .hopfield import HopfieldMemory, CompartmentalMemory
from .encoder import TextEncoder
__version__ = "0.1.0"
