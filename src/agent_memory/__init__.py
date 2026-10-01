"""agent-memory: persistent, learnable memory for AI agents.

Public surface::

    from agent_memory import MemoryStore, Recaller, Consolidator

    store = MemoryStore("agent-memory.db")
    store.remember("the deploy script is ./deploy.sh", tags=["ops"])
    hits = Recaller(store).recall("how do I deploy?")
    Consolidator(store).consolidate()

The three halves of the name are three modules: :mod:`agent_memory.store` stores,
:mod:`agent_memory.recall` retrieves, :mod:`agent_memory.consolidate` evolves.
"""

__version__ = "0.1.0"

from .consolidate import ConsolidationResult, Consolidator
from .recall import Recall, Recaller
from .store import KINDS, Memory, MemoryStore

__all__ = [
    "KINDS",
    "ConsolidationResult",
    "Consolidator",
    "Memory",
    "MemoryStore",
    "Recall",
    "Recaller",
    "__version__",
]
