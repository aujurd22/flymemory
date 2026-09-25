"""FlyMemory backend for LongMemEval-V2 (v4-rfc §7 dataset integration).

Implements the LME-V2 Memory interface:
  insert(trajectory)   -- called once per selected trajectory of the haystack
  query(query, query_image) -> [{"type": "text", "value": ...}, ...]

FlyMemory mapping: every trajectory's steps become turn-granularity entries
(text chunks, dated by the step timestamp when present); the V4 overlay
layer (consolidated/timeline entries generated offline) can be stacked by
pointing --overlay-file at a JSON of {"entries": [...]} per trajectory.

The class defers engine construction until the first insert so the harness
can instantiate it cheaply at config time.

Status: interface-complete, UNTESTED against real LME-V2 data (dataset
download is GB-scale and the reader needs a Qwen endpoint -- run from a
machine with those available). See v4-rfc.md §7.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, List, Optional

from memory_modules.memory import Memory, register_memory


def _step_text(step: Any) -> str:
    """Best-effort text extraction from a trajectory step dict."""
    if isinstance(step, str):
        return step
    for key in ("text", "content", "thought", "action", "observation"):
        v = step.get(key) if isinstance(step, dict) else None
        if isinstance(v, str) and v.strip():
            return f"{step.get('role', 'step')}: {v}"
    return json.dumps(step, ensure_ascii=False)[:2000]


def _step_ts(step: Any) -> Optional[float]:
    v = step.get("timestamp") or step.get("ts") if isinstance(step, dict) else None
    if isinstance(v, (int, float)):
        return float(v)
    return None


@register_memory("flymemory")
class FlyMemoryV4Backend(Memory):
    def __init__(self, **kwargs: Any) -> None:
        self._n_bits = int(kwargs.get("n_bits", 4096))
        self._overlay_file = kwargs.get("overlay_file")
        self._pkl_path = kwargs.get("pkl_path")
        self._mem = None
        self._loaded = False

    # -- lazy engine (loads persisted pkl only on first insert) --
    def _ensure_engine(self):
        if self._loaded:
            return
        from flymemory.v3 import SmartMemory, load
        if self._pkl_path and os.path.exists(self._pkl_path):
            self._mem = load(self._pkl_path, enable_hopfield=False)
        else:
            self._mem = SmartMemory(n_bits=self._n_bits)
        self._loaded = True

    def insert(self, trajectory: Any) -> None:
        self._ensure_engine()
        traj_id = (trajectory.get("trajectory_id")
                   if isinstance(trajectory, dict) else None) or "traj"
        steps = (trajectory.get("steps") or trajectory.get("turns")
                 or trajectory.get("messages") or [])
        if isinstance(steps, dict):
            steps = list(steps.values())
        for st in steps:
            text = _step_text(st)
            if not text.strip():
                continue
            self._mem.remember_text(
                text, tags=["lmev2", traj_id], source="import",
                timestamp=_step_ts(st), force_new=True,
            )

    def query(self, query: str, query_image: Optional[str] = None) -> List[dict]:
        self._ensure_engine()
        hits = self._mem.recall(query, top_k=5)
        out = []
        for m, _sim, _eff in hits:
            out.append({"type": "text", "value": m.text})
        # V4 direct state lookup: entity-state entries answer "what is X now"
        # style questions without competing for top-k slots
        sl = getattr(self._mem, "state_lookup", None)
        if sl and query.lower().startswith(("what is", "what's", "which")):
            cur = self._mem.recall(query, top_k=20)
            for m, _s, _e in cur:
                if m.state_key and m.state_value:
                    out.append({"type": "text",
                                "value": f"CURRENT [{m.state_key}]: "
                                         f"{m.state_value}"})
                    break
        return out or [{"type": "text", "value": "(no memory)"}]

    def _save_backend(self, path: str) -> None:
        if not self._loaded or self._mem is None:
            return
        from flymemory.v3 import save
        save(self._mem, os.path.join(path, "flymemory_v4.pkl"))

    def _load_backend(self, path: str) -> None:
        from flymemory.v3 import load
        p = os.path.join(path, "flymemory_v4.pkl")
        if os.path.exists(p):
            self._mem = load(p, enable_hopfield=False)
            self._loaded = True
