# FlyMemory

**A long-term memory layer for personal AI agents: chunked semantic + lexical recall, time decay, semantic dedup, and model-driven supersede.**

FlyMemory gives a coding agent a persistent, self-managed memory: every user message is captured by a hook, stored as per-sentence chunks, deduplicated, and recalled into later conversations with age and provenance stamps. Stale facts are not deleted — the calling model marks them *superseded*, so history stays queryable while current state stays clean.

> Note on the name: v1 of this project was a pure Hopfield associative memory
> inspired by the Drosophila mushroom body (kept below as [v1/v2](#v1v2-hopfield-experiments)).
> In v3 the production recall path is semantic + lexical retrieval; the Hopfield
> matrix survives only as an *experimental* associative-expansion layer, and
> measurements (see [Benchmarks](#benchmarks)) currently argue against enabling it.

## How it works (v3)

```text
user message
   ↓  UserPromptSubmit hook (mechanical, ~10ms)
chunked store: one block per sentence (multi-topic messages stay separable)
   ↓
semantic dedup, length-tiered thresholds
   > 0.95 (short) / 0.92 (long)  → strengthen existing entry
   > 0.85 (short) / 0.75 (long)  → merge (longer text wins)
   else                          → new entry (source=hook)
   ↓
model precision-store: the calling agent stores conclusions it judges
important via flymemory_remember (source=model)

recall (per query):
   multilingual embedding cosine, max over query chunks   (vectorized matmul)
 + IDF lexical boost (part numbers, paths, IDs — invisible to embeddings)
 × power-law decay  R(t) = (1 + t/τ)^-0.5, rehearsal-resistant
 - superseded entries excluded (include_superseded=True for history queries)
```

**Decay is a power law, not an exponential half-life**: R(τ) ≈ 0.707 and the
true half-life is 3τ (90 days at the default τ = 30 days). The heavy tail is
intentional — old memories fade slowly instead of vanishing.

**Why semantic consolidation is not threshold-based**: on a multilingual
embedder, a true paraphrase pair can score 0.62 while a same-structure pair
differing only in a detail ("meeting at 15:00 tomorrow" vs "today") scores
0.93. No threshold can merge the first without merging the second, so
near-identical dedup is automatic but *meaning-level* consolidation is left to
the calling model via `flymemory_supersede(old_id, new_id)` — judgment where
the judgment capability lives.

## MCP server + hook (the integration that makes it "just work")

The server runs as a persistent stateless streamable-HTTP MCP service, so
restarting it never breaks connected sessions:

```bash
python flymemory/mcp_v3.py --http        # serves http://127.0.0.1:8765/mcp
```

MCP registration (ZCode `~/.zcode/cli/config.json`, adjust paths):

```json
{
  "mcp": {
    "servers": {
      "flymemory": {
        "type": "http",
        "url": "http://127.0.0.1:8765/mcp",
        "timeoutMs": 120000
      }
    }
  }
}
```

A `UserPromptSubmit` hook forwards every user message to the server and
injects the recall results back into the conversation — capture does not
depend on the model remembering to call a tool:

```json
{
  "hooks": {
    "enabled": true,
    "events": {
      "UserPromptSubmit": [
        { "hooks": [ { "type": "process",
          "command": "C:/path/to/python.exe",
          "args": ["D:/path/to/flymemory/flymemory/hook_auto.py"],
          "timeoutMs": 10000 } ] }
      ]
    }
  }
}
```

`flymemory/flymemory_supervisor.pyw` keeps the server alive (restart on crash,
idle when healthy, never exits); run it headless at login via a Startup
shortcut or a scheduled task.

## The memory rules

| Rule | Mechanism |
|---|---|
| Capture everything mechanically | hook → `flymemory_auto`, silent no-op when the server is down |
| Don't lose novel facts | length-tiered dedup prefers *not* merging |
| Don't resurrect stale states | `flymemory_supersede` marks them; default recall skips them |
| Distinguish "captured" from "judged important" | `source`: hook / model, stamped on every entry and shown in recall output |
| Recall exact identifiers | IDF lexical channel (part numbers, file paths, IDs) |
| Forget slowly, never abruptly | power-law decay + rehearsal; `flymemory_cleanup` prunes below threshold |
| Multi-topic messages stay separable | per-sentence chunking on store, per-chunk max on query |

## Requirements & install

```bash
pip install -r requirements.txt   # torch, sentence-transformers, mcp, uvicorn, numpy, pytest
python -m pytest tests/           # 26 behavioral rule tests
```

The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim,
~470MB) downloads automatically from HuggingFace on first run, then works
offline. CPU inference by default (~10ms/sentence); set `FLYMEMORY_DEVICE=cuda`
to override. The library file (`flymemory_v3.pkl`) is local data and is
git-ignored — no conversation content ships with this repo.

## Benchmarks

Measured on a real 1370-entry library, CPU (see `bench_recall_speed.py`,
`bench_hopfield.py`):

| Path | Result |
|---|---|
| Query encoding (embedder, CPU) | ~10.6 ms/query — dominates end-to-end at this scale |
| Scoring: per-entry Python loop | 8.5 ms/query |
| Scoring: vectorized matmul (shipped) | 1.2 ms/query (**7×**) |
| End-to-end recall | ~11.5 ms/query |
| Hopfield associative expansion vs pure vector recall | Recall@5 0.189 → 0.043 — **hurts**; disabled by default |

The Hopfield result is a capacity story: 1370 patterns far exceed what a
4096-bit binary matrix can separate, so crosstalk dominates. The layer remains
available behind `SmartMemory(enable_hopfield=True)` for small-N experiments —
run `bench_hopfield.py` before trusting it.

## Contradiction-resolution benchmark

`bench_contradiction.py` — 14 temporal scenarios ("user runs Windows" →
"user switched to Fedora"), each with adversarial traps: a *newer off-hand
mention of the old state* ("the old Windows VM is slow" — not a state change),
old states that paraphrase the query better than the new state, and new states
missing the attribute keyword. Five policies share one embedder and one
chunker; supersede markings simulate the calling model's judgment (the
architecture under test: judgment in the caller, mechanical resolution in the
server).

| policy | current@1 | current@3 | stale top1 |
|---|---|---|---|
| dense (plain RAG) | 2/14 | 14/14 | **12/14** |
| dense + recency | 3/14 | 14/14 | 0/14 |
| bm25 | 4/14 | 14/14 | **9/14** |
| bm25 + recency | 10/14 | 14/14 | 0/14 |
| **flymemory (full)** | 6/14 | 13/14 | **0/14** |

Reading:

- Plain similarity retrieval puts a **superseded fact first in 9–12 of 14
  current-state queries** — the mechanical origin of the "stale memory
  confusion" agents suffer from.
- Recency heuristics avoid stale picks but get baited by recent non-state
  mentions (dense+recency: 3/14); a recency-ordered BM25 is the strongest
  mechanical top-1 here (10/14) — an honest result we document, not hide.
- FlyMemory's design intent is *model-resolution from a stamped top-3*, not
  mechanical top-1: with age + provenance stamps on every hit, the calling
  model resolves 13/14 correctly, never sees a superseded fact at rank 1
  (0/14), and recovers full history on demand (`include_superseded=True`,
  6/6 — archived, not forgotten, immune to decay in history mode).
- n=14: indicative, not statistical.

## API sketch

```python
from flymemory.v3 import SmartMemory

mem = SmartMemory()                      # decay_tau=30d, Hopfield off
mem.remember_text("今天讨论了X。还决定了Y。", source="hook")
mem.remember("重要结论：Y 优于 X", source="model")
mem.supersede(old_id, new_id)            # judgment made by the caller

mem.recall("X 的结论是什么")              # hybrid semantic+lexical, decay-weighted
mem.recall("以前是不是用过 X", include_superseded=True)  # history query
```

## v1/v2 Hopfield experiments

The original design (kept in `flymemory/v2.py`, `hopfield.py`, `encoder.py`,
`memory_store.py`, `demo*.py`): text → 5% sparse binary code → multi-
compartment Hopfield network (`W += s^T s`, recall by iterating
`s ← sign(W·s)`), motivated by the fly mushroom body's sparse coding and
recurrent loops. Its claims (20% cue → 100% recovery, O(1) recall) hold only
for that small-N experimental setting, not for the production v3 path above.

## Status

Personal, single-user, evolving. Scale target: the exact vectorized scan is
fine to ~10⁵ entries; beyond that, add an ANN index. Not a Mem0/Letta
competitor — it is the minimal mechanism set that made one agent's memory
reliable across months of daily use.

## License

MIT
