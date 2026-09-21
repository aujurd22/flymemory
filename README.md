# FlyMemory

*Last updated: 2026-09-21 · v3.3*

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
   ↓  UserPromptSubmit hook (mechanical: capture + recall + store)
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

## Design positioning

FlyMemory is an **explicit, inspectable memory state machine** — not a
model-driven memory synthesizer. Every state transition is mechanical and
auditable: `superseded_by` lineage, `source` provenance, power-law decay,
directed forgetting, evidence-linked consolidation, and a recovery pack after
context compaction. Capture is mechanical (hook), judgment is the calling
model's (precision-store, supersede, consolidate, forget) — the server never
runs an LLM, and everything is readable in one small file.

That is a deliberate contrast with hosted "memory synthesis" approaches
(background model-side consolidation of raw chats, e.g. ChatGPT's memory):
those optimize synthesis quality at service scale; FlyMemory optimizes
**inspectability, verifiability and data locality** for a personal agent.
Raw entries are never deleted — consolidation adds higher-order entries with
`evidence_ids` back-links (abstraction without loss), and superseded states
remain queryable via `include_superseded=True`.

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
| Fragmented knowledge gets abstracted | `flymemory_consolidate(ids, conclusion)` builds a higher-order entry with `evidence_ids` back-links; raw entries kept as evidence |

## Install

### 0. Prerequisites

- Python **3.10+** (3.13 tested), Git
- ~5 GB disk for dependencies (torch is the big one) + a ~470 MB embedding model
- OS: developed on Windows; POSIX should work (pure Python + uvicorn), feedback welcome
- A GPU is **not** required — CPU inference is the default and is fast enough

### 1. Clone and install

```bash
git clone https://github.com/aujurd22/flymemory.git
cd flymemory
pip install -r requirements.txt
```

or install as a package (adds a `flymemory-server` command):

```bash
pip install git+https://github.com/aujurd22/flymemory.git
```

`requirements.txt` covers torch, sentence-transformers, mcp (pinned `>=1.30,<2`),
uvicorn, numpy, pytest.

### 2. Run the tests (also downloads the embedding model)

```bash
python -m pytest tests/
```

behavioral rule tests. The first run downloads
`paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) from HuggingFace into the
standard HF cache; after that everything works offline. If HuggingFace is
unreachable from your network, set a mirror endpoint first:

```bash
export HF_ENDPOINT=https://hf-mirror.com    # PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
```

### 3. Start the server

```bash
python flymemory/mcp_v3.py --http          # or: flymemory-server --http
# → serves the MCP endpoint at http://127.0.0.1:8765/mcp
# → logs to flymemory/server.log (or server.<pid>.log if that file is locked)
```

Smoke test (writes one test entry into a fresh library, and reads it back):

```bash
python flymemory/test_http_client.py
```

Without `--http` the server speaks MCP over stdio (for clients that spawn it
per-session).

### 4. Register the MCP server

Any streamable-HTTP MCP client works. ZCode example
(`~/.zcode/cli/config.json`, adjust paths):

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

Restart the client afterwards; you should see the `flymemory_*` tools
(remember / recall / auto / supersede / consolidate / forget / cleanup /
stats / session_pack).

### 5. (Recommended) mechanical capture hooks

Two optional hooks make memory work without the model having to remember
anything. Same config file:

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
      ],
      "SessionStart": [
        { "matcher": "compact",
          "hooks": [ { "type": "process",
          "command": "C:/path/to/python.exe",
          "args": ["D:/path/to/flymemory/flymemory/hook_compact.py"],
          "timeoutMs": 15000 } ] }
      ]
    }
  }
}
```

- `hook_auto.py` (UserPromptSubmit): stores every user message and injects
  recall results into the turn.
- `hook_compact.py` (SessionStart on `compact`): injects a recovery pack
  (recent trail + latest conclusions) right after the client compresses the
  conversation.

Both are silent no-ops when the server is down.

### 6. (Optional) keep-alive

`flymemory/flymemory_supervisor.pyw` restarts the server if it dies and idles
while it is healthy. Run it headless at login — on Windows, a Startup-folder
shortcut to `pythonw.exe flymemory_supervisor.pyw`, or a per-user scheduled
task every few minutes as a second layer of protection.

### 7. Data & configuration

- The library lives next to the code as `flymemory/flymemory_v3.pkl`
  (git-ignored — your conversation content never leaves the machine unless
  you copy it).
- Environment variables: `FLYMEMORY_DEVICE` (default `cpu`, set `cuda` to use
  the GPU), `FLYMEMORY_MODEL` (default
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`),
  `HF_HUB_OFFLINE` (auto-set when the model cache is detected).
- Schema upgrades are automatic: new code loads older library files in place.

### Troubleshooting

- Nothing listens on `127.0.0.1:8765` → wait ~20 s (model warm-up), then check
  the newest `flymemory/server*.log`.
- First tool call right after startup can block for a few seconds while the
  embedder loads — that is the warm-up gate, not a hang.
- Library embedded with a different model name → the server logs a warning;
  re-embed (delete the `.pkl` and re-import, or switch `FLYMEMORY_MODEL` back).

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

## Two-stage retrieval (Hamming prefilter + dense rerank)

For large libraries, recall can run in two stages: a Hamming-distance prefilter
over packed 4096-bit sparse codes (bitwise ops, 512 B/entry), then dense cosine
rerank on the top-C candidates (default C=100).

```python
mem = SmartMemory(two_stage=True, hamming_candidates=100)
# or per-call: mem.recall(q, two_stage=True)
```

### Usage preconditions — read before enabling

1. **Library size N ≥ ~5000.** Below that, dense scoring is already ~1 ms
   (measured 1.2 ms at N≈1.4k via BLAS matmul) and the prefilter + rerank
   overhead makes two-stage a net loss. The pay-off region is 10⁴–10⁵+
   entries, where full-matrix scoring approaches 0.1 s/query.
2. **Code stability must hold on YOUR data.** The quality premise comes from
   the FlyPoet sparse-code retrieval experiment: Hamming lookup hit@1 0.123 vs
   dense 0.128 (Δ ≈ 0.005) on *trained char-level codes* with a measured
   address stability of 1.52×. FlyMemory's codes come from a *random
   projection* — before trusting the prefilter, verify on your own library:
   `recall(q, two_stage=True)` must agree with `recall(q, two_stage=False)`
   on realistic queries (the tests pin this only for the small-N case).
3. **Memory overhead**: +512 B per entry for the packed codes.
4. **Rehearsal side effect is scoped to returned candidates** in prefilter
   mode (non-candidates are never scored, so their access counters do not
   refresh). With `include_superseded=True`, history mode stays decay-immune
   as in the dense path.

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
  mechanical top-1. Measured: 13/14 of the correct current answers are present
  in the model-facing top-3 (with age + provenance stamps); **a superseded
  fact never appears at rank 1 (0/14)**, and full history is recovered on
  demand (`include_superseded=True`, 6/6 — archived, not forgotten, immune to
  decay in history mode). Note: the benchmark measures *presence in top-3*,
  not an actual LLM resolution step.
- This is a **system-level** comparison: the FlyMemory arm runs the whole
  pipeline (supersede + decay + source weighting + lexical + chunking + dedup)
  while baselines run bare similarity. Component ablation is future work.
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
