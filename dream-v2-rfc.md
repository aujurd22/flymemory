# RFC: Dream v2 — from consolidation to prediction (dream → new relations → verifiable improvement)

Status: DRAFT
Date: 2026-09-27
Supersedes: dream.py v1 (consolidation-only, deployed hourly as FlyMemoryDream)

## 1. The gap

Dream v1 proves the pipeline runs (distill → audit → overlay via MCP), but its
output is **summaries of what is already known** — nothing that wasn't in the
window. The next stage must make dreaming produce **new structure**:

```text
dream v1:  window → summary → store        (restating)
dream v2:  window → relations → predictions → verifiable deltas
```

This connects dreaming directly to FlyLoop (the experiment loop that already
consumes pre-registered predictions and produces verdicts), closing the loop:

```text
dream (night)  →  predictions (P-registry)  →  FlyLoop (day)  →  verdicts
                       ↑                                            │
                       └──────────────── feeds next dream ←─────────┘
```

## 2. Three output kinds (beyond summary)

Each dreaming pass emits three kinds of entries, distinguishable by tag:

1. `dream:summary` — what v1 already does (durable facts, audited)
2. `dream:relation` — cross-entry RELATIONS the window supports:
   - co-occurrence of two topics across sessions (candidate links)
   - contradictions resolved/superseded (already partially in insights)
   - temporal sequences ("X happened after Y")
3. `dream:prediction` — forward predictions in the Mushroom-Body Program
   register format, each with a FALSIFIABLE criterion:
   ```text
   {"kind": "prediction", "text": "...", "criterion": "...",
    "expires": "<iso>", "source_ids": [...]}
   ```
   Only emitted when the window supports one; never speculative filler.

## 3. Quality gates (same discipline as v1)

- Every output passes the faithfulness audit against source turns
  (predictions additionally must cite `source_ids` and an `expires` date —
  predictions without a falsifiable criterion and expiry are dropped).
- Predictions are stored with tag `dream:prediction` and surfaced by
  `flymemory_insights` (the proactive trigger) so the next session sees
  "a dream predicted X, criterion Y, expires Z" — and FlyLoop can pick it
  up as a pre-registered experiment.
- Verdict loop: when a prediction's criterion is evaluated (by any later
  session), the prediction entry is superseded by a `dream:verdict` entry
  (CONFIRMED/FALSIFIED), keeping the lineage inspectable.

## 4. What this changes in code

- `dream.py`: add a second distill pass per window (relations) and a third
  (predictions) — both audited like v1; ~100 LOC.
- `bench_dream_eval.py` (new): measures whether v2 dreaming helps —
  precision of relations (manual sample), and prediction hit-rate over
  a follow-up window. Without this measurement, v2 is unfalsifiable and
  therefore not shippable.
- FlyLoop integration: prediction entries use the same registry format the
  MB program already uses (P-ids), so the loop closes without new tooling.

## 5. Non-goals

- No LLM inside the engine (predictions are entries stored/queried like any
  other; the engine stays mechanical).
- No automatic experiment execution — FlyLoop stays the human-gated loop.

## 6. Status

Interface design only; implementation pending a working session with
DEEPSEEK_API_KEY available (the distill/audit calls are the same protocol
as dream v1, already verified).
