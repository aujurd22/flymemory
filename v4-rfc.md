# RFC: FlyMemory v4 — Temporal-Evidence Memory

Status: DRAFT
Date: 2026-09-25
Supersedes: nothing (extends v3.4)
Motivation anchor: this RFC was triggered by five external review rounds and
one overnight experiment series (2026-09-23/25) whose evidence is cited
inline as [E#].

## 1. Problem statement

v3 proved that retrieval works (RRF+CE 45.8% evidence-hit@3 on 200k entries)
and that state mutations are judged correctly (supersede/forget P/R 1.00 on
124 judgment cases). The residual failures are structural, not tuning
problems:

- [E1] Answer conversion is ~43-50% over retrieved evidence; the losses
  concentrate on multi-turn aggregation questions (arithmetic over turns,
  counting) that NO single stored entry can answer
  (bench_lme_e2e attribution; 8-wrong manual review: 6/8 cross-turn arithmetic).
- [E2] A rewriting merge fixes the state but destroys history — the old text
  is gone and `include_superseded=True` cannot recover it (6/20 state updates
  in bench_state_fidelity.py).
- [E3] v3's single `timestamp` conflates creation, last-update, and temporal
  validity: after a merge, an entry reads "user lives in Hangzhou" with a
  2024-01 creation stamp, so displayed age and temporal reasoning are wrong.
- [E4] Judgment coverage is single-update only; multi-update, reversal,
  temporary state, and partial correction are untested dimensions (v1.4 adds
  the first 20 cases — all passing, but they probe the model, not the schema:
  v3 has no first-class notion of an entity state to test).

## 2. Design principles (unchanged, now formalized)

1. Raw entries are never deleted — abstraction overlays ([E: granularity A/B,
   replace 22% vs overlay 42.8%]).
2. Decay/rehearsal are maintenance signals, not ranking signals
   ([E: post-fusion eff rerank hurts]).
3. Mechanical invariants are enforced in code; the model only judges semantics.
4. Everything is inspectable: lineage, provenance, evidence links.

## 3. Data model

### 3.1 MemoryEntry v2 (additive — v3 fields are kept)

```text
MemoryEntry:
  # --- v3 fields (unchanged) ---
  memory_id, text, response, embedding, timestamp, last_accessed,
  access_count, tags, source, superseded_by, evidence_ids

  # --- v4 additions ---
  created_at: float          # == v3 timestamp (renamed conceptually;
                             #  timestamp stays as an alias for compat)
  updated_at: float | None   # set when a rewriting merge changes text
  valid_from: float | None   # state starts being true at this time
                             # (defaults to created_at)
  valid_to:   float | None   # state stops being true at this time
                             # (set on supersede / rewriting merge;
                             #  None = currently valid)
  entity:    str | None      # optional entity key, e.g. "user.phone"
  attribute: str | None      # optional attribute key, e.g. "number"
```

Rules:
- `timestamp` keeps its v3 meaning (creation). New code reads `created_at`.
- `updated_at` is the display/recency key when set; answer-side age stamps
  should prefer it. This fixes [E3] age-display without breaking v3 readers.
- `valid_from/valid_to` are set by supersede and rewriting merge. They make
  temporal questions answerable from stored structure instead of inference.

### 3.2 Entity-state entries (new entry kind via tags)

```text
tags include "entity_state"; text format:
"ENTITY <entity>.<attribute>: <current value>
 valid_from=<iso> evidence=[#id, #id]
 superseded: <old value> (valid <from>..<to>)"
```

The text is the storage/embedding surface; the machine-readable form lives in
two new optional fields:

```text
  state_key:   str | None   # "user.phone"
  state_value: str | None   # current value
```

### 3.3 What does NOT change

- Storage stays a single-file pickle + numpy matrix. No database, no graph
  store ([E: the plan is CPU/API-type work; GPU and infra stay free]).
- Consolidation overlays; it never replaces raw turns ([E: replace 22% vs
  overlay 42.8%]).
- The judge/actor split in benchmarks stays (offline JSON protocol first).

## 4. Mechanical invariants (enforced in code, verified by tests)

- **I1 active-unique**: for each `state_key`, at most one entry has
  `valid_to is None` and `superseded_by is None`.
- **I2 no-dangling-evidence**: every id in `evidence_ids` exists in the store.
  Enforced by `forget()` (shipped) and `decay_cleanup()` (shipped 2026-09-25).
- **I3 lineage-acyclicity**: `superseded_by` chains never cycle; edges only
  point at active entries (shipped).
- **I4 merge-tombstone**: every rewriting merge creates a superseded tombstone
  of the old text (shipped, 2026-09-24).
- **I5 supersede-contract**: a supersede without a matching remember of the
  new state is rejected at the protocol layer (shipped in the benchmark
  harness; to be enforced in `mcp_v3.flymemory_supersede` too).
- **I6 zero-score lex filter**: lexical RRF candidates must have lex_vec > 0
  (shipped).
- **I7 candidate eligibility**: superseded entries are excluded before
  candidate scoring in BOTH the main path and two-stage (main path: pending
  — see §6; two-stage: shipped).

## 5. Migration

- v3 pickles load unchanged (new fields default to None).
- Backfill `created_at = timestamp` on load; `valid_from = timestamp`,
  `valid_to = None` for entries with `superseded_by is None`; for tombstones
  `valid_to = the successor's created_at` (approximation, documented).
- Entity-state entries are generated going forward (from consolidation and
  from state-update judgment cases); no backfill of unstructured old entries.

## 6. Retrieval changes

- Main path: superseded entries are excluded BEFORE candidate scoring (fixes
  the "dead nodes still compete then get disqualified" pattern; pending — the
  shipped fix only covers two-stage).
- Entity-state entries are indexed by `state_key` for direct lookup queries
  ("what is the user's phone number") without RRF.
- Answer layer (out of engine scope, benchmark lever): calculator tool
  ([E: A1 probe — DeepSeek barely used it; GLM interactive arm pending]) and
  structured-timeline presentation.

## 7. Benchmark plan (per external review rounds 4-5)

- Judgment v1.4 (144 cases) is the floor; batch 5 adds ambiguous-reference
  and indirect-update cases.
- Four-layer e2e: Retrieval / State / Derivation (consolidation faithfulness)
  / Answer — each reported separately.
- Judge: sliding-window protocol (shipped in the audit); cross-model judge
  agreement to be measured once a second API key is available.
- LongMemEval-V2: cited by the review (arXiv 2605.12493) — VERIFY EXISTENCE
  before planning; earlier reviews fabricated citations.

## 8. Non-goals (unchanged)

No knowledge graph, no graph database, no server-side LLM, no distributed
storage, no FAISS/HNSW until N > 1e5 in production.
