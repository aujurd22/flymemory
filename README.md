# FlyMemory

*Last updated: 2026-09-23 · v3.4*

**A long-term memory layer for personal AI agents: hybrid retrieval (dense + lexical → RRF, optional cross-encoder) + a memory state machine (supersede lineage, evidence-linked consolidation, power-law decay, rehearsal, directed forgetting) + model-driven judgment — the server maintains state, the calling model decides.**

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
   dense ranking: multilingual embedding cosine, max over query chunks
 + lexical ranking: IDF boost (part numbers, paths, IDs — invisible to embeddings)
        ↓ RRF fusion (k=60, pool 200) → candidate ORDER
 - superseded entries excluded (include_superseded=True for history queries)
 optional: cross-encoder rerank of the fused top-10 (enable_rerank=True)

decay R(t) = (1 + t/τ)^-0.5 is a MAINTENANCE signal (drives cleanup and
rehearsal), not a ranking feature — ordering by it was measured to lose to
plain RRF (see the LongMemEval and state-aware sections below).
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

## LongMemEval-oracle retrieval benchmark

`bench_longmemeval.py` adapts the LongMemEval oracle edition (500 questions,
940 evidence-haystack sessions, 10,866 turns, 2021-2024) to FlyMemory: all
sessions ingested at turn granularity into one shared memory, questions
answered by recall, scored by evidence-session hit@3 (retrieval-level metric;
answer generation + LLM judging not included).

| policy | evidence-hit@3 |
|---|---|
| **RRF fusion — production recall** | **339/500 = 67.8%** |
| **RRF + cross-encoder rerank (optional)** | **360/500 = 72.0%** |
| BM25-only (IDF lexical) | 314/500 = 63% |
| legacy full scoring (sim × decay × source *ordering* — retired, see below) | 303/500 = 61% |
| recency-only (newest turns) | 1/500 = 0% (dates span 3 years — recency is uninformative here) |

The bold rows are the current production path (`recall()` = RRF fusion of the
dense and lexical rankings; rerank is opt-in). The "legacy full" row is the
original eff-ordering pipeline this benchmark first measured, kept for
continuity — ordering by decay × source was later measured to *lose* to plain
RRF (see "Where do decay and source weights act?" below), so it is no longer
the shipped ranking.

Per ability (legacy path, kept as the per-type breakdown): knowledge-update
**83%** (the supersede/lineage strong suit),
single-session-assistant 98%, multi-session 56%, single-session-user 54%,
temporal-reasoning 44%, single-session-preference 40%.

Honest reading, updated: on the *retired* eff-ordering path BM25-only
statistically tied the pipeline (63% vs 61%) — exact-token overlap carried
most retrieval weight on chit-chat style sessions. The shipped RRF fusion
fixes that: 67.8% vs BM25's 63% (+4.8pp), and cross-encoder rerank adds
another +4.2pp on top. The multilingual embedder still earns less on English
casual text than on Chinese technical content, but the hybrid no longer
depends on that gap. n=500, retrieval-only, no LLM layer: indicative, not
comparable to published end-to-end LongMemEval scores (which include an
answering LLM).

## Cross-encoder rerank

`bench_rerank.py` scores the fused RRF top-10 with a cross-encoder
(`ms-marco-MiniLM-L-6-v2`) and returns the top-3 after reranking. On the
oracle edition above (500 questions, 9,729 entries):

| policy | evidence-hit@3 |
|---|---|
| RRF fusion top-3 (no rerank) | 338/500 = 67.6% |
| **cross-encoder rerank of RRF top-10** | **360/500 = 72.0%** |
| oracle: answer anywhere in RRF top-10 | 397/500 = 79.4% |

Rerank converts about a third of the top-10 ceiling into top-3 hits (+4.4pp);
it can also drop an answer that raw RRF had surfaced — the oracle row shows
what a perfect pool-level fix would be worth. Enable per library:

```python
mem = SmartMemory(enable_rerank=True, rerank_pool=10)
```

Off by default: one CE pass costs ~0.3 s CPU per recall at pool 10. The
reranker loads lazily and is never persisted. Superseded entries are excluded
*before* pooling so a dead entry cannot burn a rerank slot.

The gain transfers to production scale. `bench_rerank_full.py` on the
S-edition library (500 questions, **199,509 turn-granularity entries**):

| policy | evidence-hit@3 |
|---|---|
| dense (max-over-chunks cosine) | 137/500 = 27.4% |
| full production scoring (sims × decay × source) | 145/500 = 29.0% |
| BM25-only (IDF lexical) | 168/500 = 33.6% |
| RRF fusion top-3 | 204/500 = 40.8% |
| **cross-encoder rerank of RRF top-10** | **229/500 = 45.8%** |
| oracle: answer anywhere in RRF top-10 | 277/500 = 55.4% |

Same +5.0pp from rerank, same ~1/3 ceiling capture — the effect is stable
across a 20× library-size change. Two forensics notes from this run: an
earlier "production scoring = 0/500" row was a **bench artifact** (the
S-edition date parser silently returned None for every session, NaN-ing all
decay weights; fixed in `parse_lme_date` + `repair_lme_s_timestamps.py`), and
`bench_lex_ab.py` shows the lexical channel's ratio normalization vs
cumulative IDF is a **wash inside RRF** (339 = 339/500) — `_lex_scores` needs
no change. Thread-cap note: on a host running a full-core training job,
torch's default thread count livelocks the CE forward (220s+ vs 0.3s
capped to 1-4 threads); benches and the reranker itself cap threads.

### Where do decay and source weights act? Not after fusion.

`recall()` ranks by RRF over the raw dense and lexical channels; the decay ×
source × eff score is an auxiliary column, not the ordering key. Reranking
the fused pool by eff makes things *worse* — `bench_state_oracle.py`
(oracle edition, 500 questions):

| policy | evidence-hit@3 |
|---|---|
| RRF top-3 (production order) | 339/500 = 67.8% |
| RRF top-10 reranked by eff | 335/500 = 67.0% |
| RRF top-20 reranked by eff | 329/500 = 65.8% |

The eff score is a *maintenance* signal (what survives decay cleanup, what
the rehearsal refresh touches), not a ranking signal: applying it to an
already-fused candidate pool trades retrieval quality for recency bias. The
same eff ordering run over the WHOLE library is the "full" arm above — and it
loses to plain RRF by ~12pp at S scale. Production recall therefore keeps
RRF order, and `bench_rerank_full.py` carries the state10/state20 arms as a
standing regression guard for this decision. S-edition spot check (first
150 questions, run under a saturated host): rrf 21.3%, state10 22.0%,
state20 20.7% — within noise of plain RRF, same ordering by pool size.

## Memory Judgment Benchmark (Phase 1)

Retrieval metrics answer "can the evidence be found"; they say nothing about
whether an LLM placed in front of the state machine will *operate* it
correctly. `bench_memory_judgment.py` measures exactly that, with an offline
JSON protocol (the actor sees the memory snapshot + the new user message and
emits strict JSON; the harness executes and validates). The engine is frozen
while this benchmark exists. Mechanical validity (id existence, active
supersede target) is checked in code; unsupported inference in consolidation
conclusions goes to an LLM judge.

Dataset v1.4 (144 cases, four batches): 62 supersede (14 from the
contradiction scenarios + 20 from the state-fidelity pairs + 28 fresh
themes), 30 adversarial no-ops ("I fixed something on my old Windows VM"
must NOT supersede the Fedora entry), 14 consolidation (summary requests
over topic fragments; 2 with an outdated-distractor to test stale leakage),
8 forget (6 wrong-fact deletions + 2 traps where the right action is
supersede, never forget), and batch 4 adding four NEW capability dimensions
(5 each): multi-update (two prior states both need superseding), reversal
(value reverts -- must create a new entry, never resurrect the superseded
one), temporary state (a time-boxed fact is new info, not an update), and
partial correction (one field of a compound entry changes). Generators:
gen_dataset_batch2.py / batch3.py / batch4.py.

First results, 2026-09-24 (oracle = gold replay, harness sanity check, all
1.0 with zero mechanical errors; actor = `deepseek-chat`, temperature 0;
supersede/forget scores are execution-aware — an operation that was decided
but failed to execute leaves the state unchanged and cannot count as a hit):

| metric | value |
|---|---|
| supersede precision / recall | **1.00 / 1.00** (n=77, incl. multi-update/reversal/partial-correction dimensions) |
| forget precision / recall | **1.00 / 1.00** (n=8) |
| unnecessary mutation rate | **0/36** (all adversarial no-ops held) |
| consolidation evidence exact-match | 8/14 |
| unsupported inference 0/14; stale leakage 1/14 (LLM judge, the known
  cons_08 case -- judge calibration: 2 distilled sessions manually verified
  faithful) |

Failure modes worth keeping: (1) one supersede was issued without the
required `remember` entry holding the new state — the state transition was
understood but the protocol contract was missed; (2) one consolidation
included an already-superseded detail (the old printer cartridge) — faithful
to the evidence, but stale leakage into a "current state" summary; the
unsupported-inference judge does not catch this, it is a separate axis.
Oracle vs autonomous shows **zero judgment gap on supersede and forget** at
this scale; the gap concentrates in consolidation timing and evidence
selection — the target for the next iteration.

**Phase 1.5 (real tool-calling, `bench_memory_judgment_tools.py`)** repeats
the same 38 cases through the actual tool surface — the model must handle the
id flow itself (remember first, take the returned id, then supersede). Same
model, same data, execution-aware scoring: supersede **1.00 / 0.81**,
forget 1.00 / 1.00, unnecessary mutation 0/10. The recall gap is 3 cases
where the model's `remember` of the new state was **merged into the old
entry** (sim > 0.75, longer text wins: the entry text is rewritten to the
new state in place) — the model then issued a redundant supersede
(old == new) that was correctly rejected. Replaying all three shows the
final store state is CORRECT in every case: the merge *was* the state
transition. Strict P/R therefore undercounts; the honest statement is that
2/16 supersedes needed the explicit supersede tool and 3/16 were absorbed by
merge semantics, and the model cannot distinguish "stored as new #N" from
"strengthened/merged into existing #N" from the tool's return text — the one
genuine protocol gap found at this layer.

**Phase 2 (end-to-end, five arms, `bench_e2e_answer.py`)**: the same 30 state
cases, scored at the ANSWER level — apply a maintenance policy, retrieve,
`deepseek-chat` answers the question from the recalled entries, LLM judge
classifies the answer. The arms separate retrieval quality from state
maintenance:

| arm | retrieval | maintenance | current | stale | unknown (correct) |
|---|---|---|---|---|---|
| no memory | – | – | 0% | 0% | 28/30 (93%) |
| dense naive RAG | cosine top-3 | store-only | 26/30 = 87% | **4/30 = 13%** | 0 |
| RRF naive (store-only) | production RRF | store-only | 25/30 = 83% | **5/30 = 17%** | 0 |
| FlyMemory + oracle state | production RRF | gold ops | 26/30 = 87% | **0%** | 4/30 = 13% |
| FlyMemory + autonomous state | production RRF | DeepSeek ops | 26/30 = 87% | **0%** | 4/30 = 13% |

The 5 naive-RAG stales are exactly the deleted-wrong-fact cases (frt_01–04):
without `forget`, the assistant keeps confidently answering with facts the
user explicitly retracted. Autonomous matches the oracle ceiling exactly —
zero judgment gap end-to-end — and the stale contamination that motivates
the state machine disappears entirely under it. Judge calibration: the LLM
verdicts agree with a mechanical keyword check on all 150 arm-answer pairs
(150/150) -- the judge is neither stricter nor looser than the observable
evidence.

Note on the no-memory arm: the blind model is 3% current with a 10% stale
answer rate and 20% outright wrong -- it confabulates where it should
abstain. With a naive store it reaches 83-87% current but still leaks
17% stale; only the state-maintenance arms reach 0% stale while staying
at the 87% ceiling.

### TOOL2: time-scoped agentic search (56% strict)

`bench_toolanswer.py --tool2` (TOOL2): the answer model's search_memory
gains a `time_range` parameter ("YYYY-MM..YYYY-MM") and the system prompt
instructs scoping for time-bounded questions. Same 50 questions:

| answer mode | strict (n=50 / n=500) | weighted |
|---|---|---|
| single-round top-5 (three-layer) | 40.0% / 40.0% | 40.0% |
| TOOL1 agentic (free-form search) | 46.0% / 43.2%* | 48.6%* |
| **TOOL2 agentic + time_range** | **56.0% / 44.6%** | **63.0% / 50.0%** |
| TOOL3 + calculator (n=50) | 54.0% | 58.0% -- no gain, dropped |

(*TOOL1 was only run at n=50; at n=500 TOOL2 scores 44.6% strict / 50.0%
weighted vs turn-only 37.0% / 39.6% and prose overlay 42.8% / 44.9% --
agentic time-scoped retrieval is the best configuration at both scales.
By type at n=500: temporal-reasoning remains the weakest slice at 50/133
correct; multi-session 56/133 correct.)

The registered prediction (research/RESEARCH.md, TOOL2) confirmed with a
large margin: temporal-reasoning wrongs dropped 67% and no other type
regressed (+10pp strict overall). Time-scoped retrieval is a first-order
lever and ships as the default tool shape. TOOL3 probe: adding a
calculator tool on top gave no further gain (54.0%, within noise) --
with time-scoped search the residual bottleneck is evidence collection
(reading across many turns), not arithmetic.

Cross-model check (external-review suggestion): the same 50 questions
answered by a second model (GLM, via interactive session) score 30% strict
/ 31% weighted vs DeepSeek's 32% / 35% -- the ~50% answer-conversion
ceiling over retrieved evidence is model-agnostic, confirming it is a
task-structure limit (multi-turn aggregation), not a DeepSeek quirk.
Grading script: bench_glm_answers.py.

`bench_lme_e2e.py` extends the protocol to the public LongMemEval-oracle
questions (all 500): recall top-5 → deepseek answers → judge vs gold.
**Strict correct 37.0% (185/500), weighted with partials 39.6%.** Full-scale
attribution: retrieval hit@5 = 73.2%; answers are 43% correct when evidence
was retrieved vs 22% when not (the 22% are questions answerable without the
specific evidence -- generic or inferable). Raising top-k to 10 changed
nothing on a 50-question sample. The conversion losses (43% not 100% after a
hit) concentrate on multi-turn aggregation questions (arithmetic over two
turns, counting across sessions) — turn-granularity retrieval cannot answer
them, which is the concrete argument for session-level consolidation
(below). Not comparable to official LongMemEval end-to-end scores: they feed
the full haystack (long-context setting), this is a memory-augmented top-k
setting.

### Granularity A/B: consolidation must OVERLAY, never replace

`bench_granularity.py` distills each of the 940 evidence sessions into 1-3
DeepSeek-consolidated durable-fact entries and reruns the same 50 questions
under two ingestion policies:

| store | strict correct | weighted score |
|---|---|---|
| turn entries only (baseline, n=500) | 185/500 = 37.0% | 0.396 |
| consolidated entries only (replace, n=50) | 11/50 = 22% | 0.270 |
| **turn + consolidated (overlay, n=500)** | **214/500 = 42.8%** | **0.449** |

Replacing raw turns with summaries loses the concrete details most questions
ask about (22% -- worse than baseline). Overlaying the consolidated entries
on the untouched turns gains **+5.8pp strict at full scale** (verified on all
500 questions, not just the 50-question sample; per-question verdict flips:
40 improved vs 11 regressed, net +29). The summaries act as
retrieval entry points while the turns keep the details. This is the
measurement behind the design rule "raw entries are never deleted --
consolidation adds abstraction without loss".

Full hallucination audit (`bench_consolidation_audit.py`): all 703
successfully consolidated sessions judged against their source conversations
— 5 flagged on first pass, 4 still flagged after a head+tail re-check
(0.57--0.71% session-level). MANUAL REVIEW of all four: 2 are judge
artifacts (long conversations truncated at 6000 chars; the D&D stat-block
details and the 400k-dollar mortgage are verifiably in the source), 1 is a
mild over-inference ("user prefers romantic lyrics" inferred from one
revision request -- judge correctly flagged), 1 remains ambiguous without
the full conversation. Adjusted true hallucination rate: ~0--0.14%
(0--1 session out of 703). LLM-consolidated entries are SAFE to ingest at
this reliability level; the audit judge needs a sliding-window protocol
for long conversations.

### Engine state-fidelity audit (found and fixed a real dedup bug)

`bench_state_fidelity.py` pushes 20 realistic single-edit state updates
("server is 192.168.1.50" → "…1.99", "meeting at 15:00" → "…16:00") through
the dedup ladder and checks whether the store ends up holding the NEW state.
Pre-fix: **5/20 updates were silently dropped** — two strengthened at
sim > 0.92 (text untouched, only the access time refreshed) and three merged
without rewrite because the new text was not longer than the stored one. The
affected updates are exactly the most common kind: changed numbers, times,
names. Fix (in `remember()`): a restatement carrying tokens the stored entry
lacks, and any differing text in the merge zone, now rewrites the entry in
place — pure restatements keep the old behavior. Post-fix: 20/20 updates
end with the correct state; all 51 behavior tests, the contradiction
benchmark (stale top-1 still 0/14) and the QA benchmark (17/20 before AND
after — the three misses are its pre-existing baseline) confirm no
regression.

Known trade-off: 6/20 of those updates rewrite the entry in place (merge),
which fixes the state but loses the old text — `include_superseded=True`
cannot recover it because no lineage entry was created. Candidate v4 change
(DESIGNED, NOT IMPLEMENTED): on a rewriting merge, park the old text as a
superseded tombstone pointing at the updated entry, so history recovery
works uniformly. Cost: +1 entry per rewriting merge. Also open from the
end-to-end run: a full attribution of the 265 overlay-wrong answers —
114 had the evidence turn in top-5 (no consolidated entry hit), 53 had
BOTH the evidence turn and the consolidated entry and still failed
(answer-side multi-turn aggregation), 13 hit only the consolidated entry
(summary lacked the detail), 85 were full retrieval misses. By question
type, temporal-reasoning dominates (106/265 wrongs; 80% of that type),
followed by multi-session (92). The next lever is answer-side multi-turn
aggregation for temporal/multi-session questions, and preference/user
retrieval quality for the 32 missing ones. Manual review of 8 sampled
wrongs confirms the mechanism: 6 of 8 require cross-turn arithmetic
(subtracting membership durations, summing trip mileage, counting flights)
with the evidence turns ALL present in top-5; the model answers "I don't
know" because no single entry states the derived number. The concrete v4
lever is structured timelines (entity+date tables) produced at consolidation
time, or a calculation tool at answer time. A 6-sample review of the 85
full-miss wrongs shows their best evidence sits at rank 1708--11867 with
sim 0.18--0.64: aggregation/statistics questions ("how many doctors",
"total gift spend", "what did I do with Rachel two months ago") whose
answers span many turns -- structurally out of reach for top-k semantic
retrieval, confirming they need aggregated entities or answer-side tools.

### Timeline overlay: the structured form wins (v4 lever validated)

`bench_timeline.py` re-runs the granularity A/B with TIMELINE-form
consolidated entries ("YYYY-MM: plain fact with numbers/names",
chronological) instead of prose summaries -- same 50 questions (seed 7):

| store (n=50, seed 7) | strict correct | weighted score |
|---|---|---|
| turn entries only | 16/50 = 32% | 0.350 |
| prose-summary overlay | 19/50 = 38% | 0.400 |
| timeline overlay (50-question sample) | 22/50 = 44% | 0.440 |

At FULL SCALE (n=500) the timeline overlay scores **41.6% strict / 43.9%
weighted** vs the prose overlay's 42.8% / 44.9% -- statistically a wash.
Honest reading: the 50-question sample overestimated the timeline form by
~2pp; what survives at full scale is the overlay principle itself (either
consolidated form beats turn-only by ~+4.6pp). Timelines for all 940
sessions are cached in reports/timeline_entries.json. The memory format
shapes answer quality, but format choice between prose and timeline is
second-order; the first-order lever was overlaying consolidation on top of
turns at all.

Format matrix completion (same 50 questions, seed 7): a fourth form --
ENTITY-STATE records ("entity.attribute = current value (changed from ...)")
generated from the same 940 sessions -- scores **36% strict / 39% weighted**,
statistically indistinguishable from turn-only. Reading: structured timeline
lines keep a date plus a narrative fact (retrievable), while heavily
compressed key-value records lose the contextual surface the embedder needs,
so they neither retrieve nor answer better. Consolidation formats are now a
measured spectrum: narrative (turns) -> prose summary -> timeline -> key-value
(entity-state), with the optimum at timeline.

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
| Forget slowly, never abruptly | power-law decay (tau doubled for model-stored entries -- dopamine-gated) + rehearsal; `flymemory_cleanup` prunes below threshold |
| Rehearsal stays scarce | only entries injected into context refresh; a wide sim>0.5 rule measured 100% of entries pinned at retention 1.0 (immortal chatter, decay inert) |
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

51 behavioral rule tests. The first run downloads
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
   Measured fidelity also degrades with N (0.73 @1k → 0.55 @20k in
   `bench_two_stage_scale.py`) — at 20k entries, 45% of dense answers leak
   past the prefilter.
3. **Keep fraction matters — a lot.** The prefilter code sparsity follows a
   U-curve like FlyPoet's channel sparsity: measured fidelity@100 was 0.691
   at the biology-derived 5%, 0.849 at 25%, 0.861 at 50% (adopted default,
   `code_keep=0.5`). The biology-derived 5% was too sparse for retrieval.
4. **Memory overhead**: +512 B per entry for the packed codes.
5. **Rehearsal side effect is scoped to returned candidates** in prefilter
   mode (non-candidates are never scored, so their access counters do not
   refresh). With `include_superseded=True`, history mode stays decay-immune
   as in the dense path.
6. A C-extension popcount or a real binary ANN (faiss/hnswlib) would change
   the latency picture, but fidelity remains the binding constraint.

### Measured verdict (2026-09-22, diverse 10k-entry library)

| N | prefilter fidelity@100 | Hamming scan | dense scan |
|---|---|---|---|
| 1 000 | 0.922 | 6.35 ms | 0.21 ms |
| 2 000 | 0.965 | 11.90 ms | 0.34 ms |
| 5 000 | 0.903 | 31.94 ms | 0.56 ms |
| 10 000 | 0.878 | 55.93 ms | 0.94 ms |

**The numpy Hamming scan is ~60x slower than the BLAS dense matmul at every
scale** (no SIMD popcount in numpy; BLAS is heavily optimized), and fidelity
decays as N grows. Two-stage retrieval is therefore **rejected for the numpy
implementation at every scale** -- the dense path is both faster and more
faithful. The code stays behind the flag as a documented negative result; the
sensible scale path is a real ANN index (faiss/hnswlib) at N > 1e5.

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
