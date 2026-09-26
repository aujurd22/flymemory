# The Mushroom-Body Program

**Selection, Compression, and Emergent Structure**

> The big question: **When does compression induce structure?** — when a
> system replaces a keep-everything baseline with lossy compression (memory
> consolidation) or sparse competition (k-WTA), when do downstream
> capabilities rise, when do they collapse, and in what form?

The insect mushroom body is the classic neuroscience structure for "sparse
high-dimensional coding + selective compression -> generalized memory".
This program claims that circuit is an abstract prototype of a class of
engineering problems — **trading simple local selection and compression
mechanisms for global capability** — and that when such mechanisms work or
fail is experimentally measurable.

## Testbeds (two experimental axes of the same question)

| Testbed | Compression form | Local mechanism | Downstream capability |
|---|---|---|---|
| **FlyMemory** | memory compression (dedup / supersede / merge / consolidation) | similarity selection + time decay | QA accuracy, stale rate |
| **FlyPoet** | activation compression (k-WTA sparse competition) | winner-take-all + gradient optimization | math / reasoning performance |

## Laws (established, all one-command reproducible)

**L1 · Form law** — abstraction must OVERLAY, never replace.
Replace-style consolidation -10pp (22% vs 32%); overlay +5.8pp (42.8% vs
37.0%, n=500). Repro: `bench_granularity.py --mode replace|overlay`

**L2 · Second-order law** — the presentation FORM of consolidation (prose
vs structured timeline) is a second-order variable given overlay (41.6% vs
42.8%, n=500, statistically tied). The first-order variable is overlaying
at all. Repro: `bench_timeline.py --sample 500` vs
`bench_granularity.py --mode overlay --sample 500`

**L3 · Phase-transition law** — the emergence window for sparse competition
exists, but is phase-transition-like and non-monotonic in scale (the k-WTA
sweet spot inverts at 216M). Repro: FlyPoet repo sweet-spot grid.

**L4 · Failure law** — selection mechanisms silently swallow "small-edit
state updates" (5 of 20 pairs dropped by dedup when only a digit/date
changes); lineage (tombstone/supersede) must backstop them or the state is
lost unrecoverably. Repro: `bench_state_fidelity.py` (pre-fix git history)

**L5 · Aggregation law** — aggregation-type questions ("how many X", "total
Y") have answers scattered across turns; single-shot top-k retrieval is
structurally insufficient; agentic multi-round retrieval (self-rephrased
queries) lifts strict from 40% to 46%. TOOL3 addendum: a calculator adds
nothing over time-scoped search (54.0% vs 56.0%, noise) — the residual
bottleneck is evidence collection, not arithmetic. Retrieval FORM must
match question TYPE (lookup -> top-k, aggregation -> toolized multi-round).

## Registered Predictions (registered before running; git timestamps are the proof)

**P-2026-09-24-TL2 · structured timeline overlay**
- Registered 2026-09-24 (see git history, before the experiment ran)
- Intervention: consolidation entries switch from prose summaries to a
  structured format (`SUBJECT: attr = value (date)`, verbatim numbers/names)
- Motivation: LME e2e attribution showed temporal-reasoning is 106/265
  (40%) of wrong answers, and 6/8 manually reviewed wrongs needed
  cross-turn time/quantity arithmetic — exactly what prose summaries drop
- P1 (direction): structured timeline > prose timeline (50q strict, seed 7)
- P2 (magnitude): temporal-reasoning error rate drops more than non-temporal
- P3 (safety): overall strict >= turn-only baseline (32%)
- Falsification: structured <= prose => L2 revised to "all consolidation
  forms are second-order; only overlay matters"
- Status: **VERDICT IN (2026-09-24 15:20) — P1 falsified, P3 holds, L2 revised**
  - Result: structured overlay 38.0% strict / 40.0% weighted (n=50, seed 7)
  - Comparison: turn-only 32.0% | prose overlay 38.0% | timeline 44.0%
  - P1 falsified: structured (38.0%) <= prose (38.0%), below timeline (44.0%)
  - P3 holds: 38.0% > 32% (third confirmation of the overlay effect, +6pp)
  - **L2 revised (the registered falsification condition triggered)**: the
    three consolidation forms (prose/timeline/structured) show no stable
    mutual differences — form is second-order, only overlay matters. The
    timeline's 44% @50 was small-sample optimism (41.6% @500, tied with prose).
  - Extra datapoint: structured-JSON generation succeeded on 52% of
    sessions (488/940) vs prose 74.8% — stricter output format, lower
    consolidation coverage: an engineering-cost dimension
  - trace: reports/structured_50_1790233989.json

**P-2026-09-24-AGG1 · aggregate-style entries overlay**
- Registered 2026-09-24 (before the experiment)
- Intervention: add aggregate/list-style entries on top of consolidated
  ("TOPIC: item1; item2 (total: N)") — directly targeting the
  aggregation/statistics class among the 85 full-miss wrongs (best
  evidence rank 1708-11867, structurally beyond single-shot top-k)
- P1: three-layer (turns+cons+agg) overall strict >= two-layer (38.0%, 50q seed 7)
- P2 (targeted): multi-session / temporal-reasoning / knowledge-update wrongs drop
- P3 (safety): detail classes (single-session-*) do not degrade (L1 overlay principle)
- Falsification: three-layer <= two-layer => aggregate entries add nothing
  at the current generation quality (session-level aggregation != the
  cross-session aggregation the 85 misses actually need)
- Status: **VERDICT IN (2026-09-24 16:58) — P1 weakly confirmed, P3 holds**
  - Result: three-layer 40.0% strict (n=50, seed 7)
  - Comparison: two-layer 38.0% | turn-only 32.0% | timeline 44.0%
  - P1 weakly confirmed: +2pp (2 questions, edge of n=50 noise; direction
    as predicted, magnitude small)
  - P3 holds: single-session classes show no degradation (assistant 4/5, user 6/9)
  - Key insight: aggregate entries only cover WITHIN-session aggregation
    (20% of sessions had countable topics), while the 85 misses need
    CROSS-session aggregation (evidence rank 1708-11867, scattered far
    outside top-k) — session-level lists cannot reach them. Confirms the
    README diagnosis: the next lever for residuals is on the ANSWER side
    (query-time aggregation/tools), not more ingest forms
  - trace: reports/aggregate_50_1790240186.json

**P-2026-09-24-TOOL1 · agentic answer (toolized retrieval)**
- Registered 2026-09-24 (before the experiment)
- Intervention: the answer LLM gets a `search_memory(query)` tool
  (function-calling loop), free to run multiple rounds with rephrased
  queries — targeting "aggregation questions need multi-angle multi-round
  retrieval that a single top-5 cannot hold" (AGG1 verdict + 85-miss
  attribution)
- Library: three-layer (turns+cons+agg, same as AGG1)
- P1: agentic strict >= single-round three-layer (40.0%, 50q seed 7)
- P2 (targeted): multi-session / temporal-reasoning wrongs drop
- Falsification: <= 40.0% => the bottleneck is the answer model's
  aggregation capability itself, not retrieval form
- Status: **VERDICT IN — P1 confirmed, P2 confirmed**
  - Result: agentic answer **46.0% strict / 51.0% weighted** (n=50, seed 7,
    avg 3.7 searches/question)
  - vs single-round three-layer 40.0% / 40.0% → **+6pp strict / +11pp weighted**
  - P2 holds: temporal-reasoning 8/14 correct (57% vs structured's 36%);
    multi-session wrongs 9→7
  - **Conclusion: agentic retrieval (multi-round self-directed search)
    breaks the single-shot top-5 structural limit** — as predicted by the
    85-miss attribution. Full ladder (same 50 questions): turn-only 32% ->
    two-layer 38% -> three-layer 40% -> agentic 46%
  - L5 · aggregation law (new): see above; TOOL3 addendum: calculator adds
    nothing over time-scoped search (54.0% vs 56.0%, noise) — residual
    bottleneck is evidence collection, not arithmetic; calculator dropped
  - Note: answer/judge share the same model (DeepSeek); cross-judge
    calibration still pending
  - trace: reports/toolanswer_50_1790263915.json

**P-2026-09-24-TOOL2 · time-scoped search**
- Registered 2026-09-24 (before the experiment)
- Intervention: search_memory gains a time_range parameter
  ("YYYY-MM..YYYY-MM" filter); system prompt instructs scoping for
  time-bounded questions; compared against TOOL1 (46.0% strict, temporal
  slice 8/14 correct)
- P1 (targeted): temporal-reasoning wrongs drop (TOOL1: 6/14)
- P2 (safety): overall strict >= TOOL1 (46.0%)
- Falsification: temporal wrongs do not drop => the "time-structure
  retrieval" hypothesis is cleared; v4 RFC retrieval core shrinks to pure
  agentic (no time_range)
- Status: **VERDICT IN — P1 confirmed (large), P2 confirmed**
  - Result: **56.0% strict / 63.0% weighted** (n=50, seed 7) vs TOOL1
    46.0%/51.0% → +10pp strict / +12pp weighted
  - P1 confirmed with a large margin: temporal wrongs 6→2 (-67%), correct
    8→11 (79%)
  - P2 holds: no other type regressed (multi-session wrongs 7→6)
  - Avg 4.4 searches/question (the model used scoping on its own)
  - **The program's first registered-prediction HIT**: registered
    (e040885 chain) -> run -> confirmed with a large margin. Conclusion:
    time-scoped retrieval is a first-order lever; ships as the default
    agentic tool shape
  - trace: reports/toolanswer_50_1790274503.json

## Reproduction

All benchmark scripts are in the repo root, seeds fixed, datasets
versioned (`data/memory_judgment.json` v1.3). Reproduction index: README
and `OVERNIGHT_20260924.md`.

## T0: Structure-Induction Window (Intuition-Mechanism track, first results)

New track repo: `aujurd22/intuition-mechanism` (plan: docs/RESEARCH_PLAN.md
there). T0 measures, on synthetic families, whether a compressed encoder's
latent makes "same structure, different surface" samples become nearest
neighbours (SD = kNN same-family-different-instance fraction).

First sweep, two domains (graphs n=12 six families; algebraic identities
six templates with name-renaming + term-shuffle + side-swap as surface),
MLP-AE bottleneck b swept so compression c spans 1..1664:

- **Over-compression wall confirmed in BOTH domains**: at extreme
  compression the gap SD - surface_dom collapses (+10.8pp graphs,
  +12.1pp algebra) vs +24..67pp in the working range.
- **Under-compression end does NOT collapse**: algebra SD peaks at c=1
  (+67.4pp). The planned "under-compression dominated by surface" half of
  the window is CONDITIONAL on surface salience -- with 2 name chars as
  surface, structure dominates even with zero compression. Sub-finding for
  L5: the window is one-sided unless surface signal is engineered to be
  strong.
- Local-vs-global: kNN-based SD is high while global k-means ARI stays low
  (0.05-0.23) -- latent geometry is locally structured, globally unseparated.

## Intuition-Mechanism track: P3/P5/P7/P8 verdicts (2026-09-26)

New track repo `aujurd22/intuition-mechanism` (plan + T2 math pipeline).
Registered predictions, judged:

- **P3 PARTIAL** (compression vs text-embedding on real math objects):
  AE wins on q-expansion SHAPE objects (Eisenstein hard: 0.567 vs embed
  0.088), loses on digit-IDENTITY objects (CM j-digits: embed 0.946 vs AE
  0.411 -- same-d instances are digit rolls; MSE is shift-sensitive, the
  embedder is lexical/shift-invariant). Sub-law: the representation must
  match the invariant type (shape vs identity).
- **P5 CONFIRMED** (T2 identity-search A/B, hard space 949 monomials,
  2929 truth pairs, seeded start): first fresh hit 7.75x faster (1.0 vs
  7.75 steps), post-recombination hit rate 1.0 vs baseline 0.128 (7.8x,
  registered >=3x); coverage advantage decays to 1.7x by the 10th relation
  (recombination candidates exhaust). v0 easy-space (baseline 10.2%)
  showed NO first-hit advantage -- space difficulty is a prerequisite for
  the judgment to be decidable at all.
- **P7 REVERSED to CONFIRMED** (notation forging): the original
  falsification was an implementation artifact -- the straight-through
  estimator was inverted (values from z_e, gradients into the buffer
  codes), so the encoder never received quantizer gradients. With a
  correct VQ: forged alignment 0.693 vs inherited raw 0.674 / embed
  0.142, and reconstruction improves 20x. Task loss is unnecessary
  (lambda sweep flat; lambda=10 harms reconstruction). Honest limitation:
  forged codes do not transfer to eta products (0.494 vs inherited 1.000)
  -- self-forged notation is domain-specific. The revival experiment's
  null result (lambda-insensitivity) was the diagnostic clue that exposed
  the bug: a null result should trigger an implementation audit before a
  hypothesis verdict.
- **P8 FALSIFIED** (surface salience): a 40-char per-instance surface tag
  does NOT create an under-compression wall (c=1 keeps SD 0.545);
  instead it creates a MID-compression interference valley (SD dips to
  0.35-0.42 at c=13-52) while structure still beats surface even with
  the tag occupying 31% of the signal.

Also: T2 math pipeline landed -- validator (50-digit gate, both anchor
series PASS), family_gen (Heegner j-values exact; d=163 anchor caught a
nome-squared bug), series_gen (D1 gate PASS: Chudnovsky coefficients
(13591409, 545140134) reproduced from d=163 alone at 96.9 integer digits;
two derivation bugs caught numerically, incl. the inverted identity
T = 640320^(3/2)/(12*pi)).

Blocked items and why: P4/P6 need 1/pi series for general d -- the (A,B)
coefficients for non-class-number-1 d come from modular-form theory beyond
what we can derive honestly without the Borwein reference; fetching the 17
literature series is the alternative. T3 (miniF2F) deferred to phase 3.

## Related

- FlyPoet (k-WTA x Transformer, 216M inversion): architecture-axis experiment log
- FlyMemory README: memory-system engineering doc (a testbed of this program)
- LongMemEval / LongMemEval-V2: external benchmarks (integration pending)
