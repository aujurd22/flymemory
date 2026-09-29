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

**Role of this repo (fixed 2026-09-28, round-12 review):** FlyMemory is
the program's **memory substrate and experimental infrastructure**, not a
third parallel cognitive-theory line. New benchmarks are built only as
instruments for cross-testbed law validation (the P32-i transplant,
P-ANCH-1, P-PHASE-1 pattern), not to accrete a self-standing benchmark
zoo. Theory flows intuition-mechanism -> law ledger (here) -> testbeds;
flyloop consumes this substrate.

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

**L6 · Anchoring law (established single-model, P-ANCH-1 paired
replication)** — the solvability of a semantic discretion task is governed
by ANCHORING, not by task surface: with a concrete reference value to
compare against, discretion becomes near-mechanical (47/48 = 97.9%, the
one miss the predicted declarative-intent residual); unanchored
linguistic-behavior classification is where discretion truly lives (same
48 turns through kind-classification + naive rule: 70.8%, hitting the 75%
structural ceiling; discordant pairs 13:0, exact McNemar p ~ 2e-4). P-PHASE-1 adds the first quantitative
edge: on a textual-delta ladder the anchored verdict stays perfect down
to digit-swap updates (delta_emb ~ 0.2), so the L4 small-edit fragility
is engine-layer, not judgment-layer. Evidence: atomic decomposition -- state-change
detection WITH an anchor (stored value vs assertion) is 30/30, including
exactly the three "considering switching back" assertions that fail
end-to-end; unanchored assertion-kind classification is 26/32 with ALL
confusions biased toward fact (1/5 intent, 5/7 question). Cross-testbed
support: P32-i scaffold with an OBJECTIVE sufficient statistic restores
novelty 20/40 -> 39/40, while the same scaffold shape transplanted onto
unanchored flymemory judgment is a registered negative (3f87703).
Worded as candidate: two arms, n=92, single model (deepseek-chat).
Repro: `bench_atomic_judgment.py`

## Sister ledger: Memory Geometry laws (2026-09-29, P40b-P72 synthesis)

intuition-mechanism's `docs/MEMORY_GEOMETRY.md` distills six laws of
recognition/novelty geometry with production anchors measured on THIS
repo's live store (7,325 active memories: NN-cosine band [0.44, 0.93],
zero isolates; self-retrieval 100% hit@1 near-verbatim, 91.7% at
half-text cues). Compressed ledger (full versions with artifacts in that
doc):

- **G1 · coverage** — recognition = coverage within THRESH = mind/2;
  prototype viable iff R/mind <= 0.5, exemplar iff NN/mind <= 0.5
  (interventional: 50% crossing at exactly 0.500).
- **G2 · two scales** — prototype/exemplar divergence needs dense-core +
  sparse-fringe (two-scale) classes; single-scale families lock the two
  ratios together.
- **G3 · plateau** — capacity saturates at the THRESH-reachable fraction;
  coverage is a mixture, not an exponential; cold start is first-class.
- **G4 · novelty** — needs coverage AND clearance; the lever is CONTRAST
  features, not resolution (refining the same family closes the window).
- **G5 · query floor** — below l_min ~ #free-statistic-entries no
  detector works at any training size; production knee 10-25% of text
  (~6-12 tokens). **Adopted hook rule: user messages under ~10 tokens
  cannot reliably retrieve their target; above ~25 they can.**
- **G6 · storage policy** — evict redundant CORES, keep the fringe
  (streaming Hart): core-first 100% vs LRU 92.4% vs fringe-first 53.6%
  on two-scale corpora; replicated on SQuAD (P90: core-first 42.5% >
  random 38.2% > LRU ~ fringe 30% -- note LRU ties fringe there, and the
  probe-coverage metric differs from ours); our production pre-study is
  the boundary cell: heavy-tail redundancy, benefit-cost mirror
  (+20.8pp recoverability / -21.0pp top-5 keep). CROSS-SUBSTRATE CAUTION:
  the three measurements use different metrics (recognition-rate /
  probe-coverage / recoverability+top-5-keep) -- ordering claims are
  within-substrate; the metric口径 has not been unified. NOTE: FlyMemory's decay_cleanup is currently
  LRU-family (last_accessed); **candidate experiment P-CLEANUP (not yet
  registered)** would test core-first vs LRU on a production-geometry
  library -- contingent on first measuring whether the production store
  is two-scale (the anchor band suggests spread, but the core/fringe
  structure is unmeasured; **pre-study DONE 2026-09-29,
reports/cleanup_geometry_1790659478.json: the production store is NOT
two-scale -- it has a HEAVY-TAIL REDUNDANCY structure** (1-NN unimodal,
median 0.691; 55% zero-coverage @cos>=0.70 with a small high-redundancy
head, max 35; one 1,906-entry component = 25% of the store). Eviction
simulation (budget 15%, 500 holdout anchors) returned a clean BENEFIT-COST
MIRROR: core-first recoverability 27.1% vs LRU 6.3% (+20.8pp, the
pre-registered REGISTER band) BUT holdout top-5 keep 70.0% vs 91.0%
(-21.0pp) -- high-coverage entries are BOTH the redundant region AND the
retrieval hubs; the same structure shows its two faces in the two
metrics. fringe-first 0.0% validates the metric; LRU's old entries are
nearly unrecoverable (6.3%) but retrieval-transparent. DECISION:
registration POSTPONED and reshaped -- a registered P-CLEANUP must carry
BOTH metrics in its criteria and add a mixed arm (evict high-coverage
NON-hub entries); the free-improvement narrative is rejected for this
geometry).

Design rules 1-6 of that doc apply here verbatim (memory type by
geometry; capacity never substituted by structure; store until
fringe-reachability; budget novelty screening by l_min and open the
window with contrast; evict by redundancy; report arrival profile).

## Cross-testbed evidence chain (2026-09-28, round-9/10)

The program-level question "which cognitive tasks compress into an
objective sufficient statistic, and which must retain semantic
discretion" now has one measured cell per testbed:

| testbed | task | sufficient statistic? | result |
|---|---|---|---|
| intuition-mechanism P32-i | novelty vs family | objective (numeric signature) | scaffold restores 39/40 |
| intuition-mechanism P35-a | structure discovery | found BY search (MDL, unlabeled) | rediscovers the 4-feature tree 4800/4800 |
| FlyMemory (Arm-S transplant) | assertion kind | none found in fact/intent/question schema | scaffold NEGATIVE (3f87703) |
| FlyMemory (atomic) | state change WITH anchor | anchored (stored value) | 30/30 |
| FlyMemory (atomic) | assertion kind WITHOUT anchor | unanchored | 26/32, bias-to-fact |
| FlyPoet | which ingredient works | dissection: stable subset, update throttling (not competition, not surprise selectivity) | mechanism ≠ biology's mechanism |

**Restraint policy (round-10 review, adopted)**: candidate conclusions are
written scoped -- "in this family / under this schema / at this scale" --
never as existence claims. "No mechanical sufficient statistic" means
"none found in this schema", not "cannot exist"; P36's "interestingness is
mechanical" is written "algebraic depth is a strong candidate mechanical
component of observed interestingness in this family". The program-level
jump condition: replicate the sufficient-statistic -> capability chain on
a second, structurally different math family, or land a mechanical
sufficient statistic for memory judgment (the anchoring result is the
first candidate).

## Registered Predictions (registered before running; git timestamps are the proof)

**P-2026-09-29-TREE · topic-tree navigation for multi-session evidence
collection** (registered BEFORE the run; verdict PENDING). From the
PageIndex analysis: its "vectorless" navigation works, by our reading,
because it rewrites retrieval into a chain of ANCHORED comparisons
(C1-suppliable, C2-free). Question: does the same retrieval FORM lift
evidence collection on our shared turn store, where single-shot RRF
top-5 structurally under-collects multi-session evidence (the L5
aggregation-miss family)?

- Zero-LLM first version (tests the retrieval FORM, not LLM summary
  quality): two-level spherical k-means tree (32 x <=16, seed 42) over
  the 9,729-entry production turn store; navigation = query -> top-2 L1
  clusters -> top-2 L2 children each -> collect all -> exact rerank
  top-30. Arm RRF = exact cosine top-5 on the same store (collection
  shape is the comparison; reranker quality is not).
- Metrics: session-hit (collected set contains >=1 entry from any
  answer session) and session-coverage (fraction of the answer sessions
  covered), both from the dataset's answer_session_ids annotation.
- Sets: PRIMARY multi-session (133); GUARD single-session (156).
- SUPPORTED : multi-session session-hit(TREE) >= RRF + 15pp AND guard
  pass (TREE >= RRF - 3pp) -> proceed to end-to-end answer verification.
  PARTIAL : +5..15pp. NOT SUPPORTED : < +5pp (L5 remedy stays the TOOL3
  multi-round route). Guard regression caps at PARTIAL.
- Known scope: LME-S has no `aggregation` question_type (the 85-miss
  family was human-attributed by question wording) -- multi-session is
  the closest annotated proxy; a positive result proceeds to answer-level
  verification before any production adoption. VERDICT: PENDING.

**VERDICT (2026-09-29, reports/tree_navigation_1790690064.json +
tree_e2e_1790690354.json): stage-1 PARTIAL / stage-2 TIE -- NOT ADOPTED.**
Stage 1 (evidence collection): multi-session session-hit RRF 63.9% ->
TREE 74.4% (+10.5pp, inside the PARTIAL band); guard single-session
+4.5pp; all-500 +4.8pp -- topic guidance finds the right REGION, nowhere
hurt. Stage 2 (end-to-end, 50 multi-session questions, strict): RRF 6.0%
vs TREE 6.0% -- TIE per the addition-lock. Wrong-answer autopsy: TREE's
errors are mostly "don't know" -- the 30-entry pool reaches the right
topic region but misses the few KEY turns inside it, and the 6x context
dilutes the answerer. **Session-level hit does not compose into turn-level
evidence: the missing turns are sparse needles inside a found haystack.**
This also re-frames the 85-miss attribution: the difficulty is not topic
targeting but needle concentration -- consistent with the L5 verdict that
the remedy is multi-round/toolized collection, not a better index shape.
Line closed under the addition-lock; tree collection stays behind a flag
as an optional collector; revisit condition: an LLM-refined tree (real
node summaries) or needle-targeted reranking inside the collected pool.



**P-2026-09-29-CLEANUP · eviction policy experiment** (registered BEFORE
the run; verdict PENDING). The reshaped P-CLEANUP, admitted now that G6
has three-substrate support (P90) and our production pre-study supplies
the boundary cell (benefit-cost mirror). Question: is there an eviction
policy that beats LRU on BOTH axes the pre-study exposed?

- Protocol: identical to the pre-study (production snapshot, budget 15%,
  500 holdout anchors, seed 7), so pre-study numbers serve as the
  baseline reference.
- Arms: LRU (current rule), core-first (cov70 desc), random,
  fringe-first (controls, replicate the pre-study), and the NEW
  **backup-first**: eviction candidates are entries having a twin at
  cos >= 0.85; within candidates, evict the older last_accessed first
  (keep the newer of each pair); if the budget is unfilled, top up by
  cov70 desc and record the top-up count. Rationale: the pre-study's
  mirror came from evicting hub entries; a twin-backed entry is by
  construction recoverable, and evicting PAIRS (not hubs) should spare
  the retrieval order.
- Metrics (BOTH in the criteria, per the pre-study lesson):
  evicted-recoverability (top-1 surviving cos >= 0.75) AND holdout
  top-5 keep.

PRE-REGISTERED CRITERIA (all deltas vs LRU, same run):
- SUPPORTED : backup-first recoverability >= LRU + 10pp AND
              backup-first top-5 keep >= LRU - 2pp
              (hub-sparing twin eviction dominates LRU on both axes);
- PARTIAL   : recoverability band met but top-5 keep drops > 2pp
              (mirror persists -- the twin definition needs revising);
- NOT SUPPORTED : recoverability < LRU + 5pp OR top-5 keep drops > 10pp
              (LRU is already a reasonable policy on this geometry).
- Mechanistic prediction: backup-first recoverability near 100% by
  construction; its top-5 keep loss concentrates on twin pairs where the
  evicted member was itself a hub -- recorded if observed.
- Script: bench_cleanup_registry.py. VERDICT: PENDING.

**VERDICT (2026-09-29, reports/cleanup_registry_1790686148.json):
PARTIAL -- mirror persists; the twin definition needs revision.** But the
failure is itself the finding: backup-first found only **69 twin pairs**
in 7,025 pool entries (6% of the budget) -- the remaining 1,059 evictions
were deg70 top-ups, so the arm degenerated into core-first (27.5% /
71.4% vs core-first 27.2% / 71.6%). The upstream dedup (merge threshold
0.92) has ALREADY harvested the twin pairs; at cos >= 0.85 the
post-dedup store has almost none left. Two conclusions: (1) the mirror
is not caused by evicting "unbacked" entries -- evicting ANY high-degree
entry perturbs the retrieval order; hub-ness and redundancy are not
separable in this geometry; (2) **upstream dedup already collected the
redundancy dividend -- there is no second free lunch downstream**: G6
applies to NOT-yet-deduplicated corpora (synthetic families, raw SQuAD);
on a post-dedup production store the high-redundancy region is
retrieval-critical, and LRU's ranking-transparency (89.8% top-5 keep,
best of all arms) makes it the right default. Practical rule adopted:
future space savings on this store must come from tightening the dedup
threshold, not from eviction policy.



**P-2026-09-28-COMP1 · compression scoring harness** (registered BEFORE
the run; verdict PENDING). Question from the round-11 review discussion:
can the program's own selection theory (k-WTA sweet spot, elite-channel
concentration, stable-subset dissection) compress a TRAINED model at
inference time, and at what quality cost? First experiment is
training-free inference-time compression on the flypoet 92.6M k-WTA
checkpoints.

- Arms: D = dynamic dial (lower k_frac at inference); S = static elite
  mask (per-layer channel win-rates calibrated on the train.bin tail,
  keep-set frozen to top-r, no retraining). Baseline: native dynamic 0.25.
- Ratios r in {0.25, 0.15, 0.10, 0.05}. Checkpoints: k25_24k, k25, k25_s8.
- Scoring protocol (the scoring system): deterministic fixed-window val
  loss (320 windows x 256 tokens, batch 1, bf16 -- zero RNG; the
  +/-0.06-noise lesson forbids random-batch protocols for <0.05 effects);
  per-cell delta vs same-checkpoint baseline; LOSSLESS (delta<=+0.01) /
  HALF-COST (<=+0.05) / DEGRADED verdicts; theoretical activation-FLOP
  saving (1-r) and sparse-storage bound (W_O rows only, architecture
  keeps d_model dense) reported separately. No invented scalar score
  (restraint policy): the headline number is the per-checkpoint
  "lossless point".
- P1: |delta(S@0.25)| <= 0.01 (freezing to the earned elite set is
  functionally equivalent to dynamic top-k).
- P2 (exploratory): S@0.10 vs D@0.10 at equal activation budget.
- Scope: 92.6M char-level, data-starved regime; results do NOT transfer
  to LLM scale by L3 (sweet spot is scale-x-data dependent).
- Script: flypoet/bench_compress_score.py. VERDICT: PENDING.

**VERDICT (2026-09-28, flypoet/logs_v2/compress_score_1790601490.json +
sanity_mask_coverage.py): P1 NOT SUPPORTED -- negative result with a clean
mechanism.** Scoring card (deterministic 320x256 windows, delta vs native
dynamic 0.25): arm D (dynamic dial) degrades gracefully -- k25_24k:
+0.022 @0.15 / +0.051 @0.10 / +0.357 @0.05 (k25 and s8 agree within
0.01); arm S (static elite mask) collapses at EVERY ratio on ALL three
checkpoints (S@0.25 delta +4.1 to +7.8). Sanity check closes the
implementation loophole: the calibrated static set covers only **56.8%**
of each token's actual dynamic top-192 on val (per-layer 0.42-0.89,
shallower = more concentrated). The selection pattern is per-token and
co-adapted with the weights; the marginal win-rate statistic cannot
reconstruct it. P2: D@0.10 (+0.04..0.05) beats S@0.10 (+2.6..4.2) by two
orders of magnitude -- **how channels are chosen (per-token) matters far
more than how many are chosen**; freezing the set post-hoc is not the
same as training with a fixed set (the fix25 control trained fine --
co-adaptation, not subset sufficiency). Practical scoring verdict: the
training-free compression frontier on this architecture is the dynamic
dial at r=0.15 (activation-FLOP saving ~40% at delta ~ +0.01..0.02);
no LOSSLESS cell at delta<=+0.01 was found at any tested ratio. True
storage reduction would require TRAINING with a static mask (the natural
P-COMP-2) -- the theoretical sparse-storage bound of post-hoc masking is
only ~6.9% of params (W_O rows). Scope unchanged: 92.6M char-level.



**Verdict discipline (2026-09-28, adopted from the intuition-mechanism P47
integrity incident)**: a registration's verdict field is written `PENDING`
until the result artifact has been read; verdict text must never contain
numbers transcribed from memory or pre-filled placeholders -- numbers
enter only in the backfill commit that cites the artifact path.

**P-2026-09-28-PHASE1 · anchor phase boundary** (registered BEFORE the
run; verdict PENDING). Borrows P47's phase-boundary method: the SAME
anchored-judgment task at graded task difficulty, everything else fixed.
Question: WHERE does L6's near-mechanical anchored discretion degrade?

- Task: (6-entry store, message) -> does the message invalidate entry #1?
  Same prompt shape as P-ANCH-1 condition A, deepseek-chat temp 0.
- 36 trials: 18 YES (invalidates) at three construction levels of textual
  delta between old and new state -- 6 SMALL (digit/date swap inside an
  otherwise near-identical sentence; the L4 small-edit class), 6 MID
  (value swap + natural rewording), 6 LARGE (different wording + value
  domain; the B-arm class that scored 15/15) -- and 18 NO (6
  consideration, 6 question, 6 old-object mention WITH a near-collision
  distractor entry in the field).
- Frozen per-trial measurements (computed at construction, before the
  LLM run): delta_emb = 1 - cos(A, E) on YES trials; sim-gap
  cos(A,E) - max_j cos(A,D_j) on all trials.
- Script: bench_phase_boundary.py. Artifacts: reports/phase_*.json.

PRE-REGISTERED CRITERIA:
- P1 (phase gradient): YES-accuracy non-decreasing across small -> mid
  -> large AND acc(large) - acc(small) >= 15pp (the band threshold) =>
  CONFIRMED: the small-edit fragility (L4) reproduces at the JUDGMENT
  level -- L4 and L6 connect; the anchor does not rescue small deltas.
- P2 (sigma side): NO false-positive rate in the near-collision subtype
  exceeds the rate in consideration + question subtypes.
- Direction holds but gap < 15pp => PARTIAL; reversed or non-monotone =>
  NOT SUPPORTED (judgment level does not inherit L4's fragility).
- **VERDICT (2026-09-28, reports/phase_1790586567.json): P1 PARTIAL,
P2 NEGATIVE -- no phase boundary located in the tested range.** 36/36
perfect: YES accuracy flat at 6/6 across SMALL (delta_emb 0.214) / MID
(0.496) / LARGE (0.655); NO false-positives 0/18 with no
near-collision concentration. Per the registered bands (gap 0 < 15pp,
direction non-decreasing) P1 is PARTIAL; P2's concentration prediction
fails with zero FPs anywhere. **Interpretation: the anchored judgment
layer does NOT inherit L4's small-edit fragility -- the model reads
digit-swap updates and negative-sim-gap targeting (NEAR-COLL trials,
mean sim(A,D_near) 0.631 > sim(A,E)) perfectly when asked the anchored
binary question. L4 stays a MECHANISM-layer law (dedup/merge geometry
cannot see value swaps); L6 strengthens: the anchor flattens the
gradient down to delta_emb ~ 0.1.** Boundary, if any, lies below the
tested range or in a different variable (multi-anchor ambiguity,
unrestated slots) -- next-phase design, not a rerun of this ladder.



**P-2026-09-28-ANCH1 · anchor pairing experiment** (registered BEFORE
running; criteria locked now). Turns L6 from an observational A-vs-B
comparison into a paired controlled experiment: the SAME 48 turns, two
conditions, one dependent variable (does this message invalidate the
stored state?).

- Set (n=48): 12 balanced-stale (anchor=YES), 12 supersede-type from the
  v1.5 dataset (anchor=YES), 12 balanced-resist (anchor=NO), 12 noop-type
  (anchor=NO -- all assert facts about old objects; the structural
  false-positive source).
- Condition U (unanchored): turn alone -> asserts_fact true/false; a
  naive rule (asserts_fact -> invalidate) formalizes "classify first,
  decide freely". Structural ceiling 36/48 = 75%.
- Condition A (anchored): (stored entry, turn) -> invalidates true/false.
- Model: deepseek-chat, temperature 0. Script: bench_anchor_pairing.py.

PRE-REGISTERED CRITERIA:
- SUPPORTED  : acc(A) >= 44/48 (91.7%) AND acc(A) on the resist class
               >= 10/12 -- anchoring makes the discretion near-mechanical.
- NOT SUPPORTED : acc(A) < 38/48 (79.2%) OR resist-class repair < 8/12.
- otherwise PARTIAL.
- Mechanistic prediction: U's errors concentrate in the noop class (all
  12 assert facts -> structural false positives) and the resist class;
  any residual A errors are predicted to be declarative-intent statements
  ("I'm switching back to X") -- if so, verdict is recorded
  SUPPORTED-with-scope: anchoring repairs consideration and old-object
  classes; the declarative class is an annotation-disagreement residual.

**VERDICT (2026-09-28, reports/anchor_pairing_1790534250.json):
SUPPORTED-with-scope.** acc(A) = 47/48 = 97.9% (>= 44 required);
resist-class repair 11/12 (>= 10 required); acc(U) = 34/48 = 70.8% with
the errors EXACTLY where predicted (noop 12/12 structural false positives,
resist 2). Discordant pairs 13:0 in A's favor (exact McNemar p ~ 2e-4).
The single anchored error is bal_resist_01 ("I'm switching back to VS
Code") -- the predicted declarative-intent residual, recorded as scope:
anchoring repairs the consideration and old-object classes; the
declarative class is an annotation-disagreement residual, not an
anchoring failure.



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
- **P4 CLEAN RERUN (three-tier, post-review)**: an implementation bug
  (hyper factor added as linear term instead of multiplied) invalidated
  the first negative. Clean re-run with correct coefficients:
  Tier A blind raw: hit@1 5/17 (chance 0.368), Tier A' embed: 5/17,
  Tier B ratio-norm: **hit@1 17/17, mean_frac3 0.784**,
  Tier C template (aligned+normalized): **1.000**,
  c_s extraction (eq28-34, no c0): **7/7 exact**.
  The signature is FULLY accessible in the ratio-normalized representation;
  blind and embed representations cannot see it (both at chance).
  This closes the "invariant discovery" question for this dataset: the
  invariant IS discoverable, but only through the correct ratio transform.

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

## Intuition-Mechanism: the three-level mechanization map (2026-09-26)

The track's central question -- "which steps of mathematical discovery can be
mechanized?" -- now has a measured three-level answer:

| Level | Question | Verdict | Evidence |
|---|---|---|---|
| **L1 Recognise** | given the invariant representation, identify the structure | **MECHANIZED** | P4 clean: ratio-norm kNN 17/17; c_s extraction 7/7; P1-tag: family signal survives 2x-content noise tags (inst recall 0.046 vs chance 0.017) |
| **L2 Discover removal** | given the operator family, find the nuisance-removing transform | **MECHANIZED (with supervision caveats)** | P14: family-known LSQ selection (no labels, form given) 0.975; label-supervised DANN/GRL (s labels in training) 0.750, s-acc 0.883 -- the DANN arm is SUPERVISED invariance extraction, not blind factorization; P2 registered-caliber variant: structure ARI beats instance ARI at all 8 bottlenecks (+0.109..+0.218) |
| **L3 Discover family** | discover the operator family itself, blind | **OPEN (gap fully mapped overnight 09-27; see arc below)** | P13 v1+v2 + corrected grid (90 cells): nothing beats chance; P9: objectives learn the nuisance. CRACK (P15): a generic per-sample log-domain envelope fit (no H_s form, no labels) lifts hit@1 to 0.946 on counterfactual data (chance 0.247) -- "estimate the envelope, cancel it, read the residual" IS mechanizable. Arc follow-ups: P15-b refinement FALSIFIED (same-span refit is vacuous; obstruction = z-modulated fit bias on the invariant's own axis); P15-c ratio-domain reconstruction CONFIRMED 1.000 on one family / NEGATIVE on the other -- the two envelope schemes are complementary, neither dominates; P15-d blind scheme selection PARTIAL -- silhouette picks the right scheme when the signal is strong and falls into the salience trap (picks raw by 0.0009) when weak. Sharpest remaining statement: representations are salience-robust (P1-tag) but LABEL-FREE SELECTION criteria are not; separating invariant-structure from salient-structure by purely internal criteria is open |

The honest reading: L2 succeeds only when (a) the data contain explicit
counterfactual nuisance variation, and (b) the machine is HANDED part of the
answer -- the family form (arm c, 0.975) or the per-sample invariant labels
(arm b, 0.750). The truly blind end: recon-only AE 0.475, fixed-transform
menus at chance (P13 x2), prediction objectives learn the nuisance (P9) --
and P15's crack: a generic 3-parameter envelope FIT (no family form, no
labels) recovers the invariant's neighborhood structure at 0.946, pulling
"estimate-and-cancel" out of the un-mechanized column and leaving two
smaller residuals there: exact envelope identification (fit error keeps
adjacent classes overlapping for global readouts) and the act of CHOOSING
which decomposition to fit. Reviewer framing adopted: the open question is
not "more transforms" but whether a system can perform NUISANCE
IDENTIFICATION -> INVARIANCE CONSTRUCTION -> STRUCTURE RECOGNITION as one
data-driven loop. The current boundary of "intuition mechanization" is
therefore: the outermost step -- inventing the right operator family
without being handed it -- remains un-mechanized; everything downstream of
it, and now one layer inside it (generic envelope estimation), does not.

## Related

### Overnight arc 2026-09-27 (P15-e/f, P16, D1-eta; intuition-mechanism 050aca4/b1d2b73)

- **P15-e FALSIFIED**: 2D (z, B/A) confound regression-out -- random control
  (0.379) matches oracle (0.395) and blind (0.360); bias is idiosyncratic,
  representation-side cleanup exhausted (kNN 0.946 shows information present).
- **P15-f NEGATIVE + REGIME THEOREM**: PC1 salience-correction fixes the true
  family (ratio ARI 0.087 -> 0.394) and destroys the ratio family (1.000 ->
  0.291): the invariant may be dominant or subordinate; internal geometry
  does not encode the regime.  With P15-d, all three internal-criterion
  families are closed with distinct mechanisms (salience capture /
  idiosyncratic bias / regime ambiguity).  Remaining openings: external
  information (counterfactual access) or mild structural priors.
- **P16 PARTIAL (substance confirmed)**: Gamma-proposition -- after exact
  envelope removal the invariant axis IS C_s = 1/(Gamma(1/2)Gamma(1/s)-
  Gamma(1-1/s)), verified at 17 digits with exact-Bernoulli-coefficient
  constrained extrapolation; confusability ordering confirmed (rho=-0.956,
  most-confused pair (2,3) = smallest-Gamma-gap pair).  Registered band
  1e-25 was an instrument-range error -> meta-lesson on band-setting.
- **D1-eta GATE PASS**: level-6 eta-quotient machinery mechanized
  (documented values j6C=32, j6D=81, j6A=39200 reproduced at 1e-56..59;
  Conway-Norton relation constant corrected 22 -> 14); integer CM-value
  sweep finds 3/36 true hits incl. two NEW small class invariants (8, 9);
  D1's 0/27 explained: the parameters live in eta quotients, not j.
- **Literature anchors** (docs/LITERATURE.md): Kolmogorov structure
  function (what-to-keep), Fisher ancillarity (nuisance), Rosenfeld IRM
  critique (invariance does not identify the invariant), nonlinear-ICA
  identifiability (external assumptions required), ML symmetry discovery
  (labels/priors needed), Ramanujan Machine (search WITH human ansatz).

### Overnight continuation 2026-09-27 (P16-P22; intuition-mechanism to 480b377)

- **P16 PARTIAL (substance confirmed)**: Gamma-constant proposition
  proven to 17 digits (constrained exact-Bernoulli extrapolation);
  confusability ordering rho = -0.956, most-confused pair (2,3) =
  smallest Gamma-gap.  Band-setting meta-lesson recorded (three
  threshold errors across its registration history).
- **P17 INCONCLUSIVE**: 27B reasoning model at chance on both
  literature-real (0.235) and counterfactual (0.250) sequences; with
  wrong few-shot labels it follows labels (C = 0.000).  Discriminator
  open for stronger models.
- **P18 CONFIRMED**: Z(X) level-6 machinery implemented and verified
  (ODE 1e-63, differential identity 1e-68 on the cusp-side sheet); X0
  table at CM points = exact rationals (1/36, 1/54, 1/100, ...).
- **P19 CONFIRMED (generation mechanized end-to-end)**: 59 identities
  for N = 2..60 generated and re-verified at machine precision
  (docs/IDENTITIES.md); lambda solved numerically exactly as the CWZ
  paper prescribes; lambda algebraicity recognized (quadratics at
  N = 2, 11, 19, 23, 25, 35, 43, 47).  Novelty: absent from CWZ Table 1
  and Chan-Cooper 2012 tables.
- **P20 CONFIRMED + ERRATUM FINDING**: census caught the arXiv CWZ
  Table-1 N=17 lambda = 143/238 as inconsistent with their own eq (3.9)
  (correct value 43/238; identity closes at 1e-56 only with 43/238).
- **P21 NEGATIVE (informative)**: deg(x0) is bounded (<= 9) with clean
  residue-class structure; NOT h(-24N) -- level-12 conductor mix;
  hypothesis recorded.
- **P22 CONFIRMED**: negative-branch Table-1 identities verified
  (4/5 at <= 1e-62, the fifth is the |x| = 1/8 radius boundary);
  the A +- 2Num antisymmetry IS the 1/x(q) + 1/x(-q) = 4 identity
  under the phase-corrected half-period translate.

- **P23 CONFIRMED (07:12)**: class-group orbit generation -- the identity
  holds at EVERY conjugate CM point (~180 verified across N=2..30, zero
  failures); the published rational rows are the trace-degenerate tips of
  full orbits.  Orbit-vs-one-representative-degree mismatch is conceptual
  (K-conjugacy vs Q-degree).  P23-b (N=31..60 completion) registered.

- **P24 HALTED / P26 PARTIAL (07:30-07:45)**: the orbit lambda action test
  was ill-posed as registered (orbit lambdas are complex; category error
  vs the cusp-side quadratic); P26 measured the t-family asymptotic
  exponent alpha = 3/2 cleanly, but the constant K = 1.4344 has no closed
  form in the tested Gamma bases -- the ODE-local-analysis route at the
  ramification point x = 1/8 is the recorded next step.

- **P28-b orbit extension CONFIRMED / P29 NEGATIVE (08:40)**: orbit
  generation extended to N=61..90 (430 identities, all verified; census
  patterns hold out-of-sample); the universal lambda function G(x0) has
  NO simple low-parameter closed form (best tested basis 8e-2 residual)
  -- G is non-elementary, consistent with lambda involving z(x0) and
  z'(x0) at CM points.

- FlyPoet (k-WTA x Transformer, 216M inversion): architecture-axis experiment log
- FlyMemory README: memory-system engineering doc (a testbed of this program)
- LongMemEval / LongMemEval-V2: external benchmarks (integration pending)


### Overnight arc 2026-09-28 (P32-g..P36; intuition-mechanism to 086e59a)

The LLM-subject line (P32-h..P36-c) turned the "insight = compression"
motto into measured results, and the math line produced the night's
headline:

- **P32-h balanced novelty detection PARTIAL**: blind judge
  (doubao, thinking disabled) existing-family 16/20 (80%, p~4e-7)
  vs new-family 4/20 (20%, at the 25% guess line) -- recognition
  WITHOUT novelty; the judge says NEW only 7/40 times and absorbs
  misses indiscriminately (A x9/B x5/C x2).  Native structural
  subject 38/40 -- the information IS in the window.
- **P32-i causal cue intervention**: supplying the P34-d extraction
  procedure lifts the blind judge to 39/40 (new-arm 20/20, p=9e-13);
  merely truncating sequences to 8 terms does NOT teach extraction
  (new 11/20 but existing collapses 16->10/20; NEW-answer rate flips
  7/40 -> 19/40).  The novelty deficit lives in the EXTRACTION step;
  the representation-to-comparison wiring is the missing piece.
- **P35-a blind MDL discovery**: exhaustive MDL over threshold-free
  features (equality-to-mode + flip), no labels/class-count, on the
  4800-family corpus, rediscovers EXACTLY the P34-d tree features
  {eq2=6, eq3=20, eq4=70, flip}; 8/9 winner cells align with truth.
  P35-b: doubao induces the right SCHEMA from 24 examples (55.8%
  executed; chance 12.5%), one recalibration round -> 76.2%; MDL
  code 400/400 on the same test -- where-to-look is inducible from
  tiny data, calibration is a data requirement.
- **P36-a lambda arithmetic census (HEADLINE; N=2..160)**: the
  algebraic degree of the hidden parameter lambda EXACTLY reproduces
  the human publication boundary of the family.  Degree 1 (rational)
  = {3,5,7,13,17} -- precisely the published rows; degree 2 = {2,11,
  19,23,25,35,43,47,55,73}; degree 3 = {9,27,29,31,37,41,49,53};
  deg(lambda) = deg(x0) on 20/20 resolved rows (two-row census with
  P20/P21).  Showcase N=9: 96l^3-192l^2+114l-17 (height 192), exact
  radical lambda_9 = 2/3 - (3sqrt2+19)^(1/3)/12 -
  7/(12(3sqrt2+19)^(1/3)), identity verified <= 5e-51 -- one
  algebraic level deeper than anything published.  All relations
  verified by unique-root test + direct substitution.  (Precedence:
  x0-side census is P20/P21; this row is the lambda side, extended
  to 160, with strict verification.)
- **P36-c depth perception**: a blind judge shown only x0 to 12
  digits separates deep/plain 15/15 TWICE (independent samples),
  names four rationals exactly (1/12, 1/32, 1/104, 1/200), full
  ranking |rho| = 0.79 / 0.72 (p = 4e-4 / 3e-3), above a mechanical
  rational-detector baseline (0.61) -- arithmetic depth is visible
  on the surface.

Law candidate sharpened across the three repos: intuition is
extracting a low-dimensional, predictively valid sufficient
structure from high-entropy experience -- and in this testbed that
structure is provably small (four features), mechanically
discoverable (MDL), causally the bottleneck (P32-i), and its
"depth" axis is what human mathematicians were tracking all along
(P36-a).

- **P36-d/e perception pooled + P38 theory (cross-model)**: deep/plain
  partition 15/15 in FIVE consecutive runs (two model families, four
  seeds; the n=20 explicit-ask run gives an exact PLAIN set, zero
  FP/FN); direction of "special" is unstable (convention), partition
  is robust.  P38: naive class-number theory NEGATIVE (h(D_K)=1 at 12
  non-rational rows; Spearman(h(order), deg(x0)) = 0.146 ns) --
  rationality is a level-12-specific degeneration condition (P23's
  orbit tips); lambda-algebraicity + deg(lambda)=deg(x0) recorded as
  the standing theory sketch (docs/THEORY_LAMBDA.md); CCL Thm 2.1
  scoped out (levels {1..9} exclude level 12).; P33-d hull-axis NULL at n=15 (rho=-0.06): parameter-space novelty
  does not scale as an interestingness axis -- the lambda-degree axis is
  the one with signal; P32-i Arm T on GLM gave no usable signal
  (parse-failure dominated), truncation conclusion stands on doubao.

- **P39 capstone (04:55)**: end-to-end mechanized recognition + novelty
  pipeline (support-pattern rule -> class -> novelty comparison) scores
  40/40 on the same P32-h trials the blind LLM judges score 50-52% on.
  The family's map is now: recognition MECHANIZED, novelty MECHANIZED
  (given the discovered code), depth MEASURED (lambda degree census),
  interestingness-as-value OPEN (hull axis falsified at n=15,
  rho = -0.06).


- **P43 MI/MDL convergence QUANTIFIED**: rank correlation between
  mutual information and MDL gain rises monotonically with sample
  size (rho 0.565 -> 0.871 -> 0.890 -> 0.849 across m=10..60, all
  p < 1e-5) -- the discovery threshold m* is WHERE the two ranking
  criteria converge.  Below m*, the compressor finds features that
  compress but do not inform.
- **P47 phase boundary MAPPED**: accuracy rises monotonically with
  Delta/sigma for both memory arms (STR >= EPI-ALL at every ratio);
  transition between ratio 1 and 2.  On matched geometry the
  memory-type choice is secondary to the gap-to-noise ratio.
- **P17-B FORMAT-BOUND resolved with 6-model fleet**: all six model
  families fail raw-decimal format on both legs (A-lit AND
  B-counterfactual at chance); R2 ratio-table scaffold doubles
  deepseek-v4-flash (0.25 -> 0.50) -- the extraction lesson (P32-i)
  reproduces at the L3 layer.
- **P48 Markov authors (non-math)**: memory law REPLICATES (STR
  87.6% vs EPI 83.4%, recovery 10/10); NEW-author detection FAILS
  (STR 2/12, EPI 1/12) -- novelty failure is family-robust.
- **P49 oscillators (4th family, FFT transform)**: recognition
  CONFIRMED both arms (89.8% / 90.9% -- FFT is the correct blind
  transform); novelty still poor (4/12, 2/12).
- **P50 two-condition law CONFIRMED across four families**: C1
  (extraction quality) determines recognition; C1 AND C2 (novelty
  hull geometry) jointly determine novelty detection; no family
  violates the law.

- **P44 negative-q lambda census (partial)**: negative-q lambda values
  DIFFER from positive-q (confirms the +/- sign structure).  Degree
  detection has a float-precision bug (fix queued): N=7 lambda = 5/21
  (rational!) was misclassified as degree 2 because float precision is
  insufficient for tol = 1e-35 PSLQ.

- **P46 memory comparison on SECOND family (envelope, continuous
  ladder)**: EPI-3 65.5% > STR-centroid 55.7% -- the advantage
  INVERTS.  Discrete-support families: structural memory wins.
  Continuous-overlapping families: instance memory wins.  This IS
  the prototype-vs-exemplar dissociation from cognitive psychology,
  replicated with a mechanistic pipeline.