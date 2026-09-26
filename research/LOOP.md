# LOOP — a registered discovery loop across the three repos
(2026-09-27, design v1; the 09-27 overnight session is its manual
existence proof: 9 registered items, executed, verified, recorded,
pushed.)

## The three repos as organs

| Repo | Organ | Contract |
|---|---|---|
| intuition-mechanism | **WORLD / VERIFIER** | mechanical 50-digit gates, series validators, registered thresholds — ground-truth feedback that cannot be argued with |
| flymemory | **MEMORY / STATE** | registered experiments + verdicts as memories; recall/supersede/consolidate as the loop's persistent state across sessions |
| flypoet | **PROPOSER** | k-WTA sparse readouts, codebook mechanisms — the candidate-mechanism pool for representation experiments |

## The cycle (one loop iteration = one registered experiment)

```
1. PROPOSE   pick the sharpest open edge from the gap map
             (PAPER.md §8 / registry verdicts)
2. REGISTER  write the prediction + falsification band + metric into
             docs/RESEARCH_PLAN.md BEFORE running (commit, push,
             ls-remote verify)
3. RUN       the script (serial; CPU-level design; memory-check gate
             before any launch)
4. VERIFY    bands applied as registered — no post-hoc threshold edits;
             amendments only via recorded REVISION entries
5. RECORD    verdict into the registry row + JSON artifact + commit
6. REMEMBER  flymemory_remember (findings) + flymemory_supersede
             (superseded earlier states) + consolidate when >= 3
             related entries
7. PUSH      git push + ls-remote verification (never trust local log)
```

## Data contract: a registered experiment record

```json
{
  "id": "P15-e",
  "registered_commit": "324d0df",
  "band": {"metric": "ARI", "confirm": ">= 0.6", "kinds": "..."},
  "result_json": "p15e_confound_results.json",
  "verdict_commit": "3bc6b4f",
  "verdict": "FALSIFIED (mechanism)",
  "supersedes": null,
  "lessons": ["random control matched oracle -- confound removal was no-op"]
}
```

## Sandbox run plan (10h overnight, the user's "丢东西进去" idea)

- Inputs ("丢进去"): the registry's open edges + one model/one generator
  + the memory service as state.
- Serial discipline: one experiment at a time; RAM check before any
  GPU/model launch; artifacts small (JSON), pushed after every verdict.
- Termination: time box OR three consecutive inconclusive runs (then
  the loop writes a "blocked — needs human" note instead of thrashing).
- The 09-27 night: PROPOSE→RECORD loop executed manually with
  flymemory hooks active; next step is wrapping steps 1-7 into an
  orchestrator script (propose stays human/LLM-assisted; everything
  else mechanical).

## What the loop must NOT do

- No unregistered experiments (that is how p-hacking enters).
- No verdict without a JSON artifact + commit hash.
- No memory writes without a supersede check (stale states poison
  recall).
- No push without ls-remote verification.
