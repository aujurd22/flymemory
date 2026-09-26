# Overnight Research Handoff — 2026-09-24

Directive: continue the FlyMemory follow-up work, research until 9:00.
All core results landed; this doc is the handoff record (originally written
in Chinese, translated per repo language policy).

## Result chain of the night (all pushed, CI green)

| Commit | Content |
|---|---|
| 0b773f2 | cross-encoder rerank integration (oracle +4.4pp) + S-library timestamp repair |
| 9ee71d0 | S-library full-scale six-arm benchmark: rerank 45.8% vs RRF 40.8% (+5.0pp) |
| 84bbe63 | external-review fixes x5 (consolidate/supersede/forget/copy/scoring semantics) |
| 4218e53 | state-aware rerank negative result (post-fusion eff reranking hurts) |
| 7f3cc1e + 4ad09f7 | README semantic consistency fixes (production = RRF, decay = maintenance) |
| f2a6af3 | **Phase 1 Memory Judgment Benchmark** (38 scenarios) |
| a5293ab | **Phase 2 five-arm end-to-end** (answer-level) |
| 7388e86 | **engine dedup repair** (5/20 dropped state updates -> 0/20) |
| c278bac + 2f6dbbe + 84324d9 | LME e2e + granularity A/B + five-arm extension |

## Core numbers (all reproducible; scripts in the repo root)

- Retrieval (S library, 200k entries, evidence-hit@3): dense 27.4 / bm25 33.6 / RRF 40.8 / **RRF+CE 45.8**, oracle@10 55.4
- Judgment layer (38 scenarios): DeepSeek supersede P/R **1.00/1.00** (execution-aware), forget 1.00/1.00, **unnecessary mutation 0/10**
- End-to-end (30 state scenarios): naive RAG **13-17% stale answers**; FlyMemory (oracle and autonomous) **0% stale, 87% current**; no-memory arm 93% unknown (hardcoded harness abstention at the time; the TRUE blind baseline now lives in bench_e2e_answer.py)
- Granularity A/B (50 questions): turn-only 32% / **replace-style consolidation 22% (negative)** / **overlay 38% (+6pp)**
- LME e2e attribution: hit@5=64% x answer conversion 50% = 32% strict; top-k 10 adds nothing
- State-fidelity audit: pre-fix 5/20 real state updates silently dropped by dedup; post-fix 20/20 correct

## Design principles now measurement-backed

1. "Raw entries are never deleted — abstraction without loss": replace-style
   consolidation 22% vs overlay 38%.
2. "Decay is a maintenance signal, not a ranking signal": post-fusion eff
   reranking hurts (oracle -0.8/-2.0pp).
3. "Whatever is mechanically verifiable stays out of the model": mechanical
   validity in code; unsupported inference goes to an LLM judge.

## Known issues / next steps (by priority)

1. ~~judge lacks a stale dimension~~ -> DONE (detected in cons_08)
2. ~~flymemory_remember tool return does not distinguish new/merged~~ -> DONE
3. ~~merge-inplace history loss~~ -> tombstone SHIPPED (ad54abe)
4. ~~judgment dataset to 100+~~ -> DONE (v1.3, 124 scenarios)
5. Phase 4 long-running verification — pending

## Ops state

- Service: current PID (8765 LISTENING), running the latest engine, guard enabled
- Benchmark discipline: the engine is frozen while benchmarks exist (the
  state-fidelity fix was a correctness exception); reports/ is gitignored

## Reproduction commands

```bash
PY="C:\Users\djr82\AppData\Local\Programs\Python\Python313\python.exe"
$PY bench_state_fidelity.py                      # state-fidelity audit
$PY bench_memory_judgment.py --actor oracle      # harness check (all 1.0)
DEEPSEEK_API_KEY=sk-... $PY bench_memory_judgment.py --actor deepseek --judge
DEEPSEEK_API_KEY=sk-... $PY bench_memory_judgment_tools.py
DEEPSEEK_API_KEY=sk-... $PY bench_e2e_answer.py
DEEPSEEK_API_KEY=sk-... $PY bench_lme_e2e.py --sample 50
DEEPSEEK_API_KEY=sk-... $PY bench_granularity.py --mode overlay
```
