# Bench harness checklist (flyloop preflight discipline, adapted 2026-09-29)

Run through this list when adding a new benchmark OR adding a new dimension
(compartment, state_key, supersede semantics, a new recall field) to an
existing one. The flyloop V7 incidents behind each rule are noted so the
rule is not mistaken for ceremony.

1. **Write-path isolation** — a bench must never load the production pkl
   (`flymemory/flymemory_v3.pkl`) with write intent. Read-only studies use
   a snapshot copy (`shutil.copy` then load the copy), like
   `bench_cleanup_geometry.py`. Default arguments must not point at the
   production pkl unless the script provably never calls `.save()`
   (current state: `bench_recall_speed.py`, `bench_connectome_params.py`
   are load-only / no-load — verified 2026-09-29).
2. **Read-path check on every new dimension** — when a new field or flag
   is added (state_key, compartment, valid_from, ...), add a READ-path
   test (recall returns/filters it correctly), not only a store-path
   test. Flyloop's V7A failure was a new arm checked on writes but not on
   reads. Current coverage: include_superseded, compartment, state
   fields all have read-path tests in tests/test_rules.py.
3. **Single-writer discipline** — the live service owns the pkl; offline
   imports/repairs require the scheduled task disabled first (the
   standing rule: 改库先停服). A bench that needs a *writable* store
   builds its own `SmartMemory()` in a temp dir, never the production
   file.
4. **Determinism flags** — scoring benches set thread caps
   (`OMP_NUM_THREADS`/`torch.set_num_threads`) and fixed seeds/slices;
   any comparison below ~0.05 must use the deterministic fixed-window
   protocol (the ±0.06-noise lesson), never random batches.
5. **Trace everything** — every case's raw output, parsed decision, gold,
   and the decision bands (if pre-registered) persist to `reports/`
   before the verdict is written anywhere. Verdict text carries no
   numbers read from memory or placeholders (the PENDING-marker rule).
