---
phase: 01
title: Golden parity corpus
status: Not started
---

# Phase 01: Golden parity corpus

## Goal

Stand up the regression harness the repo currently lacks: pinned
reference arrays for **all 41 consumed compiled entry points** (of 43
public defs — the two unimported `sigma_xx_to_all` exports are
excluded here and dropped in Phase 05), generated from the pre-port
Cython, wired into pytest and CI. This is the gate
every later phase swaps against, and it doubles as the before/after
evidence `docs/versioning.md` requires for numerical changes.

## Prerequisites

- Phase 00 complete (corpus covers survivors only).
- Read `../references/numerics-replacements.md` (call-site tolerances)
  and `../rules.md` rules 1–3.
- Context: today **zero** pinned-value tests over compiled code execute
  anywhere — CI's `pytest` collects only `hazma/**` (setup.cfg
  `testpaths = hazma`), and every `.npy`-backed suite under `test/` is
  skipped, collect_ignored, or never collected
  (`test/spectra/integration.py` matches no pytest filename pattern;
  `test/positron/test_positron.py` is 0 bytes).

## Tasks

### Task 1.1: Corpus specification and generator

**Task note:** [`../task-notes/phase-01/task-1.1-corpus-generator.md`](../task-notes/phase-01/task-1.1-corpus-generator.md)
**Depends on:** —

**Exit criteria:**

- `test/parity/generate.py` produces `test/parity/data/*.npz` covering
  every **consumed** entry point in `../references/cython-inventory.md`
  ("Entry points by module" — 41 functions; the two unimported
  `sigma_xx_to_all` are excluded, with the exclusion asserted by an
  import re-check in the generator): log-spaced energy grids
  bracketing thresholds and
  kinematic endpoints (`E → m/2`, table edges), ≥4 parent
  energies per spectrum (rest frame + ε, mildly and strongly boosted),
  and for the mediators ≥3 model-parameter points including a
  near-resonance configuration; thermal ⟨σv⟩ over an x grid spanning
  freeze-out.
- A manifest (JSON) records generator git SHA, package versions, and
  per-array hashes. Total data size ≤ ~10 MB.
- Grids deliberately include the known NaN/negative-prone kinematic
  edges; captured values are stored as-is (edge behavior is part of the
  contract).

### Task 1.2: Pytest runner and tolerance budgets

**Task note:** [`../task-notes/phase-01/task-1.2-parity-runner.md`](../task-notes/phase-01/task-1.2-parity-runner.md)
**Depends on:** Task 1.1

**Exit criteria:**

- `test/parity/test_parity.py` parametrizes over the manifest and
  compares live imports against stored arrays.
- Per-function budgets live in `test/parity/tolerances.py` (or `.toml`)
  with a one-line justification each: exact (bit-equal) for pure
  closed-form kernels against the capturing commit, documented budgets
  for quad-backed kernels (start 1e-8 rel, tighten after Phase 03
  measurement; nested-ρ gets its own line).
- Running against unmodified Cython passes bit-exact.

### Task 1.3: Wire both suites into one gate

**Task note:** [`../task-notes/phase-01/task-1.3-test-wiring.md`](../task-notes/phase-01/task-1.3-test-wiring.md)
**Depends on:** Task 1.2

**Exit criteria:**

- pytest config moved to `pyproject.toml`
  (`[tool.pytest.ini_options]`), collecting `hazma` **and** `test`
  (the `test/` suite is green post-PR #31: 51 passed / 20 skipped).
- `test/spectra/integration.py` renamed to be collected; its property
  assertions pass.
- CI and `scripts/agents/preflight.sh` run the same collection;
  `docs/agents/` env notes updated.

### Task 1.4: Retire or regenerate the legacy `.npy` suites

**Task note:** [`../task-notes/phase-01/task-1.4-legacy-npy.md`](../task-notes/phase-01/task-1.4-legacy-npy.md)
**Depends on:** Task 1.2

**Exit criteria:**

- The skipped `test/scalar_mediator/` and `test/vector_mediator/`
  classes (159 reference arrays, `skip("Needs to be updated")`) are
  either regenerated-and-unskipped or deleted with their intent
  explicitly mapped to corpus coverage in the task note.
- `test/positron/test_positron.py` (0 bytes) deleted or filled.
- No `@pytest.mark.skip` remains whose reason is "needs update".

## Exit Criteria

- All tasks complete; `pytest` (bare) runs unit + property + parity
  suites and is green in CI on all matrix entries.
- Corpus regeneration is documented and reproducible
  (`python test/parity/generate.py --check` verifies hashes).
- Phase learnings written to `../learnings/phase-01-parity-corpus.md`.
