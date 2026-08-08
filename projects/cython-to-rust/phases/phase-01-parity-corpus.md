---
phase: 01
title: Golden parity corpus
status: Complete
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
- Context: when this phase was drafted, **zero** pinned-value tests over
  compiled code executed anywhere — CI's `pytest` collected only
  `hazma/**` (setup.cfg `testpaths = hazma`), and every `.npy`-backed
  suite under `test/` was either skipped or never collected
  (`test/spectra/integration.py` matched no pytest filename pattern;
  `test/positron/test_positron.py` was 0 bytes). Task 1.2 ended the first
  half of that: `test/parity/test_parity.py` ran under `pytest test`.
  Task 1.3 ended the second — pytest is configured in `pyproject.toml`
  with `testpaths = ["hazma", "test"]`, a bare `pytest` is the whole
  suite, and CI runs it on every matrix entry. Task 1.4 closed the phase
  by retiring the legacy `.npy` suites themselves. `collect_ignore` is
  not part of any of this; `test/conftest.py` has listed only the repo's
  `setup.py` since Task 0.2.

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
- The manifest's per-block `raises` records are replayed, not skipped:
  where the corpus says an entry point raised, the runner asserts the
  live implementation raises the same exception type at the same
  argument. (Task 1.1 found two —
  `sigma_xx_to_v_to_pipi` and `sigma_xx_to_v_to_pi0v` raise `TypeError`
  exactly at `e_cm = 2·mx`.) A runner that only compared the stored
  `nan` would pass against an implementation that silently returned a
  number there.
- Running against unmodified Cython passes bit-exact.

### Task 1.3: Wire both suites into one gate

**Task note:** [`../task-notes/phase-01/task-1.3-test-wiring.md`](../task-notes/phase-01/task-1.3-test-wiring.md)
**Depends on:** Task 1.2

**Exit criteria:**

- pytest config moved to `pyproject.toml`
  (`[tool.pytest.ini_options]`), collecting `hazma` **and** `test`
  (bare `pytest -q` → 935 passed / 30 skipped as of 2026-08-07, on the
  capturing environment. The `test/` root alone was 51/20 when this
  phase was drafted, then 244/20 after PR #41 and Task 0.3, then 870/20
  once Task 1.2 landed the parity suite — re-derive rather than quoting
  any of them. Expect 934/31 off the capturing environment: the parity
  suite skips its `test_running_on_the_capturing_tree` marker there and
  enforces the declared budgets instead.)
- `test/spectra/integration.py` renamed to be collected; its property
  assertions pass.
- CI and `scripts/agents/preflight.sh` run the same collection **on the
  capturing platform**; `docs/agents/` env notes updated. Two things the
  plan did not anticipate, both measured in Task 1.3:
  - Widening `testpaths` is not sufficient on its own: CI's non-editable
    `pip install .` leaves no extension inside the checkout, which
    `cases.assert_module_is_repo_tree` refuses, so the test job
    reinstalls editable first.
  - The corpus does not survive a change of libm. Enabling it in CI
    measured that for the first time: Linux/glibc fails ~70-75 of the
    626 blocks, six of them at cancellation points where the pinned
    value flips sign (`sigma_xl_to_xl[closed_resonance.mu]`: -1.504e-02
    on macOS, +5.624e-07 on Linux). The Linux entries therefore run
    `pytest --ignore=test/parity`, and making the corpus
    platform-robust — which is also what the Rust port will need — is
    [`docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md`](../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md).

### Task 1.4: Retire or regenerate the legacy `.npy` suites

**Task note:** [`../task-notes/phase-01/task-1.4-legacy-npy.md`](../task-notes/phase-01/task-1.4-legacy-npy.md)
**Depends on:** Task 1.2

**Exit criteria:**

- The skipped `TestScalarMediator` and `TestVectorMediator` classes
  (`skip("Needs to be updated")`), which read 90 `.npy` reference
  arrays from the eight `data/sm_*` and `data/vm_*` directories they
  name, are
  either regenerated-and-unskipped or deleted with their intent
  explicitly mapped to corpus coverage in the task note.
  **Realized: deleted**, with their two `generate_test_data.py` producers.
  Regeneration was rejected on evidence, not preference — the arrays
  encode a superseded convention (six scalar cross sections differ from
  the current tree by exactly ×2), `vm_5`/`vm_6` duplicate `vm_3`/`vm_4`
  by a generator bug, the scalar class loaded `sm_1` twice so `sm_2` was
  never read, and 11 of the 17 tests fail against the current tree. The
  non-redundant half of their intent — the pure-Python aggregation in
  `hazma/theory/`, which no corpus case reaches — moved to
  `test/test_theory_aggregation.py` as identities rather than as a second
  set of golden arrays.
- `test/positron/test_positron.py` (0 bytes) deleted or filled.
  **Realized: deleted.**
- No `@pytest.mark.skip` remains whose reason is "needs update".
  **Five** unrelated marker sites survive — two "Known to be broken" in
  `hazma/form_factors/vector/{_eta_gamma,_pi_gamma}_test.py` and three in
  `test/vector_mediator/test_form_factors.py` — and they account for all
  13 skipped tests the suite reports (5 + 5 + 3; the two `hazma/` markers
  sit on parametrized classes). A sixth marker,
  `test/agents/test_resolve_phase.py:47`, is a `skipif` that does not
  fire while the script it guards is present. All are outside this
  criterion and untouched.
- Folded in from Task 1.3's Open Questions: `test/rh_neutrino/integration.py`
  and `test/rh_neutrino/widths.py` matched no `python_files` pattern.
  **Realized:** the first is a real suite and was renamed to
  `test/rh_neutrino/test_rh_neutrino_integration.py` (2 tests, collected);
  the second is a matplotlib plotting script under `if __name__ ==
  "__main__"`, not a test, and was deleted. The rename cannot use the
  obvious `test_integration.py`: `test/spectra/test_integration.py`
  already holds that basename, and with no `__init__.py` under `test/`
  the two collide on pytest's import-file-mismatch check.

## Exit Criteria

- All tasks complete; `pytest` (bare) runs unit + property + parity
  suites and is green in CI. Realized on the capturing environment at
  Task 1.4: `1006 passed, 13 skipped` from 1019 collected (67 from
  `hazma`, 952 from `test`), against Task 1.3's 935/30 — the delta is
  +69 aggregation tests, +2 from the `rh_neutrino` rename, and −17
  skips as the two legacy mediator classes left. Re-derive rather than
  quoting. The parity portion runs on the **capturing platform** only —
  the other matrix entries run the rest of the suite and skip
  `test/parity`. That is a Task 1.3 amendment to this
  criterion, not the original intent: the corpus pins six
  cancellation-dominated points that no tolerance can carry across a
  libm change, so "green on all matrix entries" is unreachable until
  the follow-up above lands. Restore this bullet when it does.
- Corpus regeneration is documented and reproducible
  (`python test/parity/generate.py --check` verifies hashes).
- Phase learnings written to `../learnings/phase-01-parity-corpus.md`.
