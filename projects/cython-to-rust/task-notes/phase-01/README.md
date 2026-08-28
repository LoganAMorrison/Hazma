# Working Memory: Phase 01 — Golden parity corpus

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 01
**Status:** Complete (2026-08-08) — all four tasks landed; learnings at
`../../learnings/phase-01-parity-corpus.md`. One follow-up against the
phase's output has since landed: see the Follow-ups row below.
**Plan References:** `../../phases/phase-01-parity-corpus.md`
**Related ADRs:** none
**Depends On:** Phase 00 complete

## Objective

Track live per-task status and phase-scoped findings for the parity
corpus.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 1.1 | Corpus specification + generator | — | **Complete (2026-08-07)** | [task-1.1-corpus-generator.md](task-1.1-corpus-generator.md) |
| 1.2 | Pytest runner + tolerance budgets | 1.1 | **Complete (2026-08-07)** | [task-1.2-parity-runner.md](task-1.2-parity-runner.md) |
| 1.3 | Wire both suites into one gate | 1.2 | **Complete (2026-08-07)** | [task-1.3-test-wiring.md](task-1.3-test-wiring.md) |
| 1.4 | Retire/regenerate legacy `.npy` suites | 1.2 | **Complete (2026-08-08)** | [task-1.4-legacy-npy.md](task-1.4-legacy-npy.md) |

## Follow-ups against this phase's output

Not plan tasks — the phase is complete and these repair its artifacts.

| Item | Landed | Note |
| --- | --- | --- |
| [parity corpus pins ill-conditioned points](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md) | 2026-08-18 | [followup-parity-corpus-stability.md](followup-parity-corpus-stability.md) |

## Exit Criteria

- ~~All rows Complete; phase file frontmatter `status: Complete`.~~ — met
  2026-08-08.
- ~~Phase learnings at `../../learnings/phase-01-parity-corpus.md`.~~ —
  written 2026-08-08. **Read the learnings, not this file or the four task
  notes**; they are history, it is the distillation.

## Inputs Reviewed

- `../../phases/phase-01-parity-corpus.md`; `../README.md`;
  `../../references/numerics-replacements.md` (tolerance table);
  `../../references/cython-inventory.md` (entry-point table).

## Findings

- **The corpus exists and is self-checking.** 41 cases / 623 blocks /
  1,580 arrays / 179,695 pinned values, 2.9 MiB, generated from the
  pre-port Cython identified by kernel digest `f5e6e269be47` (the
  manifest's recorded SHA is whichever commit was HEAD at generation —
  `010747c` after the round-1 review fixes — but `hazma/` is byte-
  identical to `origin/master`, which is what the digest certifies).
  `python test/parity/generate.py --check` re-verifies it in under a
  second and needs no built tree. Two full regenerations produced a
  byte-identical manifest.
- **Coverage is derived from the tree, not transcribed.**
  `assert_full_coverage` walks the surviving `.pyx` for top-level
  `def`s and fails both ways — an unpinned `def`, or a case naming a
  `def` that no longer exists. Later phases therefore cannot delete a
  Cython module without the corpus noticing. `assert_no_rust_core`
  enforces rules.md rule 2 in code.
- **Two entry points raise rather than return at a kinematic edge.**
  `sigma_xx_to_v_to_pipi` and `sigma_xx_to_v_to_pi0v` raise `TypeError`
  at exactly `e_cm = 2·mx`; three blocks carry `raises` records. Not in
  the inventory's bug list. Task 1.2's exit criteria were patched to
  require replaying them.
- **The two `thermal_cross_section` implementations disagree above
  `x = 300`** — scalar returns `0.0`, vector saturates at `xnew = 300`.
  Both behaviors are pinned; Phase 05 must reproduce both or declare the
  unification.
- **Edge values are contract.** 26 negatives in `spectra.photon.muon`,
  123 negatives + 5 infinities in `cross_sections.scalar.sigma_xl_to_xl`.
  Stored as returned; do not "fix" them during a port without declaring
  it.
- **`--check`'s independence from a built tree is fragile.** It holds
  only because `cases.py` imports `HiggsPortal` / `KineticMixing` (and
  `hazma` itself, in `hazma_package_path`) inside functions rather than
  at module scope. Hoisting those imports would silently make the
  integrity check require a full build.
- **`import hazma` and `REPO_ROOT` are independent, and were not tied
  together until round-1 review.** The digest measures the repository;
  imports resolve through `sys.path`, which a site-packages install can
  shadow. `Case.resolve` now refuses any module whose `__file__` falls
  outside `REPO_ROOT`, and the manifest records where `hazma` actually
  resolved from (`hazma_package`). Task 1.2's runner gets the guard for
  free — it goes through `resolve()`.
- **Mechanism, not physics, sets a tolerance.** Task 1.2 classified all
  41 entry points by reading the live `.pyx`: 19 closed-form (exact), 1
  closed form through `spence` (1e-13), 7 tabulated boost integrals
  (1e-12), 5 single-`quad` (1e-8), 9 nested-`quad` (1e-6) — counts
  re-derived from `tolerances.BUDGETS`, not tallied by hand. Two of those
  would have been misfiled from the import list alone —
  `hazma/spectra/_photon/_muon.pyx` and
  `hazma/spectra/_positron/_muon.pyx` both `import quad` and never call
  it on the live path.
- **The nested class is not just ρ.** All seven mediator-spectrum entry
  points cimport a quad-backed pion kernel into a cos θ quad integrand,
  so they share ρ's subdivision sensitivity and its 1e-6 budget.
- **A budget is the wrong gate on the capturing tree.** There, the
  corpus pins an implementation against itself, so any difference is a
  regression and a 1e-8 budget would swallow it.
  `tolerances.effective_budget` therefore demands bit-equality whenever
  the kernel digest, toolchain and numerics libraries all match the
  manifest, and falls back to the declared budgets otherwise — which is
  also what keeps a Linux CI runner from failing an exactness claim it
  was never positioned to meet.
- **`spectra.photon.neutral_pion[rest]` has exactly one non-zero value
  and it is `inf`** — the rest-frame two-body delta. Worth knowing
  before using that block to check anything: a multiplicative
  perturbation of `inf` is invisible (found while negative-testing the
  runner, Task 1.2).
- **Widening `testpaths` would not have put the corpus in CI.** The test
  job installs non-editable, and `python -m pytest` from the repo root
  puts the source tree first on `sys.path`, so `import hazma` in CI
  resolves to a checkout with no compiled extensions in it. The existing
  jobs pass only because every in-package test is pure Python. Even if
  site-packages won, `cases.assert_module_is_repo_tree` would refuse it.
  Task 1.3 therefore had to change how CI installs, not only what it
  collects — a reinstall as editable, after the outside-the-repo import
  smoke test that is the only per-PR check of the *installed*
  distribution.
- **`pyproject.toml` outranks `setup.cfg` for pytest config.** The
  search order is `pytest.ini` → `pyproject.toml` → `tox.ini` →
  `setup.cfg`, so a re-added `[tool:pytest]` section would be silently
  ignored rather than winning. Read the `configfile:` line in the pytest
  header to see which file is live (Task 1.3).
- **The corpus does not survive a change of libm, and six of its points
  are not reproducible anywhere.** Task 1.3 wired the suite into CI,
  which ran the corpus off macOS/arm64 for the first time: macOS passes,
  all five Linux entries fail ~70-75 of 626 blocks. Most of that is
  last-bit `libc.math` noise (35 at ≤4.5 ulp), but **six are
  catastrophic-cancellation points** where the pinned value is one
  platform's rounding residue —
  `cross_sections.scalar.sigma_xl_to_xl[closed_resonance.mu]` is
  `-1.504e-02` on macOS and `+5.624e-07` on Linux, from identical
  Cython. No tolerance absorbs a sign flip. **This blocks Phases 04-06,
  not just CI**: a faithful Rust port with a different instruction order
  will also land elsewhere in that region, so those blocks gate nothing.
  Tracked in
  [`../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md),
  which ripens **before Phase 04** (Task 1.3).
- **The CI scoping added in Task 1.3 disabled the corpus everywhere, and
  stayed green doing it** (found in Task 1.4 while reading the PR's own
  job log). `PARITY: ${{ runner.os == 'macOS' && '' || '--ignore=test/parity' }}`
  cannot select the macOS branch: Actions' `&&`/`||` return values, the
  empty string is falsy, so `true && ''` collapses to `''` and the `||`
  then yields `--ignore=test/parity`. Every entry, macOS included, skipped
  `test/parity` from PR #52 until PR #53 fixed it. **Nothing went red,
  because removing a gate never does** — the tell was the job reporting
  `380 passed, 13 skipped` where a run including the corpus collects
  ~1019. Keep the non-empty value on the true branch of an Actions
  ternary, and verify a gate by watching it *run*, not by watching CI
  stay green.
- **The corpus passes on the macOS CI runner in budget mode** (Task 1.4,
  the first time this has been observed — the Task 1.3 measurement below
  predates the scoping commit). With `PARITY` empty the macOS entry
  reports `1005 passed, 14 skipped` against 1006/13 locally: exactly one
  test moves from passed to skipped, which is the documented budget-mode
  signature (`test_running_on_the_capturing_tree` skips when the
  toolchain differs from the manifest, and the declared per-function
  budgets are enforced instead of bit-equality). The job log does not
  print skip reasons, so that attribution is an inference from the
  arithmetic and the mechanism, not a quoted reason.
- **`EXACT_RTOL = 0.0` applies in budget mode too**, because
  `effective_budget` returns the *declared* budget off the capturing
  tree and the EXACT class declares zero. That is what turned 35 last-bit
  differences into failures. `provenance` already separates `platform`
  and `machine` from the kernel digest, so the class could tell "a
  different platform" from "a different implementation" — deliberately
  not changed in Task 1.3, since it would not have made Linux green on
  its own (Task 1.3).
- **`test_utils.py` exists in both roots** — `test/test_utils.py` and
  `hazma/form_factors/vector/test_utils.py` — and collecting them
  together does not trip pytest's import-file-mismatch check only
  because `test/` has no `__init__.py` while the `hazma` copy sits in a
  real package. Adding one under `test/` would break the merged
  collection (Task 1.3). **Task 1.4 hit the other half of this:** two
  files sharing a basename in different `test/` subdirectories *do*
  collide, and the collision aborts the whole run at collection rather
  than failing one module. `test/rh_neutrino/test_integration.py` had to
  become `test_rh_neutrino_integration.py` because
  `test/spectra/test_integration.py` already held the name.
- **The legacy `.npy` suites were not merely stale — they were
  structurally rotten** (Task 1.4), which is what settled the
  regenerate-vs-delete call on evidence:
  - Unskipped against the current tree, **11 of their 17 tests fail**.
    The differences are structured, not noisy: six scalar cross sections
    (`g g`, `pi0 pi0`, `pi pi`, `s s`, `total`) are off by *exactly* ×2,
    `partial_widths["e e"]` by ×4 — a superseded identical-particle
    symmetry-factor convention, not drift.
  - `test_scalar_mediator.py`'s `load_sm2_data` reads `sm1_dir` for all
    twelve arrays, so `sm_2` was never loaded and `sm2` was a duplicate
    of `sm1`.
  - The vector generator's `mvs = 2 * [125.0, 550.0]` makes `vm_5`/`vm_6`
    exact duplicates of `vm_3`/`vm_4`. Eight directories, six distinct
    parameter points, four distinct vector ones.
  - Every loader docstring misdescribes its own point (all six vector
    ones claim `mx = 250`, `eps = 0.1`; the stored params say
    `mx = 125` and four of them are `VectorMediator`, not
    `KineticMixing`). The scalar pair have their `ms` values swapped and
    wrong.
  - The generators still in the tree disagree with the data they
    produced (`mx = 250.0` for the vector points vs `125.0` stored), and
    write `gamma_ray_lines.npy` that no test reads.
- **The corpus stops where Cython stops** (Task 1.4). Everything in
  `hazma/theory/__init__.py` and the model packages — the dict assembly,
  the `"total"` sums, the branching-fraction division, the
  branching-fraction weighting of each spectrum, the line `bf` — is pure
  Python, and no corpus case reaches it. That is the non-redundant half
  of the deleted classes' intent and why `test/test_theory_aggregation.py`
  exists.
- **A total-is-the-sum check does not catch a lost weight** (Task 1.4,
  found by negative-testing rather than by reading). Multiplying *every*
  positron channel by `1.0` instead of its branching fraction leaves
  `total == sum(channels)` true, so the first draft of the suite passed
  the mutation. The per-channel `bf × kernel` identity is the assertion
  that fires; the total-sum check is not a substitute for it.
- **Both mediator positron kernels return `nan` at exactly `0.510998928`**
  (Task 1.4) — the legacy `MASS_E` in
  `hazma/_utils/legacy_parameters.pxd:18`, against `0.5109989461` in
  `constants.pxd` and `hazma/parameters.py`. One point, not a window: a
  2,000,001-point sweep of `[0.5109988, 0.5109990]` finds that single
  value, with `0.0` on both sides, for the scalar *and* vector kernels.
  The corpus does not pin it (zero `nan` across 19,610 pinned positron
  values), so a Rust port can land anywhere there and still pass. The
  constants divergence itself was already on the record
  (`../../references/cython-inventory.md` §Bugs item 3); this consequence
  was not. Filed as
  [`../../../../docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md`](../../../../docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md),
  ripening **before Phase 05/06**.
- **`Theory.spectra` and `Theory.positron_spectra` reject scalar
  energies** (Task 1.4) despite documenting `float or float numpy.array`
  and `AGENTS.md`'s arrays-in-arrays-out contract. Two causes:
  `spectra` calls `len()` on a float in a channel wrapper
  (`_scalar_mediator_spectra.py:20`), and `positron_spectra` hits the
  compiled `np.ndarray` signature. `total_spectrum` and
  `total_positron_spectrum` accept a scalar, which is why nobody has hit
  it. Filed as
  [`../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md`](../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md).

## Decisions and Implementation Notes

- Specification (`test/parity/cases.py`) and generator
  (`test/parity/generate.py`) are separate modules so Task 1.2's runner
  imports the call convention rather than re-deriving it — Task 1.1.
- Provenance is recorded twice: the git SHA rules.md rule 2 asks for
  (necessarily with `dirty: true`, since the generating commit does not
  exist yet) and a `kernel_digest` over all 44 `.pyx`/`.pxd`/photon-CSV
  files, which is what actually identifies the kernels — Task 1.1.
- The Cython version in the manifest is read off the generated `.c`
  header, not `importlib.metadata`: build isolation keeps Cython out of
  the runtime environment — Task 1.1.
- Raises are captured as `nan` plus a manifest record of index,
  argument and exception *type*; the message is deliberately omitted
  because Cython rewords its errors between releases — Task 1.1.
- Five parent energies per spectrum rather than the four required; the
  extra is `M·(1 + 1e-12)`, straddling the `E − M < DBL_EPSILON`
  rest-frame short-circuit — Task 1.1.
- The runner evaluates through `generate.evaluate_block` — the same
  function that produced the stored numbers — rather than its own loop,
  so a harness difference cannot masquerade as an implementation
  difference. The raise replay falls out of that: the function returns
  the same `{"index", "argument", "type"}` records the manifest stores,
  so one `==` catches both a swallowed raise and a new one — Task 1.2.
- One test per block (623), not per case or per array: a block is one
  grid at one fixed argument set, which is the granularity at which a
  failure is diagnosable — Task 1.2.
- `atol = 0.0` everywhere. One absolute floor cannot serve spectra at
  ~1e-3 MeV⁻¹ and cross sections at ~1e-20 MeV⁻²; it is also
  unnecessary, since the out-of-support regions return exactly `0.0`
  — Task 1.2.
- Abscissae (`grid`, `scalar_grid`) get their own budget, not the
  case's: no tolerance on a *value* compensates for having moved where it
  was measured — Task 1.2. Exact in **both** modes originally; Task 1.3
  split it to bit-exact on the capturing tree and 1e-13 elsewhere, after
  the first Linux CI run failed all 623 blocks by exactly one ulp because
  `numpy.geomspace` goes through the platform libm. The premise that
  "grids are arithmetic on constants" held within a platform and not
  across one.
- Parity in CI was scoped to the **capturing platform**: the `Run tests`
  step carried a `PARITY` env, empty on macOS and `--ignore=test/parity`
  elsewhere. `--ignore` rather than a marker, so the Linux entries also
  stopped paying the corpus's ~9 minutes. A workaround for the symptom,
  explicitly not a fix — the corpus follow-up was — and both the phase
  Exit Criteria and Task 1.3's exit criteria were amended to say so
  rather than left to be rediscovered — Task 1.3. **Reverted 2026-08-18**
  when that follow-up landed; see the Follow-ups table above.
- No measurement/reporting hook. `pytest_addoption` is only honored in
  an *initial* conftest, which `test/parity/conftest.py` is not under
  `pytest test`, and `assert_allclose` already prints the max relative
  difference on breach — so Phase 03's tightening loop is "set the
  budget you want and read the failure" — Task 1.2.
- `testpaths = ["hazma", "test"]`, two explicit roots rather than one:
  it keeps the bare command self-documenting and preserves the
  in-package `*_test.py` convention the form-factor and phase-space
  suites use — Task 1.3.
- `preflight.sh`'s `--tests` default is now **empty**, not `test`. A
  literal default is what drifts from `testpaths` the next time the
  collection changes; an empty one delegates to the config CI reads.
  `--tests` survives as an explicit narrowing for iteration — Task 1.3.
- No marker and no split job for the parity suite, closing the policy
  question Task 1.2 left open. A marker that must be opted into is a
  gate nobody runs, and a separate job would break the "CI and preflight
  run the same collection" criterion — Task 1.3.
- CI installs twice (non-editable, smoke test, then editable) rather
  than swapping to editable outright. The smoke test is the only per-PR
  check of the installed distribution, and a missing
  `[tool.setuptools.package-data]` entry is invisible from the source
  tree — Task 1.3.
- The legacy `.npy` suites are **deleted, not regenerated** — Task 1.4.
  Regeneration would have minted a second golden corpus with no manifest,
  no `--check`, no provenance and a 1e-4 tolerance, overlapping the real
  corpus on everything compiled. The evidence for "rotted, not merely
  stale" is under Findings; the two `generate_test_data.py` producers went
  with their data, following Task 0.3's precedent for `test/decay/`.
- The replacement (`test/test_theory_aggregation.py`) asserts
  **identities, not stored values** — Task 1.4. `total` is the channel
  sum; a branching fraction is a cross-section ratio; a spectrum is
  `bf × kernel`; a line's `bf` is its channel's. Plus three two-body
  kinematic closed forms (`e e` and `g g` at `e_cm/2`, `pi0 g` at
  `(e_cm² − m_π0²)/(2 e_cm)`). Identities need no data files, so they
  cannot rot the way the arrays did, and they hold bit-for-bit on every
  platform — making this the one numerical gate in the repo that is *not*
  scoped to the capturing platform. Re-pinning kernel numbers at a loose
  tolerance was rejected: that is the corpus's job.
- Sixteen of the suite's 21 test functions parametrize over four model
  points
  — Task 1.4 — two per model class straddling the mediator threshold
  (`s s` / `v v` open vs closed), with `gvdd` flipped on the vector pair.
  Deliberately *not* the eight the deleted data used: two of those were
  duplicates and one was unreachable.
- `test/rh_neutrino/widths.py` is **deleted, not renamed** — Task 1.4. It
  is a matplotlib plotting script under `if __name__ == "__main__"` with
  no assertions; renaming it into the collection would have imported
  matplotlib (not a test dependency) to run nothing.

## Files Changed

### Task 1.1

- `test/parity/cases.py`, `test/parity/generate.py`,
  `test/parity/README.md` — new.
- `test/parity/data/*.npz` (41) + `test/parity/data/manifest.json` — new.
- `../../phases/phase-01-parity-corpus.md` — Task 1.2 exit criteria
  gained the raise-replay bullet; round-1 review also flipped its
  frontmatter to `In Progress` and re-derived three stale claims
  (test count, `collect_ignore`, and Task 1.4's array count 159 -> 90).
- `docs/agents/lessons.md` — one new class, one citation added
  (round-1 review).
- This file and `../README.md` — status bookkeeping.

Nothing under `hazma/` was touched.

### Task 1.2

- `test/parity/test_parity.py`, `test/parity/tolerances.py` — new.
- `test/parity/cases.py` — `rust_core_available()` extracted from
  `assert_no_rust_core`.
- `test/parity/generate.py` — `load_manifest()` extracted from `check()`;
  `_sweep_pointwise` now reports an all-points-raised block instead of
  dying in `np.concatenate`.
- `test/parity/README.md` — the runner and its two comparison modes.
- `../../phases/phase-01-parity-corpus.md` — the Prerequisites context
  bullet and Task 1.3's `pytest -q test` figure re-derived.
- `task-1.1-corpus-generator.md` and
  `../../learnings/phase-00-dead-code-purge.md` — one forward-looking
  and one present-tense claim each, both falsified by this task,
  annotated in place.
- This file, `../README.md` and `task-1.2-parity-runner.md` — status
  bookkeeping.

Nothing under `hazma/` was touched.

### Task 1.3

- `pyproject.toml` — new `[tool.pytest.ini_options]` (`testpaths`,
  `markers`); `setup.cfg` — `[tool:pytest]` replaced by a pointer
  comment.
- `test/spectra/integration.py` → `test/spectra/test_integration.py`
  (`git mv`, plus the import reorder preflight's isort gate asked for).
- `.github/workflows/ci.yml` — editable reinstall before the test step.
- `scripts/agents/preflight.sh` — `--tests` defaults to empty; usage and
  the zero-collection FAIL message follow.
- `docs/agents/preflight.md`, `docs/agents/environment.md`, `AGENTS.md`,
  `test/parity/README.md` — the bare-`pytest` contract and the
  editable-install requirement.
- `.claude/skills/{execute-single-task,review-pr,review-plan}/SKILL.md` —
  the now-false "`setup.cfg` scopes it to `hazma`" claim in each.
- `../../phases/phase-01-parity-corpus.md` — Prerequisites re-derived;
  Task 1.3 exit criteria carry the realized counts and the
  editable-install constraint.
- This file, `../README.md` and `task-1.3-test-wiring.md` — status
  bookkeeping.

Nothing under `hazma/` was touched.

### Task 1.4

- Deleted, 96 files: `test/scalar_mediator/test_scalar_mediator.py` and
  `test/vector_mediator/test_vector_mediator.py` (the two skipped
  classes); their two `generate_test_data.py` producers;
  `test/scalar_mediator/data/` (24 `.npy`) and
  `test/vector_mediator/data/` (66 `.npy`) — the 90 arrays the phase file
  names; `test/positron/test_positron.py` (0 bytes, taking its directory);
  `test/rh_neutrino/widths.py` (a matplotlib plotting script, no
  assertions).
- Renamed: `test/rh_neutrino/integration.py` →
  `test/rh_neutrino/test_rh_neutrino_integration.py` — *not*
  `test_integration.py`, which collides with `test/spectra/`. No
  assertion changed; the rename pulled the file into the lint gate, so
  it also took the import reorder, two `-> None` annotations and two
  corrected docstrings (7 configured-ruff findings → 0).
- New: `test/test_theory_aggregation.py` — 21 test functions / 69
  collected over the pure-Python aggregation layer.
- New: `docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md`
  and `docs/followups/todo/model-spectra-reject-scalar-energies.md`, with
  their two rows in `docs/followups/README.md`.
- `../../phases/phase-01-parity-corpus.md` — frontmatter `status:
  Complete`; Prerequisites re-derived; Task 1.4's exit criteria carry
  their realized outcomes and the basename-collision constraint; the
  phase Exit Criteria carry the realized suite counts.
- `../../PLAN.md` — the Phases-table row for 01.
- `../../learnings/phase-01-parity-corpus.md` — **new**, the phase
  distillation.
- This file, `../README.md` and `task-1.4-legacy-npy.md` — status
  bookkeeping and phase closure.

Nothing under `hazma/` was touched.

## Verification

- Task 1.1: `python test/parity/generate.py` → `41 cases / 623 blocks /
  2937.3 KiB`; `--check` → `corpus OK: 41 cases / 1580 arrays`. Both
  existing suites unchanged from the Phase 00 baseline (`pytest -q` →
  `57 passed, 10 skipped`; `pytest -q test` → `244 passed, 20 skipped`).
  Full command log and the guard negative-tests are in the task note.
- Task 1.2: `pytest -q test/parity` → `626 passed`, all in exact
  (bit-equality) mode — the phase file's "running against unmodified
  Cython passes bit-exact" criterion, now a standing gate rather than an
  observation. Nine negative scenarios (behind three baselines) confirm
  each assertion fires: swallowed raise, wholesale raise, single-point
  raise, value shifts against both modes, grid drift, and the three
  budget-table guards. Command log in the task note. Suites on the final
  tree: `pytest -q test` → `870 passed, 20 skipped` (244 + 626, the
  pre-existing suite untouched); bare `pytest -q` → `57 passed, 10
  skipped`, unchanged because `testpaths = hazma` still scoped it.
- Task 1.3: bare `pytest -q` → `935 passed, 30 skipped` — the merged
  collection, parity suite included and in exact mode. Roots reconcile:
  `--collect-only -q` gives 965 total, 67 from `hazma` and 898 from
  `test` (890 pre-existing + the 8 the `test_integration.py` rename
  un-hid). Task 1.2's two baselines re-measured unchanged on this tree
  before the change (`pytest -q test` → `870 passed, 20 skipped`).
  `preflight.sh` with no `--tests` runs that same bare command; full log
  in the task note.
- Task 1.4 (2026-08-08): bare `pytest -q` → `1006 passed, 13 skipped in 582.63s`,
  parity suite included and in exact mode. Roots reconcile:
  `--collect-only -q` gives 1019 total, 67 from `hazma` and 952 from
  `test`. Against Task 1.3's 935/30 the delta is +69 aggregation tests,
  +2 from the `rh_neutrino` rename, and −17 skips (9 scalar + 8 vector)
  as the legacy classes left; 935 + 71 = 1006 and 30 − 17 = 13. The new
  suite alone: `pytest -q test/test_theory_aggregation.py` → `69 passed`
  in 0.56s. Eleven implementation mutations confirm each assertion class
  fires — one of them (dropping the positron branching-fraction weight)
  passed against the first draft and is why the suite has a per-channel
  positron identity as well as a total-is-the-sum check. `python
  test/parity/generate.py --check` still reports `corpus OK: 41 cases /
  1580 arrays`. Full tables in the task note.
- **Phase closed 2026-08-08.** Read
  `../../learnings/phase-01-parity-corpus.md`.

## Open Questions

- Whether every stored value is bit-reproducible on the Linux CI matrix
  is still **unmeasured** — the corpus was captured on macOS/arm64 and
  Task 1.2 had no Linux runner to answer on. It is no longer a *risk*,
  though: a runner whose platform differs from the manifest drops to
  budget mode by construction, so CI is gated on the declared budgets
  rather than on an exactness claim nobody has evidence for. Task 1.3
  wired CI, so the PR that lands it produces the first Linux numbers;
  the plausible outcome is that the transcendental-libm kernels
  (`exp`/`log`/`spence`) differ in the last ulp, well inside every
  declared budget. Read the CI log for
  `test_running_on_the_capturing_tree`'s skip reason — that names what
  differed — before treating any Linux failure as a real drift.
- ~~Task 1.4's scope narrowed but is not decided~~ — **closed
  2026-08-08: deleted, not regenerated.** The overlap with the corpus was
  the smaller half of the answer; what settled it was that the arrays
  encode a superseded convention (six scalar cross sections off by
  exactly ×2), `vm_5`/`vm_6` duplicate `vm_3`/`vm_4`, `sm_2` was never
  loaded, and 11 of the 17 tests fail against the current tree. The
  non-redundant half of their intent — the pure-Python aggregation layer
  — moved to `test/test_theory_aggregation.py` as identities. Evidence in
  `task-1.4-legacy-npy.md`.
- ~~Two more uncollected modules surfaced in Task 1.3~~ — **closed
  2026-08-08.** `test/rh_neutrino/integration.py` passes as-is and was
  renamed into the collection as
  `test_rh_neutrino_integration.py`; `test/rh_neutrino/widths.py` is a
  matplotlib plotting script with no assertions and was deleted.
  (`test/spectra/msqrd_corpus.py` remains deliberate — it is a fixture
  module `test_dnde_photon_fsr.py` imports by name.)
- Five skip **marker sites** survive the phase with reasons outside Task
  1.4's criterion: three in `test/vector_mediator/test_form_factors.py`
  (:137, :195, :230) and two "Known to be broken" under
  `hazma/form_factors/vector/`. They account for all 13 skipped **tests**
  the suite reports — 5 + 5 + 3, the `hazma/` pair sitting on
  parametrized classes, so marker count and skip count are not the same
  number. (`test/agents/test_resolve_phase.py:47`'s `skipif` does not
  fire while its script is present.) All are pure-Python form-factor
  issues this project does not port. Recorded so the silence is not
  mistaken for coverage.

## Plan Impact

**Impact Level:** Update phase file (Tasks 1.1, 1.2, 1.3 and 1.4).

- Task 1.1: `../../phases/phase-01-parity-corpus.md`'s Task 1.2 exit
  criteria gained a bullet requiring the runner to replay the manifest's
  `raises` records rather than compare the stored `nan`.
- Task 1.2: the same file's Prerequisites context bullet said "zero
  pinned-value tests over compiled code execute anywhere ... Task 1.1
  added `test/parity/` but no pytest module, so this still holds". Half
  of that is now false — `pytest test` reaches the parity suite; CI does
  not, which is Task 1.3's job. Re-derived along with Task 1.3's
  `pytest -q test` figure, which this task moved.
- Task 1.3: the Prerequisites bullet moved to past tense and, at the
  time, named Task 1.4 as what was still open (Task 1.4 has since closed
  it). Its own exit
  criteria carry the realized counts (bare `pytest -q` → 935/30, and
  934/31 expected off the capturing environment, replacing the 870/20
  and 869/21 figures that described the `test/` root alone) and a new
  clause: widening `testpaths` is not sufficient, because CI's
  non-editable install leaves no extension in the tree
  `cases.assert_module_is_repo_tree` insists on. That constraint was not
  in the plan.

- Task 1.4: the phase file's frontmatter went `In Progress` →
  `Complete`; the Prerequisites bullet's remaining present-tense claim
  (`test/positron/test_positron.py` "is 0 bytes") and its "what is still
  open" sentence were re-derived; Task 1.4's exit criteria gained their
  realized outcomes, the two `rh_neutrino` modules Task 1.3 folded in,
  and a constraint the plan did not anticipate — the rename cannot use
  `test_integration.py`, because a duplicate basename across `test/`
  subdirectories aborts the entire collection. The phase Exit Criteria's
  first bullet carries the realized suite counts, and `../../PLAN.md`'s
  Phases-table row for 01 is marked Complete.

No ADR: nothing about the port's architecture, interfaces or ordering
changed. The tolerance table is a new contract, but the phase file
already specified it.

## Handoff to Next Task

**Phase 01 is Complete (2026-08-08). The next work is Phase 02, Task 2.1
— the Rust scaffold.** Read
[`../../learnings/phase-01-parity-corpus.md`](../../learnings/phase-01-parity-corpus.md)
rather than this file or the four task notes: it is the distillation,
they are history. Then `../../phases/phase-02-rust-scaffold.md` and
`../../rules.md`.

**Currently safe to assume:**

- **One command is the suite.** Bare `pytest -q` → `1006 passed, 13
  skipped` on the capturing environment (1019 collected: 67 `hazma` +
  952 `test`); `preflight.sh` with no `--tests` runs exactly that, and so
  does CI. Any narrower run covers strictly less than the gate. Build
  editable first (`uv pip install -e .`) — the parity suite refuses a
  `hazma` resolving outside the repository.
- The corpus reproduces bit-exactly on the capturing environment (exact
  mode, inside that 1006), and `python test/parity/generate.py --check`
  re-verifies its integrity without a built tree.
- Coverage of the 41 entry points does not need re-deriving —
  `assert_full_coverage` does it on every generation, and
  `test_every_corpus_case_has_a_budget` does the same for the tolerance
  table.
- `cases.py` is the single source of the call convention, and
  `generate.evaluate_block` the single source of the evaluation path.
  Both are already reused by the runner; do not fork either.
- The corpus pins the **post-fix** `two_body_momentum` values.
- Widening a budget is a declared act, not a fix: `tolerances.py`'s
  module docstring states rules 2 and 3 at the point of use.
- The parity suite's cost is **settled policy**: paid in full on every CI
  matrix entry and every preflight run, no marker and no split job.
  Task 1.2 left the question open; Task 1.3 closed it. Reopening it needs
  a CI measurement, not a local one.
- **`test/test_theory_aggregation.py` is the Phase 04–06 wiring gate**
  and the corpus's complement: it fires on a lost branching-fraction
  weight, a dropped channel, a detached line `bf` or a broken `total`,
  none of which the corpus sees, and it costs 0.6s. Run it either side of
  every kernel swap.
- **No skipped test in the repo is waiting on this project.** The five
  survivors are pure-Python form-factor issues the port does not touch.
- **`test/` holds no golden `.npy` corpus for the mediator models any
  more.** `test/parity/data/` is the only pinned-value store, and it has
  a manifest and a `--check`.

**Currently risky / unknown:**

- ~~**Read
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/done/parity-corpus-pins-ill-conditioned-points.md)
  before Phase 04.**~~ — **closed 2026-08-18.** Read
  [`followup-parity-corpus-stability.md`](followup-parity-corpus-stability.md)
  before Phase 05 instead: the affected blocks were four cross sections
  rather than six assorted ones, they gate nothing for the port, and
  `test/parity/stability.py` is what stops that reading as a regression.
- Two Task 1.4 follow-ups ripen inside this project: the
  [`MASS_E` `nan`](../../../../docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md)
  before Phases 05/06, and the
  [scalar-energy contract](../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  during 04–06.
- Do not hoist `cases.py`'s deferred model imports.
- **Test-file basenames must be unique across `hazma/` and `test/`
  together**, and do not add an `__init__.py` under `test/` to fix a
  collision — that breaks the merged collection instead. Both halves were
  measured (Tasks 1.3 and 1.4).
- Corpus grid density vs. repo-size budget is **settled**: 2.9 MiB
  against a ~10 MB ceiling, with `MAX_TOTAL_BYTES` failing generation
  above it.
- The aggregation suite's four model points are a sample, not a sweep.
  Widening it is one list (`_models()`) if a swap ever suggests the need.
