# Phase 01 Learnings: Golden parity corpus

Read this instead of the four task notes. They are history; this is the
distillation. Phase closed 2026-08-08 (Tasks 1.1–1.4, PRs #50, #51, #52
and the Task 1.4 PR).

## 1. Implementation Reality Check

The phase delivered what it promised — pinned reference arrays for all 41
consumed compiled entry points, a runner with per-function tolerance
budgets, one pytest gate that CI and `preflight.sh` both run — and
discovered one thing the plan did not anticipate, which is now the phase's
most load-bearing output.

**The corpus does not survive a change of libm.** It was captured on
macOS/arm64 and reproduces bit-exactly there. On the Linux CI matrix
~70–75 of its 623 blocks fail. Most of that is last-bit noise (35 at ≤4.5
ulp), but **six are catastrophic-cancellation points** where the pinned
value is one platform's rounding residue:
`cross_sections.scalar.sigma_xl_to_xl[closed_resonance.mu]` is `-1.504e-02`
on macOS and `+5.624e-07` on Linux, from identical Cython. No tolerance
absorbs a sign flip.

That is not a CI problem — it is a **gate** problem. A faithful Rust port
with a different instruction order will land somewhere else in that region
too, so those six blocks gate nothing for Phases 04–06. CI works around
the symptom (`--ignore=test/parity` off the capturing platform); the fix is
[`docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md`](../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md),
which **ripens before Phase 04**. Read it before writing any Rust.

The phase also shipped that workaround broken and did not notice for two
PRs. The Actions expression meant to scope the corpus to macOS instead
disabled it on *every* entry, macOS included, from PR #52 until PR #53 —
see the Actions-ternary entry under Quirk Log. With that fixed, the corpus
now runs on the macOS entry and **passes in budget mode**: `1005 passed,
14 skipped` against 1006/13 locally, the one-test delta being
`test_running_on_the_capturing_tree` skipping because the runner's
toolchain differs from the manifest. That is the first time the phase's
"green in CI with the corpus included" claim has actually been observed
rather than assumed.

No ADR came out of the phase. Nothing about the port's architecture,
interfaces or ordering changed; the tolerance table is a new contract, but
the phase file already specified it.

## 2. Critical Context for Future Work

- **The corpus is the gate, and it is derived, not transcribed.**
  `test/parity/cases.py` is the single source of every entry point's call
  convention; `generate.evaluate_block` is the single source of the
  evaluation path. The runner reuses both, so a harness difference cannot
  masquerade as an implementation difference. Do not fork either.
- **Coverage re-derives itself.** `assert_full_coverage` walks the
  surviving `.pyx` for top-level `def`s and fails both ways — an unpinned
  `def`, or a case naming a `def` that no longer exists. Phases 04–06
  therefore cannot delete a Cython module without the corpus objecting, and
  no later task needs to re-count the 41.
- **`assert_no_rust_core` enforces `rules.md` rule 2 in code.** The corpus
  may never be regenerated from a tree where any kernel runs on Rust.
- **Provenance is the kernel digest, not the git SHA.** The manifest's SHA
  is whatever was HEAD at generation (always `dirty: true`); the digest
  over the 44 `.pyx`/`.pxd`/CSV files is what certifies the bytes the
  values came from. `f5e6e269be47` is the pre-port digest.
- **Exactness is conditional, by construction.**
  `tolerances.effective_budget` demands bit-equality when the kernel
  digest, toolchain and numerics libraries all match the manifest, and
  falls back to the declared budgets otherwise. On the capturing tree the
  corpus pins an implementation against itself, so any difference is a
  regression and a 1e-8 budget would swallow it.
- **Mechanism sets a tolerance, not physics.** The 41 entry points classify
  as 19 closed-form (exact), 1 closed-form through `spence` (1e-13), 7
  tabulated boost integrals (1e-12), 5 single-`quad` (1e-8), 9
  nested-`quad` (1e-6). All seven mediator-spectrum entry points cimport a
  quad-backed pion kernel into a cos θ quad integrand, so they share ρ's
  subdivision sensitivity and its 1e-6 budget. Two would have been misfiled
  from the import list alone: both `_muon.pyx` files `import quad` and
  never call it on the live path.
- **Widening a budget is a declared act.** `tolerances.py`'s module
  docstring states `rules.md` rules 2 and 3 at the point of use.
- **The corpus stops where Cython stops.** Everything above it —
  `hazma/theory/`'s dict assembly, the `"total"` sums, the
  branching-fraction division, the branching-fraction weighting of each
  spectrum, the line `bf` — is pure Python and no corpus case reaches it.
  `test/test_theory_aggregation.py` (Task 1.4) covers that layer, and a
  Phase 04–06 swap can break it while the corpus stays green.
- **The public numbers the port must reproduce are the post-fix ones.** The
  corpus pins `two_body_momentum`'s factored form, which landed
  out-of-band before Phase 01 started.

## 3. Quirk Log & Edge Cases

- **Two entry points raise rather than return at a kinematic edge.**
  `sigma_xx_to_v_to_pipi` and `sigma_xx_to_v_to_pi0v` raise `TypeError` at
  exactly `e_cm = 2·mx` (Cython refusing a complex `**0.5`); the scalar
  siblings do not. Pinned as `nan` plus a manifest `raises` record, which
  the runner replays — a runner that only compared the stored `nan` would
  pass against an implementation that silently returned a number there.
  Not in the inventory's bug list.
- **The two `thermal_cross_section` implementations disagree above
  `x = 300`.** Scalar returns `0.0`; vector clips to `xnew = 300` and keeps
  evaluating. Both are pinned. **Phase 05 must reproduce both or declare
  the unification** — a shared Rust helper is the obvious design and would
  silently move published numbers.
- **Edge values are contract, including the ugly ones.** 26 negatives in
  `spectra.photon.muon`; 123 negatives + 5 infinities in
  `cross_sections.scalar.sigma_xl_to_xl`. Stored as returned. Do not "fix"
  one during a port without declaring it.
- **`spectra.photon.neutral_pion[rest]` has exactly one non-zero value and
  it is `inf`** — the rest-frame two-body delta. A multiplicative
  perturbation of `inf` is invisible, so that block cannot negative-test
  anything.
- **The `MASS_E` divergence has a measurable consequence.** Both mediator
  positron kernels return `nan` at *exactly* `0.510998928` — the legacy
  `MASS_E` in `hazma/_utils/legacy_parameters.pxd:18`, against
  `0.5109989461` everywhere else. One point, not a window: a 2,000,001-point
  sweep of `[0.5109988, 0.5109990]` finds that one value and `0.0` on both
  sides. The corpus does not pin it (zero `nan` across 19,610 pinned
  positron values), so a Rust port can land anywhere there and still pass.
  Filed as
  [`docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md`](../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md).
- **An Actions ternary cannot return the empty string on its true
  branch.** `&&`/`||` yield values, not booleans, and `''` is falsy, so
  `cond && '' || 'X'` evaluates to `'X'` for *both* outcomes of `cond`.
  The corpus's macOS scoping was written that way and silently skipped
  `test/parity` everywhere for two PRs. Keep the non-empty value on the
  true branch (`cond != X && 'flag' || ''`).
- **A change that removes a gate cannot turn CI red, so green proves
  nothing about it.** The disabled corpus above was caught by comparing
  the job's reported test count (`380 passed`) against what a run
  including the suite collects (~1019), not by any check failing. After
  enabling, disabling, or scoping a gate, read the run's counts or its
  env echo and confirm the gate *ran*.
- **`--check`'s independence from a built tree is fragile.** It holds only
  because `cases.py` imports `HiggsPortal` / `KineticMixing` (and `hazma`
  itself) *inside* functions. Hoisting those imports would silently make
  the integrity check require a full build.
- **`import hazma` and `REPO_ROOT` are independent.** The digest measures
  the repository; imports resolve through `sys.path`, which a site-packages
  install can shadow. `Case.resolve` now refuses any module whose
  `__file__` falls outside `REPO_ROOT`.
- **`pyproject.toml` outranks `setup.cfg` for pytest config** (search order
  `pytest.ini` → `pyproject.toml` → `tox.ini` → `setup.cfg`), so a
  re-added `[tool:pytest]` section is silently ignored rather than winning.
  Read the `configfile:` line in the pytest header to see which is live.
- **Test-file basenames must be unique across the whole collection.**
  `test/` has no `__init__.py`, so two files sharing a basename in
  different subdirectories collide on pytest's import-file-mismatch check
  and abort the *entire* run at collection. This bit twice: `test_utils.py`
  already exists in both roots (it survives only because the `hazma` copy
  sits in a real package), and Task 1.4's obvious
  `test/rh_neutrino/test_integration.py` collided with
  `test/spectra/test_integration.py` — hence
  `test_rh_neutrino_integration.py`. Do not add an `__init__.py` under
  `test/` to "fix" this; it breaks the merged collection instead.
- **`EXACT_RTOL = 0.0` applies in budget mode too**, because
  `effective_budget` returns the *declared* budget off the capturing tree
  and the EXACT class declares zero. That is what turned 35 last-bit Linux
  differences into failures. `provenance` already separates `platform` and
  `machine` from the kernel digest, so the class *could* tell "a different
  platform" from "a different implementation" — deliberately not changed,
  since it would not have made Linux green on its own.
- **Grids are not arithmetic on constants across platforms.** The abscissa
  budget was exact in both modes until the first Linux CI run failed all
  623 blocks by exactly one ulp: `numpy.geomspace` goes through the
  platform libm. Now bit-exact on the capturing tree, 1e-13 elsewhere.

## 4. Test Infrastructure State

- **One command is the suite.** Bare `pytest` — `pyproject.toml`'s
  `testpaths = ["hazma", "test"]` — is what CI runs and what
  `preflight.sh` runs with no `--tests`. Any narrower run covers strictly
  less than the gate. On the capturing environment at phase close:
  `1006 passed, 13 skipped` from 1019 collected (67 `hazma` + 952 `test`).
  Re-derive rather than quoting.
- **Build editable first** (`uv pip install -e .`). The parity suite
  refuses a `hazma` resolving outside the repository, and CI's non-editable
  `pip install .` leaves no extension inside the checkout — which is why
  the test job reinstalls editable after the outside-the-repo import smoke
  test, the only per-PR check of the *installed* distribution.
- **`python test/parity/generate.py --check`** re-verifies the corpus in
  under a second and needs no built tree.
- **The parity suite's cost is settled policy**: paid in full on every CI
  matrix entry and every preflight run, no marker and no split job. A
  marker that must be opted into is a gate nobody runs, and a separate job
  would break "CI and preflight run the same collection". Measured ~9
  minutes for the bare run on the capturing machine. Reopening the question
  needs a CI measurement, not a local one.
- **One test per block, not per case or per array.** A block is one grid
  at one fixed argument set — the granularity at which a failure is
  diagnosable. 41 cases / 623 blocks, collecting as 626 tests (the three
  extra are the suite's own guards).
- **`atol = 0.0` everywhere.** One absolute floor cannot serve spectra at
  ~1e-3 MeV⁻¹ and cross sections at ~1e-20 MeV⁻²; it is also unnecessary,
  since out-of-support regions return exactly `0.0`.
- **No measurement/reporting hook.** `pytest_addoption` is only honored in
  an *initial* conftest, which `test/parity/conftest.py` is not under
  `pytest test`, and `assert_allclose` already prints the max relative
  difference on breach. Phase 03's tightening loop is therefore "set the
  budget you want and read the failure".
- **`test/test_theory_aggregation.py` is the model-layer gate**, and it is
  built from *identities* (total is the channel sum; a branching fraction
  is a cross-section ratio; a spectrum is `bf × kernel`; a line's `bf` is
  its channel's) plus three two-body kinematic closed forms. No golden
  arrays, so it cannot rot the way the `.npy` corpora it replaced did, and
  it holds bit-for-bit on every platform — the one numerical gate in the
  repo that is *not* scoped to the capturing platform.
- **A golden-array suite rots silently, and the rot is invisible from a
  directory listing.** The 90 `.npy` arrays Task 1.4 deleted looked like
  coverage for years while gating nothing: 11 of their 17 tests failed
  against the current tree (six scalar cross sections off by exactly ×2 —
  a superseded symmetry-factor convention), `vm_5`/`vm_6` duplicated
  `vm_3`/`vm_4` through a generator bug, the scalar class loaded `sm_1`
  twice so `sm_2` was never read, every docstring misdescribed its own
  parameter point, and `gamma_ray_lines.npy` was written by the generator
  and read by nothing. When a corpus needs pinned numbers, pin them with
  provenance and a `--check` (as `test/parity/` does); when identities will
  do, prefer identities.

## 5. Follow-on seeds

- **The corpus pins six points nothing can reproduce** —
  [`todo/parity-corpus-pins-ill-conditioned-points.md`](../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md).
  Ripens **before Phase 04**: those blocks gate nothing for the port, not
  just for CI, and the fix (re-siting or re-conditioning the abscissae) is
  also what lets the phase's "green on all matrix entries" Exit Criterion
  be restored.

- **Mediator positron spectra return `nan` at exactly the legacy
  `MASS_E`** —
  [`todo/positron-spectrum-nan-at-legacy-electron-mass.md`](../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md).
  Ripens **before Phase 05/06**. `rules.md` rule 4 says ported code uses
  the exact constant its Cython source used, so the port either reproduces
  the `nan` or the divergence is consolidated — and that is a declared
  numerical change either way. Deciding after the swap costs a second one.

- **`Theory.spectra` and `Theory.positron_spectra` reject the scalar
  energies their docstrings advertise** —
  [`todo/model-spectra-reject-scalar-energies.md`](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md).
  Two different causes: `spectra` dies in pure Python on a `len()` of a
  float, `positron_spectra` dies at the Cython `np.ndarray` boundary. The
  second resolves itself if Phase 04–06 normalizes at the public boundary,
  which makes this cheapest to settle *during* the port rather than after.
