# Working Memory: Phase 01 — Golden parity corpus

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 01
**Status:** In Progress (Tasks 1.1-1.2 complete 2026-08-07)
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
| 1.3 | Wire both suites into one gate | 1.2 | Not started — **next** | [task-1.3-test-wiring.md](task-1.3-test-wiring.md) |
| 1.4 | Retire/regenerate legacy `.npy` suites | 1.2 | Not started | [task-1.4-legacy-npy.md](task-1.4-legacy-npy.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-01-parity-corpus.md`.

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
- Abscissae (`grid`, `scalar_grid`) are compared exactly in **both**
  modes. No tolerance on a value compensates for having moved where it
  was measured — Task 1.2.
- No measurement/reporting hook. `pytest_addoption` is only honored in
  an *initial* conftest, which `test/parity/conftest.py` is not under
  `pytest test`, and `assert_allclose` already prints the max relative
  difference on breach — so Phase 03's tightening loop is "set the
  budget you want and read the failure" — Task 1.2.

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
  skipped`, unchanged because `testpaths = hazma` still scopes it.
- Remaining: bare `pytest` green incl. the parity suite (Task 1.3).

## Open Questions

- Whether every stored value is bit-reproducible on the Linux CI matrix
  is still **unmeasured** — the corpus was captured on macOS/arm64 and
  Task 1.2 had no Linux runner to answer on. It is no longer a *risk*,
  though: a runner whose platform differs from the manifest drops to
  budget mode by construction, so CI is gated on the declared budgets
  rather than on an exactness claim nobody has evidence for. Task 1.3
  wires CI and produces the first Linux numbers; the plausible outcome
  is that the transcendental-libm kernels (`exp`/`log`/`spence`) differ
  in the last ulp, well inside every declared budget.
- Task 1.4's scope narrowed but is not decided: the 90 `.npy` arrays
  the two skipped mediator classes read overlap the cross-section and
  mediator-spectrum cases now pinned here, so the
  redundant-vs-complementary call has a concrete comparison target.
  (The phase file said 159; that was a collision with Task 0.2's
  unrelated 159-array impact check. Re-derived in the round-1 review
  fixes — `find test/{scalar,vector}_mediator/data/{sm,vm}_* -name
  '*.npy' | wc -l` → 90.)

## Plan Impact

**Impact Level:** Update phase file (Tasks 1.1 and 1.2).

- Task 1.1: `../../phases/phase-01-parity-corpus.md`'s Task 1.2 exit
  criteria gained a bullet requiring the runner to replay the manifest's
  `raises` records rather than compare the stored `nan`.
- Task 1.2: the same file's Prerequisites context bullet said "zero
  pinned-value tests over compiled code execute anywhere ... Task 1.1
  added `test/parity/` but no pytest module, so this still holds". Half
  of that is now false — `pytest test` reaches the parity suite; CI does
  not, which is Task 1.3's job. Re-derived along with Task 1.3's
  `pytest -q test` figure, which this task moved.

No ADR: nothing about the port's architecture, interfaces or ordering
changed. The tolerance table is a new contract, but the phase file
already specified it.

## Handoff to Next Task

**For the next agent working in Phase 01 (Task 1.3):** read
`test/parity/README.md` (now covers the runner and its two comparison
modes), then `tolerances.py`'s module docstring, then
`task-1.2-parity-runner.md`'s Findings and Handoff.

**Currently safe to assume:**

- The corpus reproduces bit-exactly on the capturing environment:
  `pytest -q test/parity` → `626 passed`, exact mode. Task 1.3 is
  wiring, not repair.
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

**Currently risky / unknown:**

- The parity suite costs ~4.6 min. Task 1.3 puts that on every CI matrix
  entry; if that is unacceptable, the decision (marker, split job) is
  Task 1.3's to make and to record — Task 1.2 deliberately did not
  invent a policy.
- Do not hoist `cases.py`'s deferred model imports.
- Corpus grid density vs. repo-size budget is **settled**: 2.9 MiB
  against a ~10 MB ceiling, with `MAX_TOTAL_BYTES` failing generation
  above it.
- Cross-platform behavior, per Open Questions — expect budget mode and a
  skipped `test_running_on_the_capturing_tree` on CI, not a failure.
