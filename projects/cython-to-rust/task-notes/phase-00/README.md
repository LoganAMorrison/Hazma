# Working Memory: Phase 00 — Dead-code purge

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 00
**Status:** In Progress (Tasks 0.1, 0.3 Complete)
**Plan References:** `../../phases/phase-00-dead-code-purge.md`
**Related ADRs:** ADR-0003 (accepted 2026-08-04 — Task 0.2 unblocked;
Task 0.5 still owes the execution steps)
**Depends On:** none

## Objective

Track live per-task status and phase-scoped findings for the dead-code
purge.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 0.1 | Relocate legacy constants header | — | Complete | [task-0.1-relocate-constants.md](task-0.1-relocate-constants.md) |
| 0.2 | Delete phase-space / gamma-ray slice | 0.1 (ADR-0003 accepted 2026-08-04) | Not started | [task-0.2-delete-mc-slice.md](task-0.2-delete-mc-slice.md) |
| 0.3 | Delete superseded kernels + helpers | 0.1 | Complete | [task-0.3-delete-superseded.md](task-0.3-delete-superseded.md) |
| 0.4 | Prune build and packaging config | 0.2, 0.3 | Not started | [task-0.4-prune-build.md](task-0.4-prune-build.md) |
| 0.5 | Execute ADR-0003 (`gamma_ray`) — ratified 2026-08-04 | — | In Progress (ADR accepted; docs repoint pending) | [task-0.5-gamma-ray-decision.md](task-0.5-gamma-ray-decision.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-00-dead-code-purge.md`.

## Inputs Reviewed

- `../../phases/phase-00-dead-code-purge.md`; `../README.md`;
  `../../references/cython-inventory.md` (dead-code map).

## Findings

- **Build hygiene (load-bearing for every task in this project).** A
  worktree can carry stale generated `.c`/`.cpp` from another
  environment; their mtimes suppress re-cythonization and the build
  fails deep in generated code (seen in Task 0.1 as `no member named
  'subarray' in '_PyArray_Descr'` from a `.cpp` generated against an
  older NumPy). Always run
  `find hazma -name '*.c' -o -name '*.cpp' -o -name '*.so' | xargs rm -f`
  before `uv pip install -e .`. The tree builds clean on Cython 3.2.9 /
  NumPy 2.5.1 once you do.
- **`_gamma_ray/gamma_ray_generator.pyx` compiles but has never been
  importable** — `from hazma import rambo` (line 11) and
  `hazma/rambo.py` does not exist. True on unmodified `master`. It is
  nevertheless a live `Extension` in `setup.py`, so it must keep
  *compiling* until Task 0.2 removes it; exclude it from import smoke.
- **Constants-header include sites: seven, not four** (Task 0.1). The
  four live mediator ones, plus `_gamma_ray/gamma_ray_generator.pyx`
  (built → build-breaking if skipped), plus two unbuilt `_decay/*.pyx`.
  The phase file's Task 0.1 criterion was patched accordingly.
- **The mediator cross-section `.pyx` include nothing** — closes the
  project-level open question. `_c_scalar_mediator_cross_sections.pyx`
  and `_c_vector_mediator_cross_sections.pyx` carry no `include`
  directive, so Task 6.4 has no constants-header entanglement there.
- **`git rm -r` on a package leaves it importable** (Task 0.3). An
  untracked `__pycache__` keeps the directory on disk, and an empty
  directory on `sys.path` is a *namespace package* — `import
  hazma.field_theory_helper_functions` still succeeded right after the
  `git rm`. `rm -rf` the directory as well, then re-run the negative
  import check. Any verify-after-delete that only inspects the git index
  will miss this.
- **Deleting a `.pyx` forces the matching `setup.py` edit in the same
  task** (Task 0.3). `pip install -e .` fails immediately on an
  `Extension` whose source is gone, so a deletion task cannot defer its
  own extension groups to Task 0.4. Same shape as Task 0.1's fifth
  include site. `test/conftest.py` has the same property for
  `test/decay/`: its `iterdir()` is unconditional and raises at
  *collection*, taking the whole suite with it.
- **The two `common_functions` twins are algebraically identical but not
  numerically identical** (Task 0.3). `cross_section_prefactor` differs
  by ≤2.1e-7 relative within 1e-7 of threshold (`kallen_lambda`
  cancellation vs the deleted factored form) and ≤5e-15 elsewhere;
  `minkowski_dot` by ≤2.7e-14 (the C compiler contracts `a*b - c*d`
  into an FMA). Both declared under "Numerical impact so far" in
  `../README.md`.
- **`hazma/experimental/axial_vector_mediator/__init__.py` is broken on
  the trunk** — `from hazma.theory import Theory`, but `hazma.theory`
  exports `TheoryAnn` / `TheoryDec`. Pre-existing; exclude the package
  from import smoke and load `avm_msqrd.py` by path if you need it.

## Decisions and Implementation Notes

- Task 0.1 repointed all seven `.pyx`/`.pxd` include sites but left
  `_decay/_decay_charged_pion.pyx.bak` — a `.bak` is a frozen artifact
  Cython can never compile (`cythonize()` is called only on explicit
  `Extension` objects, never a glob). Task 0.3 deletes the directory.
- Task 0.1 added a PROVENANCE paragraph to the relocated header's
  docstring (values byte-identical) so the deliberate divergence from
  `constants.pxd` is visible in-file and not "cleaned up" silently.
- Task 0.3 put `minkowski_dot` in `hazma/utils.py` (beside
  `cross_section_prefactor` and its array twin `ldot`) but deliberately
  **did not** add it to `docs/source/utils.rst` — it is a relocation of
  an existing public name, and `PLAN.md` Scope forbids API additions.
- Task 0.3 kept `boost_jac` / `boost_eng` in `_utils/boost.pyx`. They
  have zero cimporters, but unlike the three functions it deleted they
  *are* declared in `boost.pxd`, i.e. published C-level API. Whether
  they die with the rest of the header is a Task 6.4 call.
- Task 0.3 did **not** repair `cross_section_prefactor`'s threshold
  cancellation. Switching `hazma.utils` to the factored form would move
  `hazma.phase_space.PhaseSpace.cross_section`, a live public API — a
  declared numerical change that does not belong inside a dead-code
  deletion. Filed as
  [`docs/followups/todo/cross-section-prefactor-threshold-cancellation.md`](../../../../docs/followups/todo/cross-section-prefactor-threshold-cancellation.md)
  with the measured magnitudes.
- Task 0.3's notebook boundary: repoint the references *this task* made
  dangle; leave the ~20 `hazma.decay` / `hazma.positron_spectra`
  imports that were already dead on the trunk (those names only existed
  via the commented-out alias block in `hazma/__init__.py`).

## Files Changed

### Task 0.1

- `hazma/_decay/parameters.pxd` → `hazma/_utils/legacy_parameters.pxd`
  (git rename; PROVENANCE note added, constant definitions untouched).
- Include repointed in seven `.pyx`: both `scalar_mediator/*_spec*`,
  both `vector_mediator/*_spec*`, `_gamma_ray/gamma_ray_generator`,
  `_decay/decay_electron`, `_decay/_decay_muon_bak`.
- `projects/cython-to-rust/phases/phase-00-dead-code-purge.md` — Task
  0.1 exit criteria corrected (four → five built sites).
- `docs/followups/todo/legacy-parameters-width-exponent-bug.md` (+
  index row in `docs/followups/README.md`).

### Task 0.3

- Deleted `hazma/_decay/` (incl. `interpolation_data/`),
  `hazma/_positron/`, `hazma/_neutrino/`,
  `hazma/field_theory_helper_functions/`, the three `hazma/__*.py`
  shims, `hazma/spectra/_positron/_kaon.pyx`, `test/decay/`, and the
  three dead `cdef`s in `hazma/_utils/boost.pyx` (462 → 242 lines).
- `minkowski_dot` added to `hazma/utils.py`; `cross_section_prefactor`
  and `minkowski_dot` callers repointed there
  (`deprecated/rambo.py`, `gamma_ray.py`, `experimental/.../avm_msqrd.py`).
- Config: `setup.py` (3 extension groups), `test/conftest.py`,
  `pyproject.toml` + `MANIFEST.in` (`_decay` package data).
- `test/test_utils.py` added (16 pinned tests);
  `docs/followups/todo/cross-section-prefactor-threshold-cancellation.md`
  filed; nine durable docs swept. Full list in the task note.

## Verification

- Per task: importer re-check (`rg` the module path) quoted in PR body;
  `pip install -e .` + import smoke + full preflight.
- Task 0.1: `pytest -q test` → `52 passed, 20 skipped`; bare
  `pytest -q` → `57 passed, 10 skipped`; 32 `.so` built; public
  spectra bit-for-bit unchanged over 64 arrays. Details in the task
  note.
- Task 0.3: `pytest -q test` → `68 passed, 20 skipped` (52 + the 16 new
  `test/test_utils.py` cases); bare `pytest -q` → `57 passed, 10
  skipped`; **25** `.so` built (32 → 25; Task 0.2's five take it to the
  20 the phase Exit Criteria name); 26 `.pyx` / 19 `.pxd` remain; the
  171-array public compiled surface bit-for-bit unchanged. Details in
  the task note.

## Open Questions

- ~~ADR-0003 sign-off — required before Task 0.2 deletes anything~~ —
  **closed 2026-08-04: accepted.** Tasks 0.2 and (through it) 0.4 are
  unblocked; what remains in Task 0.5 is execution (record the
  replacement status in the task note, repoint docs off
  `hazma.gamma_ray`). The replacement-free `gamma_ray_fsr` case now
  lives at
  [`docs/followups/todo/msqrd-driven-fsr-generator.md`](../../../../docs/followups/todo/msqrd-driven-fsr-generator.md).
- `WIDTH_K` / `WIDTH_PI` in the legacy tables are written with `**`
  where a decimal exponent was meant (no consumer today) — deferred to
  [`docs/followups/todo/legacy-parameters-width-exponent-bug.md`](../../../../docs/followups/todo/legacy-parameters-width-exponent-bug.md).
  After Task 0.3 only `legacy_parameters.pxd` still carries the bad
  literals; the other two copies are gone.
- `cross_section_prefactor`'s threshold cancellation — deferred to
  [`docs/followups/todo/cross-section-prefactor-threshold-cancellation.md`](../../../../docs/followups/todo/cross-section-prefactor-threshold-cancellation.md).
  Sequencing matters: if Phase 01 captures the corpus first, the Rust
  port inherits the cancelling values by construction.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).
Tasks 0.1 and 0.3 each patched the phase file; those patches are
recorded in their own task notes' Plan Impact sections.

## Handoff to Next Task

**For the next agent working in Phase 00:** read `../../PLAN.md`, then
`../README.md`, then this file, then the phase file. Tasks 0.1 and 0.3
are done. **ADR-0003 was accepted 2026-08-04, so the gate that blocked
the rest of the phase is gone:** Task 0.2 is the next implementation
work, 0.4 follows it, and Task 0.5's remaining steps (repoint docs off
`hazma.gamma_ray`, record the replacement status in its task note) are
open and need no further sign-off.

**Currently safe to assume:** the tree carries 25 extensions, 26 `.pyx`,
and 19 `.pxd`. All dead Cython that could go without ADR-0003 is gone —
what is left is exactly `_gamma_ray/` (2), `_phase_space/` (3),
`rh_neutrino/_rh_neutrino_fsr_four_body.pyx`, and the 20 survivors.
`hazma.utils` is the single home for `cross_section_prefactor` and
`minkowski_dot`. `test/conftest.py` ignores only `test_gamma_ray.py`.
Clean stale `.c`/`.cpp`/`.so` before every rebuild, **and** `rm -rf` a
package directory after `git rm -r` (see Findings).

**Currently risky / unknown:** whether external user code imported the
double-underscore legacy shims. The verify-before-delete check found no
in-repo importer outside the shims themselves and two notebook cells,
but it cannot see downstream users — the removal rides the project's
`version_bump: major` for exactly that reason.
