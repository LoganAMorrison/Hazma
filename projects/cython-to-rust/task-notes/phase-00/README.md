# Working Memory: Phase 00 — Dead-code purge

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 00
**Status:** In Progress (Task 0.1 Complete)
**Plan References:** `../../phases/phase-00-dead-code-purge.md`
**Related ADRs:** ADR-0003 (proposed — Task 0.5 ratifies; gates Task 0.2)
**Depends On:** none

## Objective

Track live per-task status and phase-scoped findings for the dead-code
purge.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 0.1 | Relocate legacy constants header | — | Complete | [task-0.1-relocate-constants.md](task-0.1-relocate-constants.md) |
| 0.2 | Delete phase-space / gamma-ray slice | 0.1, 0.5 (ADR-0003 accepted) | Not started | [task-0.2-delete-mc-slice.md](task-0.2-delete-mc-slice.md) |
| 0.3 | Delete superseded kernels + helpers | 0.1 | Not started | [task-0.3-delete-superseded.md](task-0.3-delete-superseded.md) |
| 0.4 | Prune build and packaging config | 0.2, 0.3 | Not started | [task-0.4-prune-build.md](task-0.4-prune-build.md) |
| 0.5 | Ratify + execute ADR-0003 (`gamma_ray`) | — | Not started | [task-0.5-gamma-ray-decision.md](task-0.5-gamma-ray-decision.md) |

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

## Decisions and Implementation Notes

- Task 0.1 repointed all seven `.pyx`/`.pxd` include sites but left
  `_decay/_decay_charged_pion.pyx.bak` — a `.bak` is a frozen artifact
  Cython can never compile (`cythonize()` is called only on explicit
  `Extension` objects, never a glob). Task 0.3 deletes the directory.
- Task 0.1 added a PROVENANCE paragraph to the relocated header's
  docstring (values byte-identical) so the deliberate divergence from
  `constants.pxd` is visible in-file and not "cleaned up" silently.

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

## Verification

- Per task: importer re-check (`rg` the module path) quoted in PR body;
  `pip install -e .` + import smoke + full preflight.
- Task 0.1: `pytest -q test` → `52 passed, 20 skipped`; bare
  `pytest -q` → `57 passed, 10 skipped`; 32 `.so` built; public
  spectra bit-for-bit unchanged over 64 arrays. Details in the task
  note.

## Open Questions

- ADR-0003 sign-off (see project-level Open Questions) — required
  before Task 0.2 deletes anything. Task 0.3 is unblocked and does not
  depend on it.
- `WIDTH_K` / `WIDTH_PI` in the legacy tables are written with `**`
  where a decimal exponent was meant (no consumer today) — deferred to
  [`docs/followups/todo/legacy-parameters-width-exponent-bug.md`](../../../../docs/followups/todo/legacy-parameters-width-exponent-bug.md).

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 00:** read `../../PLAN.md`, then
`../README.md`, then this file, then the phase file. Task 0.1 is done.
Next is **Task 0.3** (unblocked, depends only on 0.1) or **Task 0.5**
(the ADR-0003 ratification that gates 0.2 — needs Logan, not code).

**Currently safe to assume:** the dead-code evidence table in the
inventory reference was verified against 2.1.0. Nothing under
`hazma/_decay/` is included by a *built* extension any more, so Task
0.3 can delete the directory with no include-path fallout. Clean stale
`.c`/`.cpp`/`.so` before every rebuild (see Findings).

**Currently risky / unknown:** whether external user code imports the
double-underscore legacy shims — the verify-before-delete check is the
guard.
