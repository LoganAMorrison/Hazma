# Working Memory: Phase 03 — Numerics foundation

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 03
**Status:** Not started
**Plan References:** `../../phases/phase-03-numerics-foundation.md`
**Related ADRs:** ADR-0002 (Accepted 2026-08-04 — governs the
provenance of Tasks 3.2/3.3, no longer gates them)
**Depends On:** Phase 02 complete

## Objective

Track live per-task status and phase-scoped findings for the numerics
foundation.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 3.1 | Constants module | — | Not started | [task-3.1-constants.md](task-3.1-constants.md) |
| 3.2 | Special functions | — (ADR-0002 accepted) | Not started | [task-3.2-specfun.md](task-3.2-specfun.md) |
| 3.3 | QUADPACK port (qk15/qk21/qelg/qags/qagp) | — (ADR-0002 accepted) | Not started | [task-3.3-quadpack.md](task-3.3-quadpack.md) |
| 3.4 | Interpolation + boost kernels | 3.1 | Not started | [task-3.4-interp-boost.md](task-3.4-interp-boost.md) |
| 3.5 | Dispatch and error layer | — | Not started | [task-3.5-dispatch.md](task-3.5-dispatch.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-03-numerics-foundation.md`.

## Inputs Reviewed

- `../../phases/phase-03-numerics-foundation.md`; `../README.md`;
  `../../references/numerics-replacements.md` (full).

## Findings

_None yet._

## Decisions and Implementation Notes

_None yet._

## Files Changed

_None yet — phase not started._

## Verification

- `cargo test` (foundation units); scipy-comparison pytest suite green
  in CI.

## Open Questions

- `spec_math::li2` convention vs `scipy.special.spence` — Task 3.2
  resolves.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 03:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file. Every task in this
phase is unblocked — ADR-0002 was accepted 2026-08-04 — but 3.2/3.3
must honor its provenance rule: cephes lineage and netlib QUADPACK
only, nothing GSL-derived in the tree or the dependency graph.

**Currently safe to assume:** every live integral is finite-interval —
`qagi` is out of scope.

**Currently risky / unknown:** `qelg` Fortran→Rust translation is the
fiddliest item in the project; budget review time accordingly.
