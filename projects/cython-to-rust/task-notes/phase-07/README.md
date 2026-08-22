# Working Memory: Phase 07 — Packaging cutover and close

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 07
**Status:** Not started
**Plan References:** `../../phases/phase-07-cutover.md`
**Related ADRs:** ADR-0001
**Depends On:** Phase 06 complete

## Objective

Track live per-task status and phase-scoped findings for the maturin
cutover and project close.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 7.1 | Backend switch to maturin | — | Not started | [task-7.1-maturin-backend.md](task-7.1-maturin-backend.md) |
| 7.2 | Release pipeline (abi3 wheels) | 7.1 | Not started | [task-7.2-release-pipeline.md](task-7.2-release-pipeline.md) |
| 7.3 | Documentation sweep | 7.1 | Not started | [task-7.3-docs-sweep.md](task-7.3-docs-sweep.md) |
| 7.4 | Close the project | 7.1–7.3 | Not started | [task-7.4-close.md](task-7.4-close.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`;
  release candidate publishes from CI.
- Phase learnings at `../../learnings/phase-07-cutover.md` and the
  project retrospective at `../../learnings/project-retrospective.md`.

## Inputs Reviewed

- `../../phases/phase-07-cutover.md`; `../README.md`; the drift table
  in `../numerical-impact.md` (moved out of `../README.md`'s "Numerical
  impact so far" section on 2026-08-21) — input to the CHANGELOG.

## Findings

_None yet._

## Decisions and Implementation Notes

_None yet._

## Files Changed

_None yet — phase not started._

## Verification

- Clean-clone `pip install .`; sdist content check; wheel abi3-tag +
  import check on CPython 3.10 and newest; RTD/Sphinx build;
  `scripts/agents/preflight.sh --closing`.

## Open Questions

- Add aarch64/Windows wheels now that they are cheap? (Task 7.2
  records the call; default no.)

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 07:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file. Re-verify the
"Prerequisites" packaging facts against the tree — they were recorded
at plan time (2.1.0).

**Currently safe to assume:** pytest config already lives in
pyproject (moved in Phase 01 Task 1.3).

**Currently risky / unknown:** `version = attr: hazma.VERSION` has
tooling tendrils (`preflight.sh --closing`, release docs) — grep for
every consumer before flipping the source of truth.
