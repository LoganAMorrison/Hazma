# Working Memory: Phase 01 — Golden parity corpus

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 01
**Status:** Not started
**Plan References:** `../../phases/phase-01-parity-corpus.md`
**Related ADRs:** none
**Depends On:** Phase 00 complete

## Objective

Track live per-task status and phase-scoped findings for the parity
corpus.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 1.1 | Corpus specification + generator | — | Not started | [task-1.1-corpus-generator.md](task-1.1-corpus-generator.md) |
| 1.2 | Pytest runner + tolerance budgets | 1.1 | Not started | [task-1.2-parity-runner.md](task-1.2-parity-runner.md) |
| 1.3 | Wire both suites into one gate | 1.2 | Not started | [task-1.3-test-wiring.md](task-1.3-test-wiring.md) |
| 1.4 | Retire/regenerate legacy `.npy` suites | 1.2 | Not started | [task-1.4-legacy-npy.md](task-1.4-legacy-npy.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-01-parity-corpus.md`.

## Inputs Reviewed

- `../../phases/phase-01-parity-corpus.md`; `../README.md`;
  `../../references/numerics-replacements.md` (tolerance table).

## Findings

_None yet._

## Decisions and Implementation Notes

_None yet._

## Files Changed

_None yet — phase not started._

## Verification

- `python test/parity/generate.py --check` (hash verification);
  bare `pytest` green incl. parity suite.

## Open Questions

_None yet._

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 01:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file. The corpus manifest
must record the generating git SHA — rules.md rule 2.

**Currently safe to assume:** `test/` is green post-PR #31; merging
suites is safe.

**Currently risky / unknown:** corpus grid density vs. repo-size
budget (≤ ~10 MB) — measure before committing data.
