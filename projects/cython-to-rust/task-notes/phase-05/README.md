# Working Memory: Phase 05 — Mediator cross sections

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 05
**Status:** Not started
**Plan References:** `../../phases/phase-05-mediator-cross-sections.md`
**Related ADRs:** ADR-0002
**Depends On:** Phase 03 complete (may run parallel to Phase 04 — no
shared files)

## Objective

Track live per-task status and phase-scoped findings for the
cross-section ports.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 5.1 | Vector cross sections (template) | — | Not started | [task-5.1-vector-xs.md](task-5.1-vector-xs.md) |
| 5.2 | Scalar cross sections | 5.1 | Not started | [task-5.2-scalar-xs.md](task-5.2-scalar-xs.md) |
| 5.3 | Thermal ⟨σv⟩ validation sweep | 5.1, 5.2 | Not started | [task-5.3-thermal-sweep.md](task-5.3-thermal-sweep.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-05-mediator-cross-sections.md`.

## Inputs Reviewed

- `../../phases/phase-05-mediator-cross-sections.md`; `../README.md`;
  `../../references/cython-inventory.md` (three-tier structure, bug §2).

## Findings

_None yet._

## Decisions and Implementation Notes

_None yet._

## Files Changed

_None yet — phase not started._

## Verification

- Corpus over the mediator parameter grid; relic-density end-to-end
  check (Task 5.3); benchmark per rules.md rule 12.

## Open Questions

_None yet._

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 05:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file. Transliterate the
Mathematica dumps mechanically — never retype a 90-line expression.

**Currently safe to assume:** both `_c_*` modules cimport nothing from
hazma — they are self-contained above the foundation.

**Currently risky / unknown:** near-resonance corpus points are the
most drift-sensitive for ⟨σv⟩ (integrand peaks at the `points=`
breakpoints).
