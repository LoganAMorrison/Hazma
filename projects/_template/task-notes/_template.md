# Task <N>: <Short Title>

**Date:** <YYYY-MM-DD>
**Project:** <project slug>
**Status:** <In Progress | Complete | Blocked | Superseded>
**Plan References:** <sections of `../PLAN.md`, a phase file, or `../rules.md`>
**Related ADRs:** <none | ADR-XXXX (project-scoped) | ADR-XXXX (`docs/adrs/`)>
**Depends On:** <none | Task N-1 | ADR-XXXX>

## Objective

<One or two sentences stating what this task accomplishes.>

## Exit Criteria

<Concrete, testable outcomes that define when this task is done. Fill this
in at task start, before implementation — it's the gate you're working
toward. "Verification" below captures retrospectively what you actually did
to check these.>

- <criterion 1>
- <criterion 2>

## Inputs Reviewed

- <`../PLAN.md` sections consulted>
- <Phase file, if phased project>
- <`../rules.md` sections, if any>
- <Existing code, docs, specs, external references>
- <Prior task notes or ADRs consulted>

## Findings

- <Key fact or constraint discovered>
- <Edge case, limitation, or interoperability note>

## Decisions and Implementation Notes

- <Decision made while implementing>
- <Why that decision was taken over the obvious alternative>

## Files Changed

- <Path> — <purpose>

## Verification

- <Tests run, with command — e.g., `pytest test/spectra -q`>
- <Manual checks or fixture validation>
- <Anything intentionally deferred>

## Open Questions

- <Unresolved question, or `None`>

## Plan Impact

**Impact Level:** <None | Follow-up in learnings | Update phase file or
`../rules.md` | ADR required | Both ADR and phase/rules update>

<Describe whether this task changed canonical behavior, future task
ordering, acceptance criteria, or internal contracts. If impact is
"None", say so explicitly. For canonical changes, patch the affected
`PLAN.md` / phase file / `rules.md`; promote to a repo-wide ADR under
`docs/adrs/` if the decision has implications outside this project.>

## Handoff to Next Task

- <What the next agent should read first>
- <What assumptions are now safe to make>
- <What remains risky or incomplete>
