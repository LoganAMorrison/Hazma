# Working Memory: Phase <XX> — <Phase Title>

**Date:** <YYYY-MM-DD> (created)
**Project:** <project slug>
**Phase:** <XX>
**Status:** <Not started | In Progress | Complete>
**Plan References:** `../../phases/phase-<XX>-<slug>.md` (phase file)
**Related ADRs:** <none | ADR-XXXX>
**Depends On:** <prior phases / active ADRs>

<!--
This is the phase's shared working-memory file. It is analogous to
the project-level `../README.md`, but scoped to a single phase:

- The live Tasks status table for THIS PHASE (authoritative for the
  skills' "next unfinished task in the current phase" lookup).
- Findings, decisions, and open questions that are scoped to this
  phase (cross-phase material belongs in the project-level
  `../README.md`).
- The rolling handoff for whoever picks up the next task in this
  phase.

Agents working on any task in this phase should read this file first,
append to it as they learn, and update the Tasks Status cell when a
task completes.

Length (docs/adrs/ADR-0002): append one-liners; the per-task detail
belongs in the task note. Once this phase closes and
`../../learnings/phase-<XX>-<slug>.md` exists, the learnings file is
what later phases read — this file and the phase's task notes become
history that a citation may point into.
-->

## Objective

Track cumulative context and live per-task status across the tasks in
Phase <XX> so any agent picking up work within this phase has the
facts, decisions, and handoff context needed to start immediately.

## Tasks

Canonical per-task shape (objective, exit criteria, implementation
guidance) lives in `../../phases/phase-<XX>-<slug>.md`. This table is
the live status — update the Status column as tasks progress.

| # | Task                  | Depends on | Status      | Task Note                              |
|---|-----------------------|------------|-------------|----------------------------------------|
| X.1 | <short task title>  | —          | Not started | `task-X.1-<slug>.md`                   |
| X.2 | <short task title>  | X.1        | Not started | `task-X.2-<slug>.md`                   |

<!-- Optional dependency diagram if the phase has parallel workstreams.

```text
X.1 ──► X.2 ──► X.3
```
-->

## Exit Criteria

<When is this phase's working memory retired? Typically: every row in
the Tasks table above is `Complete` and the phase file's frontmatter
has been flipped to `status: Complete`.>

- All tasks in the Tasks table above have Status `Complete`.
- Phase file `../../phases/phase-<XX>-<slug>.md` frontmatter set to
  `status: Complete`.
- Phase learnings document written to
  `../../learnings/phase-<XX>-<slug>.md`.

## Inputs Reviewed

<!-- Sources read while executing this phase. Append terse entries as
sources are consulted. -->

- `../../phases/phase-<XX>-<slug>.md` — phase file with per-task shape.
- `../README.md` — project-level working memory.

## Findings

<!-- Phase-scoped findings. Cross-phase material belongs in
../README.md. -->

- <finding 1>

## Decisions and Implementation Notes

<!-- Phase-scoped decisions with rationale. Link to ADRs. -->

- <decision 1 — rationale — ADR link if applicable>

## Files Changed

<!-- Cumulative files touched during this phase, grouped by task. -->

_None yet — phase not started._

<!--
### Task X.1
- `path/to/file.py` — <purpose>
-->

## Verification

<!-- Commands gating each task in this phase. -->

- Task X.1: `<command>`

## Open Questions

- <question 1>

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

Canonical changes to the phase file, `PLAN.md`, or `rules.md` go
through the normal ADR / plan-patch path. This file only tracks live
state.

## Handoff to Next Task

<!-- Rewrite this section at the end of each task. "Next" = whichever
task in this phase the agent picks up next. -->

**For the next agent working in Phase <XX>:**

1. Read `../../PLAN.md` (project-level), then `../README.md`
   (project-level working memory), then this file.
2. Read the phase file `../../phases/phase-<XX>-<slug>.md` for the
   specific task's shape and exit criteria.
3. Check "Open Questions" above for anything relevant to your task.

**Currently safe to assume:**

- <invariant 1>

**Currently risky / unknown:**

- <risk 1>
