# Working Memory: <Project Title>

**Date:** <YYYY-MM-DD> (created)
**Project:** <project slug>
**Status:** <In Progress | Complete | Blocked>
**Plan References:** `../PLAN.md` (all sections)
**Related ADRs:** <none | ADR-XXXX (planned) | ADR-XXXX (accepted)>
**Depends On:** <none | project/task dependency>

<!--
This is the project's shared working memory — not a per-task note.
Agents working on any task in this project should:

1. **Read this file first**, before loading per-task notes. It captures
   the live task table, cumulative findings, decisions, and open
   questions that outlive a single task.
2. **Append to it as they learn.** When a task surfaces a fact that a
   future task will need (a file path, a flake, a subtle invariant),
   record it here rather than burying it in a task note.
3. **Promote stable entries out.** Once a finding becomes canonical,
   patch `../PLAN.md` or write an ADR and leave a one-line pointer
   here.

Per-task notes (`task-N-*.md`) remain the retrospective record of a
single task's execution. This file is the running context + live
status shared across tasks.
-->

## Objective

<1-2 sentences. Usually: "Track cumulative context and live task
status across all N tasks so any agent picking up work mid-project
has the facts, decisions, and open questions needed to start without
re-discovering them.">

## Tasks

Canonical task *shape* (objectives, exit criteria, implementation
guidance) lives in `../PLAN.md` under "Task Details" (flat) or in
`../phases/phase-XX-<slug>.md` (phased). This section tracks live
*status* — update the Status column as tasks progress.

<!-- Flat projects: use THIS table. Delete the phased block below. -->

| # | Task                  | Depends on | Status      | Task Note                              |
|---|-----------------------|------------|-------------|----------------------------------------|
| 1 | <short task title>    | —          | Not started | `task-1-<slug>.md`                     |

<!-- Optional dependency diagram for parallel workstreams.

```text
1 ──► 2 ──► 3
4 ──► 5 ──► 6
```
-->

<!--
Phased projects: delete the flat Tasks table above and unfence the
block below. Per-task status lives in each `phase-XX/README.md`, not
here.

This project-level README carries the Phases status table plus
*cross-phase* findings, decisions, and handoff. Phase-scoped working
memory (per-task Tasks table, phase-scoped findings) belongs in
`phase-XX/README.md` (one file per phase, copied from
`phase-XX/README.md` template).

The phase-level status mirrors each phase file's frontmatter `status:`
field. Phase frontmatter is authoritative for phase status; this table
is a single-pane-of-glass for project-wide navigation.

The block is fenced rather than commented out so its rows stay outside
MD013 — a wide table inside an HTML comment is not parsed as a table,
so `.markdownlint.jsonc`'s `MD013 {tables: false}` cannot reach it.
-->

```markdown
| # | Phase | Phase file | Working memory | Status |
| --- | --- | --- | --- | --- |
| 01 | <phase title> | `../phases/phase-01-<slug>.md` | `phase-01/README.md` | Not started |
| 02 | <phase title> | `../phases/phase-02-<slug>.md` | `phase-02/README.md` | Not started |
```

## Exit Criteria

<When is this working-memory file retired? Typically: all tasks
complete, all anticipated ADRs accepted or dropped, no open questions
remain.>

- <criterion 1>
- <criterion 2>
- Closing PR bumps `VERSION` in `hazma/__init__.py` per `PLAN.md`'s
  `version_bump:` frontmatter and adds a `CHANGELOG.md` entry naming
  this project slug. See
  [`../../../docs/versioning.md`](../../../docs/versioning.md).

## Inputs Reviewed

<!-- Append as new sources are consulted. Keep entries terse: one line
per source, with a `path:line` or URL and a short "what this tells us"
note. -->

- `../PLAN.md` — project plan, task details, anticipated ADRs.
- <additional entries as they accumulate>

## Findings

<!-- Cumulative facts discovered while implementing. Group by theme or
subsystem once there are enough entries to warrant it. Entries here
are informational — decisions go in the next section. -->

- <finding 1>

## Numerical impact so far

<!-- Running record of whether this project has moved any value the
library returns. One line per task that touched a public code path:
the function, the grid checked, and the result ("unchanged", or the
magnitude and direction of the shift). This is what the closing PR's
CHANGELOG entry and the `version_bump:` level are derived from — do
not reconstruct it from memory at close time. -->

_No public code paths touched yet._

## Decisions and Implementation Notes

<!-- Cumulative list of decisions made, including the "why". Link to
ADRs where applicable. When a decision is formalized in an ADR, leave
the one-liner here pointing to the ADR file. -->

- <decision 1 — rationale — link to ADR-XXXX if applicable>

## Files Changed

<!-- Update as tasks land. Group by task. One line per file with a
short purpose. When a task is complete, the per-task note
(`task-N-*.md`) should have the authoritative list; this section is
a cross-task roll-up for quick orientation. -->

_None yet — project not started._

<!--
Example format (fill in as tasks land):

### Task 1
- `path/to/new_file.py` — <purpose>
- `path/to/modified.py` — <what changed>
-->

## Verification

<!-- Accumulate the commands that gate each task. Useful for picking up
mid-project: "which test command corresponds to which task?" -->

- Task 1: `<command>`
- <additional entries>

## Open Questions

<!-- Keep only currently open items. When a question resolves, move
the resolution into "Decisions and Implementation Notes" or an ADR. -->

- <question 1, with enough context that a future agent knows why it
  matters and where to find the relevant code/spec>

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

This working-memory file does not by itself change `../PLAN.md`. When
a task's findings DO require a canonical change (e.g. ADR, task-table
reordering, new in-scope/out-of-scope item), patch `../PLAN.md`
directly and record a one-line pointer here under "Decisions and
Implementation Notes".

## Handoff to Next Task

<!-- Rewrite this section at the end of each task. "Next" = whichever
task the agent picks up next, not necessarily N+1. Three sub-sections:
where to start, what's safe, what's risky. -->

**For the next agent starting any task in this project:**

1. Read `../PLAN.md` end-to-end once, then this file.
2. Read the specific task's detail block in `../PLAN.md` ("Task N:
   ..." under "Task Details").
3. Check "Open Questions" above for anything relevant to your task.
4. If your task is gated by an ADR, confirm the ADR is accepted
   before implementing.

**Currently safe to assume:**

- <invariant 1>

**Currently risky / unknown:**

- <risk 1>
