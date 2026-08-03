---
phase: <XX>
title: <Phase Title>
status: <Not started | In Progress | Complete>
---

# Phase <XX>: <Title>

<!--
Phased-project file. Delete `phases/` entirely for flat projects.
One file per phase: `phase-XX-<slug>.md`.
-->

## Goal

<What this phase delivers.>

## Prerequisites

- <Phase X-1 complete, or specific task outputs required>
- <Active ADRs or `../rules.md` sections to read first>

## Future Phases (read-only)

<Optional: pointers to later phase files that build on this one. List
them for context so you can skim what's coming, but do NOT edit those
files while working on this phase — they belong to their own phase.>

## Parts

### Part 1: <Name>

<Overview of what this part groups and why.>

## Tasks

<!--
Each task heading below carries the same canonical hooks the flat
`PLAN.md` task table gives: a pointer to the task note's canonical
path and a dependency field. Replace `XX` with the phase number and
`X.Y` with the phase/task number (e.g. `phase-03/task-3.2-<slug>.md`).
-->

### Task X.1: <Title>

**Task note:** [`../task-notes/phase-XX/task-X.1-<slug>.md`](../task-notes/phase-XX/task-X.1-<slug>.md)
**Depends on:** <— | Task X.0 | Task (X-1).N | ADR-XXXX>

**Exit criteria:**

- <Concrete, testable outcome>
- <Gate test or verification>

**Notes:**

<Any detail too specific for the project `PLAN.md` but needed to
implement.>

### Task X.2: <Title>

**Task note:** [`../task-notes/phase-XX/task-X.2-<slug>.md`](../task-notes/phase-XX/task-X.2-<slug>.md)
**Depends on:** Task X.1

**Exit criteria:**

- <Concrete, testable outcome>

**Notes:**

<...>

## Exit Criteria

- All tasks in this phase complete.
- <Specific gate: tests pass, integration validated, etc.>
- Phase learnings written to `../learnings/phase-<XX>-<slug>.md`.
