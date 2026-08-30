---
name: review-plan
description: "Perform a read-only, findings-first review of a Hazma project PLAN before implementation, covering feasibility, numerical correctness, architecture, Python conventions, and testability. Use when a user asks whether a project plan is ready or asks to review projects/<slug>/PLAN.md."
---

# Review a project plan

Review the plan without editing it. Normalize the supplied slug, project path,
or `PLAN.md`; require a non-empty `projects/<slug>/PLAN.md`. An unknown focus
category is an input error. Missing plan-linked context is a finding, not a
reason to fabricate it.

## Read and ground the plan

Read `PLAN.md`, `rules.md`, phase files, references, ADRs, live task-notes
README, [`AGENTS.md`](../../../AGENTS.md),
[`docs/workflow.md`](../../../docs/workflow.md),
[`docs/versioning.md`](../../../docs/versioning.md),
[`docs/PR_GUIDELINES.md`](../../../docs/PR_GUIDELINES.md), and
[`docs/agents/lessons.md`](../../../docs/agents/lessons.md), plus files the
plan cites. Ignore `_template.md`.

Verify load-bearing current-state claims with `rg`, exact file reads, and the
commands the plan asks an implementer to run. Distinguish pre-existing paths
from paths the work will create. A green-but-noop test command is blocking;
record every verified claim with concrete evidence.

## Evaluate

Assess every requested category (or all of them) and cite the exact plan
location for each finding.

- **Feasibility:** one-PR task sizing, explicit and acyclic dependencies,
  testable exit criteria, and no unresolved decision that blocks dependents.
- **Numerics:** a realistic Numerical impact section; correct version-bump
  expectation; units, normalization, frame, and oracle for every new number;
  endpoint, interpolation, and tolerance risks identified.
- **Correctness:** error paths, physical limits, scalar/array inputs, NaN and
  mass edge cases, dispatch-table fan-out, and downstream model interactions.
- **Architecture:** layering, public/private placement and exports, existing
  abstractions, justified additions to the Rust crate, and ADR placement.
- **Idiomatic Python:** public annotations, NumPy docstrings with units,
  broadcast contract, appropriate Hazma errors, and black/isort/ruff
  conventions from `AGENTS.md`.
- **Other:** real pytest targets, scope boundaries, performance measurement,
  dependency hygiene, documentation surface, and Conventional Commit scope.

## Report

Emit only this report. Use `[P1]` for a blocker, `[P2]` for needs work, and
`[P3]` for a suggestion; any P1 means `REQUEST MAJOR REVISION`.

```md
# Plan Review: <project>

## Findings
<severity-ordered, cited findings and concrete resolutions>

## Summary
| Category | Verdict | Top finding |
|---|---|---|
| ... | Pass / Needs work / Blocker | ... |

## Grounded-facts audit
| Claim (plan location) | Verified? | Evidence |
|---|---|---|

## What the plan does well
<concrete, cited strengths>

## Verdict
<APPROVE | APPROVE WITH CHANGES | REQUEST MAJOR REVISION>
```
