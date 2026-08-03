---
name: review-pr
description: "Review a Hazma pull request against its project task or ad-hoc scope using one focused lens: generalist, completeness, logic, document consistency, or numerics. Use when a user asks for a PR review or when a Codex review cycle assigns a single reviewer lens."
---

# Review one pull request

Review one PR and return one structured verdict. Do not implement fixes or
orchestrate other reviewers; use `$review-cycle` for the bounded multi-agent
loop and `$review-plan` for a pre-implementation plan.

## Gather a fresh PR view

Normalize a PR number, URL, or pushed branch. Fetch its metadata and fresh
head ref; do not judge the ambient checkout:

```sh
git fetch origin pull/<N>/head:refs/remotes/origin/pr/<N>
```

Verify that SHA against `gh pr view <N>`. Treat a truncated `gh pr diff` as
incomplete; read changed files from the fetched ref. Parse project branches as
`<agent>/<slug>/<task>` for both agent prefixes. For project work, read
`lessons.md`, the PLAN's numerical-impact section, optional rules, target
phase, task note, relevant ADRs, and upstream learnings. For ad-hoc work,
review against `AGENTS.md`, the PR guidelines, and the change itself.

Read every changed non-trivial file in full context. Check common diff omissions:
public exports, sibling dispatch tables, Cython rebuild evidence, stale Sphinx
references, package data registration, and ignored tests.

## Apply one lens

Read the assigned section of
[`docs/agents/review-lenses.md`](../../../docs/agents/review-lenses.md) and
apply its baseline duties before the lens rubric. Reproduce PR-body claims,
check a real pytest collection/result, run changed examples or observable
claims, and require a rebuild after Cython edits.

- `default`: public API, units/docstrings, layering, broadcasting, debug
  residue, task completion, and PR conventions.
- `completeness`: explicit Exit-Criterion-to-pinning-test map and scope.
- `logic`: adversarial mutation, boundary/error cases, and test validity.
- `doc-consistency`: execute every applicable check in
  [`docs/agents/doc-consistency.md`](../../../docs/agents/doc-consistency.md)
  with line-numbered evidence.
- `numerics`: measure public before/after values, dimensional analysis,
  limiting cases, stability, integration/interpolation bounds, tolerance, and
  hot-path performance.

Use `REQUEST CHANGES` for any blocking finding and `APPROVE` otherwise.
`COMMENT` is only for a standalone advisory review. On a verification pass,
classify every prior comment as `RESOLVED`, `PARTIALLY RESOLVED`, or
`UNRESOLVED`.

```text
## Verdict
<APPROVE | REQUEST CHANGES | COMMENT>

## Issues (<lens> Focus)
### Blocking
<numbered, file-and-line-specific findings or None>
### Non-blocking
<numbered findings or None>

## Cross-cutting
<only clearly blocking issues outside the lens>

## Lens-Specific Observations
<required evidence for the selected lens>
```

If a required command or dependency cannot run, report the verification gap;
never invent a result or turn environment noise into a code defect.
