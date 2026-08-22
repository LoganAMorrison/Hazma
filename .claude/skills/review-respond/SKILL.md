---
name: review-respond
description: As the implementer, synthesize existing PR review feedback from one or more reviewers, implement the justified fixes behind the preflight gate, and produce per-reviewer, verification-ready response summaries. Use when reviews already exist and you need to address them — not to collect reviews.
---

**Role:** Act as the engineer who implemented the changes under review.
You critically evaluate feedback from multiple independent reviewers,
implement the justified fixes, and produce a structured response each
reviewer can verify.

## When to use this skill

- The user (or an orchestrator) provides review feedback and asks you to
  process, address, or respond to it.
- Called by `/review-cycle` (its B.2 step) after collecting a round.

## When NOT to use this skill

- No reviews exist yet and you need to *spawn* reviewers and loop →
  `/review-cycle`.
- You are the reviewer, not the implementer → `/review-pr`.
- You have only a working-tree diff to critique with no feedback →
  `/code-review`.

## Inputs

Reviews may arrive as pasted text, file paths, or subagent results. You
also need a handle for the changes under review:

- **PR number** or **URL** (extract the trailing number), or
- **Branch name** — resolve via
  `gh pr list --head <branch> --json number --jq '.[0].number'`,
  otherwise fall back to the local branch diff.

When a caller (`/review-cycle`, `/task-pipeline`) hands you `WT_PATH` /
`BRANCH`, work in that shared worktree; fixes must land there. Use
`git -C <WT_PATH>` for every git command.

## Workflow

### Step 1: Gather context

- Obtain the diff: `gh pr diff <N>`, or `git diff <base>...HEAD`. Confirm
  `<base>` is the true merge-base —
  `git merge-base --fork-point origin/master HEAD` — so a stacked branch
  does not pull unrelated commits into the diff.
- Parse the head branch for the project slug. Project branches are
  `<agent>/<project-slug>/<task-slug>`; ad-hoc are
  `<agent>/<short-description>`, `<agent>` ∈ `{claude, codex}` — **parse
  both prefixes**.
- For **project work**, read in order: `projects/<slug>/PLAN.md` (check
  `phased`, `version_bump`, and the **Numerical impact** section);
  `rules.md` if present; the phase file (phased); the task note; any
  referenced ADRs.
- Read [`lessons.md`](../../../docs/agents/lessons.md), then every review
  input provided.

### Step 2: Evaluate every comment

Assess each comment on technical merit:

- **Correctness:** does the claim hold against the code, spec, and tests?
- **Scope:** is the fix within the task boundary, or scope creep?
- **Severity:** would ignoring it cause a bug, a wrong number, a spec
  violation, a test failure, or a canonical-rule violation? Use the
  severity anchors in
  [`review-lenses.md`](../../../docs/agents/review-lenses.md).
- **Conflicts:** if reviewers disagree, choose the path aligned with
  `rules.md`, active ADRs, and the repo conventions in `AGENTS.md`; cite
  which source resolved the conflict.

A **numerics finding gets a measurement, not an argument.** If a reviewer
claims a value moved or is wrong, evaluate the function and paste the
numbers. Rejecting a numerics comment without running the comparison is
not a rejection you can defend in verification.

Categorize each comment:

| Category      | Meaning                                      | Action                           |
|---------------|----------------------------------------------|----------------------------------|
| `fix`         | Valid, in-scope issue                        | Implement the fix                |
| `fix-partial` | Valid concern, suggested fix wrong/overkill  | Fix differently, explain how     |
| `acknowledge` | Valid but out of scope for this task         | Defer to a follow-up (see below) |
| `reject`      | Incorrect, already handled, or out of scope  | Explain why, with evidence       |

For `acknowledge`, follow the follow-up lifecycle in
[`workflow.md#follow-ups`](../../../docs/workflow.md#follow-ups): grep
`docs/followups/` for an existing entry and link it; else copy
`_template.md` into `docs/followups/todo/<slug>.md` and add a README row.
**Dedup against open PRs too** (`gh pr list --state open`, then
`gh pr diff <n> --name-only | grep followups`). Cite the follow-up file
in the response.

### Step 3: Implement accepted changes

- Apply all `fix` and `fix-partial` changes. Stay scoped — do **not**
  drift into adjacent cleanup a reviewer did not flag.
- **Class-fix, not point-fix.** Every fix re-runs the *class* of check
  across the whole touched artifact set, not just the cited line. A
  correctness fix that guards one entry point re-checks every sibling
  entry point; a fixed count re-derives every sibling count; a corrected
  unit or contract phrase is swept across all durable docs. A point fix
  that leaves an adjacent stale sibling is what drives extra rounds.
- **Stale-sibling sweep.** Before fixing any factual claim (count,
  identifier, command, line number, unit, or qualitative prose claim),
  run the class-wide sweep in
  [`doc-consistency.md`](../../../docs/agents/doc-consistency.md) §11:
  `rg -n '<old-value>' projects/ docs/ hazma/ test/ README.md
  CHANGELOG.md`, paste under `### Pre-fix occurrences`; fix every
  occurrence or justify each skip; re-run and paste under
  `### Post-fix occurrences`. Numeric fixes sweep the bare digit
  (`\b<old>\b`). Re-derive a corrected fact from first principles rather
  than confirming it against an adjacent copy.
- **Re-measure after any behavior fix.** If your fixes changed a code
  path a public function reaches, re-run the numerical comparison and
  update the task note and (on a closing PR) `CHANGELOG.md`. Then
  re-check the PR body against the post-fix diff — a stale body is a
  blocking finding.
- **Rebuild before gating** if you touched `.pyx` / `.pxd` / `rust/` /
  `setup.py`: `pip install -e .` (a `cargo` run does not republish the
  extension).
- **Run the preflight gate.** `scripts/agents/preflight.sh --paths
  "<touched>"` — bare, so its pytest gate is the same collection CI runs
  (or the manual list in
  [`preflight.md`](../../../docs/agents/preflight.md)). Read the pytest
  summary line — exit 5 is zero collected, a FAIL not a pass. Never
  silently revert a fix to make a gate pass; diagnose it.
- Update the task note to reflect the review fixes — a brief bullet under
  its decisions/implementation-notes section is enough.

### Step 4: Produce per-reviewer response summaries

One block per reviewer (downstream verification parses these — keep the
shape). If the caller specifies a condensed shape (e.g. `/review-cycle`'s
single decisions table), use the caller's shape; the invariants are one
block per reviewer, a category per comment, and a verdict line.

```text
## Response to <Reviewer ID>

### Accepted

| # | Comment Summary | Category | Action Taken |
|---|-----------------|----------|--------------|
| 1 | ... | fix | Changed X in `spectra/_muon.py:42` |
| 2 | ... | fix-partial | Addressed by Y instead of Z because ... |

### Acknowledged (deferred)

| # | Comment Summary | Reason | Follow-up file |
|---|-----------------|--------|----------------|
| 3 | ... | Out of scope for this task | `docs/followups/todo/<slug>.md` |

### Rejected

| # | Comment Summary | Reason (with evidence) |
|---|-----------------|------------------------|
| 4 | ... | Value is unchanged: `<command>` → `<output>` |

### Verdict Requested: <ACCEPT | ITERATE>
```

`ACCEPT` = every blocking concern from this reviewer is addressed;
`ITERATE` = open questions remain. If you were stuck mid-fix, say
`ITERATE` and state which fixes landed and which did not — never report a
clean verdict over a half-applied round.

### Step 5: Produce the combined summary

```text
## Combined Summary

- **Reviews processed:** N reviewers, M total comments
- **Breakdown:** X fixed, Y partially fixed, Z acknowledged, W rejected
- **Files modified:** [list]
- **Tests:** <literal pytest summary line, e.g. "43 passed, 2 skipped">
  — quote it verbatim from the run, never paraphrase
- **Numerical impact:** <none (verified: <command>) | <function>:
  <magnitude and direction>>
- **Open questions:** [tradeoffs or ambiguities needing user input]
```

### Step 6: Append to the lessons ledger

If any addressed finding is **class-shaped** — it could recur on
unrelated tasks, not a one-off typo — append a one-line entry to
[`lessons.md`](../../../docs/agents/lessons.md) in the *same commit* as
the fix: `- [class] one-line rule (PR #N)`, **and** the worked example
(what the PR did, what review caught, the command that exposed it) under
a matching `### <class>` heading in
[`lessons-examples.md`](../../../docs/agents/lessons-examples.md). If an
existing entry covers the class, add this PR to its citation list in
`lessons.md` and append the example under its heading rather than
duplicating. Cite a real PR; an uncited lesson is a guess.

### Commit boundary

- **Inside `/review-cycle` or `/task-pipeline`** (you were given
  `WT_PATH` / `BRANCH`): commit and push the fixes to the PR branch per
  the caller's instructions, but only **after** the Step 3 preflight gate
  passes and the branch/worktree assertion in
  [`preflight.md`](../../../docs/agents/preflight.md) holds (intended
  branch, never `master`; `git -C <WT_PATH>`). The orchestrator then
  verifies the push landed.
- **Standalone:** leave the changes uncommitted and **say so** in the
  combined summary — hand off to `/commit-and-pr`. Do not improvise a
  commit.

## Guardrails

- Do not accept comments uncritically — reviewers can be wrong. Do not
  reject defensively — if the code has a bug, fix it.
- Reject a numerics comment only with a pasted measurement.
- Do not expand scope beyond the task boundary even if a reviewer
  suggests it; acknowledge as deferred and file/link a follow-up.
- Be specific: file path, line number, what changed and why.
- If two reviewers flag the same issue, credit both, implement once.
- If a comment is ambiguous, note the ambiguity rather than guessing.
- Do not "fix" things reviewers did not mention — it muddies
  verification; out-of-scope cleanup is a separate follow-up.
- If a fix conflicts with another or introduces a regression, document
  the tradeoff and choose the safer path.
- If a gate starts failing after applying fixes, diagnose it before
  proceeding; never silently revert a fix to make it pass.
