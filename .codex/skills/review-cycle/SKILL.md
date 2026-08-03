---
name: review-cycle
description: "Orchestrate a bounded, multi-agent Codex review loop for a pushed Hazma pull request: select independent reviewer lenses, collect reviews in parallel, apply and push justified fixes, verify them, post durable round summaries, and stop at convergence, escalation, or the iteration cap."
---

# Run a PR review cycle

Use this skill only for a pushed PR or branch. It owns the single review loop;
reviewer and responder agents must not invoke `$review-cycle` or
`$task-pipeline` recursively.

## Setup

Normalize the PR number, URL, or pushed branch; record its head branch and
SHA. Parse project metadata when applicable and create or reuse one worktree
for fixes. The managed worktree is shared by Codex agents: give every editing
agent its absolute path and branch, and tell it not to create another worktree.

Read these canonical sources rather than reproducing their rules:

- [`review-lenses.md`](../../../docs/agents/review-lenses.md) for lens choice,
  Codex reasoning effort, verdicts, and rubrics.
- [`preflight.md`](../../../docs/agents/preflight.md) for every commit.
- [`doc-consistency.md`](../../../docs/agents/doc-consistency.md) and
  [`lessons.md`](../../../docs/agents/lessons.md) for review evidence.

Select the reviewer set once: normally A and D; B for explicit criteria; C for
runtime behavior; E for any change that can move a number; prefer all lenses
on project closeout. Record chosen and omitted lenses and retain this exact set
for every round. Respect a caller override.

## Execute each round (default cap: 3)

1. Spawn the selected reviewers concurrently through Codex collaboration.
   Assign one stable task name and one lens per agent. Each prompt must name the
   PR, require `$review-pr`, require a fresh fetched PR head, prohibit edits,
   and require only `APPROVE` or `REQUEST CHANGES`.
2. If every internal reviewer approves with no comments, post the shortcut
   round summary and converge. Otherwise spawn a fresh responder in the shared
   worktree with the full tagged reviews and `$review-respond`. Require it to
   commit, push, report `FINAL_COMMIT_SHA`, and quote the literal pytest
   summary.
3. Verify the PR head equals the reported pushed SHA and advanced from the
   round's prior SHA. Stop if it did not; do not verify stale code.
4. Spawn the same lenses concurrently as verification reviewers. Give each
   reviewer its original comment and corresponding response; require a fresh
   ref fetch, re-run any volume/numerics claim, status every original comment,
   and report new issues.
5. Converge only if all verifiers approve with no new blocking issue. Loop only
   when at least one blocking issue was resolved. Escalate if the same blocker
   persists for two rounds; stop un-converged at the cap.
6. Post a PR comment for every round containing the lens selection (round 1),
   verdicts, verification statuses, decisions, numerical impact, and fix SHA.
   External reviews are advisory: surface them but do not count them toward
   convergence or invent a verification round for them.

Use this response shape so a caller can route the result:

```text
## Review Cycle Summary
TARGET: PR #<N> / <branch>
SELECTED_REVIEWERS: <A, B, ...>
STATUS: CONVERGED | NOT_CONVERGED | ESCALATE
ITERATIONS_USED: <N>
UNRESOLVED: <list or none>
TESTS: <literal pytest summary>
NUMERICAL_IMPACT: <measurement or verified no-change>
FINAL_COMMIT_SHA: <verified SHA>
REVIEW_SUMMARY: <one paragraph>
```

Never force-push, skip the preflight gate, mutate the selected reviewer set,
or treat an unmeasured numerical assertion as a verdict.
