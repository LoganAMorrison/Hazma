---
name: task-pipeline
description: "Orchestrate one Hazma project task end to end in a dedicated Codex worktree: resolve, implement, open a draft PR, run a bounded review cycle, and finalize or retain the PR draft. Use when a user asks to implement and ship one project task; it requires projects/<slug>/PLAN.md."
---

# Run one task pipeline

Keep the orchestrator small and use fresh Codex agents for implementation and
finalization. All editing agents share the one pipeline worktree, so give them
the absolute path and branch, prohibit nested worktrees, and make phase
boundaries hard stop/go gates.

## Phase A — resolve and create the worktree

Resolve `--project <slug>` or a project branch (parse both prefixes), then use
`scripts/agents/resolve_task.py --project <slug> [--task <id>]` to select the
requested task or `next`. Stop on `done` or unresolved project state.

Create the shared worktree from fresh trunk:

```sh
scripts/agents/setup_task_worktree.sh \
  --project <slug> --task-slug <task-slug> --agent codex
```

Record `WT_PATH`, `BRANCH`, task ID, title, and task slug. Confirm the new
worktree is clean and based on the reported trunk SHA.

## Phase B — implement

Spawn one implementation agent in the shared worktree. Direct it to use
`$execute-single-task`, skip only that skill's worktree-creation step, execute
every other gate and bookkeeping step, then stage intentionally, validate a
Conventional Commit header, commit, push, and end with:

```text
## Pipeline Report
STATUS: COMPLETE | BLOCKED
PROJECT: <slug>
TASK_ID: <id>
TASK_TITLE: <title>
TASK_NOTE_PATH: <path>
BRANCH: <branch>
COMMIT_SHA: <pushed SHA>
FILES_CHANGED: <list>
SUMMARY: <paragraph>
NUMERICAL_IMPACT: <measurement>
PLAN_IMPACT: <value>
BLOCKER: <none or reason>
```

Parse this block, then independently verify `origin/<branch>` equals its SHA.
Stop for `BLOCKED`, a missing report, or a mismatched SHA.

## Phase C — draft PR

Unless `skip-pr`, open a draft PR against `master` with a valid
`chore(pipe): pipeline draft ...` placeholder title and a minimal task/body
stub. Capture `PR_NUMBER`. A review still needs this draft even when final PR
creation is skipped.

## Phase D — review

Unless `skip-review`, run `$review-cycle` inline with `PR_NUMBER`, `WT_PATH`,
`BRANCH`, external advisory reviews, and the iteration cap. Route
`CONVERGED` to finalization, retain a draft for `NOT_CONVERGED`, and stop for
`ESCALATE`. Do not spawn a second orchestration agent for this phase.

## Phase E — finalize

Spawn one finalizer in the same worktree. It reads the PR guidelines and task
note, reconciles the current diff, validates the final title, and updates the
draft body with Summary, Project, Numerical impact, Review, Test plan, and
Versioning when closing a project. It makes a converged PR ready and watches
CI to a bounded conclusion; it leaves an unconverged PR as a draft. It reports:

```text
## Pipeline Report
STATUS: PR_READY | PR_DRAFT | PR_FAILED
PR_URL: <url or none>
PR_TITLE: <title>
CI_STATUS: passing | failing | pending | not-watched
ERROR: <none or reason>
```

Verify the PR head against the final commit before reporting. Leave the
worktree and branch for the user to inspect; do not clean them up implicitly.

## Final summary

Report project, task, branch, worktree, implementation SHA and summary,
numerical impact, review status/rounds/unresolved issues, PR URL/title/CI, and
plan/versioning impact. Never batch tasks, commit to `master`, accept a
missing pipeline report, or let a public numerical change go unmeasured.
