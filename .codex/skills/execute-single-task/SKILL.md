---
name: execute-single-task
description: "Implement exactly one numbered task from a Hazma project plan, including task-note updates, numerical-impact measurement, documentation consistency, and the preflight gate. Use when a user asks to take a project task or the next project task; stop before committing unless the caller expressly owns the commit."
---

# Execute one project task

Implement one bounded project task while preserving the repository's durable
project memory. This skill is for project branches only; handle an ad-hoc
change directly, then use `$commit-and-pr` when it is ready to ship.

## Resolve and isolate the task

1. Resolve the project slug from an explicit `--project <slug>` or a branch
   named `<agent>/<project-slug>/<task-slug>`. Parse both `claude/` and
   `codex/`; create only `codex/` branches. Confirm
   `projects/<slug>/PLAN.md` exists.
2. Resolve the requested task with
   `scripts/agents/resolve_task.py --project <slug> [--task <id>]`. Its live
   task-notes table, not `PLAN.md`, is authoritative for status. Work on one
   task or tightly coupled, testable cluster; never cross a phase boundary.
3. Before editing, create an isolated worktree from fresh trunk:

   ```sh
   scripts/agents/setup_task_worktree.sh \
     --project <slug> --task-slug <task-slug> --agent codex
   ```

   Run all later commands in the reported `wt_path`, preferably as
   `git -C <wt_path> ...`. If a pipeline already supplied its managed
   worktree, skip this creation step and use that exact path.
4. If the task edits `.pyx`, `.pxd`, or `setup.py`, rebuild inside that
   worktree and confirm `python -c "import hazma; print(hazma.__file__)"`
   resolves there before trusting any result.

## Read the task context

Read only the context needed for the chosen task, in this order:

1. `projects/<slug>/PLAN.md`, including Numerical impact.
2. `projects/<slug>/task-notes/README.md`, then `rules.md` if present.
3. For phased projects, the target phase file, its per-phase task-notes
   README, prerequisite files, and upstream learnings.
4. The current task note and active relevant ADRs.
5. [`docs/agents/lessons.md`](../../../docs/agents/lessons.md) and
   [`docs/agents/environment.md`](../../../docs/agents/environment.md).

Treat `_template.md` files as references, not live artifacts. If a bounded
codebase survey is genuinely necessary, use a narrowly scoped Codex subagent;
all agents share the filesystem, so direct it to read only and avoid edits.

## Implement and record evidence

Create or update the task note from `projects/_template/task-notes/_template.md`
before implementing. Copy concrete Exit Criteria first. Keep its verification,
files-changed, decisions, plan-impact, and handoff sections current.

Apply the numerical-correctness and public-API rules in
[`AGENTS.md`](../../../AGENTS.md). In particular:

- Pin numerical behavior with an analytic limit, cited value, independent
  implementation, or regression array; explain tolerance.
- Cover relevant thresholds, endpoints, scalar/array broadcasting, error
  paths, and dispatch-table siblings.
- For every public path the diff can reach, compare representative before/after
  values and record the grid and result. A verified no-change result is valid;
  an unmeasured result is not.
- Run the durable-doc and stale-state checks in
  [`docs/agents/doc-consistency.md`](../../../docs/agents/doc-consistency.md).
  Update a task note, phase file, ADR, or `PLAN.md` only according to its
  canonical role; never turn `PLAN.md` into a live status log.

For canonical architectural, interface, unit, normalization, or ordering
changes, create the appropriate ADR and correct affected canonical documents in
the same task. For deferred work, follow the deduplication and filing process in
[`docs/workflow.md`](../../../docs/workflow.md).

## Finish without committing

Update the per-task note and the appropriate working-memory README. Complete
phase or project closeout bookkeeping, including the version bump and
`CHANGELOG.md`, when this is the final task. Run the mandatory gate:

```sh
scripts/agents/preflight.sh --paths "<touched paths>" --tests "<test targets>"
```

Add the `## Stale-state sweep` evidence block required by the doc-consistency
checklist, then self-review the full diff against the task spec. Do not commit
or push unless a caller explicitly directs it.

End with:

```text
STATUS: Complete | Blocked | Superseded
PROJECT: <slug>
TASK_ID: <Task N or Task X.Y>
TASK_NOTE: <path>
FILES_CHANGED: <short list>
TESTS: <command> — <literal pytest summary>
NUMERICAL_IMPACT: <measurement or verified no-change>
PLAN_IMPACT: None | Task note only | Phase file patched | ADR-XXXX | Both
NEXT: <first item for the next agent to read>
```
