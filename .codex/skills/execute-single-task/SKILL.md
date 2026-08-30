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
4. If the task edits `rust/` or `pyproject.toml`, rebuild
   inside that worktree (`pip install -e .` — not `cargo build`, which
   publishes nothing to Python) and confirm
   `python -c "import hazma; print(hazma.__file__)"` resolves there
   before trusting any result.

## Read the task context

Read only the context needed for the chosen task, in this order
([ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)):

1. `projects/<slug>/PLAN.md`, including Numerical impact.
2. `projects/<slug>/task-notes/README.md` — the head file only (status
   tables, open questions, handoff); not its `history-*.md` archive, and its
   `numerical-impact.md` log only if your diff can reach a public code
   path. Then `rules.md` if present.
3. For phased projects, the target phase file, its per-phase task-notes
   README, prerequisite files, and the learnings of every closed upstream
   phase. **The learnings replace a closed phase's task notes** — open
   one of those only when a learnings entry, the handoff, or a citation
   sends you to a specific detail. Read task notes of the current phase
   only: the previous task's handoff and open questions first.
4. The current task note and active relevant ADRs.
5. [`docs/agents/lessons.md`](../../../docs/agents/lessons.md) (the
   one-line rules; `lessons-examples.md` only for a rule you cannot act
   on as written) and
   [`docs/agents/environment.md`](../../../docs/agents/environment.md).

Treat `_template.md` files as references, not live artifacts.

Context discipline (ADR-0002 carries the measurements): reading a whole
`.rs` or test module to answer a bounded question is a
narrowly scoped, read-only Codex subagent's job — take back the
conclusion with `file:line` citations, not the file, and read a file
yourself only when you are about to edit it, by symbol rather than by
sweeping line windows. Never echo a generated artifact (a transpiled
expression, a disassembly, a long test log) into the transcript: write it
to a scratch file and inspect a narrow range. Write a file once and
confirm it landed with `wc -l`/`grep -n`, not by reading it back.

## Implement and record evidence

Create or update the task note from `projects/_template/task-notes/_template.md`
before implementing. Copy concrete Exit Criteria first. Keep its verification,
files-changed, decisions, plan-impact, and handoff sections current, and keep
it within the template's length budget: findings plus decisions under ~100
lines, the whole note under ~500, with the pasted-evidence sections
(verification summary lines, numerical impact, stale-state sweep, handoff)
exempt. Compress prose to meet it; never weaken a gate.

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

Update the per-task note and the appropriate working-memory README (append
one-liners; the project-level README is a head file of roughly 5k tokens).
Record any moved public value in the project's numerical-impact log
(`task-notes/numerical-impact.md` where one exists, else the README's
"Numerical impact so far" section). Complete phase or project closeout
bookkeeping when this is the final task: the learnings file, and for a phase
close the verbatim sweep of the closed phase's README accretions into
`task-notes/history-<section>.md` (shape as in
`projects/cython-to-rust/task-notes/`),
plus the version bump and `CHANGELOG.md` on project close. Run the mandatory
gate:

```sh
scripts/agents/preflight.sh --paths "<touched paths>"
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
