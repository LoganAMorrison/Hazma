---
name: solo-task
description: Execute one project task end-to-end in the current session — resolve, implement, gate, note, ship, one review pass — while keeping the main context small by delegating every heavy read, search, and gate to a subagent that returns a summary. Use when running a task by hand instead of through task-pipeline; execute-single-task is task-pipeline's implementer and reads its inputs in full, which is the wrong shape for a long-lived main session.
---

**Role:** You are the implementer for exactly one task from a project
under `projects/<slug>/`, working in the main session. The procedure is
`execute-single-task`'s, unchanged: the task-note template, the one
status invariant, the numerical-impact measurement, the ADR escalation
rule, the preflight gate, and `commit-and-pr` owning every commit. This
skill decides only who reads and who runs each step: a subagent reads
and summarizes what `execute-single-task` reads in full, and every step
not repeated verbatim here is delegated to its owning skill by name, so
that skill's steps apply in full.

## When to use

- One task, or a tight cluster, from a project, done by hand in this
  session while the user watches and steers.
- The user wants the result sooner than `task-pipeline`'s isolated phases
  and multi-round review convergence deliver it.

## When NOT to use

- Unattended end-to-end work: use `task-pipeline`.
- A task spanning more than one subsystem or more than two architectural
  decisions: split it in the plan first.
- Ad-hoc fixes outside a project: do the work and use `commit-and-pr`.

## Context budget

[ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)
measured three task runs that each ended between 513k and 644k tokens of
context, and the mandatory documents were under 35k of it; the agent's
own output and ad-hoc source reads were the rest. A main session that
also runs `review-respond` and `commit-and-pr` in the same window grows
further, and most of its cost is re-reading a window that never shrinks.
These rules hold for the whole run:

- The main session never reads `PLAN.md`, a phase file, a task-notes
  README, `docs/agents/lessons.md`, `docs/agents/lessons-examples.md`,
  `docs/agents/doc-consistency.md`, `docs/agents/environment.md`, or
  `docs/agents/review-lenses.md` in full. A subagent reads them and
  returns a brief. When a step has you edit one of those files, read only
  the section you are editing, or hand the edit to the subagent the step
  names.
- Read code in bounded ranges: `Read` with `offset`/`limit`, or
  `sed -n A,Bp`, at most about 120 lines at a time, located by symbol
  first (`rg -n 'def name' file`). Never re-read a file after editing it;
  the edit result already confirms the change.
- A search wider than one package directory (`hazma/spectra/`, `rust/`,
  `test/`) goes to an `Explore` subagent that returns `file:line` hits,
  not excerpts. Bound your own searches with `rg -l` or `| head -n 20`.
- Generated output — a captured array, a before/after spectrum diff, a
  `pytest -v` log — goes to a scratch file, and you inspect a narrow range
  of it, as `execute-single-task` Step 4's context discipline says.
- `git diff --stat` before any diff, then diff one path at a time.
- Git commands, script paths, and subagent prompts name the worktree
  explicitly, in the forms Step 2 sets, because the tool shell's working
  directory resets between calls and `scripts/agents/preflight.sh`
  changes to the repository that contains it. See
  `docs/agents/environment.md` under "Shell and filesystem".
- Test, lint, and build output reaches you only through a subagent.
- Every subagent call names its model: Haiku for gates, builds, and
  relays; Sonnet for briefs, bookkeeping, and shipping; Opus for the
  numerics review, as `docs/agents/review-lenses.md` assigns it.
- If the window passes roughly 300k tokens, bring the task note current
  first (status, the blocker or the point reached, and the handoff), then
  emit the structured report and tell the user to resume from the task
  note in a fresh session. The report's `BRANCH` and `WORKTREE` fields
  are what the fresh session enters in Step 2.

## Workflow

### Step 1 — Resolve the task (main session)

Run `git fetch origin -q`. Determine the project slug from `--project`,
else from a branch named `<agent>/<slug>/<task-slug>`, else ask. Run
`scripts/agents/resolve_task.py --project <slug>` (add `--task <id>` when
the user named one); its one-line JSON is the only output you keep. A
`status` of `blocked`, `done`, or `error` ends the run: report it and
stop. If the checkout is behind `origin/master`
(`git rev-list --count HEAD..origin/master` is nonzero), the answer is
provisional and Step 3 confirms it.

### Step 2 — Enter the worktree and build it

If a worktree already exists for the branch `<agent>/<slug>/<task-slug>`
(`git worktree list`), enter it; a resumed session must not rerun the
setup script, which finds the branch taken and mints `<task-slug>-2` from
`origin/master`. Otherwise run

```sh
scripts/agents/setup_task_worktree.sh \
  --project <slug> --task-slug <task-slug> --agent claude
```

and record the `branch` and `wt_path` from its JSON line. From here on
every git command is `git -C <wt_path>`, every script runs as
`<wt_path>/scripts/agents/<name>`, and every subagent prompt opens with
"In `<wt_path>`".

A fresh worktree has no `hazma/_core.abi3.so`, because the extension is
gitignored, so nothing in it imports until it is built. Spawn a Haiku
subagent: "In `<wt_path>`, run
`pip install -e . --config-settings build-args="--features test-probes"`,
then report only whether it succeeded, the first error if it did not,
and the output of
`python -c "import hazma, hazma._core; print(hazma.__file__, hazma._core.__file__)"`."
Both paths must lie inside `<wt_path>`; if either does not, stop and
report it, because every later result would describe a different tree.
Repeat this build after any edit under `rust/` or to `pyproject.toml`
before trusting a Python-side result, as `execute-single-task` Step 3
says.

### Step 3 — Task brief and note skeleton (Sonnet subagent, then you)

Spawn one Sonnet subagent: "In `<wt_path>`, re-run
`scripts/agents/resolve_task.py --project <slug> --task <id>` and stop if
it no longer names this task. Then read, in `execute-single-task` Step 4's
order and under its skip rules, `PLAN.md` (including its **Numerical
impact** section), the working-memory README, `rules.md`, the target
phase file only, the per-phase README, prerequisite files, the learnings
of closed phases, the current phase's prior task notes as that step
limits them, ADRs, any existing note for this task, then
`docs/agents/lessons.md` and `docs/agents/environment.md`." It returns a
**task brief** of at most 400 words: task id and slug; objective; exit
criteria verbatim; whether this task is the last in its phase and
whether it is the last in the project; the project's `version_bump:`
level and numerical-impact constraints; the files and public functions
likely touched; whether the task touches `rust/`; the plan rules and ADR
constraints that bind this task; each applicable lessons entry as its
class name plus one line; environment traps for the tools involved; open
questions. The brief is the only project prose in your window.

Then create the task note from `projects/_template/task-notes/_template.md`
at the path `execute-single-task` Step 5 prescribes, with `**Status:**`
`In Progress` and the exit criteria filled in first, and keep it current
while working, within that step's length budget. The note is the resume
point for a session that stops early, so it must exist before any work
does.

### Step 4 — Implement (main session)

Do the task against the brief, under the context budget. The discipline
is `execute-single-task` Steps 6a through 6d: the testing checklist
(pin a number, state the tolerance, the boundaries, scalar and array
inputs, the stash-proof check), the numerical-impact measurement, claim
verification, and the implementer's durable-doc sweep. When a rule
matters, have a Haiku subagent quote the applicable rule rather than
reading the file. Iterate on Rust with

```sh
cargo test --manifest-path <wt_path>/rust/Cargo.toml \
    --no-default-features --features test-probes
```

and send narrow `pytest` runs through a Haiku subagent that returns the
summary line and the first failure only.

The Step 6b measurement is mandatory whenever the diff can reach a public
code path. Hand it to a Sonnet subagent: "In `<wt_path>`, evaluate
`<functions>` on `<grid>` at `origin/master` and on the working tree,
writing both arrays to scratch files, and return the largest absolute
and relative change per function and the command used." It needs a
second, built checkout of `origin/master`; the subagent creates it in a
scratch directory and removes it afterward. Record its answer in the
note, and in the numerical-impact log when a value moved.

Re-derive every claim about the code (a count, a call site, a value)
from the live tree before it enters a note or a doc. If the exit criteria
prove wrong or the task needs a canonical plan change, apply
`execute-single-task` Step 7's decision tree: a task-note finding, a
`docs/followups/todo/` file, or an ADR (project-scoped by default;
`docs/adrs/` only when it stands without the project) plus the patched
phase file or `rules.md`. The canonical-contract diff runs in this
skill's Step 7, not here. Before leaving this step, sweep your own diff
for `TODO`, `FIXME`, `breakpoint()`, `pdb`, and stray `print()`, as
`execute-single-task` Step 8 requires.

### Step 5 — Gate (Haiku subagent)

Spawn a Haiku subagent: "In `<wt_path>`, run
`<wt_path>/scripts/agents/preflight.sh --paths "<touched paths>"` with no
`--tests`, so pytest runs the whole suite as CI does. Report every gate
row verbatim, the pytest summary line, and the first failing diagnostic
per failed gate, at most 60 lines." If the diff touched `rust/` or
`pyproject.toml` since the last Step 2 build, the prompt rebuilds first.
A `WARN` row is a gate that did not run, not a pass. Fix and re-run until
green; the gate runs the whole suite plus the cargo gates, so re-run it
only after a change to a file some gate reads.

### Step 6 — Task note, status, and closure

Complete the task note yourself: findings, decisions, files changed, the
gate's verification lines, the numerical-impact result, open questions,
plan impact, and handoff. Then spawn a Sonnet subagent: "In `<wt_path>`,
task `<id>` of project `<slug>` has its note at `<note path>`. Perform
`execute-single-task` Step 8's bookkeeping: record status in the
working-memory README, and, if this task is the last in its phase or the
last in the project, the phase or project closure procedure from the
same step, including the version bump and `CHANGELOG.md` entry on a
project close. Do not commit. Return the files you changed and, only if
you performed a closure, the result of
`<wt_path>/scripts/agents/preflight.sh --closing --paths "<touched paths>"`."
The working-memory README is the project's `task-notes/README.md`, or
the phase's `task-notes/phase-XX/README.md` in a phased project; the
rest of this skill means that file by the name.

A task-note status alone does not close a phase:
`scripts/agents/resolve_task.py` reads the phase file's frontmatter, so
without the `status: Complete` flip it keeps offering the finished phase
and never reaches the next phase's first task.

### Step 7 — Self-review and doc consistency (Sonnet subagent)

This runs after the last write of Step 6, so the note, the status cells,
and any closure artifacts are in the tree it checks. Spawn a Sonnet
subagent: "In `<wt_path>`, `git add -N` every untracked file, then
perform `execute-single-task` Step 9's self-review for task `<id>` of
project `<slug>` over the diff that step names, including Step 7's
canonical-contract diff against the phase file and every active ADR, the
`## Stale-state sweep` block with its numerical-impact row appended to
the note at `<note path>`, and the durable-doc pass from
`docs/agents/doc-consistency.md` (AGENTS.md, `docs/`, `docs/source/`,
project files, followups). Make the doc edits. Return the files you
changed, the plan impact in `execute-single-task`'s vocabulary, and any
gap you could not fix." A gap that needs a code change is yours: fix it
and return to Step 5.

### Step 8 — Ship (Sonnet subagent)

Spawn a Sonnet subagent: "In `<wt_path>`, use the `/commit-and-pr` skill
for task `<id>` of project `<slug>`, running its preflight gate with
`--md` for the changed Markdown files and `--closing` if the diff closes
the project. Open the PR and return without watching CI; it is watched
once, on the final head." It runs the gate again, writes the
Conventional Commit, pushes, opens the PR, and returns the PR URL and
title. Never run `/commit-and-pr` in the main session; it and the guides
it loads are many times this skill's size.

### Step 9 — One review pass (subagents, then a Sonnet subagent)

Choose reviewers by `docs/agents/review-lenses.md`'s selection rules,
narrowed to one round of at most two:

- **A, the Generalist** (Sonnet, lens `default`), always. Upgrade it to
  Opus on a project-closing PR, as the roster allows.
- **E, Numerics** (Opus, lens `numerics`), whenever the diff can move a
  number: any change under `hazma/` or `rust/` that is not purely a
  rename, a docstring, or a type annotation.

Spawn them in parallel, each with: "Use the `/review-pr` skill with lens
`<lens>` to review PR #N. Return ranked findings only, each with
`file:line`." Triage the findings yourself, since you hold the
implementation: accept or reject each with a one-line reason. One round
only: a finding that needs another round goes in the report.

Then spawn a Sonnet subagent: "In `<wt_path>`, use the `/review-respond`
skill on PR #N with these findings and this triage: `<list>`. Then use
`/commit-and-pr` to commit and push the fixes to the existing PR (it
exists, so edit it rather than open another), with `--md` and, on a
closing diff, `--closing`, and watch CI to a conclusion." That path keeps
the class-fix sweep, the PR-body re-check, the lessons append, and the
branch and worktree assertion that a hand-rolled commit would drop. A fix
that moves a number also re-runs the Step 4 measurement and updates the
PR body's Summary. Before the commit, the same subagent reconciles the
task note and status with what the fixes changed: the files-changed,
verification, and numerical-impact lines, any decision a fix reversed,
and, for a finding left open, an entry under the note's open questions
and the working-memory README's handoff. If an open finding leaves an
exit criterion unmet, it sets `**Status:**` to `Blocked` in the note and
in the working-memory README, and a closing task that ends `Blocked` is
not closed: the subagent reverses every closure artifact Step 6 wrote,
in the same commit, so the PR does not ship a phase or project that is
still open.

If the triage accepts nothing, spawn a Haiku subagent to run
`gh pr checks <N> --watch --fail-fast` and return the verdict; the report
carries it either way.

### Step 10 — Hand off and end the session

Emit the structured report, then tell the user to start a fresh session
for the next task. Once the task note and the PR exist, nothing in the
window is worth carrying forward.

## Structured report

```text
STATUS: Complete | Blocked | Superseded
PROJECT: <slug>
TASK_ID: <id>
TASK_NOTE: <path>
BRANCH: <branch>
WORKTREE: <wt_path>
PR: <url>
CI: passing | failing | not watched
FILES_CHANGED: <list>
TESTS: <command> — <literal pytest summary line>
NUMERICAL_IMPACT: <none (verified: <command>) | <function>: <magnitude>>
REVIEW: <reviewers run; findings applied / deferred>
PLAN_IMPACT: None | Task note only | Phase file patched | ADR-XXXX | Both | Phase closure | Project closure
NEXT: <next task id, in a fresh session>
```

## Guardrails

- Status lives in the note's header, the working-memory README, and the
  closure frontmatter. `PLAN.md` carries no per-task status; its
  `## Phases` cell flips only through the Step 6 closure.
- Never report a numerical result from a tree that was not rebuilt after
  a Rust edit, or from a hazma imported from outside `<wt_path>`.
- Never paste a subagent's raw tool output into the window, and never ask
  a subagent for detail you will not act on.
- A blocked task hands off with the report after the note is current; it
  does not widen scope.
