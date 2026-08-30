---
name: execute-single-task
description: Execute exactly one task (or tightly related task cluster) from a project's PLAN, keep scope narrow, update the task note, run the preflight gate, and escalate only canonical plan changes into ADRs and plan files. Stops before commit — the caller (or /commit-and-pr) owns the commit.
---

**Role:** Act as the engineer responsible for completing exactly one task
from a project's plan with minimal scope drift and durable project
memory.

## When to use this skill

- The user asks to work on one task from a project, or to take the next
  task in a project.
- The user wants a bounded execution pass with a clean handoff.

## When NOT to use this skill

- **Full task lifecycle (implement → review → ship)** → `/task-pipeline`,
  which wraps this skill in a context-isolated subagent and owns the
  review loop and PR.
- **Ad-hoc, non-project work** (a one-off fix, dep bump, or a branch that
  is not `<agent>/<project-slug>/<task-slug>`) → this skill errors out in
  Step 1. Do it directly and use `/commit-and-pr`.
- **Committing / opening the PR** → out of scope. This skill runs the
  preflight gate and leaves the tree ready.

## Inputs

- **project slug** (optional) — explicit `--project <slug>`, or inferred
  from the current branch.
- **task identifier** (optional) — `Task 3` (flat) or `Task 2.4`
  (phased). If omitted, resolve the next unfinished task (Step 2).

## The one status invariant (stated once)

`PLAN.md` holds canonical task **shape** — objectives, scope, ordering,
Exit Criteria. It is **never** a live status log. Live task **state**
lives in the working-memory README under
`projects/<slug>/task-notes/`. Every step below assumes this split; never
edit a `Status` column inside `PLAN.md`. You touch `PLAN.md` only when
canonical shape, ordering, or task definition changes (Step 7).

## Workflow

### Step 1: Resolve the project slug

Project branches are `<agent>/<project-slug>/<task-slug>`, where
`<agent>` is `claude` or `codex`; ad-hoc branches are
`<agent>/<short-description>`. **Parse both prefixes.** The slug is the
first segment after the prefix.

Precedence: explicit `--project` → branch-name inference → stop and ask.
A branch that does not match the three-segment project pattern is ad-hoc
— error out and ask the user for the intended project.

Confirm `projects/<slug>/PLAN.md` exists. If not, stop and report the
missing scaffold.

### Step 2: Determine the task boundary

Work on one numbered task, or one tightly coupled cluster. A unit sized
for one task touches ≲1 subsystem and carries ≲2 architectural
decisions. If the requested task is broader, split it to the smallest
meaningful testable unit and record the split in the task note. Do not
cross phase boundaries in one pass.

If the user did not specify a task, resolve it deterministically:

```sh
scripts/agents/resolve_task.py --project <slug>
```

It reads the live Tasks table (flat → `task-notes/README.md`; phased →
the current phase's `phase-XX/README.md`), skips `_template.md`, and
emits single-line JSON: `{status, task_id, task_title, task_slug, phase,
reason}`. Pass `--task <id>` to pin a specific row. Trust `task_slug`.

**Fallback (script unavailable):** read the table by hand — the lowest-
numbered row whose `Status` is not `Complete`/`Superseded`.

### Step 3: Enter a worktree (branch off origin/master)

Create an isolated git worktree **before** any code change. Never modify
the main checkout directly, and never branch from ambient `HEAD` — Step 9's
`git diff origin/master --` is only sound if the branch was cut from a
freshly-fetched trunk.

```sh
scripts/agents/setup_task_worktree.sh \
  --project <slug> --task-slug <task-slug> --agent claude
```

It fetches origin, resolves the trunk from `origin/HEAD` (falling back to
`master`), picks a collision-free name, verifies `HEAD == origin/<trunk>`,
and prints `{"branch":…,"wt_path":<absolute>,"head_sha":…}`. If the
`EnterWorktree` tool is available, prefer it.

All edits and gates happen inside the worktree. The Bash-tool cwd can
reset between calls — use `git -C <worktree>` with absolute paths (see
[`environment.md`](../../../docs/agents/environment.md)).

**If the task touches compiled code** (`rust/`, `pyproject.toml`),
rebuild in the worktree before running anything:
`pip install -e .`, then confirm
`python -c "import hazma; print(hazma.__file__)"` points inside the
worktree. A stale extension makes every later result meaningless, and
`cargo build` / `cargo test` do not count as the rebuild — they work out
of `rust/target/`, which nothing Python imports.

### Step 4: Read only the required context

The reading list is a budget, not a syllabus
([ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)).
In this order; skip what does not apply:

1. `projects/<slug>/PLAN.md` — always. Frontmatter `phased:` picks the
   branch below. Read the **Numerical impact** section; it constrains
   what you are allowed to change.
2. `projects/<slug>/task-notes/README.md` — **always**, and only this
   head file: the live Tasks table (flat) or Phases table (phased), the
   open questions, and the rolling `## Handoff to Next Task`. Its pointer
   sections name the archive (`task-notes/history-*.md`) and, where the
   project keeps one, the numerical-impact log
   (`task-notes/numerical-impact.md`). Do not read the archive; read the
   log only when your diff can reach a public code path (6b) or you are
   closing the project.
3. `projects/<slug>/rules.md` — cross-cutting project rules (optional).
4. `projects/<slug>/phases/phase-XX-<slug>.md` — **phased only**, and
   **only the target phase**. Find the exact `### Task X.Y`, its Exit
   Criteria, and Prerequisites.
5. `projects/<slug>/task-notes/phase-XX/README.md` — **phased only**,
   the current phase. Per-phase working memory; authoritative for "next
   unfinished task".
6. Files named in that phase file's Prerequisites block.
7. `projects/<slug>/learnings/phase-XX-*.md` for every **closed**
   upstream phase (phased only). **These replace the closed phases' task
   notes and per-phase READMEs** — do not open those; the one exception
   is a learnings entry, the current handoff, or a citation that sends
   you to a specific note for a specific detail.
8. Task notes of the **current phase only**
   (`task-notes/phase-XX/task-*.md`): the previous task's
   `## Handoff to Next Task` and `## Open Questions` first, the rest of a
   note only when the handoff points into it. Flat projects: the same
   rule, for the preceding task's note.
9. Active ADRs touching the same area: `projects/<slug>/adrs/ADR-*.md`
   and `docs/adrs/ADR-*.md`.
10. The existing task note if this task is already in progress.
11. [`lessons.md`](../../../docs/agents/lessons.md) — check your plan
    against each recurring class before writing code. The worked
    examples behind each class are in
    [`lessons-examples.md`](../../../docs/agents/lessons-examples.md);
    open a class's section only when its one-line rule is not enough to
    act on.
12. [`environment.md`](../../../docs/agents/environment.md) — skim; it is
    load-bearing for every command you will run.

**Do NOT** read other phase files unless a Prerequisites link pulls you
there, and do not read the whole `projects/` tree "to get oriented".

**Context discipline.** Three rules, each from a measured transcript
(ADR-0002 carries the numbers: three Phase 04–05 tasks each ended
between 513k and 644k tokens of context, and the mandatory documents
above were under 35k of it — the agent's own output and ad-hoc source
reads were the rest):

- **Delegate survey reads.** Reading a whole `.rs` or test
  module to answer a bounded question — which constants a kernel reads,
  where a symbol is dispatched, what a generated file contains — is an
  `Explore` subagent's job: hand it the question and the paths and take
  back the conclusion with `file:line` citations, not the file. Read a
  file yourself only when you are about to edit it, and then by symbol
  (`rg -n 'def name' file`, then `sed -n 'a,bp'` on the range you need),
  not by sweeping `sed -n` windows through it.
- **Never echo generated artifacts.** Write generated output — a
  transpiled expression, a disassembly, a captured array, a full
  `pytest -v` log — to a file under your scratch directory, then inspect
  a narrow range (`wc -l`, `grep -c`, `sed -n '1,20p'`, a `diff` against
  the expected form). A multi-thousand-character expression printed into
  the transcript is paid for on every later step and read by nothing.
- **Write once; do not re-read what you wrote.** A heredoc or `Write`
  payload is already in context; confirm it landed with `wc -l` or
  `grep -n` on the lines you care about, not `cat`. The same goes for
  the task note: a note written in ten chunks and re-read between them
  costs its size twice.

**Template-file awareness:** files named `_template.md` are references,
not artifacts. Glob for artifact patterns explicitly:

| Directory     | Artifact glob                                             |
|---------------|-----------------------------------------------------------|
| `adrs/`       | `ADR-[0-9][0-9][0-9][0-9]-*.md`                           |
| `task-notes/` | `task-[0-9]*-*.md` (flat) or `phase-*/task-*.md` (phased) |
| `phases/`     | `phase-[0-9][0-9]-*.md`                                   |
| `learnings/`  | `phase-[0-9][0-9]-*.md` or `project-retrospective.md`     |

### Step 5: Create or update the task note

- **Template:** `projects/_template/task-notes/_template.md`.
- **Path:** flat → `projects/<slug>/task-notes/task-N-<slug>.md`;
  phased → `projects/<slug>/task-notes/phase-XX/task-X.Y-<slug>.md`.
- Keep it current **while working**, not only at the end.
- **Length budget** (ADR-0002): a task note is evidence, not narrative.
  `## Findings` + `## Decisions and Implementation Notes` together stay
  under ~100 lines; `## Inputs Reviewed` is one line per source;
  `## Open Questions` one paragraph per question; the whole note under
  ~500 lines. The measurement tables, `## Verification` (commands and
  their summary lines — never whole logs), `## Numerical impact`, the
  `## Stale-state sweep` block and `## Handoff to Next Task` are exempt
  because they are pasted evidence. A section about to exceed its budget
  is either a phase-level fact (one line here, the rest in
  `phase-XX/README.md`) or prose to compress — do not weaken a gate to
  meet it. Measured 2026-08-21 (`wc -l`): the three longest notes ran
  775–876 lines (`phase-04/task-4.4` 876, with a 254-line
  `## Findings`; `phase-01/task-1.3` 815; `phase-04/task-4.5` 775), and
  the Phase 04 learnings condensed all six Phase 04 notes into 245 lines.

**Fill in Exit Criteria first**, before you implement — copy the concrete
bullets from the phase file's `**Exit criteria:**` block (phased) or the
task row and PLAN prose (flat). That is what "done" means.

`## Verification` is the retrospective record of what you ran, and must
contain the exact `pytest` command(s) and their real summary lines
(`23 passed, 1 skipped`), a categorized list of what the tests cover (not
just a count), and anything intentionally deferred with a reason.

### Step 6: Execute the task

Keep changes scoped to the chosen task. Verify against the Exit Criteria
you wrote in Step 5. If blocked, record the blocker and current state in
the task note before stopping.

#### 6a: Testing discipline

Every public function and every non-trivial path needs at least one test
that would fail if the implementation were wrong.

- **Pin a number, not a shape.** For any physics change, assert against
  an analytic limit, a published value (cite the source in the test), an
  independent implementation, or a stored regression array. A test that
  only checks `shape`, `> 0`, or `np.isfinite` pins nothing.
- **State the tolerance and why.** An unexplained `rtol` invites silent
  loosening later. Choose it from the expected numerical error, not from
  what makes the test pass.
- **Boundaries:** threshold energy, the spectrum endpoint, zero mass,
  equal masses, the massless limit, an empty array, a scalar where an
  array is expected and vice versa, negative and NaN input.
- **Broadcasting:** a spectrum-shaped function gets a scalar test *and*
  an array test.
- **Error paths:** every raise and documented failure mode has a test
  that triggers it.
- **Test validity (stash-proof):** for every new/modified test asserting
  a behavior change, prove it fails without your fix — `git stash` the
  production change, run, confirm failure, `git stash pop`. A test that
  still passes with the fix reverted is not testing what its name
  implies.
- **Symmetry with prior tasks:** if a prior task covered one channel or
  one direction, the symmetric variant should exist unless the spec
  defers it.

Do not count tests by hand:

```sh
pytest test/spectra --collect-only -q | tail -n 1
```

Guard against a false green: `pytest` exits **5** on zero collected, and
a `-k` filter matching nothing exits 0 with `no tests ran`. A bare
`pytest` is the full suite — `pyproject.toml`'s
`testpaths = ["hazma", "test"]`, the same collection CI runs — so that is
the run that gates your commit, and it takes minutes rather than seconds
because of the parity corpus. Cite the command you ran and record the
real count in the task note.

**Correctness shapes to defend:** adding an entry to a dispatch table (a
final-state → spectrum-function map, a channel list) requires sweeping
every sibling lookup, `__all__`, and test; a constant becoming a
user-supplied argument needs a validity guard; an invariant must hold at
every public entry point, not just the one the happy-path test hits;
validate before writing to any module-level cache or table.

#### 6b: Measure the numerical impact

**Mandatory whenever the diff can reach a public code path.** Before and
after your change, evaluate every public function the diff can reach on a
representative grid and diff the arrays. Record in the task note:

- the functions checked and the grid used,
- whether any value moved, and by how much,
- whether that is intended.

`No public value changes (verified: <command>)` is a valid result.
Silence is not. If a value moved, say so in the task note, append a line
to the project's numerical-impact log — `task-notes/numerical-impact.md`
where the project keeps one (cython-to-rust does), otherwise the
working-memory README's **Numerical impact so far** section — and
re-check the project's `version_bump:` level against
[`versioning.md`](../../../docs/versioning.md) — a moved published number
is `minor`, not `patch`.

#### 6c: Verify your claims

Every factual claim in comments, docstrings, the task note, and any PR
description needs a citation or a command + output — never an estimate.
Run [`doc-consistency.md`](../../../docs/agents/doc-consistency.md) §1,
§2, and §7 against your diff. Three implementer-only checks:

- **Docstring parameters** match the actual signature, and **every
  physical quantity states its units**.
- **Performance claims** ("vectorized", "2x faster") are backed by a
  measurement with a stated grid size — never by intent.
- **External-behavior assertions need a primary source in the same
  paragraph.** "Matches PPPC4DMID", "reproduces Eq. 14 of
  arXiv:1907.11846" must carry a DOI, arXiv id + equation number,
  permalink, or command + output. Confident prose without a citation is a
  fabrication risk — if you cannot cite, soften or remove the claim. A
  self-citing "23 passed via `pytest test/spectra`" already satisfies
  this.

#### 6d: Durable-doc sweep (implementer pass)

Before staging, run
[`doc-consistency.md`](../../../docs/agents/doc-consistency.md) over
every durable doc your change touches or references — you run it
shift-left; Reviewer D re-runs it at review. Load-bearing points:
re-derive every numeric claim from the live tree and replace **every**
occurrence via the §11 stale-sibling sweep; treat the task note's
`## Verification` as regenerated, not curated; reconcile ADR §Body ↔
§Consequences, phase-file gate text, phase README row, and the `PLAN.md`
summary line; grep `docs/source/` for any renamed public object; and
sweep **new** files too (§12).

### Step 7: Handle plan changes correctly

- **Local implementation detail** → task note only.
- **Durable context later tasks need** → task note now; fold into phase
  learnings or the project retrospective at closure.
- **Out-of-scope deferred work this task surfaced in its touched area**:
  1. `rg` [`docs/followups/`](../../../docs/followups/) for the
     identifier. If an entry covers it, link that file in the task note's
     `## Open Questions` and stop.
  2. Dedup against open PRs — `gh pr list --state open`, then
     `gh pr diff <n> --name-only | grep followups`.
  3. Otherwise drop a stub in `docs/followups/todo/<slug>.md` (from
     `_template.md`), add a row under "Open" in
     `docs/followups/README.md`, and note it in `## Open Questions`.

  Trivial cleanups you could do in the same PR need no follow-up file —
  just do them. Drive-by `TODO`s outside your touched area are out.
  **Wrong-premise guard:** a followup can prescribe a mechanism that no
  longer matches the code — before building on one, `rg` its cited
  symbols and run its cited command against current code.
- **Canonical change to architecture, invariants, interfaces, task
  ordering, units, or normalization conventions** → write an ADR, patch
  the affected phase file or `rules.md`, and update the one-line summary
  in `PLAN.md` only if it is now wrong.

**ADR placement:** could someone read this ADR without knowing which
project produced it and still find it useful? Yes → `docs/adrs/`; No →
`projects/<slug>/adrs/`. Default bias: start project-scoped.

**Canonical-contract diff.** Before reporting `PLAN_IMPACT: None`, open
the canonical phase file and every active ADR, read each gate sentence
and exit-criterion bullet, and check it against what you shipped. If any
is now factually wrong, patch it in this same task — do not defer a
canonical-contract patch to a follow-up.

### Step 8: Finish with a handoff

Re-run the 6c/6d checks, then grep for `TODO`, `FIXME`, `breakpoint()`,
`pdb`, and stray `print()` you introduced — resolve, document, or file a
follow-up.

**Run the preflight gate:**

```sh
scripts/agents/preflight.sh --paths "<touched paths>"
```

Rationale and manual fallback:
[`preflight.md`](../../../docs/agents/preflight.md). A `WARN` row means a
tool is missing and its gate did not run — that is a hole, not a pass.
Non-zero exit is a blocked handoff.

Then **record task status in two places**:

1. **Per-task note:** set `**Status:**` to `Complete` / `Blocked` /
   `Superseded`. Record files changed, verification performed, the
   numerical-impact result, `## Plan Impact`, and
   `## Handoff to Next Task`.
2. **Working memory:** flat → `task-notes/README.md`; phased → this
   task's `Status` cell in `task-notes/phase-XX/README.md` (touch the
   project-level README only for cross-phase material and the Phases
   table). Every status change updates: the Tasks-table cell;
   **Findings** (only what outlives this task); the numerical-impact
   log (`numerical-impact.md`, or the README's **Numerical impact so
   far** section where no separate log exists); **Decisions**
   (one-liners with rationale, linking an ADR if written); **Open
   Questions**; **Files Changed** (a `### Task N` roll-up); and a
   rewritten `## Handoff to Next Task`. The project-level README is a
   head file of roughly 5k tokens — tables, open questions, handoff and
   pointers — so append one-liners, not narrative, and keep the handoff
   to what the next task needs (ADR-0002).

Then handle closure:

- **Phased, last task in the phase:** synthesize
  `learnings/phase-XX-<slug>.md`, set the phase file frontmatter
  `status: Complete`, update its cell in `PLAN.md`'s `## Phases` table.
  Then sweep the project README: move the closed phase's Findings,
  Decisions, Files Changed and Verification entries **verbatim** into
  `task-notes/history-<section>.md` (same directory as the README so the
  moved text's links keep resolving; shape and provenance header as in
  `projects/cython-to-rust/task-notes/history-findings.md`), leaving the pointer
  paragraph in place — the learnings file you just wrote replaces them
  for every later reader (ADR-0002). Nothing is deleted or summarised.
- **Last task overall:** synthesize `learnings/project-retrospective.md`.
  For every substantive §5 follow-on seed, file a
  `docs/followups/todo/<slug>.md` stub with an index row, and
  cross-check **all** of `docs/followups/todo/` for entries sourced from
  this project. Set `PLAN.md` `status: Complete` and move the row from
  Active to Completed in `projects/README.md` with the Shipped date.

  **Then bump the version (mandatory closure step).** Read
  `version_bump:` from `PLAN.md`, re-confirm the level against the
  project's realized **Numerical impact** and
  [`versioning.md`](../../../docs/versioning.md), set `version` in
  `pyproject.toml`'s `[project]` table, and add a
  `## [X.Y.Z] — YYYY-MM-DD` section to
  `CHANGELOG.md` naming the slug and stating any numerical change and its
  magnitude. Verify with `scripts/agents/preflight.sh --closing`.

### Step 9: Self-review before the commit boundary

**Commit boundary:** this skill does **not** commit or push unless its
caller instructs it. Either way the preflight gate and the
branch/worktree assertion in
[`preflight.md`](../../../docs/agents/preflight.md) MUST run immediately
before that commit — confirm `git rev-parse --abbrev-ref HEAD` is the
intended branch (**never `master`**) and `--show-toplevel` is the
worktree. Never batch edit→gate→commit→push into one parallel tool block.

After Step 8 and before staging, do a fresh-eyes pass on your **full**
diff. Step 8's bookkeeping is part of the deliverable and gets the same
scrutiny.

1. **Re-read the task spec end-to-end** and capture every deliverable,
   gate sentence, and constraint as an explicit checklist.
2. **Inventory the full change** with `git status --short` and
   `git diff origin/master --` (single-arg: self-review runs before the
   commit). That does not show untracked files — for each `??`,
   `git add -N <path>` or walk it as a fresh creation.
3. **Walk each checklist item against the change.** Flag anything
   uncovered, scope creep, or divergence from a stated approach.
4. **Sanity-check correctness** on the changed surfaces: branches, error
   paths, empty/NaN inputs, and whether the new tests would actually fail
   without the production change.
5. **Re-confirm the doc sweep and bookkeeping are internally
   consistent.** The Tasks-table cell, the `**Status:**` header, any
   phase `status:` flip, the `projects/README.md` row on closure, and the
   synthesized learnings must agree with each other and the diff. On
   closing PRs re-run `scripts/agents/preflight.sh --closing`.
6. **Produce a `## Stale-state sweep` block in the task note.** This is
   the forcing function for items 1–5 — describing what to check is not
   enough; paste actual command output. The canonical block shape is in
   [`doc-consistency.md`](../../../docs/agents/doc-consistency.md) under
   "The sweep block". Append it above `## Handoff to Next Task`, running
   each command against the current branch, and include the
   **Numerical-impact statement** row. For a zero-touch sweep rows may
   read "no occurrences" — but still produce the block to prove each
   command ran.

If self-review surfaces a real gap, fix it in the same working tree now.
Do not record "noted for later"; either fix it or set the task-note
status to `Blocked` with a concrete blocker description.

## Output checklist

- Code, tests, or docs changes for exactly one task.
- Updated per-task note with Exit Criteria, Verification, the numerical-
  impact result, Plan Impact, and the `## Stale-state sweep` block.
- Updated working-memory README: Tasks Status cell, Findings, the
  numerical-impact log (`numerical-impact.md` or the README section),
  Decisions, Open Questions, Files Changed roll-up, and a rewritten
  `## Handoff to Next Task` — one-liners, within the head-file budget.
- ADR (project-scoped or repo-wide) if the decision is canonical.
- `PLAN.md` / phase file updated only if canonical scope, ordering, or
  task *shape* changed — never for live status.
- Phase learnings / project retrospective synthesized on closure.
- On project closure: `PLAN.md` `status: Complete`, working-memory
  `**Status:**` set, `projects/README.md` row moved, **and** the version
  bump + CHANGELOG entry (verify with `preflight.sh --closing`).
- Preflight gate green.

## Structured report

End with a block the caller can scrape:

```text
STATUS: Complete | Blocked | Superseded
PROJECT: <slug>
TASK_ID: <e.g., Task 3 or Task 2.4>
TASK_NOTE: <path>
FILES_CHANGED: <count and/or short list>
TESTS: <command> — <literal pytest summary line>
NUMERICAL_IMPACT: <none (verified: <command>) | <function>: <magnitude>>
PLAN_IMPACT: None | Task note only | Phase file patched | ADR-XXXX | Both
NEXT: <what the next agent should read first>
```

## Guardrails

- Do not silently broaden scope.
- Do not start the next task in the same pass unless asked.
- Do not commit or push unless the caller instructs it.
- Do not bury durable findings only in PR text or chat.
- Do not report a numerical result from a tree you did not rebuild after
  a Rust edit, or from an installed hazma rather than the worktree.
- Prefer files and tests over long explanations.
