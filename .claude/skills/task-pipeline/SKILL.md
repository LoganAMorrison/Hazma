---
name: task-pipeline
description: Orchestrate one project task end-to-end — resolve, implement, review, ship — delegating each phase to a context-isolated subagent over a shared worktree.
---

**Role:** Act as the lightweight orchestrator that drives one task from
implementation through review to a shipped PR, delegating each phase to a
fresh subagent with full context isolation. The orchestrator stays slim:
it reads structured `## Pipeline Report` blocks, makes go/no-go
decisions, and points at the shared agent layer
([`docs/agents/`](../../../docs/agents/README.md)) rather than restating
its rules.

**When to use this skill**

- The user asks to "run the pipeline", "implement and ship", or "do one
  task end-to-end".
- The user wants automated implement → review → PR without manual
  `/clear` steps.

**When NOT to use this skill**

- **Ad-hoc work with no project** — the pipeline hard-requires a
  `projects/<slug>/PLAN.md` (Phase A stalls without one). Route these to
  a plain commit + PR via `/commit-and-pr`.
- **A review loop with no implementation** — reviewing an already-pushed
  PR is `/review-cycle`; reviewing a working-tree diff is
  `/code-review`.
- **Stress-testing a plan before implementation** — `/review-plan`.

## Inputs

Required:

- A **task identifier** (`Task 5.2` phased, `Task 3` flat), or the
  literal string `next`. Phase A resolves it with
  [`resolve_task.py`](../../../scripts/agents/resolve_task.py).
- A **project slug**, via `--project <slug>` or parsed from the current
  branch (both `claude/` and `codex/` prefixes; the slug is the first
  segment after the prefix). If neither resolves, stop and ask.

Optional:

- **skip-review** — skip Phase D. Default `false`.
- **skip-pr** — skip PR creation and finalization (Phases C and E). If
  review is not skipped, a draft PR is still created for the review phase
  but not finalized. Default `false`.
- **external-reviews** — file paths or pasted text from reviewers outside
  Claude Code. Passed through to `/review-cycle` as **advisory context
  only**. The full contract lives there.
- **max-review-iterations** — cap on review rounds (default `3`).

---

## Workflow

### Subagent preamble

Every subagent prompt that operates inside the worktree (Phase B
implementer, Phase E finalizer, and the review subagents spawned by
`/review-cycle`) opens with the preamble below. Substitute `<WT_PATH>`
and `<BRANCH>` from Phase A.

> Your working directory is `<WT_PATH>`. Change to it immediately with
> `cd <WT_PATH>` before doing anything else. All file operations, git
> commands, and Python commands must run inside this directory.
>
> **CRITICAL: Do NOT create your own worktree or call any
> worktree-creation tool.** You are already in an isolated git worktree
> on branch `<BRANCH>`, branched from the trunk. This worktree is managed
> by the pipeline orchestrator.
>
> Hazma ships one Rust extension, `hazma._core`. If your work touches
> `rust/` or `pyproject.toml`, run `pip install -e .` inside this
> worktree before running tests — `cargo build` alone publishes nothing
> to Python — and confirm `python -c "import hazma; print(hazma.__file__)"`
> resolves inside `<WT_PATH>` — otherwise every result you report comes
> from a different tree.

### Phase A: Setup (orchestrator — no subagent)

#### A.1: Resolve the project slug and task

Resolve the **project slug** (precedence: explicit `--project` → branch
parse → error). Confirm `projects/<slug>/PLAN.md` exists and read its
frontmatter for `phased: true|false`. `PLAN.md` is **not** the source of
live task status — the Tasks tables under `task-notes/` are.

Resolve the task **before** creating the worktree:

```sh
scripts/agents/resolve_task.py --project <slug> [--task <id>]
```

It reads the live Tasks status table (flat → `task-notes/README.md`;
phased → the current phase's `task-notes/phase-XX/README.md`), skips
`_template.md`, and prints JSON: `{status, task_id, task_title,
task_slug, phase, reason}`. Omit `--task` for the lowest-numbered
non-Complete row. On `status: done` the project has no open task — stop
and report. On `status: error` fall back to reading the table by hand.

Record `TASK_ID`, `TASK_TITLE`, and `TASK_SLUG`.

#### A.2: Create the shared worktree

```sh
scripts/agents/setup_task_worktree.sh \
  --project <slug> --task-slug <TASK_SLUG> --agent claude
```

It fetches origin, resolves the trunk from `origin/HEAD` (falling back to
`master`), always branches from it (never ambient HEAD), picks a
collision-free name, and verifies HEAD before reporting success.

**Branch policy (canonical).** Project branches are
`<agent>/<project-slug>/<task-slug>`. This skill runs under Claude Code,
so it **creates** with `--agent claude` and a matching
`.claude/worktrees/<slug>/<task-slug>/` base. Always *parse* both
prefixes (A.1); *create* with the running agent's.

The script prints one line of JSON:
`{"branch":…,"wt_path":<absolute>,"head_sha":…}`. Record `BRANCH` and the
absolute `WT_PATH`.

#### A.3: Verify the worktree

```sh
git -C "${WT_PATH}" rev-parse HEAD
git -C "${WT_PATH}" status --short
```

Confirm HEAD matches the trunk SHA the script reported and the worktree
is clean.

---

### Phase B: Implementation (fresh subagent)

```text
Agent(
  subagent_type: "general-purpose",
  model: "opus",
  description: "Pipeline — implement",
  prompt: <implementation prompt below>
)
```

**Implementation agent prompt:** open with the
[Subagent preamble](#subagent-preamble), then continue with:

> You are the implementation engineer. Use the `/execute-single-task`
> skill exactly as written, with these pipeline-specific overrides:
>
> - **Skip Step 3** (entering a worktree) — already done.
> - **Project slug:** `<project-slug>`.
> - **Task:** `<TASK_ID>` — `<TASK_TITLE>`.
> - Execute every other step. `/execute-single-task` carries the full
>   gate itself — the numerical-impact measurement, the durable-doc
>   sweep, test-validity preflight, canonical-contract diff, Step 9
>   self-review, the `scripts/agents/preflight.sh` gate, and the
>   project-closure version bump are all defined there. Do not
>   re-implement or second-guess them here.
> - **Commit and push.** Stage only files you intentionally changed
>   (never `git add -A` blindly; never commit build output). Immediately
>   before committing, run the preflight gate and the branch/worktree
>   assertion from `docs/agents/preflight.md` (`git rev-parse
>   --abbrev-ref HEAD` is `<BRANCH>`, never `master`; cwd is
>   `<WT_PATH>`). Write a Conventional Commits message — validate the
>   header with `scripts/agents/check_pr_title.py "<header>"` before
>   committing. Then `git push -u origin HEAD`.
> - **Record the exact commit SHA** via `git rev-parse HEAD` after push;
>   report it as `COMMIT_SHA`.
>
> **Output format — you MUST end your response with this exact section:**
>
> ```text
> ## Pipeline Report
>
> STATUS: <COMPLETE | BLOCKED>
> PROJECT: <project slug>
> TASK_ID: <e.g. Task 5.2 or Task 3>
> TASK_TITLE: <short title>
> TASK_NOTE_PATH: <projects/<slug>/task-notes/... path>
> BRANCH: <branch name>
> COMMIT_SHA: <output of git rev-parse HEAD after push>
> FILES_CHANGED: <comma-separated list of changed files>
> SUMMARY: <one-paragraph summary of what was implemented>
> NUMERICAL_IMPACT: <none (verified: <command>) | <function>: <magnitude>>
> BLOCKER: <description if BLOCKED, "none" if COMPLETE>
> PLAN_IMPACT: <None | Task note only | Phase file patched | ADR-XXXX | Project closure>
> ```

#### B.2: Extract implementation results

Parse **only** the `## Pipeline Report`. Do not read the full narrative —
the orchestrator stays slim.

- `STATUS == BLOCKED`: stop the pipeline. Report the blocker. The
  worktree remains for inspection.
- `Pipeline Report` missing: treat as failed. Report the raw output
  (first and last ~60 lines).

#### B.3: Verify the branch and commit landed

Do not trust the self-reported SHA:

```sh
git -C "${WT_PATH}" fetch origin "${BRANCH}"
git -C "${WT_PATH}" rev-parse "origin/${BRANCH}"
```

The `origin/<BRANCH>` HEAD must equal `COMMIT_SHA`. If the branch is
absent from origin, or the SHAs disagree, report failure and stop.

#### B.4: Check skip flags

If both `skip-review` and `skip-pr` are `true`, skip to Phase F.

---

### Phase C: Draft PR (orchestrator — no subagent)

The orchestrator opens the draft PR **inline**. The draft lets Phase D
reviewers use native PR tooling (`gh pr view`, `gh pr diff`).

Safe placeholder title (scope `pipe`):

```sh
DRAFT_TITLE="chore(pipe): pipeline draft <TASK_SLUG>"
```

Shorten the slug if the header would exceed 69 characters (validate with
`scripts/agents/check_pr_title.py "${DRAFT_TITLE}"`). Phase E rewrites
the title with a real scope before marking the PR ready.

Write a temporary body file inside the worktree:

```markdown
## Task
<TASK_ID>: <TASK_TITLE>
Task note: `<TASK_NOTE_PATH>`

## Summary
- Draft — pipeline in progress. Title and body finalized after review.
```

Open the draft and capture its number:

```sh
cd "${WT_PATH}" && \
  gh pr create --draft --base master --head "${BRANCH}" \
    --title "${DRAFT_TITLE}" --body-file pr_draft_body.md && \
  rm pr_draft_body.md
PR_NUMBER="$(gh pr view --json number --jq .number)"
```

Record `PR_NUMBER`. If `skip-review` is `true`, skip to Phase E.

---

### Phase D: Review (delegate to `/review-cycle`)

`/review-cycle` is the single review-loop implementation. The
orchestrator runs its workflow **itself** — it does not spawn a level-2
orchestration subagent — passing the shared worktree so fixes land in
`WT_PATH`. Reviewer, review-respond, and verification subagents are
spawned by `/review-cycle`'s own phases; that keeps subagent nesting at
one level.

Run the [`/review-cycle`](../review-cycle/SKILL.md) workflow with
`PR_NUMBER`, `WT_PATH`, `BRANCH`, `external-reviews`, and
`max-review-iterations`.

`/review-cycle` owns reviewer selection (per
[`review-lenses.md`](../../../docs/agents/review-lenses.md)), the
per-round PR comment, the commit-and-push of review fixes, the
verification rounds, and the convergence shortcut. Do not restate the
roster here.

#### D.2: Capture the review outcome

From `/review-cycle`'s final summary, capture: `STATUS`
(`CONVERGED | NOT_CONVERGED | ESCALATE`), `ITERATIONS_USED`,
`UNRESOLVED`, `NUMERICAL_IMPACT`, `FINAL_COMMIT_SHA` (already verified
against origin; falls back to Phase B `COMMIT_SHA` when the all-APPROVE
shortcut fired), and `REVIEW_SUMMARY`.

Route on `STATUS`:

- `CONVERGED` → Phase E.
- `NOT_CONVERGED` → Phase E, but Phase E leaves the PR a draft and lists
  the unresolved items in the body.
- `ESCALATE` → stop the pipeline. Report the stuck items. The worktree
  and draft PR remain for manual resolution.

If `skip-pr` is `true`, skip to Phase F.

---

### Phase E: PR finalization (fresh subagent)

```text
Agent(
  subagent_type: "general-purpose",
  model: "sonnet",
  description: "Pipeline — finalize PR",
  prompt: <finalization prompt below>
)
```

**PR finalization agent prompt:** open with the
[Subagent preamble](#subagent-preamble), then continue with:

> You are finalizing draft PR #`<PR_NUMBER>`. Do NOT use
> `/commit-and-pr` — the PR already exists and the code is already
> committed.
>
> **Context:**
> - Project: `<project-slug>` — Task `<TASK_ID>`: `<TASK_TITLE>`.
> - Branch: `<BRANCH>`; task note: `<TASK_NOTE_PATH>`.
> - Implementation summary: `<SUMMARY from Phase B>`.
> - Numerical impact: `<NUMERICAL_IMPACT>`.
> - Review: `<CONVERGED | NOT_CONVERGED | skipped>`; unresolved:
>   `<UNRESOLVED or "none">`; summary: `<REVIEW_SUMMARY or "skipped">`.
> - Plan impact: `<PLAN_IMPACT from Phase B>`.
>
> **Steps:**
>
> 1. Read `docs/PR_GUIDELINES.md` and the task note.
> 2. Run `git diff --stat origin/master...<BRANCH>` and reconcile the
>    Summary bullets you write against it.
> 3. Compose a Conventional Commits title (`type(scope): subject`) using
>    the guidelines' scope table. Do NOT include task IDs. **Validate
>    before setting it:** `scripts/agents/check_pr_title.py "<title>"`.
>    Rewrite until it passes; do not hand-count.
> 4. Compose the body to `pr_body.md`:
>    ```markdown
>    ## Summary
>    - <bullets from implementation and review outcomes>
>
>    ## Project
>    `projects/<project-slug>/` — <TASK_ID>: <TASK_TITLE>.
>    See `<TASK_NOTE_PATH>` for detail, decisions, and verification.
>
>    ## Numerical impact
>    <NUMERICAL_IMPACT — name the functions and the magnitude, or state
>    "no public code path touched". Never omit this section.>
>
>    ## Review
>    - Internal review: <converged in N rounds | not converged — N
>      unresolved items | skipped>
>    - <list unresolved items if any>
>
>    ## Test plan
>    - <verification commands + real output, or cite the task note's
>      ## Verification section — do not invent green results>
>    ```
>    **If `<PLAN_IMPACT>` is `Project closure`,** insert a `## Versioning`
>    section between `## Project` and `## Numerical impact`. Read the new
>    version from `pyproject.toml`'s `[project] version`; the prior is in
>    `git show origin/master:pyproject.toml`:
>    ```markdown
>    ## Versioning
>    Closing project — version bumps `<OLD>` → `<NEW>`
>    (`<patch | minor | major>` per `PLAN.md` `version_bump:`). New
>    `CHANGELOG.md` entry under `## [<NEW>]`. See `docs/versioning.md`.
>    ```
> 5. Update the PR: `gh pr edit <PR_NUMBER> --title "<title>"
>    --body-file pr_body.md`, then `rm pr_body.md`.
> 6. If review is `CONVERGED` (or skipped): `gh pr ready <PR_NUMBER>`,
>    then watch CI to a bounded conclusion:
>    `gh pr checks <PR_NUMBER> --watch --fail-fast`. Report the outcome;
>    **fixing a CI failure is out of scope** — surface it, do not loop on
>    it. If review is `NOT_CONVERGED`, leave the PR a draft and ensure the
>    body's `## Review` section lists the unresolved items.
>
> **Output format — you MUST end your response with this exact section:**
>
> ```text
> ## Pipeline Report
>
> STATUS: <PR_READY | PR_DRAFT | PR_FAILED>
> PR_URL: <URL or "none">
> PR_TITLE: <the title used>
> CI_STATUS: <passing | failing | pending | not-watched>
> ERROR: <description if PR_FAILED, "none" otherwise>
> ```

Parse the report: `PR_READY` → record URL and `CI_STATUS` (report a
failing CI to the user; the pipeline does not fix CI). `PR_DRAFT` →
record the URL and the unresolved items. `PR_FAILED` → report the error;
the worktree and draft PR remain for manual recovery.

Before Phase F, verify `gh pr view <PR_NUMBER> --json headRefOid` equals
`FINAL_COMMIT_SHA`.

---

### Phase F: Summary (orchestrator — no subagent)

Report the final pipeline state. Do **not** clean up the worktree or
branch — the user decides when to delete them.

```text
## Pipeline Summary

**Project:** <project-slug>
**Task:** <TASK_ID> — <TASK_TITLE>
**Branch:** <BRANCH>
**Worktree:** <WT_PATH>

### Implementation
- Status: <COMPLETE | BLOCKED>
- Commit: <COMMIT_SHA>
- Files changed: <FILES_CHANGED>
- Summary: <SUMMARY>

### Numerical impact
<NUMERICAL_IMPACT>

### Review
- Status: <CONVERGED | NOT_CONVERGED | ESCALATE | skipped>
- Iterations: <ITERATIONS_USED>
- Unresolved: <UNRESOLVED or "none">
- Final commit: <FINAL_COMMIT_SHA>

### PR
- Status: <PR_READY | PR_DRAFT | PR_FAILED | skipped>
- URL: <PR_URL>
- Title: <PR_TITLE>
- CI: <CI_STATUS>

### Plan Impact
<PLAN_IMPACT from Phase B; note if a phase file or PLAN frontmatter was
flipped to Complete, or a project moved from Active to Completed in
projects/README.md.>

### Versioning (only when PLAN_IMPACT is `Project closure`)
- Package version: <OLD> → <NEW> (<patch | minor | major>)
- CHANGELOG entry: `## [<NEW>]` added to `CHANGELOG.md`
```

---

## Guardrails

- **Orchestrator stays slim.** For Phases A, B, C, E, and F, read only
  `## Pipeline Report` sections and make go/no-go decisions. Phase D runs
  the `/review-cycle` workflow, which does its own reading. The
  orchestrator never reads source code or implements fixes.
- **Never commit to `master`.** The pipeline always branches in Phase A
  and every write uses `git -C <worktree>`. The branch/worktree assertion
  runs immediately before every commit. Note the trunk is `master`, not
  `main` — an assertion written against `main` protects nothing here.
- **Trust no self-reported SHA.** B.3 verifies `COMMIT_SHA` against
  `origin/<BRANCH>`; Phase E verifies the final HEAD against
  `FINAL_COMMIT_SHA`.
- **No nested worktrees.** The orchestrator creates the one worktree in
  Phase A. Every subagent prompt tells the agent to `cd` into `WT_PATH`
  and **not** call any worktree-creation tool.
- **Rebuild discipline is stated in every worktree prompt.** A subagent
  reporting green tests against a stale `hazma._core` is a failed phase,
  not a pass.
- **Bounded subagent nesting (one level).** The orchestrator spawns the
  Phase B implementer and the Phase E finalizer directly; Phase D's
  subagents come from the inline `/review-cycle` workflow. No subagent
  may spawn another orchestration skill.
- **The preflight gate is mandatory before every commit** —
  implementation and review-fix alike. There are no pre-commit hooks in
  this repo and CI's lint pass is advisory.
- **Numerical impact is a required field**, not an optional one. A Phase
  B report with `NUMERICAL_IMPACT` missing or hand-waved is a failed
  phase — the whole point of the pipeline in a physics library is that no
  number moves silently.
- **Draft PR before review** so Phase D reviewers can use `gh pr view` /
  `gh pr diff`.
- **Review rounds are posted to the PR** by `/review-cycle`, every round
  — including rounds that loop, hit the cap, or escalate. The PR timeline
  is the durable record.
- **Do not invoke `/commit-and-pr` for finalization.** That skill assumes
  uncommitted changes and a non-existent PR; Phase E uses `gh pr edit`.
- **Phase boundaries are go/no-go gates.** A missing or failed Pipeline
  Report stops the pipeline. Never silently skip a failed phase.
- **One task per pipeline run.** Do not batch tasks.
- **Subagent prompts are self-contained** — worktree path, branch,
  project slug, task context, explicit skill instructions. Subagents do
  not rely on orchestrator session state.
- **No worktree cleanup; reclaim failed runs.** The orchestrator leaves
  `WT_PATH` and the branch in place. When a run ends BLOCKED / ESCALATE /
  PR_FAILED and the user is done inspecting, reclaim the stale worktree
  with `git worktree remove <WT_PATH>` (and delete the unused branch)
  before re-running, so collision suffixes do not pile up.
- **Project lifecycle bookkeeping belongs to the implementation agent.**
  If the closed task ends a phase or the project,
  `/execute-single-task` handles the frontmatter flips, learnings
  synthesis, the `projects/README.md` move, **and** the version bump +
  CHANGELOG entry. Phase E only surfaces it in the PR body.
- **Template files are not tasks.** `resolve_task.py` skips
  `_template.md`; never pass one to a subagent as a task target.
