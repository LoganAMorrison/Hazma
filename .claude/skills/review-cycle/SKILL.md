---
name: review-cycle
description: Orchestrate the full PR review loop — select reviewers, spawn them in parallel, commit and push fixes, verify, post a round comment, and iterate to convergence or a cap. The single review-loop implementation; task-pipeline Phase D delegates here.
---

**Role:** Orchestrate a multi-agent review loop for a pushed PR. You
select reviewers, spawn them in parallel, drive a review-respond agent
that **commits and pushes** fixes to the PR branch, verify the push
landed, run verification reviewers, post a durable round comment, and
iterate until reviewers approve or a cap is reached. This is THE
review-loop implementation in the repo — `/task-pipeline` Phase D
delegates to it rather than re-implementing it.

## When to use this skill

- The user wants a full automated review cycle for an open PR.
- `/task-pipeline` reaches its review phase.

## When NOT to use

- **Uncommitted local work.** Reviewer agents run in isolated contexts
  and fetch the PR by number; they cannot see your working tree. Push the
  branch or open a draft PR first.
- **A single advisory review** → `/review-pr`; **a plan** →
  `/review-plan`.
- **Never recurse.** A review-respond or reviewer subagent must not
  invoke `review-cycle` or `task-pipeline`. There is exactly one
  orchestration level.

## Inputs

Required:

- A **PR number**, **PR URL**, or **pushed branch name**.

Optional:

- **`WT_PATH` / `BRANCH`** — when a caller already owns the shared
  worktree, pass them so review-respond's fixes land there. When absent,
  Phase A sets up a worktree on the PR branch.
- **External reviews** — file paths or pasted text from reviewers outside
  Claude Code. **Advisory only:** they never gate convergence, receive no
  verification pass, do not participate in the all-APPROVE shortcut, and
  are surfaced verbatim in the round comment.
- **Max iterations** — cap on review rounds (default `3`).
- **Reviewer override** — a caller may pin `SELECTED_REVIEWERS`.

## Shared references (do not restate — point)

- [`review-lenses.md`](../../../docs/agents/review-lenses.md) — roster,
  models, effort, `--lens` flags, selection rules, verdict rule, baseline
  duties, per-lens FOCUS rubrics.
- [`preflight.md`](../../../docs/agents/preflight.md) — the pre-commit /
  pre-push gate.
- [`doc-consistency.md`](../../../docs/agents/doc-consistency.md) — the
  checklist Reviewer D runs and review-respond sweeps.
- [`lessons.md`](../../../docs/agents/lessons.md) — recurring
  review-defect classes every reviewer checks first.

---

## Workflow

### Phase A: Setup

1. **Normalize to a PR number.** A number is used directly; a URL yields
   its trailing number; a branch resolves via
   `gh pr list --head <branch> --json number --jq '.[0].number'` (if
   none, ask the user to open one, even a draft). Fetch the
   orchestrator's view (`gh pr view <N>`, `gh pr diff <N>`) and record
   the head branch and SHA (`--json headRefName,headRefOid`).
2. **Resolve the project slug** from the head branch, parsing **both**
   agent prefixes. `<agent>/<slug>/<task-slug>` → project work;
   `<agent>/<short-description>` → ad-hoc; skip the project-spec reads.
3. **Read the task spec** (project work only), for judging reviews and
   detecting scope creep: `PLAN.md` (frontmatter, `version_bump`, the
   **Numerical impact** section, the task row), `rules.md` if present,
   the phase file (phased), the task note, and active ADRs.
4. **Ensure a worktree for fixes.** If `WT_PATH` / `BRANCH` were passed,
   use them. Otherwise `git fetch origin` and add a worktree on the PR
   head branch. Record `WT_PATH` and `BRANCH`.

### Phase B: Review round (repeat up to max iterations)

#### B.0: Select reviewers

Select once, at the start of the loop, per the **selection rules** in
[`review-lenses.md`](../../../docs/agents/review-lenses.md) —
default-include A and D; add B for explicit Exit Criteria, C for
runtime-behavior changes, **E whenever the diff can move a number**
(any change under `hazma/` that is not purely a rename, docstring, or
annotation); always include D on project-closing PRs. Or use the caller's
`reviewer override`.

Record `SELECTED_REVIEWERS` (e.g. `{A, C, D, E}`) with a one-sentence
justification per chosen and per omitted reviewer. The set is **fixed for
every round** (convergence assumes a stable set); surface it in the Round
1 PR comment only.

#### B.1: Spawn selected reviewers in parallel

Use the **Agent tool** to spawn one `general-purpose` agent per ID in
`SELECTED_REVIEWERS`, **in a single message** (parallel tool calls).
Substitute `<ID>`, `<Role>`, `<MODEL>`, `<EFFORT>`, `<LENS>` from the
review-lenses roster row; when `--lens` is `(none)`, omit the
`with lens:` clause.

```text
Agent(
  subagent_type: "general-purpose",
  model: "<MODEL>",       # from the roster: sonnet for A–D, opus for E
  description: "Reviewer <ID> (<Role>)",
  prompt: <template body below>
)
```

**Template body:**

> You are Reviewer `<ID>` (`<Role>`). Use the `/review-pr` skill[ with
> lens: `<LENS>`] to review PR #`<PR_NUMBER>`.
>
> Apply your lens's FOCUS rubric and the baseline duties (fresh-eyes
> PR-head recipe, zero-collection guard, empirical execution, rebuild
> awareness, lessons.md read, verdict rule, new-code correctness shapes)
> from `docs/agents/review-lenses.md` — read your section there. Reviewer
> D runs `docs/agents/doc-consistency.md`.
>
> **Pipeline-managed placeholders — do NOT flag.** A draft placeholder
> title (`chore(pipe): pipeline draft …`) and stub body (`"Draft —
> pipeline in progress."`) are rewritten before the PR leaves draft. Skip
> them; if `gh pr view` shows a real title/body, evaluate it against
> `docs/PR_GUIDELINES.md` as normal.
>
> Your verdict MUST be APPROVE or REQUEST CHANGES — never COMMENT inside
> this loop. Non-blocking-only ⇒ APPROVE and list them. Return your full
> structured review.

Collect all reviewer results (external reviews stay alongside as advisory
context). One branch fires:

- **Shortcut — every selected internal reviewer APPROVES with zero
  comments:** converged; no fix round needed. Post a brief round comment
  (B.5 shortcut form, carrying the Round 1 selection block) and go to
  Phase C. Do **not** run a pointless verification round.
- **Otherwise** (any REQUEST CHANGES, or all APPROVE with at least one
  non-blocking suggestion): proceed to **B.2**.

#### B.2: Review-respond — implement, commit, and push

Spawn a fresh **opus** subagent. Open its prompt with the worktree
directive (`cd <WT_PATH>`; do not create a worktree; branch `<BRANCH>`).

```text
Agent(
  subagent_type: "general-purpose",
  model: "opus",
  description: "Review respond",
  prompt: <review-respond prompt below>
)
```

> Use the `/review-respond` skill to process these reviews for PR
> #`<PR_NUMBER>` (Project `<slug>`, Task `<TASK_ID>` — `<TASK_TITLE>`,
> task note `<TASK_NOTE_PATH>`):
>
> <paste each selected internal reviewer's result, tagged with its ID>
>
> <!-- Only if external reviews were provided: advisory only, no
>      verification pass, do not gate convergence. -->
> <paste any external reviews here>
>
> Follow `/review-respond`: the categorization table, the scope-of-fix
> sweep (`docs/agents/doc-consistency.md` §11), the class-fix rule
> (re-run the CLASS of check across the whole touched artifact set —
> never point-fix the cited line), the measurement rule (reject a
> numerics comment only with pasted numbers), and the lessons-ledger
> append when a finding is class-shaped.
>
> **You MUST commit and push.** Stage only files you intentionally
> changed. Rebuild first (`pip install -e .`, never `cargo build` alone)
> if you touched `rust/` or `pyproject.toml`.
> Run the preflight gate (`docs/agents/preflight.md`) before staging;
> assert a real `N passed` count. Commit with a Conventional Commits
> message (validate with `scripts/agents/check_pr_title.py`) and
> `git push`; report `FINAL_COMMIT_SHA` from `git rev-parse HEAD` after
> the push. Quote the literal pytest summary line — a bare "tests pass"
> is insufficient.
>
> Emit one `## Response to Reviewer <ID>` block per selected reviewer (a
> decisions table: `# | Comment | Category | Action | Rationale`), a
> block per external reviewer using an `Ext-<Label>` tag, then a
> `## Pipeline Report` with STATUS
> (`FIXES_APPLIED | NO_FIXES_NEEDED | BLOCKED`), COMMENTS_FIXED,
> COMMENTS_REJECTED, COMMENTS_ACKNOWLEDGED, NUMERICAL_IMPACT,
> FINAL_COMMIT_SHA.

**Verify the push landed** before spawning verifiers — the whole loop
reviews stale code otherwise.
`gh pr view "${PR_NUMBER}" --json headRefOid --jq .headRefOid` must equal
the agent's `FINAL_COMMIT_SHA` and differ from the pre-round head. If it
did not advance and the agent claimed fixes, stop and report. If the push
was rejected (branch behind), the review-respond agent fetches and
rebases onto `origin/<BRANCH>` and re-pushes; never force-push a shared
branch.

#### B.3: Verification reviewers in parallel

Spawn one verification agent per **selected** reviewer, same model and
lens as Round 1, in a single parallel message. Each fetches the PR head
fresh (not the worktree).

```text
Agent(
  subagent_type: "general-purpose",
  model: "<same model as the original reviewer>",
  description: "Verification — Reviewer <ID>",
  prompt: "You are Reviewer <ID> (<Role>). You previously reviewed
           PR #<PR_NUMBER> and raised these comments:

  <your Round N review, verbatim>

  The implementer responded:

  <that reviewer's ## Response to Reviewer <ID> block from B.2>

  Fetch the PR head fresh (baseline fresh-ref recipe in
  docs/agents/review-lenses.md): delete any stale ref, then
  `git fetch origin pull/<PR_NUMBER>/head:refs/remotes/origin/pr/<PR_NUMBER>`,
  verify the SHA against `gh pr view`, never verify the ambient checkout.

  **Live-tree verification (not snapshot).** Before REQUEST CHANGES on a
  volume claim ('N stale refs survive', 'M criteria uncovered'), re-run
  the command against the freshly-fetched diff and quote the output —
  review-respond may have committed since your first read. Such a verdict
  with no pasted command output is not blocking.

  **Numerics claims need numbers.** If you asserted a value was wrong and
  the implementer pasted a measurement, re-run it yourself before holding
  the comment UNRESOLVED.

  **Stale-state sweep** (per docs/agents/doc-consistency.md §11): for any
  count / command / identifier / unit / qualitative claim you raised, `rg`
  it across the rest of the touched docs; the original value surviving
  anywhere — even in an untouched section — is a NEW blocking issue.

  Mark each original comment RESOLVED / PARTIALLY RESOLVED / UNRESOLVED,
  then list NEW issues in your area. Verdict MUST be APPROVE or
  REQUEST CHANGES.

  ## Verification — Reviewer <ID> (<Role>)
  | # | Original Comment | Status | Notes |
  |---|-----------------|--------|-------|

  ## New Issues
  ### Blocking
  <numbered list, or 'None'>
  ### Non-blocking
  <numbered list, or 'None'>

  ## Verdict: <APPROVE | REQUEST CHANGES>"
)
```

If review-respond returned `NO_FIXES_NEEDED`, verification still runs —
verifiers independently judge whether each rejection was justified, so
the responder cannot unilaterally overrule blocking feedback.

#### B.4: Convergence

| Condition | Action |
|-----------|--------|
| Every selected verifier APPROVES, no new blocking issues | **Converged** → B.5, then Phase C |
| Any UNRESOLVED blocking comment or new blocking issue | Loop to B.2 with verification results as the new input |
| Same blocking comment persists two consecutive rounds | **Escalate** to the user — needs human judgment |
| `max-review-iterations` reached | **Stop**, not converged; carry the unresolved list forward |

Each round must resolve at least one blocking issue; a round with zero
progress on blocking items escalates rather than looping.

#### B.5: Post the round comment to the PR

Only after B.4 has decided the round outcome, post one durable comment so
the PR timeline records why the code ended up as it did. This runs
**every** round — looping, cap-reached, escalated, and the shortcut path
(brief form). Use `<details>` blocks so the verdicts table stays visible.

```markdown
## Automated Review — Round <N>

<!-- Round 1 only: the B.0 reviewer-selection block (chosen + omitted,
     one reason each). -->

### Verdicts
| Reviewer | Role | Model | Initial | Verification |
|----------|------|-------|---------|--------------|
<!-- One row per SELECTED reviewer, A→E order. -->

**Round outcome:** <Converged | Not converged — looping to Round <N+1>
| Not converged — iteration cap reached | Escalated — blocking comment
persisted two rounds>

<!-- One <details> block per selected reviewer (A→E): initial key
     comments, then the verification status table + New Issues. -->

<!-- External Reviews section only when provided — advisory, no
     verification column, not counted toward convergence. -->

### Decisions
| # | Source | Comment | Decision | Rationale |
|---|--------|---------|----------|-----------|
<!-- Built from review-respond's per-reviewer tables; Source = <ID>.<n>
     or Ext-<Label>.<n>. -->

### Numerical impact
<review-respond's NUMERICAL_IMPACT for this round, or "no public code
path touched">

### Fix commit
<short SHA + one-line message from B.2, or "NO_FIXES_NEEDED — <reason>">
```

Post via `gh pr comment "${PR_NUMBER}" --body-file <file>` from the
worktree, then remove the temp file. Increment `<N>` each round.

### Phase C: Final summary

Report to the user; do not clean up the worktree or branch. When called
by `/task-pipeline`, these fields are the review phase's return value.

```text
## Review Cycle Summary

**Target:** PR #<N> / branch `<name>`
**Project / Task:** <slug> — <TASK_ID> / "<title>"
**Selected reviewers:** <SELECTED_REVIEWERS + models>

### Per round
- Round N — <per-reviewer verdicts, resolved/new-issue counts, changes>.

### Final state
- **STATUS:** CONVERGED | NOT_CONVERGED (iteration cap) | ESCALATE
- **ITERATIONS_USED:** <N>
- **UNRESOLVED:** <list with reviewer attribution, or "none">
- **FILES_MODIFIED:** <list>
- **TESTS:** <literal pytest summary line from the final round>
- **NUMERICAL_IMPACT:** <none (verified: <command>) | <function>:
  <magnitude>>
- **FINAL_COMMIT_SHA:** <sha; MUST equal
  `gh pr view <N> --json headRefOid` — verify against origin>
- **REVIEW_SUMMARY:** <one-paragraph synthesis, e.g. "Converged in 2
  rounds. Round 1: 3 REQUEST CHANGES, 7 fixes applied. Round 2: all
  APPROVE.">
- **External review re-verification needed:** yes / no

Callers (`/task-pipeline` Phase D.2) parse the literal `STATUS`,
`ITERATIONS_USED`, `UNRESOLVED`, `FINAL_COMMIT_SHA`, and
`REVIEW_SUMMARY` keys — keep the tokens exact.
```

Handoff: this skill does not finalize the PR title/body. Standalone, the
user (or `/commit-and-pr`) finalizes; under `/task-pipeline`, Phase E
does — report `STATUS` so the caller routes accordingly.

---

## Guardrails

- **Iteration cap / progress required:** default 3 rounds; never loop
  indefinitely. Each round must resolve at least one blocking issue —
  zero progress ⇒ stop and escalate.
- **Fixes must be pushed before verification.** B.2 commits and pushes;
  the orchestrator verifies the PR head advanced before spawning
  verifiers.
- **Preflight gate before every commit.** review-respond runs
  [`preflight.md`](../../../docs/agents/preflight.md), including a
  rebuild when `rust/` or the build config changed. There are no
  pre-commit hooks here.
- **No fabricated test claims.** Every TESTS line quotes the literal
  pytest summary; a zero-collection run is a false green.
- **No unmeasured numerics verdicts.** Neither a reviewer nor the
  responder settles "did the number change?" by argument. Numbers or it
  did not happen.
- **Scope discipline:** reviewer suggestions do not expand the task
  beyond its boundary. Acknowledge out-of-scope suggestions (or file a
  followup); do not implement them here.
- **Test gate / no silent retries:** never proceed to verification with
  failing tests, and never revert a fix just to make tests green.
- **Reviewer identity is stable across rounds** so verification maps back
  to the original comments; the selected set is fixed once in B.0.
- **Roster and models live in review-lenses.md** — this skill selects a
  subset and never re-defines identities or rubrics inline.
- **External reviews are advisory only** — synthesized, never
  auto-verified; the summary flags when re-verification is needed.
- **Template files are not artifacts:** never flag `_template.md` as
  missing content.
- **No recursion:** no subagent may invoke `review-cycle`,
  `task-pipeline`, or another orchestration skill.
