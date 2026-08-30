---
name: commit-and-pr
description: Branch, run the preflight gate, commit with Conventional Commits, push, and open a PR that follows the repo's PR guidelines and passes CI.
---

**Role:** Ship the current working-tree changes as a well-formed branch,
commit, and pull request conforming to
[`docs/PR_GUIDELINES.md`](../../../docs/PR_GUIDELINES.md). This skill is
the **sole commit authority** — when a review skill lands fixes, it
follows the same preflight gate and title rules, not a hand-rolled
`git commit`.

## When to use this skill

- The user asks to commit and open a PR, or says "ship it".
- A standalone `/execute-single-task` run finished and its working-tree
  changes need to become a branch, commit, and PR.

## When NOT to use this skill

- You are already inside a `/task-pipeline` or `/review-cycle` run that
  owns the branch and PR — commit to the branch the caller established;
  do not mint a second one.
- The changes span two unrelated concerns that should be two PRs — split
  them first, then run this once per PR.

## Workflow

### Step 1: Read the PR guidelines

Read [`docs/PR_GUIDELINES.md`](../../../docs/PR_GUIDELINES.md) **before**
composing any commit message or PR title. It is authoritative; the rules
below are a reminder.

- Header: `type(scope): subject`, max **69 characters total**.
- `type` ∈ `feat fix chore ci docs test refactor perf style build revert`.
- `scope` is **required**, `^[a-z0-9-]+$`, max 10 chars, not a type name.
  (`phase-space` is 11 chars → rejected; use `phase`.)
- `subject` starts alphanumeric, no trailing `.` or space, lowercase
  first word by convention.

The scope table lives in the guidelines file — read it there rather than
inventing a one-off scope.

### Step 2: Assess the working tree

- `git status` and `git diff` to see what will be committed.
- `git log --oneline -n 10` for recent commit style.
- If there are no changes, stop and tell the user.
- **Numerical-change check.** If the diff can reach a public code path,
  confirm the task note (or your own measurement) records whether any
  returned value moved. A PR that silently shifts a published spectrum is
  the failure mode this repo cares most about — it belongs in the Summary
  and, on a closing PR, in `CHANGELOG.md`.
- **Version-bump check (project-closing PRs only).** If the diff flips a
  `projects/<slug>/PLAN.md` `status:` to `Complete`, the PR must carry
  the version bump in `pyproject.toml`'s `[project] version` and a
  `CHANGELOG.md` entry.
  `/execute-single-task` Step 8 is the canonical place to do it; if you
  arrive here with a closing diff that lacks it, stop and add it.

### Step 3: Create a branch (if needed)

- If already on the caller-established feature branch for **this** task,
  stay on it. Do not pile a commit onto an unrelated stale feature
  branch — if the current branch is not this task's branch, branch fresh.
- If the current branch is `master`, create a new branch. **Never commit
  directly to `master`.** The trunk here is `master`, not `main`.

Branch naming (parse both agent prefixes; create with `claude/`):

- **Project work:** `claude/<project-slug>/<task-slug>`. The slash
  separator is load-bearing — `review-pr` and `task-pipeline` parse it.
- **Ad-hoc work:** `claude/<short-description>`.

### Step 4: Run the preflight gate

Run it **before** you stage anything:

```sh
scripts/agents/preflight.sh --paths "<touched paths>" \
    [--tests "<narrow targets>"] [--md "<changed .md files>"] [--closing]
```

See [`preflight.md`](../../../docs/agents/preflight.md) for the
rationale, the zero-collection trap, and the manual fallback. There are
no pre-commit hooks in this repo, and CI's lint pass is narrow
(`black --check` plus `ruff check --isolated --select E9,F63,F7,F82`, so
no import-order or configured-rule check) — this gate catches more than
CI will. A non-zero exit is
a blocked commit; a `WARN` row is an unrun gate, not a pass.

If the diff touched `rust/` or `pyproject.toml`, rebuild
(`pip install -e .`) **before** the gate, not after. `cargo build` is
not that rebuild: it refreshes `rust/target/`, not the
`hazma/_core.abi3.so` Python imports.

### Step 5: Stage and commit

- Stage only the files relevant to the change. Do **not** `git add -A`
  blindly — it sweeps in scratch files, build artifacts, editor configs,
  or secrets. Compiled `.c` / `.so` output is never committed.
- Compose the header and **validate it before committing**:

  ```sh
  scripts/agents/check_pr_title.py "type(scope): subject"
  ```

  Deterministic — do not eyeball the 69-char count. For complex changes,
  add a body (blank-line-separated) explaining the "why".
- **Branch/worktree assertion, immediately before `git commit`:** confirm
  `git rev-parse --abbrev-ref HEAD` is the intended branch (**never
  `master`**) and `git rev-parse --show-toplevel` is the intended
  worktree. The Bash-tool cwd can reset between calls, so prefer
  `git -C <worktree>` with an absolute path.
- The critical path is sequential: edit → rebuild → gate → read → stage →
  commit → push → verify. Never batch these in one parallel tool block.

### Step 6: Push and open the PR

- `git push -u origin HEAD`, then verify: `git rev-parse HEAD` must equal
  `git rev-parse origin/<branch>`.
- Open with `gh pr create --base master`. The **title** is the validated
  header from Step 5.
- **PR body:**

  ````markdown
  ## Summary
  <1-3 bullets describing what changed and why>

  ## Test plan
  - [ ] <command(s) you ran, with the real output pasted>
  ````

- The Test plan quotes **real command output** (the pytest summary line,
  a REPL evaluation) or cites the task note's `## Verification` section —
  never invented counts.
- Reconcile the `## Summary` bullets against `git diff --stat`: every
  claim must map to a change in the diff.
- **If any returned value moved,** say so explicitly in the Summary, with
  the function, the magnitude, and which value is right. Do this even
  when the tests stayed green because a tolerance absorbed the shift.

For **project work**, add a `## Project` section between Summary and Test
plan:

````markdown
## Project
`projects/<slug>/` — Task N: <title>.

See `projects/<slug>/task-notes/task-N-<slug>.md` for detail.
````

(Phased: `task-notes/phase-XX/task-X.Y-<slug>.md`.) Put task IDs, issue
links, and extra context in the body, never in the title. For reverts,
add a `Refs:` trailer linking the original PR.

### Step 7: Watch CI

```sh
gh pr checks <N> --watch --fail-fast
```

If a check fails, read the failure and either fix it (new commit → push,
back through Steps 4–6) or, if the fix is out of scope, report the
failing check to the caller. Do not declare done on a red PR.

### Step 8: Validate before finishing

Title passed `check_pr_title.py`; body has `## Summary` + `## Test plan`
(+ `## Project` for project work); branch matches the naming policy; the
push verified; CI green or its failure reported.

## Error recovery

- **`git push` rejected (diverged).** `git fetch origin`, rebase onto
  `origin/<branch>`, re-run the preflight gate, re-push. **Never
  `--force`.** If the divergence is not clearly yours, stop and report.
- **A landed commit needs redoing.** `git reset --soft origin/<branch>`
  then create a fresh commit — do not `--amend` a pushed commit.
- **`gh pr create` says a PR already exists.** Use `gh pr edit` /
  `gh pr view` on the existing PR.

## Structured report

```text
STATUS: Pushed | PR opened | Committed-only | Failed
BRANCH: <branch name>
COMMIT_SHA: <short sha>
PR_URL: <url or n/a>
TITLE: <final PR title>
```

`Committed-only` = the commit landed but the push or PR did not;
`Failed` = the gate or commit itself blocked. Report the reason in prose
above the block.

## Guardrails

- Never force-push; never push to `master` directly; never `--no-verify`.
- Never commit files that look like secrets (`.env`, credentials,
  tokens, `*.pem`) or build output (`*.c`, `*.cpp`, `*.so`, `build/`).
- Do not blindly `git add -A`; stage the specific files you intended.
- Do not include unrelated changes. Pull drive-by cleanup in deliberately
  (and note it in Summary) or stash it for a follow-up.
- Do not fabricate Test-plan output or Summary claims — both reconcile
  against real command output and `git diff --stat`.
