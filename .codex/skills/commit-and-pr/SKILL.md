---
name: commit-and-pr
description: "Run Hazma's preflight gate, create or validate a Codex branch, commit intentionally with a valid Conventional Commit header, push, and open or update a pull request. Use when a user asks to commit, ship, or open a PR for current work."
---

# Commit and open a pull request

Ship the current worktree without bypassing Hazma's gates. This is the commit
authority for standalone work; a pipeline-owned branch keeps its existing
branch and PR instead.

1. Read [`docs/PR_GUIDELINES.md`](../../../docs/PR_GUIDELINES.md), inspect
   `git status --short`, the full diff, and recent commit style. Stop if there
   are no intended changes.
2. Confirm numerical-impact evidence for any public code path, and confirm a
   closing project carries its required `VERSION` and `CHANGELOG.md` updates.
3. Work only on a feature branch. If a new branch is needed, create
   `codex/<project>/<task>` for project work or `codex/<description>` for
   ad-hoc work; never commit to `master`.
4. Rebuild first when the Rust crate changed, then run:

   ```sh
   scripts/agents/preflight.sh --paths "<touched paths>" \
     [--tests "<narrow targets>"] [--md "<changed markdown>"] [--closing]
   ```

   Treat a non-zero result or `WARN` gate as a blocker. Read the literal pytest
   summary and use the sequential critical path in
   [`docs/agents/preflight.md`](../../../docs/agents/preflight.md).
5. Stage only intended source, test, and documentation files. Do not stage
   build outputs, credentials, scratch files, or generated C/C++.
6. Compose and validate the commit/PR header before committing:

   ```sh
   scripts/agents/check_pr_title.py "type(scope): subject"
   ```

   Immediately before the commit, verify the intended non-`master` branch and
   worktree. Commit with an explanatory body when it helps.
7. Push without force, verify `HEAD` equals `origin/<branch>`, and create or
   update a PR against `master`. Use the validated header as the title. The PR
   body must contain `## Summary` and `## Test plan`, plus `## Project` for
   project work. Reconcile all prose claims with the current diff and actual
   command output; state every published numerical change explicitly.
8. Watch CI to a bounded conclusion. If it fails, report the failing check or
   fix it through the full gate again; never report a red PR as complete.

End with:

```text
STATUS: Pushed | PR opened | Committed-only | Failed
BRANCH: <branch>
COMMIT_SHA: <short sha>
PR_URL: <url or n/a>
TITLE: <validated title>
```
