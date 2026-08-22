# Agent shared layer

Agent-neutral, single-source guidance for any coding agent working in
this repo — Claude Code, Codex, or otherwise. The skills under
`.claude/skills/` and `.codex/skills/` are thin: they carry role,
workflow, and gates, and they **point here** for the shared rules rather
than restating them.

## The contract

- **One copy per invariant.** Each rule, checklist, roster, or command
  block lives in exactly one file here. Skills reference it; they do not
  paste it. This is what keeps parallel skill copies from drifting.
- **Precedence.** When a rule in this layer conflicts with a skill, this
  layer and [`AGENTS.md`](../../AGENTS.md) win. When two files here
  disagree, `AGENTS.md` is the tie-breaker.
- **Update in place.** Fix a rule where it lives, once. Never fork a
  corrected rule back into a skill — that recreates the drift these files
  exist to kill. If a skill needs an exception, state the exception in
  the skill and link the canonical rule.
- **Read before you work.** Implementers and reviewers skim the files
  their task touches (below) before editing or reviewing.

## Files

| File | Purpose |
| --- | --- |
| [`preflight.md`](preflight.md) | The before-every-commit / before-every-PR gate: format, lint, the cargo gates, tests, import smoke, the sequential critical path. |
| [`doc-consistency.md`](doc-consistency.md) | The canonical doc-consistency checklist — implementers run it pre-PR, Reviewer D verifies it. |
| [`review-lenses.md`](review-lenses.md) | Reviewer roster, per-lens rubrics, selection rules, and the verdict rule. |
| [`environment.md`](environment.md) | Environment and test-infra gotchas (fish shell, cwd resets, Cython rebuilds, pytest collection traps). |
| [`lessons.md`](lessons.md) | The living review-lessons ledger: one line per class-shaped mistake that recurs, cited by PR. Read on every task. |
| [`lessons-examples.md`](lessons-examples.md) | The worked examples behind each ledger class, one `###` section per class. Read a section only when its rule is not enough to act on. |

## Scripts

Deterministic helpers live under
[`scripts/agents/`](../../scripts/agents/) — `preflight.sh`,
`check_pr_title.py`, `setup_task_worktree.sh`, `resolve_task.py`,
`resolve_phase.py`,
`check_doc_citations.py`. They are the executable form of the rules in
this layer; a skill calls the script rather than re-describing its steps.
The same one-copy-per-invariant principle applies: the logic lives in the
script, and the doc explains when and why to run it.
