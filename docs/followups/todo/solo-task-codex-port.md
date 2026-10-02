# `solo-task` has no Codex counterpart

- **Added:** 2026-10-01
- **Source:** the PR that added `.claude/skills/solo-task/SKILL.md`
- **Scope:** commit
- **Status:** open
- **Triggers / blockers:** a Codex session that runs one project task by
  hand and grows its context past the point where the run stays cheap.

## Why

Every other workflow skill exists in both `.claude/skills/` and
`.codex/skills/`, and `AGENTS.md` describes the two trees as parallel.
`solo-task` breaks that symmetry. Its whole design is a delegation
shape: the main session keeps a brief while subagents, each pinned to a
Claude model tier, read the full documents, run the preflight gate,
measure the numerical impact, ship the PR, and run one review pass. That
shape assumes Claude Code's subagent tool and its per-call model choice,
so a line-for-line copy would not run under Codex.

## What

Write `.codex/skills/solo-task/SKILL.md` in the condensed style of the
other Codex skills, keeping the procedure `execute-single-task` defines
and replacing the delegation with whatever Codex offers for a
context-isolated helper. If Codex has no equivalent, the port may
instead state the context-budget rules alone (bounded reads, no
whole-document reads of the guides, gate output summarized), which carry
most of the saving. Then drop the Claude-only qualifiers from
`AGENTS.md`'s `## Skills` section and `docs/workflow.md`'s `## Skills`
list.

## Entry points

- `.claude/skills/solo-task/SKILL.md` — the Claude version to port.
- `.codex/skills/execute-single-task/SKILL.md` — the Codex procedure it
  would reference.
- `.codex/skills/task-pipeline/SKILL.md` — how the Codex tree already
  phrases orchestration without per-call model choice.
