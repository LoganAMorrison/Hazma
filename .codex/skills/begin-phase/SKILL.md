---
name: begin-phase
description: "Safely determine whether a phased Hazma project can begin its next or a requested phase, then produce a guarded Codex kickoff prompt. Use when a user asks which phase is ready, asks to begin a phase, or needs a phase handoff; this skill does not implement the task."
---

# Begin a phased project phase

Prepare a handoff only. Do not edit project files or start implementation.

1. Resolve `--project <slug>` from an explicit argument or a project branch
   (`claude/` and `codex/` both parse). Read `projects/<slug>/PLAN.md` and
   refuse non-phased projects; direct those to `$execute-single-task`.
2. Map an optional completed phase to `--completed-phase` and a requested one
   to `--target-phase`, then run the readiness oracle from the repository root:

   ```sh
   python3 scripts/agents/resolve_phase.py --project <slug> --agent codex \
     [--completed-phase <id>] [--target-phase <id>]
   ```

3. Follow its JSON result exactly:

   - `ready`: state why it is eligible and paste `prompt` verbatim in a fenced
     block. It directs the next agent to `$execute-single-task`, requires a
     re-check, and specifies a `codex/` worktree.
   - `choose`: list each eligible choice and its reason, then stop for the
     user's selection.
   - `blocked`: explain each blocker. A missing learning document means a
     prerequisite is marked complete but was not closed out; `(none)` means no
     phase remains in `Not started` state.
   - `error`: report the message and request corrected input when needed.

Never infer readiness by hand, choose among parallel phases automatically, or
generate a prompt for a blocked phase. Ignore `_template.md` files.
