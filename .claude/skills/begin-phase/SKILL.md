---
name: begin-phase
description: Prepare a guarded, ready-to-paste kickoff prompt for the next eligible phase of a phased project, verifying strict prerequisite closeout first and refusing blocked or ambiguous phase starts.
---

**Role:** Prepare a safe, ready-to-paste prompt for the next phase of a
**phased** project without letting an agent start a blocked or ambiguous
phase.

**When to use this skill**

- The user asks to begin the next phase of a phased project.
- The user asks which phase is actually ready to start next.
- The user asks to prepare a specific phase safely.

This skill **prepares a handoff prompt only**. It does not start
implementation or edit project files. First-task selection inside the
chosen phase is delegated to `/execute-single-task`.

**When NOT to use**

- Flat (non-phased) projects — refuse and point at
  `/execute-single-task` (see Step 2).
- Actually implementing a task — that is `/execute-single-task`.

## Inputs

Required:

- **project slug** — explicit `--project <slug>`, or inferred from the
  current branch (see Step 1).

Optional:

- **completed phase** — the phase that just wrapped up (e.g. `2`, `3.1`),
  passed as `--completed-phase`. When set, the helper prefers phases this
  one directly unlocks.
- **target phase** — a specific phase to prepare, via `--target-phase`.

## Workflow

### Step 1: Resolve the project slug

1. **Explicit argument** (`--project <slug>`) wins.
2. **Branch-name inference.** A project branch is
   `<agent>/<project-slug>/<task-slug>`, `<agent>` ∈ `{claude, codex}` —
   **parse both prefixes**. The first segment after the prefix is the
   slug. Ad-hoc branches have no second `/` and name no project.
3. **Stop and ask** if neither resolves a slug.

### Step 2: Confirm the project is phased

Read `projects/<slug>/PLAN.md`'s frontmatter.

- `phased: true` → continue.
- `phased: false` (or missing) → **refuse.** `begin-phase` exists only
  for phased projects. Point the user at `/execute-single-task` with
  their target task number.

The helper enforces this too (it returns `status: error` for a non-phased
project), so this read is belt-and-suspenders, not the sole gate.

### Step 3: Determine the request shape

Map the user's request to helper arguments:

- "What phase is ready next?" → no phase arguments.
- "Prepare the next phase after Phase X" → `--completed-phase X`.
- "Prepare Phase Y" → `--target-phase Y`.
- Both supplied → pass both.

### Step 4: Run the shared helper

The helper is at `.claude/skills/begin-phase/scripts/resolve_phase.py`.
**Run it from the repo root** — the Bash-tool cwd can reset between
calls, so pass an absolute path or `cd` in the same command.

```sh
# Current frontier:
python3 .claude/skills/begin-phase/scripts/resolve_phase.py \
  --project <slug>

# After a just-completed phase:
python3 .claude/skills/begin-phase/scripts/resolve_phase.py \
  --project <slug> --completed-phase 4

# Explicit target:
python3 .claude/skills/begin-phase/scripts/resolve_phase.py \
  --project <slug> --target-phase 6
```

The helper emits JSON on stdout with `status`:

- `ready` — exactly one phase is eligible; a kickoff prompt is included
  under `prompt`.
- `choose` — multiple phases are eligible; `choices[]` lists them with
  reasons.
- `blocked` — the requested or frontier phase has unmet prerequisites or
  its prereq prose can't be machine-resolved; `blockers[]` lists them.
- `error` — bad input or project not found; stop and report the message.

It also carries a `notes[]` array — surface anything listed there.

### Step 5: Follow the helper output strictly

- **`ready`:**
  - Tell the user which phase is eligible, why (the helper's `reason`),
    and whether any notes were emitted.
  - Paste the `prompt` inside a fenced block. **Paste it verbatim; do not
    rewrite, trim, or "clean up" the prompt** — it already carries the
    guardrails (phase scope, a re-run-and-refuse-unless-`ready` check for
    the next agent, worktree setup per `/execute-single-task` Step 3).
- **`choose`:**
  - List the numbered `choices` with their `reason`s. Stop. Do not
    auto-select — the user must choose before a kickoff prompt is
    generated.
- **`blocked`:**
  - Explain each blocker clearly (which phase, which issue). A phase is
    only eligible when every prerequisite is **both** `status: Complete`
    **and** has a matching learnings document
    (`learnings/phase-XX-*.md`). So `blocked: missing learnings document`
    means the prereq phase is marked done but its write-up is absent —
    say that, don't just echo the string.
  - Synthetic `(none)` blocker: when the helper reports
    `phase_id: "(none)"`, no phase is in a Not-started state. Tell the
    user the project may be complete or needs a new phase file authored.
  - Do not draft a kickoff prompt for any blocked phase.
- **`error`:** report it. Ask for the correct project slug or phase id if
  it's an input problem.

## Guardrails

- **Do not auto-start phases for flat projects.** Refuse with a pointer
  to `/execute-single-task` when `phased: false`.
- **Do not bypass the helper.** Never eyeball the phase files and decide
  a phase is ready without running `resolve_phase.py`.
- **Do not prepare a kickoff prompt for a blocked phase.** The helper's
  blockers list is the source of truth.
- **Do not auto-pick between parallel phases.** On `choose`, surface the
  options and stop.
- **Do not start implementation work yourself.** This skill only
  generates a handoff prompt.
- **Do not flag `_template.md` files as project state.** The helper
  ignores phase-file names starting with `_`; you should too.
