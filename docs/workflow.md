# Workflow

This repo uses a lightweight, project-scoped workflow for any change that
spans more than a single trivial commit. The goal: give agents (and
humans) enough structure to plan, track, and synthesize multi-task work
without drowning the repo in process.

## When to create a project

Create a project under `projects/<slug>/` when the work:

- Spans multiple commits or PRs,
- Needs explicit scope bounds (what's in, what's out),
- Would benefit from per-task notes that outlive the PR description, or
- Requires architectural decisions that should be durable.

Skip the project scaffolding for single-commit changes (bugfixes, dep
bumps, typo fixes, one-file refactors). Just commit and open a PR.

## Creating a project

1. Pick a kebab-case slug (e.g. `neutrino-spectra-parity`,
   `vectorize-boost-integrals`).
2. Copy `projects/_template/` to `projects/<slug>/`.
3. Fill in `projects/<slug>/PLAN.md` — set the frontmatter, write the
   Goal, Scope, and initial Tasks (or Phases).
4. Delete any optional files you don't need (e.g. `rules.md` for simple
   projects, the whole `phases/` directory for flat projects). Leave the
   `_template.md` reference files in place (see "Template files vs
   artifacts" below).
5. Add one line to `projects/README.md` under Active Projects.

### Template files vs artifacts

Each per-project subdirectory ships with a `_template.md` reference file
(e.g. `task-notes/_template.md`). The leading underscore is the signal:
**files named `_template.md` are in-project references, not real
artifacts.** They stay in place after scaffolding so agents can copy
their shape when creating new work.

Real artifacts follow strict, globbable patterns:

| Directory     | Artifact pattern                                         |
|---------------|----------------------------------------------------------|
| `adrs/`       | `ADR-XXXX-<slug>.md`                                     |
| `task-notes/` | `task-N-<slug>.md` (flat), `task-X.Y-<slug>.md` (phased) |
| `phases/`     | `phase-XX-<slug>.md`                                     |
| `learnings/`  | `phase-XX-<slug>.md` or `project-retrospective.md`       |
| `references/` | any `<topic>.md` (optional — see below)                  |

Tooling and skills glob for these artifact patterns; they ignore
`_template.md`. Never rename a `_template.md` to match an artifact
pattern — create a fresh file from it instead.

## The project tree

```text
projects/<slug>/
├── PLAN.md              # scope, task shape (canonical), frontmatter
├── rules.md             # OPTIONAL: project-specific cross-cutting rules
├── task-notes/          # per-task notes + shared working memory
│   ├── README.md                 # project-level live status + working memory
│   ├── task-N-slug.md            (flat projects: per-task note)
│   └── phase-XX/                 (phased projects)
│       ├── README.md             # per-phase live Tasks table + working memory
│       └── task-X.Y-slug.md      # per-task note
├── phases/              # OPTIONAL: only for phased projects
│   └── phase-XX-slug.md
├── references/          # OPTIONAL: deeper reference material for big plans
│   └── <topic>.md
├── adrs/                # project-scoped decisions
│   └── ADR-XXXX-slug.md
└── learnings/           # phase or project synthesis
    └── phase-XX-slug.md
```

**Split of concerns between `PLAN.md` and `task-notes/README.md`:**

- `PLAN.md` is the **canonical shape** of the project: goal, scope,
  per-task objectives and exit criteria, anticipated ADRs. For phased
  projects it also lists the phase table. `PLAN.md` changes when
  canonical behavior changes, and those changes need review.
- `task-notes/README.md` is the project's **live state** hub.
  - **Flat:** holds the numbered Tasks status table + cumulative
    findings, open questions, and handoff.
  - **Phased:** holds the Phases status table (mirroring each phase
    file's frontmatter) + cross-phase findings, decisions, and handoff.
    Per-task status inside a phase lives in
    `task-notes/phase-XX/README.md` (the per-phase working memory).
- `task-notes/phase-XX/README.md` (**phased only**) is the per-phase
  working-memory file: the Tasks status table for that phase's tasks,
  plus phase-scoped findings, decisions, open questions, and handoff.
  This is the authoritative source the skills consult when asked to "pick
  the next unfinished task in the current phase".

Keeping these separate means `PLAN.md` stays reviewable as a contract
while the status tables churn freely alongside the notes.

## References (`references/`)

Large projects tend to accumulate two kinds of content in `PLAN.md` that
don't belong there:

1. **Grounded facts** — exhaustive maps of where something lives today
   (file paths, line numbers, call graphs). Useful context, but stable
   enough to break out.
2. **How-to detail** — derivation write-ups, translation tables,
   checklists, output-contract specs. Scoped to one or two tasks, but too
   verbose to inline in a Task Details block. In this repo that often
   means **the physics**: a derivation, a published-value table, or the
   provenance of a data file.

When `PLAN.md` starts ballooning past ~15KB because of either, break that
content out into `references/<topic>.md` and have each Task Details entry
**point** to the references it needs. This keeps `PLAN.md` scan-friendly
and cuts token cost for agents: they load PLAN + one or two relevant
references per task, not the full plan.

Guidelines:

- **The PLAN stays canonical.** References are enrichment, not
  substitute. Task objectives, scope, and deliverable/gate must live in
  `PLAN.md`'s Task Details.
- **Each reference file starts with an Audience + Nature header.**
  Example: `**Audience:** Tasks 4, 5, 6.` `**Nature:** Derivation.` That
  way an agent can skim the top and decide whether to load it.
- **Name by topic, not by task.** `references/boost-integral-kernel.md`
  ages better than `references/task-8-notes.md`.
- **Don't create a reference for every task.** The pattern pays off when
  the same material serves multiple tasks, or when one task needs >1 page
  of supporting detail.
- **Add a reference index to `PLAN.md`** under a short "Orientation"
  section, one line per reference.

Skills that read `PLAN.md` don't automatically fetch references — they
follow the pointers when the active task's detail names a file.

## Flat vs phased

Most projects are flat — one `PLAN.md` plus one `task-notes/README.md`
holding the numbered task table and cross-task working memory. Use phases
only when **all three** hold:

1. **One shipping deliverable** — the whole effort produces one thing.
2. **Real temporal dependency** — later work can't be validated until
   earlier work lands.
3. **15+ tasks** — a flat list would be unreadable.

If any of these fails, prefer multiple projects with cross-references
(`Requires: projects/other-project/`) over phases within one project.

**Signaling which style the project uses:**

- Frontmatter: `phased: true | false` (machine-readable; skills branch on
  this).
- First body line of `PLAN.md`: `**Structure:** Flat task list.` or
  `` **Structure:** Phased — see `phases/`. ``

## ADR placement

ADRs capture canonical decisions that future work should respect.

**Heuristic:** Could someone read this ADR without knowing which project
produced it and still find it useful?

- **Yes** → `docs/adrs/ADR-XXXX-*.md` (repo-wide).
- **No** → `projects/<slug>/adrs/ADR-XXXX-*.md` (project-scoped).

**Default bias:** start project-scoped. Promote to repo-wide (by re-filing
the ADR in `docs/adrs/` with a fresh repo-wide number) only when a second
project needs the same decision.

Examples of repo-wide ADRs:

- "Spectrum functions broadcast over energy arrays and return NaN outside
  the kinematic range"
- "Cython kernels never import from pure-Python `hazma` modules"
- "Published spectra are pinned by regression arrays, not tolerances"

Examples of project-scoped ADRs:

- "The N-body integrator uses RAMBO rather than adaptive sampling for
  this project's multiplicities"
- "This form factor is interpolated from the tabulated data rather than
  evaluated analytically"

## Task notes

Every task gets a note. It captures:

- What you reviewed before starting.
- What you found (constraints, edge cases, surprises).
- Decisions made and their alternatives.
- Files changed.
- How you verified.
- Open questions.
- Plan impact (None | ADR needed | phase file or `rules.md` update).
- Handoff: what the next agent should read first.

Use `projects/_template/task-notes/_template.md` as the starting shape.

**Canonical path:**

- Flat projects: `task-notes/task-N-<slug>.md`.
- Phased projects: `task-notes/phase-XX/task-X.Y-<slug>.md`.

One task note per meaningful work unit. Tiny sub-subtasks that naturally
belong together can share one note.

## Working memory (`task-notes/README.md`)

Alongside per-task notes, each project keeps a **working-memory** file at
`task-notes/README.md`. It is the project's shared scratchpad and holds
live state that `PLAN.md` should not carry:

- Flat projects: the **numbered Tasks status table** + cumulative
  findings, open questions, and rolling handoff.
- Phased projects: the **Phases status table** (mirroring each phase
  file's frontmatter) + cross-phase findings, decisions, and handoff.

**Phased projects also need per-phase working memory.** Each phase gets
its own file at `task-notes/phase-XX/README.md`, copied from
`projects/_template/task-notes/phase-XX/README.md`.

Agents should read working-memory files before loading per-task notes,
append to them as they learn, and promote stable entries to `PLAN.md` or
an ADR once a finding is canonical.

## Learnings

When a phase closes (phased projects) or a project wraps up, synthesize
the task notes into a learnings document. This is durable memory for
future work that touches the same area — not a status log.

A retrospective should also include a **§5 Follow-on seeds** section
listing every substantive deferred item the project surfaced. Each seed
gets a one- or two-paragraph entry in the retrospective (the historical
record), and a corresponding stub in [`followups/`](followups/README.md)
(the live actionable backlog). Duplication is intentional: the audiences
differ.

## Follow-ups

[`docs/followups/`](followups/README.md) is the durable backlog of ideas
and deferred work that hasn't been promoted to a `projects/<slug>/` plan
yet. Each entry is a single markdown file with a fixed shape:
description, source, scope, status, entry points. Open items live in
[`todo/`](followups/todo/); resolved items move to
[`done/`](followups/done/), so `ls todo/` is the live backlog at a
glance.

### Lifecycle

1. **Create.** Copy `docs/followups/_template.md` to
   `docs/followups/todo/<slug>.md`; fill the fields; add a row to
   `docs/followups/README.md` under "Open".
2. **Resolve.** When picked up or dropped, set the `Status:` field, then
   `git mv docs/followups/todo/<slug>.md docs/followups/done/<slug>.md`
   and move the README row to the **Promoted / Done / Pruned** table:
   - *Promoted to a project:* scaffold `projects/<slug>/`, set
     `Status: promoted`, link the project. The new project's PLAN should
     backlink.
   - *Done ad-hoc:* do the work, set `Status: done`, link the PR.
   - *Pruned:* set `Status: pruned` plus a one-line reason. Don't
     silently delete — the historical reasoning is useful.
3. **Repoint inbound links.** The path encodes status, so resolving an
   item changes its path (`todo/` → `done/`). Before committing,
   `rg -l '<slug>\.md'` and update every reference (retrospectives, ADRs,
   PLANs, code comments, the README tables).

Items move between `todo/` and `done/`; they are never deleted.

### When to add one

- During or at the end of a project: every substantive §5 follow-on seed
  gets a stub here.
- When the current task **introduces** or **surfaces in its touched
  area** a substantive `TODO`/`FIXME` representing real deferred work
  rather than minor cleanup. Grep `docs/followups/` for the relevant
  identifier first — don't file a duplicate.
- When a code review surfaces a real out-of-scope issue.

Skip the followup file for trivial cleanup that's faster to just do, and
skip drive-by `TODO`s in unrelated files outside the task's touched area.
This directory is for items worth tracking, not every passing thought —
backlog bloat is the failure mode.

### Why not GitHub issues

GitHub issues stay for user-reported bugs and feature requests.
`docs/followups/` is for internal sequencing decisions — markdown files
version with the code, link cleanly to ADRs and retrospectives, and are
agent-grep-friendly.

## Branch and PR conventions

The trunk branch is **`master`**. Branches carry the driving agent's
identity as their prefix: Claude-driven work uses `claude/<...>`,
Codex-driven work uses `codex/<...>`. Both prefixes are valid and
permanent — all tooling (skills, helper scripts) parses either.

- **Project work** (multi-task effort under `projects/<slug>/`):
  `<agent>/<project-slug>/<task-slug>`.
- **Ad-hoc work** (single-commit bugfix, dep bump, etc.):
  `<agent>/<short-description>`.
- Commit messages and PR titles follow
  [`PR_GUIDELINES.md`](PR_GUIDELINES.md).

## Project lifecycle

1. **Active** — listed under Active Projects in `projects/README.md`;
   `PLAN.md` frontmatter `status: In Progress`.
2. **Complete** — all tasks done, final synthesis written. The closing PR
   (the one that flips `status:` to `Complete`) **must also bump the
   package version and add a `CHANGELOG.md` entry** per
   [`versioning.md`](versioning.md). Then update `PLAN.md`
   `status: Complete` and move the entry from Active to Completed in
   `projects/README.md`.
3. **Archived** — after a long time or when the project's decisions are
   fully superseded, the folder may be moved or deleted. Default: keep
   indefinitely for historical reference.

## Versioning

Hazma follows Semantic Versioning over the **public Python API** —
module paths, function and class names, keyword arguments, return shapes
and units, and the numerical values those functions produce.

Every project declares a `version_bump` field in its `PLAN.md`
frontmatter:

```yaml
---
status: In Progress
phased: false
version_bump: patch     # patch | minor | major
---
```

The closing PR for every project bumps `VERSION` in `hazma/__init__.py`
per this field and adds a `CHANGELOG.md` entry naming the project slug.
`scripts/agents/preflight.sh --closing` checks both.

See [`versioning.md`](versioning.md) for the full policy: the litmus
test, the public-surface inventory, examples, and the numerical-change
carve-out.

## Skills

This repo defines Claude skills under `.claude/skills/`:

- `execute-single-task` — scoped implementation of one task.
- `commit-and-pr` — commit, push, and open a PR following the guidelines.
- `review-pr` — focused single-lens review of a PR.
- `review-respond` — synthesize review comments and implement fixes.
- `review-cycle` — the single review-loop implementation: reviewer
  selection, parallel review, commit+push of fixes, PR round comments,
  and convergence. `task-pipeline`'s review phase delegates to it rather
  than reimplementing the loop.
- `task-pipeline` — end-to-end: implement → review → ship.
- `begin-phase` — safe kickoff for the next phase (phased projects only).
- `review-plan` — stress-test a project plan before implementation.

Each skill expects the filesystem contract described above. Read the
skill's `SKILL.md` for exact inputs, outputs, and reading order.

### Shared agent layer

Skills stay thin — role, when-to-use, workflow, gates, structured report
— and point into a shared, agent-neutral layer rather than restating it.
This layer is the one-copy-per-invariant source of truth:

- [`docs/agents/`](agents/README.md) — reference files, one per
  invariant: the preflight gate, the doc-consistency checklist, the
  reviewer roster and lens rubrics, environment/test-infra gotchas, and
  the review-lessons ledger.
- [`scripts/agents/`](../scripts/agents/) — deterministic helper scripts
  (preflight execution, PR title validation, task-worktree setup, task
  resolution, doc-citation bounds checking) that skills shell out to
  instead of re-deriving the logic inline.

When a skill needs to state an invariant already captured under
`docs/agents/`, it links there rather than duplicating the text.

## Update discipline

- Don't use `PLAN.md` as a running status log. Task progress lives in
  task notes.
- Don't stash task-level findings or transient debugging notes in an ADR.
- Don't rely on PR descriptions as the only record of project knowledge —
  those get lost.
- When a canonical decision changes mid-project, add an ADR and patch the
  affected `PLAN.md` / `phases/` / `rules.md`.
- Verify counts (tests, channels, fixtures) by actually counting in
  source, not from memory or task notes.
