---
status: In Progress
phased: false
version_bump: patch
deliverable: <one-line description of what ships when this project is done>
created: <YYYY-MM-DD>
---

<!--
`version_bump` declares the SemVer step the closing PR will take.
See `../../docs/versioning.md` for the litmus test and examples.

- `patch` (default) — no user-visible behavior change: internal
  refactor, docs, tests, packaging, or a fix whose previous output was
  an exception or an obvious crash.
- `minor` — additive (a new public function, model, or channel), OR a
  deliberate correction that MOVES A PUBLISHED NUMBER. The
  numerical-change carve-out is the one people miss: the API is
  unchanged, so it isn't `major`, but a user's plot moves, so it isn't
  `patch`.
- `major` — existing correct user code breaks or silently changes
  meaning: a public name is removed or renamed, a return shape or unit
  changes, a required argument is added, or the Python floor rises.

Raising the level mid-project is fine and expected when scope shifts.
Lowering it requires a one-line note in `task-notes/README.md`
explaining why the change is no longer user-facing.
-->

# Project: <Title>

**Structure:** Flat task list.
<!-- For phased projects, replace the line above with:
**Structure:** Phased — see [`phases/`](phases/).
Also set `phased: true` in the frontmatter.
-->

## Goal

<1-3 sentences describing what this project accomplishes and why it
exists. If there's a GitHub issue or upstream context, link it from the
Related section below, not here.>

## Scope

**In scope:**

- <item>

**Out of scope:**

- <item>

## Numerical impact

<Does this project change any value the library returns? Name the public
functions it can reach, and state the expected direction: "no public
value changes", "corrects `dnde_photon` for K-long by up to ~3% near
threshold", or "unknown — Task 1 measures it". This field drives the
`version_bump` above and the CHANGELOG entry the closing PR writes.
"Unknown" is a valid answer at plan time; it is not a valid answer at
close time.>

## Tasks

The live task table, status, and dependency diagram are tracked in
[`task-notes/README.md`](task-notes/README.md) (the project's working-
memory file). This `PLAN.md` describes the canonical *shape* of each
task in the "Task Details" section below; `task-notes/README.md` tracks
their live *state*.

<!--
Phased projects: keep the PLAN pointer above, and add a `## Phases`
table to `task-notes/README.md` mirroring each phase file's
frontmatter `status:`. Its exact shape lives once, in the phased
block of `projects/_template/task-notes/README.md` — copy it from
there rather than from a second copy here.

A phased project keeps a task breakdown inside each phase file's
`## Tasks` section (canonical task shape) and a REQUIRED per-phase
working-memory file at `task-notes/phase-XX/README.md` that holds
the live Tasks status table for that phase. Copy the per-phase file
from `projects/_template/task-notes/phase-XX/README.md` for each
phase.
-->

## Task Details

<!-- One subsection per task. Each subsection describes the canonical
shape of the work: objective, scope, files to touch, exit criteria.
Do NOT track live status here — that belongs in `task-notes/README.md`.

### Task 1: <short task title>

**Objective:** <one sentence>

**Scope / implementation notes:** <what the task covers, with concrete
file paths and API sketches where helpful>

**Deliverable / gate:** <testable outcome that defines done — a pytest
node id, a pinned numerical comparison, a measured benchmark>
-->

## Dependencies

- Requires: <nothing | `projects/<other-slug>/` complete through task N
  | specific upstream PR merged>

## Related

- GitHub Issue: <optional link>
- Background: <optional — papers, published values, upstream
  discussions, or the plan this emerged from>

## Change control

See [`../../docs/workflow.md#adr-placement`](../../docs/workflow.md#adr-placement)
for when to write an ADR and where it lives (repo-wide vs project-scoped).
Patch the affected `PLAN.md` / phase file / `rules.md` when canonical
behavior changes.

## Closing this project

The PR that flips this `PLAN.md` `status:` to `Complete` must also bump
`[project] version` in `pyproject.toml` per the `version_bump:`
frontmatter and
add a `CHANGELOG.md` entry naming this project slug. Re-check the level
against the **Numerical impact** section above before bumping — a
project that ended up moving a published number is `minor`, not `patch`.
Verify with `scripts/agents/preflight.sh --closing`. See
[`../../docs/versioning.md`](../../docs/versioning.md) for the full
policy.

### Anticipated ADRs

<Decisions you already expect to require an ADR. Empty is fine — it just
means no major architectural forks are known yet.>

- <anticipated ADR topic>
