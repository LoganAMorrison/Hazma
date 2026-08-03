# <Human-readable title>

<!--
  This is the follow-up template. Copy it into `todo/` with a
  kebab-case slug (e.g. `todo/eta-prime-endpoint-drift.md`) and
  fill in the fields below. Then add a row to `README.md` under "Open".

  When the item is resolved, `git mv todo/<slug>.md done/<slug>.md`,
  move its README row to the "Promoted / Done / Pruned" table, and
  repoint any inbound links (the path changes from todo/ to done/).

  See `docs/workflow.md#follow-ups` for the full lifecycle (promote,
  done, prune) and how this slots in alongside the `projects/` directory.
-->

- **Added:** YYYY-MM-DD
- **Source:** <projects/<slug>/learnings/project-retrospective.md §5 | TODO at file:line | GH issue #N | conversation>
- **Scope:** <project | commit | cross-cutting>
- **Status:** open
- **Triggers / blockers:** <optional — when does this ripen? what must land first?>

## Why

<1-2 paragraphs: what observation prompted this, what gap or risk it
addresses, why it's worth tracking now rather than letting it drift.>

## What

<Concrete description of the work. Files or modules touched, channels
added, ADRs to amend, expected shape of the change. Enough detail
that a future agent can scope it without re-deriving the context.>

## Entry points

- <File path : line>
- <Related project: `projects/<slug>/`>
- <Prior ADR: `projects/<slug>/adrs/ADR-XXXX-*.md`>
- <Prerequisite follow-up: `docs/followups/todo/<slug>.md`>

## Risks / open questions

<Optional. Sequencing concerns, unresolved trade-offs, dependencies on
external data (a tabulated form factor, a detector response
file), open scope questions.>
