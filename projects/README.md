# Projects

This directory holds scoped bodies of work. Each project is
self-contained with its own plan, task notes, and (optionally) phases
and architectural decisions.

See [`../docs/workflow.md`](../docs/workflow.md) for:

- When to create a project vs ship an ad-hoc commit.
- How to scaffold a new project.
- Flat vs phased projects.
- The ADR placement heuristic.
- The skills that automate the loop.

## Starting a new project

```sh
cp -r projects/_template projects/<your-slug>
# Edit projects/<your-slug>/PLAN.md (canonical task shape). Fill in the
#   Numerical impact section — it drives version_bump and the CHANGELOG.
# Edit projects/<your-slug>/task-notes/README.md (live task status
#   table + working memory). This is NOT optional — agents read it
#   every pass and treat it as the source of truth for what is done.
# For phased projects, also create a task-notes/phase-XX/README.md
#   from the template for each phase (the per-phase Tasks status
#   table lives there).
# Delete any optional files the project doesn't need (rules.md, phases/,
#   references/, ...). Leave the `_template.md` reference files in
#   the directories you keep.
# For big plans that would push PLAN.md past ~15KB, use the
#   references/ directory to break out topic-scoped reference docs —
#   see `../docs/workflow.md#references-references`.
# Add a row to the Active Projects table below.
```

See [`../docs/workflow.md#template-files-vs-artifacts`](../docs/workflow.md#template-files-vs-artifacts)
for why `_template.md` files stay in place after scaffolding.

## Active Projects

| Slug | Deliverable | Phased | Started | Status |
| --- | --- | --- | --- | --- |
| [`cython-to-rust`](cython-to-rust/PLAN.md) | Compiled layer rebuilt in Rust (PyO3, abi3 `hazma._core`, maturin); zero Cython; permanent parity corpus | Yes (8) | 2026-08-03 | In Progress |

## Completed Projects

| Slug | Deliverable | Phased | Started | Shipped |
| --- | --- | --- | --- | --- |
| _none yet_ | | | | |
