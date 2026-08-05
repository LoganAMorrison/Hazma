# Follow-ups

The durable backlog of ideas and deferred work that hasn't been promoted
to a `projects/<slug>/` plan yet.

Open items live in [`todo/`](todo/); resolved items move to
[`done/`](done/), so `ls todo/` is the live backlog at a glance. Items
move between the two directories; they are never deleted — the historical
reasoning is worth keeping.

The full lifecycle (create → resolve → repoint inbound links), when to
add one, and why this is not GitHub issues, live in
[`../workflow.md#follow-ups`](../workflow.md#follow-ups).

## Creating one

```sh
cp docs/followups/_template.md docs/followups/todo/<slug>.md
# fill in the fields, then add a row to the Open table below
```

## Open

| Item | Added | Source | Scope |
| --- | --- | --- | --- |
| [markdownlint config for templates](todo/markdownlint-config-for-templates.md) | 2026-08-03 | cython-to-rust scaffolding | cross-cutting |
| [`WIDTH_K`/`WIDTH_PI` exponent bug](todo/legacy-parameters-width-exponent-bug.md) | 2026-08-04 | cython-to-rust Task 0.1 | cross-cutting |
| [`cross_section_prefactor` threshold cancellation](todo/cross-section-prefactor-threshold-cancellation.md) | 2026-08-04 | cython-to-rust Task 0.3 | cross-cutting |
| [`msqrd`-driven Monte-Carlo FSR generator](todo/msqrd-driven-fsr-generator.md) | 2026-08-04 | cython-to-rust ADR-0003 | cross-cutting |

## Promoted / Done / Pruned

| Item | Status | Resolution |
| --- | --- | --- |
| [`black` pin diverges between pyproject and CI](done/black-pin-divergence-pyproject-vs-ci.md) | done | [PR #40](https://github.com/LoganAMorrison/Hazma/pull/40) — pins moved to a single PEP 735 `lint` dependency group that CI installs; repo reformatted with black 26.x (33 files). |
