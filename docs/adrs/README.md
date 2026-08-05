# Repo-Wide Architectural Decision Records

This directory holds ADRs that apply across the whole codebase —
decisions any future project should respect without needing context from
the original project that produced them.

For project-scoped ADRs (decisions meaningful only within one project),
use `projects/<slug>/adrs/` instead. See
[`../workflow.md`](../workflow.md#adr-placement) for the full placement
heuristic.

## Naming

`ADR-XXXX-short-imperative-title.md`

- `XXXX` — four-digit zero-padded sequential number (ADR-0001, ADR-0002,
  …).
- `short-imperative-title` — kebab-case, matches the ADR's H1.

Sequence numbers are repo-wide. Project-scoped ADRs have their own
sequence within each project's `adrs/` directory.

## When to write a repo-wide ADR

**Heuristic:** Could someone read this ADR without knowing which project
produced it and still find it useful?

- **Yes →** here.
- **No →** `projects/<slug>/adrs/`.

Default bias: start project-scoped. Promote a project-scoped ADR to
repo-wide by re-filing it here (with a new repo-wide number) and leaving
a one-line pointer in the original location that links to the new number.

In a physics library the repo-wide tier is mostly for **contracts about
what the numbers mean** — units, normalization conventions, frame
choices, behavior outside the kinematic range, and how published values
are pinned. Those outlive any single project and are exactly what a
future implementer needs and cannot re-derive.

## Template

Copy [`template.md`](template.md) and fill in the sections.

## Index

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [ADR-0001](ADR-0001-fsr-generator-takes-both-matrix-elements.md) | The FSR generator takes both matrix elements | Proposed | 2026-08-04 |
