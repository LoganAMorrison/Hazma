# The delta-declaration layer outlives the project that owns its ADR

- **Added:** 2026-09-24
- **Source:**
  `projects/parity-pinned-defect-repair/learnings/project-retrospective.md`
  §5
- **Scope:** cross-cutting
- **Status:** done. Resolved together with its first trigger,
  [`mediator-positron-line-misses-the-electron-velocity.md`](mediator-positron-line-misses-the-electron-velocity.md).
- **Triggers / blockers:** the next repair that moves a parity-pinned
  value. Three are already filed:
  [`rho-photon-outer-boost-misses-support.md`](../todo/rho-photon-outer-boost-misses-support.md),
  [`neutrino-pion-continuum-loses-its-quadrature-support.md`](../todo/neutrino-pion-continuum-loses-its-quadrature-support.md)
  and [`phi-omits-its-direct-pi0-photon-line.md`](../todo/phi-omits-its-direct-pi0-photon-line.md).

> **Resolved.** The project ADR is re-filed as repo-wide
> [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md),
> and the project copy is now a one-line pointer to it. The ADR carries
> the project's rules that bind any repair: never rewrite the corpus,
> never widen a budget, carry an independent oracle and a physics
> invariant, declare an allowlist, fail stale declarations, and compose
> rather than overlap. It adds the label rule. A post-project repair
> takes the next `C<n>` in landing order and adds it to `deltas.REPAIRS`.
> Its roster row goes in `test/parity/README.md`, "Repairs", and its
> evidence is the resolved follow-up. `C1` is the mediator positron line.
> The Group A oracle reader was not generalized, so the project's
> ADR-0003 on keeping those captures stays where it is, and the new ADR
> says it binds only them.

## Why

`test/parity/deltas.py` is now how the parity suite lets a deliberate
repair move a pinned value without rewriting
`test/parity/data/*.npz`. The rule that makes it safe is recorded in
[`projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`](../../../projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md).
It covers the allowlist, the staleness rule, one key per array,
`Composed` for overlapping repairs, and absolute floors. That ADR is
project-scoped, and the project closed on 2026-09-24. `PLAN.md`'s
"Anticipated ADRs" already named it as a candidate for promotion to
`docs/adrs/` if the layer outlived the project, and it does.

The roster is also closed. `deltas.REPAIRS` holds the ten labels `A1`
to `A4` and `B1` to `B6`, and the shape tests treat it as the set every
declaration's label must come from. A repair filed after the project
has no rule for how it gets a label, where its oracle lives, or whose
change control applies to `deltas.py`. The three open follow-ups above
will each need a label. Whoever does the first will write that rule on
the spot, unless it has been decided beforehand.

## What

- Re-file ADR-0001 under `docs/adrs/` with a new repo-wide number, and
  replace the project copy's body with a one-line pointer. That is the
  procedure the ADR template's own comment describes.
- Decide how a post-project repair joins `deltas.REPAIRS`, for example
  with a label scheme that is not tied to the closed A/B groups, and say
  so in `test/parity/README.md`.
- Carry
  [`ADR-0003-keep-the-group-a-oracle-captures-committed.md`](../../../projects/parity-pinned-defect-repair/adrs/ADR-0003-keep-the-group-a-oracle-captures-committed.md)
  along if the oracle reader is generalized beyond the four Group A
  captures.

## Entry points

- `test/parity/deltas.py` — `REPAIRS`, the four relations, `DECLARED_DELTAS`
- `test/parity/test_parity.py` — `EXPECTED_DECLARED_ARRAYS` and the shape tests
- `test/parity/README.md` — the layer's user-facing account
- `projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
