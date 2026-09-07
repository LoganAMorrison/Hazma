# ADR 0001: Express corpus repairs as declared deltas, never as rewritten arrays

**Date:** 2026-09-06
**Status:** Accepted
**Scope:** Project-scoped (applies only within `projects/parity-pinned-defect-repair/`).

<!--
If this decision turns out to apply repo-wide, re-file it under
`docs/adrs/` with a new repo-wide number, and replace this file's body
with a one-line pointer to the new ADR.
-->

## Context

`test/parity/data/` pins 179,695 values captured from the pre-port
Cython, and `projects/cython-to-rust/rules.md` rule 2 forbids
regenerating them from a tree whose kernels run on Rust — after
cython-to-rust Task 6.4 there is nothing else to generate from. Eight of
those values are wrong (the roster in
`../references/defect-blast-radius.md`), and every repair has to get
past the gate that pins the wrong value. Re-pinning the affected arrays
in place would destroy the only record of what 2.1.0 shipped, and
widening the case budget until the repaired value fits would make the
gate vacuous for that case (`../rules.md` rule 2,
`docs/agents/lessons.md` `[exemption-wider-than-its-mechanism]`).

The first repair to need the mechanism (B4, PR #87) arrived before the
mechanism existed.

## Decision

A repair is expressed as a **declaration**, in `test/parity/deltas.py`,
that named positions of named stored arrays now relate to the stored
value in a named way. The runner compares a declared position against
the declared relation and every undeclared position against the stored
value under the case's existing budget. Concretely:

- The committed arrays and manifest are never rewritten.
- A declaration names the repair (from a closed roster), the positions
  it covers, the relation, the measurement that justifies the
  relation's own budget, and the file holding the evidence.
- A relation answers one question — what the repaired array should be.
  It may say so as a term added to the stored array or as a closed-form
  transform of it; either way the runner compares against the array it
  predicts.
- A declaration covers only the positions its mechanism moves: either an
  explicit tuple every entry of which the relation moves, or `MOVED`,
  resolved at comparison time against the array the relation predicts.
  Naming a position the relation leaves at its stored value is an error,
  not a looser gate.
- A declaration whose array no longer differs from the stored one fails,
  so a reverted repair cannot hide behind it. A repair's declaration
  therefore cannot precede the repair; a delta modelled ahead of one
  waits outside the table until the array it describes has moved.
- Shape tests hold the table to real cases, blocks, arrays and
  positions, to roster labels, and to a counted size.

## Consequences

- **Positive:** the corpus stays the record of what shipped; a repair
  proves it moved only what it meant to, because everything else is
  still gated at the old budget; a revert turns the gate red; the close
  can aggregate "how many pinned values are under a declaration" from
  the table.
- **Negative:** a relation is held to a budget of its own, and how much
  that loosens depends on how it builds its prediction. An additive term
  that is its own quadrature runs three orders of magnitude looser than
  the case budget at the declared positions, and costs one extra kernel
  evaluation per declared array in the gate. A closed-form transform of
  the stored array costs neither, which is why it is the relation to
  reach for first.
- **Mitigation:** the relation's budget is measured and written beside
  it; the positions it loosens are exactly the ones the repair moved,
  and the mutation tests in `test_parity.py` keep it from loosening any
  other. If the layer outlives this project, promote this ADR to
  `docs/adrs/`.
