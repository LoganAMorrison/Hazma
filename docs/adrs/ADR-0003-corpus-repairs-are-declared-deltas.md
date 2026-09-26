# ADR 0003: Express corpus repairs as declared deltas, never as rewritten arrays

**Date:** 2026-09-24
**Status:** Accepted

Promoted from `projects/parity-pinned-defect-repair/adrs/ADR-0001`
(accepted 2026-09-06), which now points here. That project built the
layer and closed on 2026-09-24. The layer outlived it, so the decision
and the rules that governed it are recorded here.

## Context

`test/parity/data/` pins 179,695 values captured from the pre-port
Cython, and nothing can regenerate them. The Cython is deleted, and
regenerating from a tree whose kernels run on Rust would pin the port
against itself (`test/parity/README.md`, "When *not* to regenerate").
Some of those values are wrong, and every repair has to get past the gate
that pins the wrong value. Re-pinning the affected arrays in place would
destroy the only record of what 2.1.0 shipped. Widening the case budget
until the repaired value fits would make the gate vacuous for that case
(`docs/agents/lessons.md` `[exemption-wider-than-its-mechanism]`).

The project that built the mechanism repaired a fixed roster of ten
defects, labelled `A1`–`A4` and `B1`–`B6`. Defects keep being found
after it, and each one that moves a pinned value needs the same
mechanism and a label to declare under.

## Decision

A repair is expressed as a **declaration**, in `test/parity/deltas.py`,
that named positions of named stored arrays now relate to the stored
value in a named way. The runner compares a declared position against
the declared relation and every undeclared position against the stored
value under the case's existing budget. Concretely:

- **The committed arrays and manifest are never rewritten.**
  `git diff --stat -- test/parity/data` is empty in a repair's PR.
- **No budget is widened to admit a repair.** If a repaired value does
  not match its declared relation, the declaration or the repair is
  wrong, not the budget.
- A declaration names the repair (from the closed set
  `deltas.REPAIRS`), the positions it covers, the relation, the
  measurement that justifies the relation's own budget, and the file
  holding the evidence. Where two repairs move the same array it names
  both, joined by `+` in landing order, and one `Composed` relation
  covers them. Two declarations never overlap.
- A relation answers one question: what should the repaired array be?
  The runner compares against the array it predicts. That prediction may
  come from a term added to the stored array, a closed-form transform of
  it, or a value a second implementation computes. It may also be one of
  those with further repairs composed in landing order. A further repair
  may add a term or transform the preceding prediction. It may not
  replace that prediction with an unrelated reference.
- **Every repair carries an independent oracle and a physics invariant.**
  The oracle is a closed form, a captured twin, or a second integrator,
  never the kernel being repaired. The invariant is something a corpus
  comparison cannot give, such as a yield per decay, a normalization
  integral or an endpoint. It is pinned in the kernel's own test module.
- The comparison is relative. A relation may declare an absolute floor
  beside its relative budget, but only where its own arithmetic cannot
  resolve the repaired value at every magnitude the array takes. That
  happens when a composition *relocates* a term: it subtracts the term
  back out of the base, and where the base is a captured array the
  cancellation destroys whatever the term's last bit could not hold. The
  floor is one ulp of the cancelled term. A shape test caps it below what
  the case already tolerates for a stored zero, so it cannot grow into a
  budget.
- A declaration covers only the positions its mechanism moves. That is
  either an explicit tuple every entry of which the relation moves, or
  `MOVED`, resolved at comparison time against the array the relation
  predicts. Naming a position the relation leaves at its stored value is
  an error, not a looser gate.
- A declaration whose array no longer differs from the stored one fails,
  so a reverted repair cannot hide behind it. A repair's declaration
  therefore cannot precede the repair. A delta modelled ahead of one
  waits in `deltas.DELTA_MODELS`, outside the table, until the array it
  describes has moved.
- Shape tests hold the table to real cases, blocks, arrays and
  positions, to labels in `deltas.REPAIRS`, and to a counted size.

**Labels after the project.** A repair that lands outside
`parity-pinned-defect-repair` takes the next label `C1`, `C2`, … in
landing order and adds it to `deltas.REPAIRS` in the same PR. The `A`
and `B` groups stay closed. A `C` repair's roster row goes in
`test/parity/README.md`, "Repairs". Its evidence is the follow-up it
resolves, moved to `docs/followups/done/` with the measurement written
into it. Its numerical change goes in `CHANGELOG.md` with its magnitude,
as `docs/versioning.md` requires of any moved published number.

## Consequences

- **Positive:** the corpus stays the record of what shipped. A repair
  proves it moved only what it meant to, because everything else is
  still gated at the old budget, and a revert turns the gate red.
  `Counter(d.repair for d in deltas.DECLARED_DELTAS.values())`
  aggregates how many pinned arrays are under a declaration, per repair.
  A post-project repair has a label, an evidence location and a roster
  without reopening a closed project.
- **Negative:** a relation is held to a budget of its own, and how much
  that loosens depends on how it builds its prediction. An additive term
  that is its own quadrature runs three orders of magnitude looser than
  the case budget at the declared positions, and costs one extra kernel
  evaluation per declared array in the gate. A closed-form transform or
  term costs neither, which is why it is the relation to reach for
  first. A relocation is the worst case: it cancels a term it did not
  itself compute, so at positions where that term dominated the result
  the prediction is exact only down to the term's last bit. Those
  positions are gated by an absolute floor rather than relatively. The
  table only grows, and so does the gate's cost.
- **Mitigation:** the relation's budget is measured and written beside
  it. The positions it loosens are exactly the ones the repair moved,
  and the mutation tests in `test/parity/test_parity.py` keep it from
  loosening any other. The captured Group A oracles stay committed under
  `projects/parity-pinned-defect-repair/adrs/ADR-0003-keep-the-group-a-oracle-captures-committed.md`,
  which binds only those four captures; a `C` repair supplies its own
  oracle.
