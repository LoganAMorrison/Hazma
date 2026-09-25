# ADR 0003: Keep the Group A oracle captures committed

**Date:** 2026-09-24
**Status:** Accepted
**Scope:** Project-scoped (applies only within `projects/parity-pinned-defect-repair/`).

## Context

`PLAN.md`'s "Anticipated ADRs" left one question open: should the Task 2
captures under `test/parity/oracles/data/` stay committed after
`cython-to-rust` closes? They are the last evidence that a repaired value
was checked against a non-Rust implementation. The Cython they were
patched from is gone: `cython-to-rust` Task 6.4 deleted the last `.pyx` in
`f479b231`. The project closed on 2026-08-29.

When Task 2 captured them, the arrays were a measurement only. By the
project's close they had become an input to the parity gate. Tasks 4, 7,
8 and 10 declare every Group A repair as a `deltas.Reference` (or a
`deltas.Composed` built on one), and `test/parity/oracle_reference.py`
reads `oracles/data/{A1,A2,A3,A4}.npz` through the capture manifest at
test time. That is 321 of the 343 declared arrays, counted by
`Counter(d.repair for d in deltas.DECLARED_DELTAS.values())`: A1 44,
`A1+B1` 6, `A1+B2` 6, A2 1, A3 62, `A3+B4` 20, `A3+B3` 4 and A4 178.

## Decision

The four captures, their `manifest.json`, the patches under
`test/parity/oracles/patches/` and the capture harness stay committed
for as long as `test/parity/deltas.py` declares a Group A relation. None
of them is regenerated. The capture needs the deleted Cython, so no
regeneration is possible anyway, and re-deriving from the Rust would
check the port against its own output.
`python test/parity/oracles/capture.py --check` remains the self-check, and
`test/parity/test_oracles.py` keeps asserting that the roster and the
capture agree.

The manifest also records provenance. Its `follow_up` fields still name
the `docs/followups/todo/` paths that were live when the capture ran.
Task 12 moved those follow-ups to `done/` and repointed the live roster
in `test/parity/oracles/defects.py`, but left the manifest untouched,
like the arrays it describes.

## Consequences

- **Positive:** every Group A declaration stays checked against an
  independent implementation. Deleting an oracle turns the parity suite
  red, because the relation cannot load it. The oracle cannot quietly go
  stale.
- **Negative:** about 1.6 MB of `.npz` and manifest (`ls -la
  test/parity/oracles/data`) stays in the tree permanently. Four
  manifest fields name follow-up paths that have since moved.
- **Mitigation:** the four follow-ups keep the same slug in `done/`, so
  a reader who follows a manifest path finds the file one directory
  over. `docs/followups/README.md` indexes all four.
