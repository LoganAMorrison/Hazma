# Task 1: The delta-declaration layer

**Date:** 2026-09-06
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md` Task 1;
`../references/corpus-repinning.md` ("The mechanism", "Shape tests",
"Proof obligations"); `../rules.md` rules 1, 2, 5, 6, 7, 11
**Related ADRs:** `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md` (project-scoped)
**Depends On:** none

## Objective

Give `test/parity` a way to say "this array moved, here, by this much,
because of this repair" without rewriting the array, so a repair can
pass the corpus gate while the gate keeps asserting everything the
repair did not touch.

## Exit Criteria

From `../PLAN.md` Task 1, with the one deviation stated:

- `pytest test/parity` green. **Deviation:** the plan asked for green
  with an *empty* declaration set, proving the layer inert before any
  repair lands. The layer landed in the same PR as its first repair
  (B4, `docs/followups/done/scalar-decay-fsr-half-normalized.md`), so
  inertness is shown instead by construction and by test: every array
  outside the 30 declared ones goes through the unchanged code path;
  within a declared array every position the term leaves at zero is
  compared at the case budget; and `TestTheDeclaredDeltaComparison`
  runs the mutations that would exploit either property.
- A shape test that fails if a declaration names a case, block, array
  or position the corpus does not contain —
  `test_every_declared_delta_addresses_a_real_stored_array`.
- `git diff --stat -- test/parity/data` empty — it is.

## Inputs Reviewed

- `../PLAN.md` Task 1 and "Anticipated ADRs".
- `../references/corpus-repinning.md`, the declaration schema and the
  five proof obligations.
- `../rules.md`, in particular rule 5 (allowlist, not a rule over a
  shape) and rule 6 (a declaration that describes no change fails).
- `test/parity/test_parity.py` and `test/parity/stability.py`, whose
  `(case_name, block_label, array_suffix)` key the layer reuses.
- `docs/agents/lessons.md` `[exemption-wider-than-its-mechanism]`
  (PR #71), which is the failure this layer is designed not to repeat.

## Findings

- **An additive term is its own quadrature.** The scalar decay term is
  evaluated live from the repaired kernel as half its FSR-only spectrum,
  and the stored total, the term and the repaired total are three
  different adaptive `cos θ` integrals over three different integrands.
  The relation `repaired == stored + term` therefore holds only to the
  integrator's accuracy, not to the case's 1e-9: measured 3.1e-4 worst
  relative over the declared positions, at `ms_550.boosted_strong`,
  `E = 2696` MeV, where the boost window is narrow. The relation carries
  its own budget (1e-3) and `why` string for that reason. This is not a
  widening under rule 2: the case budget still binds every undeclared
  position.
- **Positions must resolve against the term, not the array.** Review
  round 1 of PR #87 injected a 0.05% regression at
  `ms_250.rest_plus_eps.default.values[193]`, where the FSR term is
  exactly zero, and the first version of the layer accepted it at the
  relation budget because the declaration said `ALL`. The layer now has
  `MOVED` — the positions where the term is non-zero — and refuses an
  explicit tuple that names a position the term does not move.
- **A revert fails twice over.** With the term far outside the
  relation budget a revert trips the relation itself; with a term
  everywhere inside the budget the declaration asserts nothing and
  fails as stale. Both are tests.

## Decisions and Implementation Notes

- `test/parity/deltas.py` holds the schema (`Delta`, `Additive`,
  `MOVED`, `REPAIRS`) and the declarations; `test_parity.py` consumes it
  in `_assert_declared_delta` after budget selection and the stability
  masks, exactly where the spec places it.
- Only the `Additive` relation is implemented. `Exact`, `Oracle` and
  `Bounded` from the spec are added by the first repair that needs
  them rather than speculatively.
- `REPAIRS` is the closed set `A1`–`A4`, `B1`–`B4`; a declaration
  naming anything else fails the shape test.
- `EXPECTED_DECLARED_ARRAYS = 30` is a literal in `test_parity.py`, for
  the same reason the mask and floor counts are: growing the set must
  show up in a diff.
- Review round 1 also asked for the widening mutation the spec's second
  proof obligation names; it is
  `test_widening_a_tuple_by_an_unmoved_position_fails`.

## Files Changed

- `test/parity/deltas.py` — the layer and the B4 declaration.
- `test/parity/test_parity.py` — `_assert_declared_delta`, three shape
  tests, `TestTheDeclaredDeltaComparison` (seven mutation tests),
  `EXPECTED_DECLARED_ARRAYS`.
- `test/parity/README.md` — the file table, the carve-out list, the
  test count.
- `../PLAN.md`, `../rules.md`, `../references/defect-blast-radius.md`,
  `../references/corpus-repinning.md`, `README.md` (this directory) —
  B4 joins the roster; counts re-derived.
- `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md` — the
  decision the plan anticipated at this task.

## Verification

Declared positions, re-derived after the last edit (rule 11):

```sh
python - <<'EOF'
import sys, warnings; warnings.simplefilter("ignore"); sys.path.insert(0, "test/parity")
import numpy as np, cases as corpus, generate as gen, deltas
CASES = corpus.build_cases(); M = gen.load_manifest()
declared = total = over = 0
for (name, label, suffix), d in deltas.DECLARED_DELTAS.items():
    case = CASES[name]; block = next(b for b in case.blocks if b.label == label)
    term = d.relation.term(case.resolve(), block)[suffix]
    mb = next(b for b in M["cases"][name]["blocks"] if b["label"] == label)
    with np.load(gen.DATA_DIR / M["cases"][name]["file"]) as npz:
        pinned = npz[mb["arrays"][suffix]["key"]]
    moved = term != 0; declared += int(moved.sum()); total += term.size
    with np.errstate(divide="ignore", invalid="ignore"):
        over += int((np.abs(term) / np.abs(pinned))[moved].__gt__(1e-3).sum())
print(declared, total, over)
EOF
# -> 3065 4305 2874
```

So 3,065 of the 4,305 positions in the 30 declared arrays are under the
B4 declaration; 2,874 of them move by more than 0.1%; the remaining
1,240 are compared at the case budget of 1e-9. Whole-corpus accounting
for close time: 3,065 of 179,695.

Gates, on the final tree:

- Bare `scripts/agents/preflight.sh` over the Python paths and every
  touched markdown file; its pytest row: `2248 passed, 15 skipped, 12
  subtests passed in 24.08s`. Every other row passes; `ruff` reports
  the 35 errors `hazma/scalar_mediator/_scalar_mediator_spectra.py`
  already carries on `master`.
- `pytest test/parity -k "TestTheDeclaredDeltaComparison or declared or
  roster or counted"` — `14 passed`.
- `git diff --stat -- test/parity/data` — empty.

## Open Questions

- Whether `Exact`/`Oracle` relations need a budget field of their own
  or inherit the case budget. Decided by the first Group A repair
  (Task 4), which has an oracle rather than a computed term.

## Plan Impact

**Impact Level:** ADR required — written
(`../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`). `../PLAN.md`
Task 1 carries a status line; Task 8's gate now names the rule 7
obligation against B4; `../rules.md` rule 7 lists the new overlap;
`../references/defect-blast-radius.md` has the B4 row and section and
re-derived arithmetic (25 slots, union 20).

## Handoff to Next Task

- Read `test/parity/deltas.py`'s module docstring, then
  `_assert_declared_delta`; a new repair adds one `Delta` per moved
  array and bumps `EXPECTED_DECLARED_ARRAYS`.
- Safe to assume: undeclared arrays and undeclared positions are gated
  exactly as before this task.
- Risky: Task 8 (A3) reaches `scalar_mediator_decay_spectrum` through
  the `pi pi` decay channel and will overlap B4's positions wherever
  both the pion continuum and an FSR channel are open at the same
  energy; expect a composite declaration, not a disjointness proof.
