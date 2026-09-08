# Task 5: Repair B1 — the η′ two-photon line weight

**Date:** 2026-09-07
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md` "Task 5", "Numerical impact"; `../rules.md`
rules 1–7, 10, 11; `../references/corpus-repinning.md` (the shape tests);
`../references/defect-blast-radius.md` (B1's row)
**Related ADRs:** `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
(amended: a declaration may name more than one repair)
**Depends On:** Task 3 (the B1 model), Task 4 (A1, which declares the same
arrays)

## Objective

Give the η′'s `η′ → γγ` line the factor of two its three `X → γγ`
siblings carry, and declare the resulting corpus shift as a composite of
this repair and the A1 boost repair that already owns the same arrays.

## Exit Criteria

- `ETAP_TO_A_A_WEIGHT` is `2 · BR(η′ → γγ)`, and the two tests that pinned
  the shipped defect are renamed as well as re-pointed.
- The line-term integral measures `2 · BR = 0.04614` photons per decay,
  against the `0.02306998 ± 1.3e-08` the follow-up measured pre-repair.
- A declared delta on `spectra.photon.eta_prime` only. `spectra.photon.eta`,
  `spectra.photon.long_kaon` and `spectra.photon.short_kaon` — the three
  siblings that were already right — do not move.
- The declaration composes with A1's rather than overlapping it
  (`../rules.md` rule 7), and reverting either repair turns the gate red.
- `git diff --stat -- test/parity/data` empty.

## Inputs Reviewed

- `../PLAN.md` — Goal, "The premise this project corrects", "Numerical
  impact", Task 5, Task 6, "Anticipated ADRs".
- `../rules.md` — all eleven.
- `task-notes/README.md` — Tasks table, Open Questions, Handoff.
- `task-3-closed-form-deltas.md` Handoff (the B1 model and its predicted
  keys), `task-4-boost-window.md` Open Questions + Handoff (the composite
  constraint this task inherits).
- `../references/corpus-repinning.md` §"Shape tests", `defect-blast-radius.md`
  (B1's row and the A1/B1 overlap).
- `docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md`.
- `test/parity/deltas.py`, `test_parity.py`, `oracle_reference.py`,
  `test_delta_models.py`, `oracles/{defects,entry_points}.py`.
- `rust/src/kernels/photon_tables.rs`, `test/test_core_photon_tables.py`.

## Findings

- **The A1 capture for this case carries B1's defect, which is what makes
  the composite well-defined.** `oracles/entry_points.py` resolves
  `spectra.photon.eta_prime` to the restored Cython `_eta_prime.pyx`
  (`_TABLES_RESTORED`, deleted at `0954e5a`), and `oracles/defects.py`
  patches exactly one `.pyx` per defect — for A1 that is
  `hazma/_utils/boost.pyx`. So `oracles/data/A1.npz` holds the boost
  repair alone, over a kernel that still writes a bare `BR`. The repaired
  value is therefore that capture plus a second copy of the line.
- **A1's and B1's positions genuinely overlap; the composite is required,
  not a convenience.** 18 of B1's 189 positions are ones A1 also moves,
  distributed unevenly: all 9 of `rest_plus_eps.values`, 3 of
  `near_rest.values`, 0 of `boosted_mild.scalar_values`, 3 of
  `boosted_mild.values`, 1 of `boosted_strong.scalar_values` and 2 of
  `boosted_strong.values`. Neither "provably disjoint" nor "one contains
  the other" was available.
- **Two of A1's eight `eta_prime` arrays keep A1's declaration alone.**
  B1 moves nothing at the scalar probe of `rest_plus_eps` or `near_rest`
  — the probe falls outside the line's boosted window at those two parent
  energies — so declaring the composite there would name a repair the
  array never saw (`../rules.md` rule 5).
- **Task 3's predicted reach held exactly.** Six arrays, 189 positions,
  the same six keys, re-derived against the repaired kernel rather than
  inherited (`../rules.md` rule 11). `test_delta_models.EXPECTED_REACH`
  needs no change: it measures the standalone B1 model against the stored
  corpus, and neither moved.
- **Every move is upward, and the `rest` block is untouched.** 189 of 189
  positions rise, by 7.7e-04 to 1.0 relative. Both `rest` arrays are
  unchanged, because the kernel short-circuits to its rest-frame spectrum
  at `E − m < DBL_EPSILON` and that arm adds no line at all — the same
  reason `rest` is absent from A1's declaration.
- **The composite is one ulp from the repaired kernel, not bit-exact.**
  The kernel folds a single `2·BR` weight into the boost's own fused
  multiply-add; the prediction adds a second `BR` copy afterwards. That
  association difference is the whole residue: 2.1e-16 worst.

## Decisions and Implementation Notes

- **A fourth relation, `Composed`, rather than a hand-summed callable.**
  `deltas.Composed(base, added, rtol, why)` predicts the base relation's
  array with each `Additive` addend's term on top. Writing the sum as a
  bespoke `Reference` callable would have worked and been shorter, but it
  would have hidden the second repair from everything that reads the
  table — Task 12's aggregation included. Tasks 6 (A1+B2 on `phi`) and
  possibly 8 (A3+B4 on the scalar decay case) need the same shape, so the
  mechanism is built once here. The addends are typed `Additive` because
  an addend has to leave room for what came before it, which `Exact` and
  `Reference` do not.
- **`Composed.expected` copies the base's dict before adding.** A
  `Reference` base hands back `oracle_reference`'s `@cache`d arrays, which
  are read-only and shared by every comparison that reads them; updating
  in place would poison the capture for the rest of the run.
- **`REPAIRS` stays the closed set of ten roster labels.** A composite
  spells its label `"A1+B1"` and `deltas.repair_labels` splits it, so the
  roster keeps exactly one meaning and the close still aggregates per
  defect. `test_every_delta_model_is_a_roster_repair_with_its_evidence`
  now checks each part against `REPAIRS`, rejects a repeated part, and —
  the point of the change — requires a multi-part label to carry a
  `Composed` with one addend per extra part. A composite spelling on a
  plain relation would otherwise claim a repair the array never saw.
- **The composite holds at `rtol = 1e-12`, the case's own
  `tolerances.TABULATED_RTOL`, not at the 2.1e-16 measured.** Same reason
  A1 gives for its own budget: the capture is one platform's, and a libm
  that moves the boost integral must not fail at declared positions while
  the undeclared positions of the same block still pass. It is not a
  widening — 1e-12 is what these six arrays were already held to under
  A1 alone.
- **`SPECTRA["eta_prime"]` in `test/test_core_photon_tables.py` had to
  move with the kernel.** `test_each_line_carries_the_photon_count_its_
  weight_declares` measures the kernel's line term against that table's
  declared weight, so the table is the independent statement of the
  physics and not a mirror of the constant. Leaving it would have turned
  that test red; changing it is what keeps the measurement a measurement.
- **Both defect-pinning tests were renamed to state the repaired physics
  over all four siblings**, per `docs/agents/lessons.md`
  `[test-name-claims-an-unmade-assertion]`:
  `the_eta_prime_line_is_missing_its_factor_of_two` →
  `every_two_photon_line_carries_twice_its_branching_ratio`, and
  `TestPhysics::test_the_eta_prime_line_carries_half_the_photons_it_should`
  → `TestPhysics::test_every_two_photon_line_carries_twice_its_branching_ratio`.
  Both now assert the η′ *with* its three siblings rather than against
  them, and both keep an `assert_ne`/explicit-value line so a revert to
  the shipped weight fails on the number.
- **Review round 1 (PR #94) caught two stale claims, both class-shaped.**
  The follow-up this task annotated still said `hazma/_utils/boost.pyx`
  "is still live" — Task 6.4 deleted it, and `PLAN.md`'s own Task 3
  bullet already recorded the correction to `hazma._core.boost`; the
  class sweep found the identical sentence in the φ follow-up, which is
  Task 6's premise, and both are fixed. And this note's handoff said
  "Four defects" while listing five; the same sweep found
  `test_delta_models.py`'s module docstring asserting "all three defects
  are still live", which this repair falsified for B1. A green
  `check_doc_citations.py` did not catch the first class because a
  citation into a deleted file is reported as EXTERNAL and skipped —
  `docs/agents/lessons.md` `[touched-doc-inherits-its-citations]` now
  says so.
- **`folded_constants_match_the_shipped_object_code` keeps pinning the η′
  weight, against twice the shipped immediate.** `0x3f97_9fa9_7e13_2b56`
  → `0x3fa7_9fa9_7e13_2b56`: doubling only increments the exponent field,
  so the mantissa is the shipped one digit for digit, and the test still
  says where the constant came from.

## Files Changed

- `rust/src/kernels/photon_tables.rs` — `ETAP_TO_A_A_WEIGHT` is
  `2.0 * pdg::BR_ETAP_TO_A_A`; its doc comment, the module docstring's
  folded-constants paragraph, the folded-constant bit pattern, and the
  renamed weight test.
- `test/test_core_photon_tables.py` — `SPECTRA["eta_prime"]`'s line
  weight, the table's comment, and the renamed `TestPhysics` test.
- `test/parity/deltas.py` — the `Composed` relation, `repair_labels`, the
  `_A1_B1` declaration, six re-pointed `eta_prime` keys, and the module
  docstring's Relations section.
- `test/parity/test_parity.py` — the model shape test now validates
  composite labels.
- `CHANGELOG.md` — the `[Unreleased] / Changed` entry for this repair,
  stating the yield rather than a percentage of it.
- `docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md` —
  status annotated as repaired; the renamed tests, the surviving Rust
  expression, and the magnitude to quote in `CHANGELOG.md` corrected.
- `projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
  — the two Decision bullets this task made incomplete.
- `../references/corpus-repinning.md` — the Relations table, which still
  named two relations that were never built (`Oracle`, `Bounded`) and
  neither of the two that were. It now points at `deltas.py`'s docstring
  as the enumeration and keeps only the guidance, so it stops going stale
  each time a relation is added.
- `projects/parity-pinned-defect-repair/task-notes/README.md`,
  `.../task-5-eta-prime-line.md` — bookkeeping.

## Numerical impact

**`hazma.spectra.dnde_photon_eta_prime` moves at every parent energy above
rest; five sibling spectra and the `E = M_η′` rest case do not.** Measured
before and after on the same worktree, by reverting the Rust constant,
rebuilding the editable install, capturing, restoring (`cmp`-verified) and
rebuilding again.

Grid: `np.geomspace(1.0, 5 * M_η′, 601)` MeV at four parent energies.

| Entry point | `E_parent` (MeV) | moved | relative shift | direction |
| --- | --- | ---: | --- | --- |
| `dnde_photon_eta_prime` | 957.78 (`= M`) | 0/601 | — | — |
| `dnde_photon_eta_prime` | 958.73778 | 7/601 | 9.64e-01 … 1.00 | up |
| `dnde_photon_eta_prime` | 1436.67 | 137/601 | 2.57e-03 … 1.00 | up |
| `dnde_photon_eta_prime` | 4788.90 | 325/601 | 1.18e-03 … 1.00 | up |
| `dnde_photon_{eta,long_kaon,short_kaon,omega,phi}` | all four | 0/601 each | — | — |

The n-body public path, `hazma.spectra.dnde_photon(E, cme, states)` on
`np.geomspace(1, 2000, 201)` MeV at `cme = 2.5 M_η′`:

| Final state | moved | relative shift |
| --- | ---: | --- |
| `["etap", "etap"]` | 36/201 | 6.48e-03 … 1.00 |
| `["etap", "eta"]` | 45/201 | 1.78e-03 … 1.00 |
| `["eta", "eta"]` | 0/201 | — |

**Yield.** `scipy.integrate.quad` over `1e-3 ≤ E ≤ E_parent` at
`E_parent = 2 M_η′`: 3.80607171 → 3.82914171 photons per decay, a rise of
**0.02307000 = `BR_ETAP_TO_A_A` exactly**, which is **0.603%** of the
repaired total. `dnde_photon_eta` is unchanged at 3.30197000. The absolute
rise is the invariant — a boosted δ-function integrates to its own weight
at any boost — while the percentage depends on the integration window,
which is why `PLAN.md`'s pre-repair 0.63% estimate does not reproduce.

**Physics invariant (`../rules.md` rule 4).** The line term alone, isolated
by subtracting the boosted continuum and integrated with QUADPACK over the
line's window at `E_parent = 2 M`:

```text
eta         line integral = 0.78820000 +/- 8.8e-15   weight 0.78820000   ratio 1.0000000000
eta_prime   line integral = 0.04614000 +/- 5.1e-16   weight 0.04614000   ratio 1.0000000000
long_kaon   line integral = 0.00109400 +/- 8.4e-17   weight 0.00109400   ratio 1.0000000000
short_kaon  line integral = 0.00000526 +/- 3.0e-17   weight 0.00000526   ratio 1.0000000000
```

`0.04614000` is the `2 · BR` the Exit Criteria name, against the
`0.02306998 ± 1.3e-08` the follow-up measured pre-repair — a factor of
2.0000009 on the follow-up's own figure, and 2 exactly on the constant.

**Corpus.** 6 arrays, 189 positions, all in `spectra.photon.eta_prime`;
the other six tabulated photon cases are bit-identical across the rebuild.

| Block | Suffix | moved / size | relative shift | also moved by A1 |
| --- | --- | ---: | --- | ---: |
| `rest` | both | 0 / 285, 0 / 8 | — | 0 |
| `rest_plus_eps` | `values` | 9 / 285 | 1.00 | 9 |
| `rest_plus_eps` | `scalar_values` | 0 / 8 | — | — |
| `near_rest` | `values` | 20 / 285 | 2.12e-02 … 1.00 | 3 |
| `near_rest` | `scalar_values` | 0 / 8 | — | — |
| `boosted_mild` | `values` | 57 / 285 | 1.47e-03 … 1.00 | 3 |
| `boosted_mild` | `scalar_values` | 2 / 8 | 2.02e-03 … 7.89e-02 | 0 |
| `boosted_strong` | `values` | 98 / 285 | 7.66e-04 … 1.00 | 2 |
| `boosted_strong` | `scalar_values` | 3 / 8 | 7.66e-04 … 1.27e-02 | 1 |

`git diff --stat -- test/parity/data` is empty (`../rules.md` rule 1).

## Verification

```text
$ env PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh \
      --paths "test/parity/deltas.py test/parity/test_parity.py \
               test/test_core_photon_tables.py" \
      --md "<the four touched markdown files>"
RESULT: PASS
  black 3 files, isort 0 new (2 fixed), ruff 0 new (2 fixed),
  cargo fmt/clippy/test, import hazma 2.2.0, markdownlint, forbidden
  tokens none added, and
  pytest  2286 passed, 15 skipped, 1 warning, 37 subtests passed in 25.30s

$ .venv/bin/python -m pytest test/parity -q -n 0
687 passed, 1 skipped in 6.39s

$ .venv/bin/python -m pytest test/test_core_photon_tables.py -q -n 0
190 passed, 1 skipped in 0.81s

$ .venv/bin/python -m pytest test/parity/test_delta_models.py -q -n 0
17 passed in 0.26s

$ cargo test --manifest-path rust/Cargo.toml --no-default-features \
      --features test-probes
test result: ok. 263 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

The preflight `pytest` row is the full suite and is the run that gates the
commit. It reads 15 skipped where a bare `.venv/bin/python -m pytest`
reads 24, because nine `test/agents/test_lint_delta.py` cases shell out to
`ruff` and `isort` on `PATH`, which only preflight's invocation puts
there. The remaining 15 are the project's standing skips — 10 "known to
be broken" form factors, 3 vector form factors, the charged kaon's absent
monochromatic line, and `test_parity`'s provenance notice — none of them
touched here. The one warning is the pre-existing `SyntaxWarning` in
`hazma/vector_mediator/_vector_mediator_fsr.py`.

What the tests cover, by category:

- **The repaired constant, in Rust:** `photon_tables::tests::
  every_two_photon_line_carries_twice_its_branching_ratio` (all four
  `X → γγ` weights are `2·BR`, and the η′ is *not* the bare `BR`),
  `folded_constants_match_the_shipped_object_code` (the doubled bit
  pattern).
- **The physics invariant, in Python:** `TestPhysics::
  test_each_line_carries_the_photon_count_its_weight_declares` integrates
  the isolated line term for all seven spectra against the weight the test
  module declares, at a derived tolerance between 1e-5 and 1e-4;
  `TestPhysics::test_every_two_photon_line_carries_twice_its_branching_ratio`
  holds the four `X → γγ` weights together and pins the η′'s at 0.04614.
- **The corpus gate:** `test_entry_point_matches_corpus` over all 41 cases,
  which is where the composite declaration is exercised — six arrays
  against `A1+B1`, two against `A1` alone, two `rest` arrays against the
  stored values under the case's own budget.
- **The declaration's shape:** `test_every_declared_delta_addresses_a_real_
  stored_array`, `test_every_delta_model_is_a_roster_repair_with_its_
  evidence` (extended here for composite labels),
  `test_every_declaration_points_at_a_delta_model`,
  `test_the_declared_arrays_are_counted` (98, unchanged — this task
  re-points keys rather than adding them).
- **The model against the stored corpus:** `test_delta_models.py::
  test_each_model_moves_what_it_says_it_moves[B1]` and
  `test_the_eta_prime_line_is_stored_at_one_branching_ratio_not_two`, both
  of which read the committed arrays and no kernel, so they keep saying
  what the corpus pinned after the repair as before it.

**Test validity (stash-proof).** Reverting the one-line Rust change and
rebuilding turns four parity cases red before the declaration is added —

```text
FAILED test_entry_point_matches_corpus[spectra.photon.eta_prime[rest_plus_eps]]
FAILED test_entry_point_matches_corpus[spectra.photon.eta_prime[near_rest]]
FAILED test_entry_point_matches_corpus[spectra.photon.eta_prime[boosted_mild]]
FAILED test_entry_point_matches_corpus[spectra.photon.eta_prime[boosted_strong]]
4 failed, 1 passed, 645 deselected
```

— and red again *with* the declaration in place, on the staleness rule
(`../rules.md` rule 6): the composite's prediction still departs from the
stored array, so the live value must too. Both directions were run on a
rebuilt extension, not on `cargo test` alone. The `rest` block passes in
every configuration, which is the "moved only what it intended" half.

Nothing deferred.

## Open Questions

- **Should `DELTA_MODELS` keep the standalone `B1` entry now that
  `A1+B1` holds the keys?** It is kept: `test_delta_models.py` gates it
  against the stored corpus, `EXPECTED_REACH["B1"]` is the re-derivable
  statement of what B1 alone reaches, and Task 12 aggregates per roster
  entry. The cost is that `DELTA_MODELS` now has eight entries for seven
  modelled defects, and a reader has to know that `A1`, `B1` and `A1+B1`
  are not three independent repairs. Task 12 should say so once when it
  aggregates rather than leaving it to be inferred.
- **Does Task 6 need a second composite or can it extend this one?** A
  second: `A1+B2` is a different case (`spectra.photon.phi`) and a
  different addend, so it is another `Composed` beside this one, not a
  third part of this label.
- Inherited and still open: whether the Task 2 oracles stay committed
  after `cython-to-rust` closes (`README.md` Open Questions).

## Plan Impact

**Impact Level:** ADR required (amendment to ADR-0001).

Two of ADR-0001's Decision bullets became incomplete with this task and
are patched in the same change:

- "A declaration names the repair (from a closed roster)" — a composite
  names more than one, which rule 7 requires wherever two repairs move the
  same array. The bullet now says so and points at `repair_labels`.
- "A relation … may say so as a term added to the stored array or as a
  closed-form transform of it" — an enumeration that was already missing
  `Reference` (added by Task 4) and would now also miss `Composed`. It is
  restated as the invariant the relations share — a relation answers what
  the repaired array should be, and the runner compares against what it
  predicts — so it stops going stale each time one is added.

No task ordering, gate, interface or exit criterion changed. `PLAN.md`'s
Task 5 block is unchanged: its scope notes and its gate figure
(`2 · BR = 0.04614` against `0.02306998 ± 1.3e-08`) both reproduce. The
project's `version_bump: minor` is unchanged and still correct — a moved
published spectrum with no name, signature, shape or unit change.

## Stale-state sweep

```text
$ git status --short
 M CHANGELOG.md
 M docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md
 M projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md
 M projects/parity-pinned-defect-repair/references/corpus-repinning.md
 M projects/parity-pinned-defect-repair/task-notes/README.md
 A projects/parity-pinned-defect-repair/task-notes/task-5-eta-prime-line.md
 M rust/src/kernels/photon_tables.rs
 M test/parity/deltas.py
 M test/parity/test_parity.py
 M test/test_core_photon_tables.py
  The new note is `git add -N`'d so the diff below walks it.

$ git diff origin/master --stat --
 CHANGELOG.md                                       |  21 +
 ...eta-prime-two-photon-line-missing-factor-two.md |  39 +-
 .../ADR-0001-corpus-repairs-are-declared-deltas.md |  13 +-
 .../references/corpus-repinning.md                 |  18 +-
 .../task-notes/README.md                           | 139 ++++--
 .../task-notes/task-5-eta-prime-line.md            | 492 +++++++++++++++
 rust/src/kernels/photon_tables.rs                  |  53 ++-
 test/parity/deltas.py                              | 131 +++++-
 test/parity/test_parity.py                         |  21 +-
 test/test_core_photon_tables.py                    |  38 +-
 10 files changed, 848 insertions(+), 117 deletions(-)

$ .venv/bin/python scripts/agents/check_doc_citations.py <the six touched docs>
docs scanned: 6
in-repo citations checked: 0
external citations skipped: 5
out-of-range or ambiguous: NONE
  The five skipped are the deleted `.pyx` this repair's history cites --
  `constants.pxd`, `_eta.pyx`, `_eta_prime.pyx` and `_kaon.pyx` -- which
  the checker cannot range-check because the port removed the files.
  `docs/followups/todo/citation-checker-skips-deleted-inrepo-files.md`
  already carries that gap.

$ git diff --stat -- test/parity/data | wc -l
0
  `../rules.md` rule 1 holds: the committed corpus is untouched.

$ git rev-parse --abbrev-ref HEAD && git rev-parse --show-toplevel
claude/parity-pinned-defect-repair/task-5-eta-prime-line
/Users/logan.morrison/dev/Hazma/.claude/worktrees/brave-curie-b1ca04

$ grep -rn "the_eta_prime_line_is_missing_its_factor_of_two" . | grep -v '.git/'
projects/parity-pinned-defect-repair/PLAN.md:345
$ grep -rn "test_the_eta_prime_line_carries_half_the_photons_it_should" . \
      | grep -v '.git/'
projects/parity-pinned-defect-repair/PLAN.md:346
  Both are PLAN.md's Task 5 scope note naming the pre-repair symbols as the
  ones this task must rename. That instruction is still a true account of
  the task, so the block is left as written; the follow-up's two copies
  were re-pointed to the new names.

$ grep -rnE "TODO|FIXME|breakpoint\(\)|import pdb" \
      rust/src/kernels/photon_tables.rs test/parity/deltas.py \
      test/parity/test_parity.py test/test_core_photon_tables.py
  (no occurrences)
  preflight's `forbidden tokens` row reports `none added` over the same
  diff, which also covers stray `print(`.

$ .venv/bin/python -c "import sys; sys.path.insert(0,'test/parity'); import deltas;
    from collections import Counter;
    print(len(deltas.DECLARED_DELTAS), Counter(d.repair for d in deltas.DECLARED_DELTAS.values()))"
98 Counter({'A1': 50, 'B4': 30, 'A1+B1': 6, 'B5': 6, 'B6': 6})
  Matches `test_parity.EXPECTED_DECLARED_ARRAYS`, which this task does not
  change: A1 falls 56 -> 50 and those six move to `A1+B1`.
  50 + 30 + 6 + 6 + 6 = 98.

$ .venv/bin/python -m pytest test/parity/test_delta_models.py -q -n 0
17 passed in 0.26s
  `EXPECTED_REACH == {"B1": (6, 189), "B2": (6, 305), "B3": (4, 350)}` still
  holds unedited: it measures the standalone models against the stored
  corpus, and neither the models nor the corpus moved.

$ .venv/bin/python -c "import hazma._core; print(hazma._core.__file__)"
/Users/logan.morrison/dev/Hazma/.claude/worktrees/brave-curie-b1ca04/hazma/_core.abi3.so
  Every Python-side number in this note was taken after the editable
  reinstall, on this worktree's own extension.

Bookkeeping consistency: this note's `**Status:**` is `Complete`; the
Tasks-table cell in `task-notes/README.md` reads
"**Complete** — declared as the composite `A1+B1`"; that file's
"Numerical impact so far" opens "Five public values have moved" and
carries the B1 bullet; "Files Changed" has a `### Task 5 (B1)` roll-up;
two Open Questions (the A1/B1 collapse, the Group B budgets) are answered
in place rather than left standing. No phase file and no
`projects/README.md` row apply — this is a flat project and not its
closing task.

Numerical-impact statement: `dnde_photon_eta_prime` rises by exactly
`BR_ETAP_TO_A_A` = 0.02307 photons per decay at every parent energy above
rest (0.603% of the repaired 3.8291 at `E_parent = 2 M`), and is unchanged
at `E_parent = M`. `dnde_photon` on an η′-bearing final state moves with
it. No other public function moves: verified by the before/after diff over
`dnde_photon_{eta,long_kaon,short_kaon,omega,phi}` at four parent energies
and `dnde_photon(["eta", "eta"])`, all bit-identical across the rebuild.
Recorded in `README.md`'s "Numerical impact so far" (`../rules.md`
rule 10).
```

## Handoff to Next Task

**Task 6 (B2, the φ line energies) is next and inherits a worked
precedent for its hardest part.**

- The composite mechanics are built: `deltas.Composed(base, added, rtol,
  why)` and `deltas.repair_labels`. Task 6 adds `_A1_B2` in the same
  shape — `base=_A1.relation`, `added=(_B2.relation,)` — keyed under
  `"A1+B2"` in `DELTA_MODELS`, and re-points the `spectra.photon.phi`
  keys B2 measurably moves. Nothing in the runner or the shape tests
  needs changing again.
- **Re-derive which of φ's eight A1 arrays B2 actually moves, then
  re-point only those.** Task 3 predicted six; this task's η′ prediction
  held exactly, but the two `rest_plus_eps`/`near_rest` scalar probes
  were the arrays that turned out to be B1-free, and B2's relocation is a
  different mechanism from B1's second copy. `EXPECTED_DECLARED_ARRAYS`
  stays 98 if the count is six again.
- The measurement recipe that produced every figure above: capture the
  corpus blocks from the defective build, revert the one constant, rebuild
  the **editable install** (not `cargo build`), capture, restore with a
  `cmp` check, rebuild, and diff those two captures. `cargo test` alone
  proves nothing about a Python-side number.
- B2's gate is a *position* assertion, not a yield one — `PLAN.md` Task 6
  says why, and this task's yield measurement is the counter-example that
  makes the point concrete: B1 moved the yield by exactly its weight,
  where B2 moves none of it.

**Currently safe to assume:**

- `test/parity/data/` is still intact and untouched by this project.
- Five defects have now moved a library value: B4, B5, B6, A1 and B1.
  `deltas.DECLARED_DELTAS` holds 98 arrays across five declarations, one
  of which is composite.
- `../rules.md` rule 7's first real test is answered: A1 and B1 overlap at
  18 positions, and the composite is the mechanism that carries it.

**Currently risky / unknown:**

- The composite's budget is the case budget, not the 2.1e-16 this platform
  measures. That is deliberate, but it means a repair that leaks a term
  into a declared position below 1e-12 would not be caught there. The
  undeclared positions of the same blocks still hold the line.
- Task 9's B3 declaration is still an ordinary two-key addition; nothing
  here changes that, and it should not be composed with anything.
