# Task 6: Repair B2 — the φ photon line energies

**Date:** 2026-09-07
**Project:** parity-pinned-defect-repair
**Status:** Complete
**Plan References:** `../PLAN.md` "Task 6", "Numerical impact"; `../rules.md`
rules 1–7, 10, 11; `../references/corpus-repinning.md` (Relations, the
shape tests); `../references/defect-blast-radius.md` (B2's row)
**Related ADRs:** `../adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
(amended: a relation may declare an absolute floor)
**Depends On:** Task 3 (the B2 model), Task 4 (A1, which declares the same
arrays), Task 5 (the `Composed` relation)

## Objective

Put both φ photon lines at the photon's rest-frame energy rather than the
daughter meson's, and declare the resulting corpus shift as a composite of
this repair and the A1 boost repair that already owns the same arrays.

## Exit Criteria

- `PHI_TO_ETA_A_ENERGY` is 362.5189975276151 MeV and
  `PHI_TO_ETAP_A_ENERGY` is 59.815040556235125 MeV, both from
  `(M_φ² − m²) / (2 M_φ)`.
- The gate is a **position** assertion, not a magnitude one: the total
  yield is unchanged at 0.013092 photons per decay, so a yield-only check
  passes on an unrepaired kernel.
- A declared delta on `spectra.photon.phi` only. `spectra.photon.omega`,
  whose two `ω → Yγ` lines were always the photon's, does not move.
- The declaration composes with A1's rather than overlapping it
  (`../rules.md` rule 7), and reverting the repair turns the gate red.
- `git diff --stat -- test/parity/data` empty.

## Inputs Reviewed

- `../PLAN.md` — Goal, "Numerical impact", Task 6, "Orientation".
- `../rules.md` — all eleven.
- `task-notes/README.md` — Tasks table, Open Questions, Handoff.
- `task-5-eta-prime-line.md` — the worked composite precedent, its Open
  Questions (which answered "second composite, not a third part").
- `../references/corpus-repinning.md` §Relations, §"How the runner uses
  it"; `defect-blast-radius.md` (B2's row and the A1/B2 overlap).
- `docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md`,
  including its unanswered `φ → π⁰γ` risk bullet.
- `test/parity/deltas.py`, `test_parity.py`, `test_delta_models.py`,
  `oracle_reference.py`, `tolerances.py`, `stability.py`.
- `rust/src/kernels/photon_tables.rs`, `test/test_core_photon_tables.py`.

## Findings

- **Task 3's predicted reach held exactly, but not the predicted arrays.**
  Six arrays and 305 positions, 233 up and 72 down — the model's own
  numbers, re-derived against the repaired kernel (`../rules.md` rule 11).
  The six are *not* the six B1 moved: B2 reaches `near_rest.scalar_values`
  where B1 did not, and reaches neither `rest_plus_eps` array where B1
  reached one. At `rest_plus_eps` the boost opens a window 2.8e-06 wide in
  relative energy, too narrow for any grid point to fall inside a shipped
  or a repaired line — a fact that holds for both, which is why that block
  keeps A1's declaration alone.
- **A1 and B2 genuinely overlap: 90 of B2's 305 positions.** Unevenly —
  19 of `near_rest.values`, 0 of `near_rest.scalar_values`, 36 and 1 of
  `boosted_mild.{values,scalar_values}`, 33 and 1 of `boosted_strong`.
  Neither "provably disjoint" nor "one contains the other" was available,
  so the composite is required rather than chosen.
- **The sign splits, which is the signature of a relocation.** B1 moved
  189 of 189 positions upward, because it added a second copy of a line.
  B2 moves 233 up and 72 down: the shipped lines vacate their windows and
  the repaired ones fill new ones. That is why the deliverable's gate had
  to be positional — see the next finding.
- **A yield-only gate would have passed on the unrepaired kernel, and this
  was confirmed rather than reasoned about.** With the two constants
  reverted and the extension rebuilt,
  `test_each_line_carries_the_photon_count_its_weight_declares[phi]` still
  passes: a boosted δ-function integrates to its own weight wherever it
  sits. The new `test_each_line_sits_where_its_rest_frame_energy_declares`
  is what fails, and only for the φ.
- **The composite cannot be exact, and the reason is representation
  rather than budget.** At two positions — `near_rest.values[195]` and
  `boosted_mild.values[193]`, both inside the shipped `φ → ηγ` window and
  outside both repaired ones — the prediction reads exactly 0.0 where the
  kernel returns 3.331476e-21 and 3.143213e-22, which is the boosted
  continuum (confirmed against `boost_integrate_linear_interp` directly).
  The addend subtracts the shipped line back out of the A1 capture, and
  the plateau there is 3.107723e-05, whose last bit is 6.8e-21 — larger
  than the continuum underneath it, so the capture never held it and no
  prediction built on the capture can return it. This is the layer's first
  relation that *relocates* a term rather than adding one, and the only
  kind that can hit this.
- **The φ is indeed missing a direct `φ → π⁰γ` line**, which the
  follow-up's risk bullet asked to be checked before this repair. The ω
  supplied the control its truncation argument lacked: normalizing each
  `pi0_a` column by its own `BR(X → π⁰γ)` gives 1.978 for the φ and 1.951
  for the ω against `2 BR(π⁰ → γγ) = 1.976`, so both columns hold the π⁰'s
  own decay photons and nothing else, and the ω's direct photon comes from
  its line. `BR_PHI_TO_PI0_A` (`rust/src/constants.rs:359`) is read by
  nothing. Filed rather than folded in — see Decisions.

## Decisions and Implementation Notes

- **A relation may declare an absolute floor, and `A1+B2` declares
  1e-20.** `../rules.md` rule 2 forbids widening a budget when a repaired
  value misses its relation, so the declaration had to change instead. The
  honest statement is not relative: at a position whose predicted value is
  exactly zero, no `rtol` describes anything. The floor is one ulp of the
  plateau the composition cancels (6.8e-21, rounded to 1e-20). It sits
  **11.6 decades under the smallest value these six arrays carry above it**
  (3.849e-09) and 8.6 decades under the tightest `zero_floor` among them
  (3.580e-12), so nothing but the two cancellation positions is inside it.
  `atol` is a field on all four relations, defaulting to `0.0`, and
  `_assert_declared_delta` compares at `max(budget.atol, relation.atol)`;
  the staleness check keeps the case's own floor, because whether the
  repair happened is a different question from how precisely it can be
  predicted. ADR-0001 is amended accordingly.
- **The floor is capped, not merely justified.**
  `test_a_declared_absolute_floor_stays_under_its_arrays_zero_floor`
  requires every declared `atol` to sit below `tolerances.zero_floor` of
  the array it covers — what the corpus already tolerates where it stored
  an exact zero — and counts them against `EXPECTED_FLOORED_ARRAYS = 6`.
  Uncapped, the field would be a tunable that grows until the gate
  passes: `docs/agents/lessons.md` `[exemption-wider-than-its-mechanism]`.
- **Both φ energies and both ω energies now go through one
  `const fn photon_line_energy(parent, daughter)`.** The defect was one
  sign in an expression written out four times — the follow-up traces its
  origin to exactly that — and one function leaves no site for the `+`
  form to be written at. The ω constants are bit-identical through it,
  which `every_folded_constant_is_the_shipped_immediate_or_its_declared_repair`
  asserts against the immediates the shipped object code loads.
- **`folded_constants_match_the_shipped_object_code` was renamed**, per
  `docs/agents/lessons.md` `[test-name-claims-an-unmade-assertion]`: with
  three of its sixteen constants now carrying a repaired value rather than
  the shipped immediate, the old name claimed a check the test no longer
  performs for all of them. It is
  `every_folded_constant_is_the_shipped_immediate_or_its_declared_repair`,
  and each repaired constant names the immediate it replaces beside it, so
  the object code stays recoverable from the test.
- **Both defect-pinning tests were rewritten to state the repaired physics
  over all four `X → Yγ` lines**, the ω's held with the φ's rather than as
  a control, since both mesons now run through the same function:
  `the_phi_line_energies_are_the_daughter_mesons` →
  `every_line_energy_is_the_photons_not_the_daughters`, and
  `TestPhysics::test_the_phi_lines_sit_at_the_daughter_mesons_energy` →
  `TestPhysics::test_every_line_sits_at_the_photons_energy_not_the_daughters`.
  Both keep an explicit-value and a not-equal line against the shipped
  energies, and both add `E_γ < M/2`, the bound no two-body photon can
  exceed and which the shipped `φ → η′γ` energy broke by 88%.
- **The missing `φ → π⁰γ` line is filed, not folded in.** The follow-up
  instructed the opposite ("the same PR should add the missing line"), and
  that instruction is superseded on its own terms: it was priced against
  regenerating the corpus once for several changes, and under ADR-0001 the
  corpus is never regenerated, so batching saves nothing. It is also a
  different defect — adding a line raises the yield where B2 relocates it
  — and it is not among the ten `deltas.REPAIRS` holds as a closed set.
  The follow-up's bullet is rewritten to record the answer and the reason.
- **`SPECTRA["phi"]` in `test/test_core_photon_tables.py` moved with the
  kernel**, as `SPECTRA["eta_prime"]` did in Task 5: that table is the
  independent statement of which lines the kernel carries.
- **The public-impact claim is scoped to a φ in flight** (review round 1).
  The first draft said `dnde_photon` moves for "any final state containing
  a φ", one sentence after recording that a φ at rest is untouched. At the
  two-body production threshold every φ is at rest, so `branch` returns
  `Branch::RestFrame`, that arm adds no line, and neither line energy is
  reachable. The sweep found the identical overstatement in the merged
  `CHANGELOG.md` entries for B1 and A1, each one sentence after its own
  rest-frame caveat; all five occurrences are corrected and the class is
  in `docs/agents/lessons.md` as
  `[composed-entry-point-inherits-the-branch-caveat]`.

## Files Changed

- `rust/src/kernels/photon_tables.rs` — the `photon_line_energy` const fn,
  both φ energies and both ω energies routed through it, their doc
  comments, the module docstring's folded-constants paragraph, the two
  renamed tests and the two repaired bit patterns.
- `test/test_core_photon_tables.py` — `SPECTRA["phi"]`'s two line
  energies, the module docstring's defect section, the renamed
  `TestPhysics` test, and the new
  `test_each_line_sits_where_its_rest_frame_energy_declares`.
- `test/parity/deltas.py` — the `atol` field on all four relations and its
  "Absolute floors" section, the `_A1_B2` declaration, six re-pointed
  `phi` keys, and the two line-energy helpers' docstrings.
- `test/parity/test_parity.py` — the runner honors a relation's floor,
  `EXPECTED_FLOORED_ARRAYS`, and the cap test.
- `test/parity/test_delta_models.py` — the module docstring's roll-call of
  which models have landed, and `MODEL_CASES`.
- `test/parity/README.md` — the declared-repair roll-call (already stale
  for B1) and the third carve-out.
- `CHANGELOG.md` — the `[Unreleased] / Changed` entry, stating the
  relocation and the unchanged yield.
- `docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md`
  — status annotated as repaired, the renamed tests, and the `φ → π⁰γ`
  risk bullet answered.
- `docs/followups/todo/phi-omits-its-direct-pi0-photon-line.md`,
  `docs/followups/README.md` — the new follow-up and its index row.
- `projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`
  — the absolute-floor decision bullet and its consequence.
- `../references/corpus-repinning.md` — the floor, under Relations.
- `../references/defect-blast-radius.md` — B1's and B2's measured reach,
  replacing "all blocks" in both.
- `projects/parity-pinned-defect-repair/task-notes/README.md`,
  `.../task-6-phi-lines.md` — bookkeeping.

## Numerical impact

**`hazma.spectra.dnde_photon_phi` moves at every parent energy above rest;
six sibling spectra and the `E = M_φ` rest case do not.** Measured before
and after on the same worktree, by reverting the two Rust constants,
rebuilding the editable install, capturing, restoring (`grep`-verified)
and rebuilding again.

Grid: `np.geomspace(1.0, 5 M, 601)` MeV at four parent energies per
spectrum.

| Entry point | `E_parent` | moved | relative shift | up / down |
| --- | --- | ---: | --- | ---: |
| `dnde_photon_phi` | `M` | 0/601 | — | — |
| `dnde_photon_phi` | `M (1 + 1e-6)` | 1/601 | 1.00 | 0 / 1 |
| `dnde_photon_phi` | `1.5 M` | 331/601 | 5.26e-05 … 1.00 | 262 / 69 |
| `dnde_photon_phi` | `5 M` | 474/601 | 1.12e-05 … 1.00 | 449 / 25 |
| `dnde_photon_{eta,eta_prime,omega,charged_kaon,long_kaon,short_kaon}` | all four | 0/601 each | — | — |

The n-body public path, `hazma.spectra.dnde_photon(E, cme, states)` on
`np.geomspace(1, 2000, 201)` MeV at `cme = 2.5 M_φ`:

| Final state | `cme` | moved | relative shift | up / down |
| --- | --- | ---: | --- | ---: |
| `["phi", "phi"]` | `2.5 M_φ` | 98/201 | 6.57e-05 … 1.00 | 73 / 25 |
| `["phi", "eta"]` | `2.5 M_φ` | 113/201 | 2.81e-05 … 1.00 | 90 / 23 |
| `["eta", "eta"]` | `2.5 M_φ` | 0/201 | — | — |
| `["phi", "phi"]` | `2 M_φ` | 0/201 | — | — |
| `["phi", "eta"]` | `M_φ + M_η` | 0/201 | — | — |

The last two rows are the two-body production thresholds, where every φ is
produced at rest: `branch` returns `Branch::RestFrame` at
`E − m < DBL_EPSILON`, that arm adds no line, and so neither repaired
energy is reachable. A composed entry point is therefore narrower in reach
than the kernel it calls, not equal to it.

**Yield: relocated, not changed.** The line term isolated by subtracting
the boosted continuum, integrated over its own window on a 2,000,001-point
grid at `E_φ = 2 M_φ`, is **0.013092203** against the
`BR(φ → ηγ) + BR(φ → η′γ) = 0.013092200` the two weights declare — the
residual is the trapezoid across the plateau edges. That figure is
invariant under the repair, because a boosted δ-function integrates to its
own weight wherever it sits, which is precisely why the gate is
positional. `PLAN.md`'s pre-repair "0.60% of the photon yield" reproduces
against the rest-frame continuum: `0.013092 / 2.1616 = 0.606%` of the
continuum, `0.602%` of the two together. Integrating the *lab-frame*
spectrum on a fixed 601-point grid does move — by −6.6e-02 at
`E_φ = M(1 + 1e-6)` — but that is the trapezoid failing to resolve a
plateau whose width and height both change with its position, not a yield
change; the window-independent statement above is the one to quote.

**Physics invariant (`../rules.md` rule 4).** The line positions recovered
from the kernel's own boosted output at `E_φ = 2 M_φ`, by inverting the
outermost edges of the isolated line term:

```text
phi -> etap gamma   recovered  59.815510 MeV   declared  59.815041   rel 7.8e-06
phi -> eta gamma    recovered 362.518896 MeV   declared 362.518998   rel 2.8e-07
```

Both inside the grid's own resolution (`cell / edge` is 6.3e-05 and
7.5e-07). Against the shipped 959.646 and 656.942 MeV these are off by
1504% and 81%, so the measurement separates the two hypotheses by five
decades. `E_γ + E_daughter = M_φ` holds for both, and both now satisfy
`E_γ < M_φ / 2 = 509.73` MeV, which the shipped `φ → η′γ` energy did not.

**Corpus.** 6 arrays, 305 positions, all in `spectra.photon.phi`; the
other six tabulated photon cases are bit-identical across the rebuild.

| Block | Suffix | moved / size | relative shift | up / down | also moved by A1 |
| --- | --- | ---: | --- | ---: | ---: |
| `rest` | `values` | 0 / 255 | — | — | 0 |
| `rest` | `scalar_values` | 0 / 8 | — | — | 0 |
| `rest_plus_eps` | `values` | 0 / 285 | — | — | 0 |
| `rest_plus_eps` | `scalar_values` | 0 / 8 | — | — | 0 |
| `near_rest` | `values` | 57 / 285 | 1.29e-04 … 1.00e+00 | 23 / 34 | 19 |
| `near_rest` | `scalar_values` | 1 / 8 | 1.00e+00 | 0 / 1 | 0 |
| `boosted_mild` | `values` | 102 / 285 | 4.48e-05 … 1.00e+00 | 83 / 19 | 36 |
| `boosted_mild` | `scalar_values` | 3 / 8 | 5.40e-05 … 2.35e-01 | 3 / 0 | 1 |
| `boosted_strong` | `values` | 139 / 285 | 2.91e-06 … 1.00e+00 | 121 / 18 | 33 |
| `boosted_strong` | `scalar_values` | 3 / 8 | 6.14e-04 … 1.47e-02 | 3 / 0 | 1 |

The composite's residue against the repaired kernel, per array: 0.0 at
`near_rest.scalar_values`, 1.2e-16 and 1.9e-16 at the two boosted scalar
probes, 2.9e-14 worst at `boosted_strong.values`, and the two
cancellation positions the 1e-20 floor covers.

`git diff --stat -- test/parity/data` is empty (`../rules.md` rule 1).

## Verification

```text
$ env PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh \
      --paths "test/parity/deltas.py test/parity/test_parity.py \
               test/parity/test_delta_models.py \
               test/test_core_photon_tables.py" \
      --md "<the ten touched markdown files>"
RESULT: PASS
  black 4 files, isort 0 new (2 fixed), ruff 0 new (2 fixed),
  cargo fmt/clippy/test, import hazma 2.2.0, markdownlint, forbidden
  tokens none added, and
  pytest  2293 passed, 16 skipped, 1 warning, 37 subtests passed in 26.80s

$ .venv/bin/python -m pytest test/parity -q -n 0
688 passed, 1 skipped in 6.41s

$ .venv/bin/python -m pytest test/test_core_photon_tables.py -q -n 0
196 passed, 2 skipped in 5.69s

$ cargo test --manifest-path rust/Cargo.toml --no-default-features \
      --features test-probes
test result: ok. 263 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

The preflight `pytest` row is the full suite and is the run that gates the
commit. It reads 16 skipped where a bare `.venv/bin/python -m pytest` reads
25, because nine `test/agents/test_lint_delta.py` cases shell out to `ruff`
and `isort` on `PATH`, which only preflight's invocation puts there. The
count rose from Task 5's 2286/15 by the seven cases this task adds — six
parametrizations of the new line-position test, the seventh skipping for
the charged kaon, plus the floor-cap test. The remaining 15 standing skips
are untouched, and the one warning is the pre-existing `SyntaxWarning` in
`hazma/vector_mediator/_vector_mediator_fsr.py`.

What the tests cover, by category:

- **The repaired constants, in Rust:**
  `photon_tables::tests::every_line_energy_is_the_photons_not_the_daughters`
  (all four `X → Yγ` energies against `E_γ + E_Y = M`, `E_γ < E_Y` and
  `E_γ < M/2`, plus the two φ values explicitly and `assert_ne!` against
  the shipped pair), and
  `every_folded_constant_is_the_shipped_immediate_or_its_declared_repair`
  (the two repaired bit patterns, with the shipped immediates named).
- **The physics invariant, in Python:**
  `TestPhysics::test_each_line_sits_where_its_rest_frame_energy_declares`
  recovers the smallest and largest rest-frame line energy from the
  kernel's boosted output for all six line-bearing spectra, at a tolerance
  derived from the grid cell;
  `TestPhysics::test_every_line_sits_at_the_photons_energy_not_the_daughters`
  holds the four `X → Yγ` energies against the identity and the numbers.
- **The corpus gate:** `test_entry_point_matches_corpus` over all 41
  cases, where the composite is exercised — six arrays against `A1+B2`,
  four against `A1` alone, two `rest` arrays against the stored values
  under the case's own budget.
- **The declaration's shape:**
  `test_every_declared_delta_addresses_a_real_stored_array`,
  `test_every_delta_model_is_a_roster_repair_with_its_evidence` (which
  already validated composite labels from Task 5),
  `test_every_declaration_points_at_a_delta_model`,
  `test_the_declared_arrays_are_counted` (98, unchanged — this task
  re-points keys rather than adding them), and the new
  `test_a_declared_absolute_floor_stays_under_its_arrays_zero_floor`.
- **The model against the stored corpus:**
  `test_delta_models.py::test_each_model_moves_what_it_says_it_moves[B2]`
  and `test_the_phi_lines_are_stored_at_the_daughter_mesons_energy`, both
  of which read the committed arrays and no kernel, so they keep saying
  what the corpus pinned after the repair as before it.

**Test validity (stash-proof).** Reverting both Rust constants to the `+`
form and rebuilding the extension turns three parity cases red *with* the
declaration in place — on the relation itself, not on the staleness rule,
because the composite's prediction no longer matches the live value:

```text
FAILED test_entry_point_matches_corpus[spectra.photon.phi[near_rest]]
FAILED test_entry_point_matches_corpus[spectra.photon.phi[boosted_mild]]
FAILED test_entry_point_matches_corpus[spectra.photon.phi[boosted_strong]]
3 failed, 2 passed, 618 deselected
```

`rest` and `rest_plus_eps` pass in both configurations, which is the
"moved only what it intended" half. On the same reverted build:

- `photon_tables::tests::every_line_energy_is_the_photons_not_the_daughters`
  and `every_folded_constant_is_the_shipped_immediate_or_its_declared_repair`
  fail, and every other test in the module passes;
- `TestPhysics::test_each_line_sits_where_its_rest_frame_energy_declares[phi]`
  fails and the other five parametrizations pass;
- `test_each_line_carries_the_photon_count_its_weight_declares[phi]`
  **passes** — the direct confirmation that a yield-only gate would not
  have caught this;
- `test_delta_models.py` passes in full (17 tests), because it reads the
  committed corpus and no kernel.

Nothing deferred.

## Open Questions

- **Will any later repair need an absolute floor?** The mechanism is
  narrow: only a `Composed` whose addend relocates a term can hit it, and
  of the four remaining repairs only B3 transforms rather than adds — as
  an `Exact` on the stored array, which cancels nothing. So the expected
  answer is no, and `EXPECTED_FLOORED_ARRAYS = 6` is what makes a second
  one visible in a diff rather than absorbed.
- **Should `DELTA_MODELS` keep the standalone `B2` entry now that `A1+B2`
  holds the keys?** Kept, for the reason Task 5 gave for `B1`:
  `test_delta_models.py` gates it against the stored corpus and
  `EXPECTED_REACH["B2"]` is the re-derivable statement of what B2 alone
  reaches. `DELTA_MODELS` now has nine entries for seven modelled defects;
  Task 12 should say once that `A1`, `B1`, `B2`, `A1+B1` and `A1+B2` are
  five entries for three repairs rather than leaving it to be inferred.
- **Does the φ omit any *other* direct photon line, and do its siblings?**
  The new follow-up asks it as its own first question — the sweep behind
  the `pi0_a` table covered only the two parents with such a column, and
  the general form is whether every `X → Yγ` branching ratio in
  `rust/src/constants.rs` has a line in `photon_tables.rs`.
- Inherited and still open: whether the Task 2 oracles stay committed
  after `cython-to-rust` closes (`README.md` Open Questions).

## Plan Impact

**Impact Level:** ADR required (second amendment to ADR-0001).

ADR-0001's Decision gains a bullet: the comparison is relative, and a
relation may declare an absolute floor only where its own arithmetic
cannot resolve the repaired value at every magnitude the array takes. The
Consequences' Negative bullet gains the case that produces one — a
relocation cancels a term it did not compute, so at positions where that
term dominated, the prediction is exact only to the term's last bit.

No task ordering, gate, interface or exit criterion changed. `PLAN.md`'s
Task 6 block is unchanged and reproduces in full: both energies, the
factor of 16, the 0.013092 relocated yield, the positional gate, and
`spectra.photon.omega` not moving. The project's `version_bump: minor` is
unchanged and still correct — a moved published spectrum with no name,
signature, shape or unit change.

## Stale-state sweep

```text
$ git status --short
 M CHANGELOG.md
 M docs/followups/README.md
 A docs/followups/todo/phi-omits-its-direct-pi0-photon-line.md
 M docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md
 M projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md
 M projects/parity-pinned-defect-repair/references/corpus-repinning.md
 M projects/parity-pinned-defect-repair/references/defect-blast-radius.md
 M projects/parity-pinned-defect-repair/task-notes/README.md
 A projects/parity-pinned-defect-repair/task-notes/task-6-phi-lines.md
 M rust/src/kernels/photon_tables.rs
 M test/parity/README.md
 M test/parity/deltas.py
 M test/parity/test_delta_models.py
 M test/parity/test_parity.py
 M test/test_core_photon_tables.py
  Both new files are `git add -N`'d so the diff below walks them.

$ git diff origin/master --stat --
 CHANGELOG.md                                       |  23 +
 docs/followups/README.md                           |   1 +
 .../todo/phi-omits-its-direct-pi0-photon-line.md   | 106 ++++
 ...i-photon-lines-use-the-daughter-meson-energy.md |  70 ++-
 .../ADR-0001-corpus-repairs-are-declared-deltas.md |  13 +-
 .../references/corpus-repinning.md                 |  10 +
 .../references/defect-blast-radius.md              |  17 +-
 .../task-notes/README.md                           | 165 ++++--
 .../task-notes/task-6-phi-lines.md                 | 570 +++++++++++++++++++++
 rust/src/kernels/photon_tables.rs                  | 169 +++---
 test/parity/README.md                              |  26 +-
 test/parity/deltas.py                              | 123 ++++-
 test/parity/test_delta_models.py                   |  13 +-
 test/parity/test_parity.py                         |  47 +-
 test/test_core_photon_tables.py                    | 148 ++++--
 15 files changed, 1288 insertions(+), 213 deletions(-)
```

**Identifier sweep** —
`rg -n --hidden '<id>' projects/ docs/ README.md hazma/ test/ rust/ .claude/ .codex/`
for every name this task added, renamed or removed:

| Identifier | Live hits outside this task's notes | Status |
| --- | --- | --- |
| `the_phi_line_energies_are_the_daughter_mesons` | none | DELETED (renamed) |
| `test_the_phi_lines_sit_at_the_daughter_mesons_energy` | none | DELETED (renamed) |
| `folded_constants_match_the_shipped_object_code` | `positron_muon.rs`, `photon_muon.rs`, `neutrino_muon.rs` — three other kernels' own same-named tests | KEPT (unrelated) |
| `photon_line_energy` | `photon_tables.rs` ×6 (the fn, four constants, one doc link) | EDITED |
| `every_line_energy_is_the_photons_not_the_daughters` | `photon_tables.rs` ×2, the φ follow-up ×1 | EDITED |
| `every_folded_constant_is_the_shipped_immediate_or_its_declared_repair` | `photon_tables.rs` ×2 (the test and the module docstring) | EDITED |
| `_A1_B2` / `"A1+B2"` | `deltas.py` ×9, project README ×1 | EDITED |
| `EXPECTED_FLOORED_ARRAYS` | `test_parity.py` ×2, project README ×2 | EDITED |

The one stale hit the sweep caught was `photon_tables.rs:128`, whose
doc link still pointed at the pre-rename test name; fixed and re-run.
`projects/cython-to-rust/` hits are closed records and are excluded.

**Line-number citation sweep.**

```text
$ .venv/bin/python scripts/agents/check_doc_citations.py \
      $(git diff origin/master --name-only -- '*.md')
docs scanned: 10
in-repo citations checked: 0
external citations skipped: 6
  hazma/spectra/_photon/_omega.pyx (2)
  hazma/spectra/_photon/_phi.pyx (4)
out-of-range or ambiguous: NONE
```

The six skips are the deleted `.pyx` quotations in the φ follow-up's
"Why" section, recoverable with `git show 665aed5:<path>` as that section
says. They are untouched by this task and were checked by eye — the
lesson `[touched-doc-inherits-its-citations]` is about exactly this blind
spot.

**Forward-looking phrase sweep.**

```text
$ rg -n '(Task [0-9]+ will|will be added|still pending|today: ?stub)' \
      projects/parity-pinned-defect-repair/ hazma/
projects/parity-pinned-defect-repair/task-notes/task-4-boost-window.md:350
projects/parity-pinned-defect-repair/task-notes/task-3-closed-form-deltas.md:384
projects/parity-pinned-defect-repair/task-notes/task-10a-neutrino-pion-line.md:324
```

All three are the sweep command quoted inside a prior task's own sweep
block, not a forward-looking claim. KEPT.

**Scratch tokens.**

```text
$ git diff origin/master -- '*.py' '*.rs' \
    | grep '^+.*\(TODO\|FIXME\|breakpoint()\|import pdb\|XXX\)'
(no output)
```

**Count sweep.**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| `EXPECTED_DECLARED_ARRAYS = 98`, unchanged | `Counter(d.repair for d in deltas.DECLARED_DELTAS.values())` | `A1: 44, B4: 30, A1+B1: 6, A1+B2: 6, B5: 6, B6: 6` = 98 | OK |
| `EXPECTED_FLOORED_ARRAYS = 6` | `sum(d.relation.atol > 0 for d in deltas.DECLARED_DELTAS.values())` | 6 | OK |
| 305 positions over 6 arrays, 233 up / 72 down | before/after capture diff (Numerical impact table) | 57+1+102+3+139+3 = 305; 23+0+83+3+121+3 = 233 up | OK |
| 90 of 305 also moved by A1 | same capture against `oracle_reference.captured("A1")` | 19+0+36+1+33+1 = 90 | OK |
| `EXPECTED_REACH["B2"] = (6, 305)` unchanged | `pytest test/parity/test_delta_models.py -k B2` | passes; the model reads the stored corpus, which did not move | OK |
| full suite 2293 passed, 16 skipped | `preflight.sh` pytest row | 2293 passed, 16 skipped, 37 subtests | OK |
| `cargo test` 263 passed | `cargo test --no-default-features --features test-probes` | 263 passed | OK |
| the 1e-20 floor's headroom (11.6 and 8.6 decades) | min live value above the floor / min `tolerances.zero_floor` over the six arrays | 3.849e-09 and 3.580e-12 | OK |
| φ `pi0_a` ratios 1.978 / 1.951 vs 1.976 | `numpy.trapezoid` over the two CSVs ÷ each `BR(X → π⁰γ)` | 0.0026115/1.32e-3 = 1.9784; 0.16275/8.34e-2 = 1.9514; `2 × 0.98823` = 1.97646 | OK |

**Numerical-impact statement.** `hazma.spectra.dnde_photon_phi` and
`hazma.spectra.dnde_photon` on a final state whose φ is **in flight**
move, by design; the grids, the per-block counts and the invariant yield
are in `## Numerical impact` above. A φ at rest is not in the repair's
reach at all — the kernel's `branch` guard takes the rest-frame arm at
`E − m < DBL_EPSILON`, and that arm adds no line, so neither line energy
can enter. Measured rather than argued: `["phi", "phi"]` at
`cme = 2 M_φ` and `["phi", "eta"]` at `cme = M_φ + M_η`, the two-body
production thresholds, are bit-identical across the rebuild, while
`["phi", "phi"]` at `cme = 2.5 M_φ` moves at 98 of 201 grid points. Six
sibling photon spectra and the `["eta", "eta"]` n-body state are
bit-identical across the rebuild
(`np.geomspace(1, 5 M, 601)` at four parent energies each, and
`np.geomspace(1, 2000, 201)` at `cme = 2.5 M_φ`). The project's
`version_bump: minor` is unchanged and still correct.

**Exit Criteria → test mapping.**

| Exit criterion | Satisfied by |
| --- | --- |
| Both constants at `(M_φ² − m²)/(2 M_φ)` | `photon_tables::tests::every_folded_constant_is_the_shipped_immediate_or_its_declared_repair` (bit patterns) and `every_line_energy_is_the_photons_not_the_daughters` (values and identity) |
| The gate is positional, not a magnitude | `TestPhysics::test_each_line_sits_where_its_rest_frame_energy_declares`; confirmed necessary by `test_each_line_carries_the_photon_count_its_weight_declares[phi]` still passing on the reverted build |
| Declared delta on `spectra.photon.phi` only; ω does not move | `test_entry_point_matches_corpus` over all 41 cases, plus the bit-identical rebuild of the other six tabulated spectra |
| Composes rather than overlaps; a revert turns it red | `test_every_delta_model_is_a_roster_repair_with_its_evidence` (composite label ⇒ `Composed` with one addend per extra part) and the three-case red run under "Test validity" |
| `git diff --stat -- test/parity/data` empty | the command, run above, prints nothing |

**Task-note self-consistency.** `**Status:** Complete` matches the Tasks
table cell in `README.md`; every file named in §Files Changed appears in
the `--stat` above, and every symbol named in §Findings and §Decisions
appears in the identifier sweep.

## Handoff to Next Task

**For Task 7 (A2 — the muon photon rest-frame endpoint), which is the
next unblocked task:** nothing here constrains it. A2 is a `Reference`
against a Task 2 capture on a single case, with no overlap to compose
(`../rules.md` rule 7's carve-out, measured by Task 2), so
`task-4-boost-window.md` remains the worked precedent rather than this
note.

Two things from this task are worth carrying anyway:

- **Re-derive *which* arrays a model reaches, not only how many.** Task
  3's counts held exactly for both B1 and B2, but B2's six arrays were
  not B1's six — it reaches `near_rest.scalar_values` and neither
  `rest_plus_eps` array. A count that matches is not evidence the reach
  matches.
- **A relation may declare an absolute floor, and only a relocation has
  needed one.** If a later repair's prediction misses at a position whose
  value is many decades below its neighbors, that is representation
  rather than budget, and `deltas`'s "Absolute floors" section says what
  to declare. It is not a route around `../rules.md` rule 2: the floor is
  capped below the array's own zero floor and counted by
  `EXPECTED_FLOORED_ARRAYS`.
