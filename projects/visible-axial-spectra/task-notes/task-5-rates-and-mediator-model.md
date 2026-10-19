# Task 5: Rates and mediator model

**Date:** 2026-10-17
**Project:** visible-axial-spectra
**Status:** Complete
**Plan References:** `../PLAN.md` Task 5 (Rates and mediator model),
"Numerical impact"
**Related ADRs:** ADR-0001 to ADR-0005 (project-scoped); ADR-0004 and
ADR-0005 are new
**Depends On:** Task 3, Task 4

## Objective

Ship `AxialVectorMediator`, a `TheoryAnn` model for a spin-1 mediator with
vector and axial couplings to dark matter and axial couplings to the Standard
Model, with declared per-channel cross sections, partial and total widths,
thresholds and resonances built on the Task 1-4 spectral functions.

## Exit Criteria

From `../PLAN.md` Task 5, "Deliverable / gate":

- At zero width the master formula reproduces the spin-1 f fbar rate and an
  independent explicit-spinor evaluation with all vector and axial couplings
  nonzero (the note reached 5e-16 and 2e-14, `sec:ax-match`).
- Threshold limits of sigma and sigma v agree.
- Particle and population factors are separate.
- No fabricated zero for an unavailable channel.
- Existing public outputs stay bit-identical
  (`references/check_public_impact.py` against
  `references/public-values-baseline.json`).

## Inputs Reviewed

- Task brief from `../PLAN.md`, `task-notes/README.md`, ADR-0001 to ADR-0003,
  the Task 1-4 notes, `docs/agents/lessons.md` and `docs/agents/environment.md`.
- Mediator note `version3/note_v3/sec_ax.tex` (`sec:ax-match`, `eq:ax-master`,
  `eq:ax-gg`) and `version3/ax_xsec.py` (`lepton_spectral`, transcribed into
  the test as the closed-form fermion densities).
- `hazma/theory/__init__.py` (`TheoryAnn`), `hazma/vector_mediator/__init__.py`,
  `hazma/form_factors/axial/` (`_poles.py`, `_three_pi.py`, `_sources.py`).

## Findings

- `TheoryAnn` declares its abstract methods through `__metaclass__`, which
  Python 3 ignores, so nothing enforces them; the model implements all of
  them anyway, raising for the continuum spectra Task 6 owns.
- `hazma.limits._relic_density` and `TheoryConstrain` mutate model parameters
  with `setattr`, so the model needs settable scalar parameters that keep the
  cached total width consistent, as `VectorMediator` does.
- The existing consumers already apply the Dirac population factor
  (`f_dm` in `hazma/theory/_theory_gamma_ray_limits.py`) and the two photons
  of `"g g"` (`hazma/parameters.py`), so the rates must carry neither.
- In the isospin limit G parity keeps the octet and singlet directions out of
  three pions in both theories (the `f1` does not decay to three pions), so
  the LO provider's isovector-only rule restricts a channel, not the model.
- The hypercharge benchmark direction has octet and singlet components, so
  the model has an `eta pi pi` channel (Task 11) open above
  `m_eta + 2 m_pi0` = 817.8 MeV that it cannot compute.

## Decisions and Implementation Notes

- **Kernel name:** `rust/src/kernels/axial_rates.rs` as the plan proposed,
  behind the new PyO3 submodule `hazma._core.axial_vector_mediator`
  (`rust/src/axial_vector_mediator.rs`, following `rust/src/axial.rs`).
- **`sigma v` is primary and `sigma` derived** (ADR-0005). The kernel
  factors `s - 4m²` and `s - M²` so neither cancels near its zero, and a
  final state with no transverse density skips the resonant term, so
  `"g g"` stays finite on a zero-width pole.
- **One theory, declared coverage** (ADR-0004): `theory` is a required
  keyword; `rcht` raises `NotImplementedError`; three pions read the T3
  component of the card; rates past a provider's range raise; totals and the
  width raise while `v v` or `eta pi pi` is open; the mixing guard
  binds the axial dark-matter coupling only; a channel with identically
  zero couplings is an exact zero that is neither range-checked nor listed
  in `annihilation_thresholds`, and the longitudinal-only `g g` and
  `pi0 pi0 pi0` vanish with `gaxx = 0`, taking the meson poles out of
  `annihilation_resonances`; a rate at an exact pole of zero width
  raises; branching fractions come from `sigma v`.
- **Parameters:** `mx`, `mv`, `gvxx`, `gaxx` positional; `quarks`
  (`AxialCurrent`), `leptons` (new `LeptonCouplings`, nine required
  couplings, neutrinos left-handed), `gacc` (charm triangle only), `theory`
  and `width` keyword-only. The mediator has no vector quark coupling;
  Task 11 adds the vector hadronic channels. Setters change a parameter and
  the width together or neither.
- **Pion mass:** `pi pi pi0` opens at three neutral pion masses because the
  LO action uses one common pion mass (ADR-0001); the class docstring and
  the Sphinx page say so, and Task 11's physical-mass work owns the fix.
- **Partial widths:** fermion pairs, three pions and `"x x"`; `"g g"` is
  absent (Landau-Yang). Resonances are the mediator plus, unless `"g g"`
  vanishes, each meson pole with a nonzero source coupling, where `"g g"`
  peaks at `e_cm = m_P`.

## Files Changed

- `rust/src/kernels/axial_rates.rs` — master formula, Moller velocity,
  `sigma` from `sigma v`, fermion-pair densities, 9 unit tests; the resonant
  term is skipped when its numerator is zero.
- `rust/src/axial_vector_mediator.rs`, `rust/src/lib.rs`,
  `rust/src/kernels.rs` — PyO3 boundary, registration and crate docs.
- `hazma/axial_vector_mediator/` (`__init__.py`, `_couplings.py`,
  `_model.py`) —
  `AxialVectorMediator` and `LeptonCouplings`; exact-zero channels, the
  zero-width pole error, the conditional mixing guard and `sigma v`
  branching fractions.
- `test/axial_vector_mediator/spinor_oracle.py`,
  `test/axial_vector_mediator/test_axial_vector_mediator.py` — explicit-spinor
  oracle (its flux comment corrected) and 40 tests.
- `test/parity/cases.py`, `test/parity/README.md` — the new submodule joins
  `_CORE_NEW_PHYSICS_MODULES`.
- `../adrs/ADR-0004-one-hadronic-theory-and-declared-channel-coverage.md`,
  `../adrs/ADR-0005-rates-per-pair-in-sigma-v-moller.md`, and their links in
  the `../PLAN.md` Orientation list.
- `docs/source/axial_vector_mediator.rst`, `docs/source/models.rst` — the
  Sphinx page and its toctree entry, as Tasks 1, 2 and 4 did for
  `form_factors.rst`.
- `CHANGELOG.md` (Unreleased, Added) — the model entry, which the fixes
  leave true; the version bump stays with Task 16.
- `.github/workflows/ci.yml`, `.github/workflows/release.yml` — the served
  submodule set in the probe-free import check names
  `axial_vector_mediator`; CI's import smoke test failed without it.
- `AGENTS.md` — `axial_vector_mediator/` in the Layout tree and the Models
  layer. Its count of six `test-probes` probe submodules is about probes,
  not domains, and stays.
- `hazma/_core.pyi` — the comment naming the non-Cython submodules now
  names `axial_vector_mediator` as the seventh.

## Verification

- Exit criterion 1: the explicit-spinor oracle, with all four couplings
  nonzero, agrees with the model's `e e`, `mu mu`, `tau tau` and `vm vm`
  rates to at worst 4.4e-15 at zero width and 3.1e-15 at `Gamma = 30` MeV,
  over 40 energies per channel and width (test tolerance 1e-12). The
  unsplit zero-width Proca expression agrees to 1e-13 for all six fermion
  pairs, and `"g g"` matches note `eq:ax-gg` to 1e-12.
- Exit criterion 2: `sigma v_Moller` from `sigma` reproduces `sigma v` to
  1e-14 at `v ~ 3e-6` and `3e-3` for `e e`, `mu mu`, `pi pi pi0` and `g g`,
  and `sigma v` is continuous at `2 mx` to 1e-9.
- Exit criterion 3: rates carry no population or multiplicity factor; the
  `"g g"` line's branching fraction is its share of the per-pair rate.
- Exit criterion 4: `rcht`, three pions past 600 MeV, a width past that
  range, open `v v` and `eta pi pi`, the mixing guard and the continuum
  spectra each raise, under a dedicated test.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes axial_rates`: 9 passed.
- `.venv/bin/python -m pytest -n 0 -q test/axial_vector_mediator`:
  `40 passed in 1.19s`.
- `scripts/agents/preflight.sh --paths "hazma/axial_vector_mediator
  test/axial_vector_mediator test/parity/cases.py" --md "<the five touched
  Markdown files>"`: black, isort, ruff (0 new), cargo fmt, clippy and test,
  pytest (`2920 passed, 16 skipped, 20 warnings, 127 subtests passed in
  53.52s`), import and forbidden tokens PASS; markdownlint over the five
  touched Markdown files PASS. Output: `preflight-task-5.txt`.
- Performance on 10^4 points, including the mixing guard: `mu mu`
  0.016 us, `pi pi pi0` 0.29 us and `g g` 0.081 us per point, against
  877 us per point for the explicit-spinor oracle. One scalar
  `annihilation_cross_sections` call costs about 296 us, against 1.5 us
  for `KineticMixing`, because every channel re-evaluates the mixing guard
  and rebuilds card objects; a thermal average stays well under a second.

## Numerical impact

Additive: no existing public function changed. `.venv/bin/python
projects/visible-axial-spectra/references/check_public_impact.py`, with hazma
imported from this worktree, reports 164 existing output samples and
`"maximum_absolute_difference": 0.0`, "unchanged on the recorded grid".

## Open Questions

- Per-call overhead of about 300 us for one scalar total
  (`annihilation_cross_sections`) comes from per-channel mixing-guard
  evaluation and rebuilt card objects. It affects time only; caching them
  is left to a later task if a consumer needs it.
- None that block the exit criteria. A model with `mv < mx` has no total
  rate until Task 14 supplies `v v`, as ADR-0004 records.

## Plan Impact

**Impact Level:** ADR-0004, ADR-0005

ADR-0004 and ADR-0005 record the two decisions the plan anticipated under
"Anticipated ADRs" for Task 5. `PLAN.md` gains only their Orientation links;
the Task 5 text and gate, and the flat plan's other tasks, are unchanged.
The canonical-contract diff found every gate sentence of `PLAN.md` Task 5
and ADR-0001 to ADR-0005 true of the shipped code.

## Review

- N1 (accepted, fixed): a zero-coupling `tau tau` listed as a threshold
  pushed the thermal window past a range-checked zero `g g`; exact-zero
  channels are now neither range-checked nor thresholds.
- N2 (accepted, fixed): `width=0.0` at an exact pole gave `inf` or NaN; the
  kernel skips a zero-numerator resonant term and Python raises on a
  non-finite rate.
- N3 (accepted, fixed): the mixing guard rejected rates mixing cannot
  affect; it now runs only when `gaxx != 0`.
- G3 (accepted, fixed): inherited branching fractions were NaN at `2 mx`;
  they now come from `sigma v`.
- G1/N6 (accepted, documented): `pi pi pi0` opens at three neutral pion
  masses by ADR-0001's common pion mass; Task 11 owns the fix.
- G2 (accepted, fixed): docs now name both `eta pi pi` and `eta pi0 pi0`.
- N5 (accepted, fixed): the oracle comment stated the wrong flux.
- G4/N4 (rejected for this PR): per-call overhead costs time only, with no
  correctness effect; recorded under Open Questions.
- Codex C1 (accepted, fixed): meson poles stayed in
  `annihilation_resonances` with `gaxx = 0`, where `g g` vanishes, and
  pushed the thermal window to the eta' pole past the three-pion range.
- Codex C2 (accepted, fixed): the LO `pi0 pi0 pi0` density is purely
  longitudinal (`A_T` measured 0.0 on 410-600 MeV), so it joins `g g` in
  `_LONGITUDINAL_ONLY` and vanishes, unchecked, with `gaxx = 0`.

## Stale-state sweep

This block is folded by hand: rows collapse many `rg` hits and carry
dispositions. Commands ran against the working tree with the new files
added by `git add -N`.

- **Identifier sweep.** `rg -n --hidden
  'AxialVectorMediator|LeptonCouplings|axial_vector_mediator|axial_rates'
  projects/visible-axial-spectra/ docs/ README.md hazma/ test/ .claude/
  .codex/ -c` found the model's own files, `docs/source/`, `hazma/_core.pyi`,
  `PLAN.md`, the ADRs, `task-notes/README.md`, this note, `test/parity/` and
  three older hits: `hazma/experimental/axial_vector_mediator/` (KEPT, a
  different import path that is excluded from lint and not public),
  `references/scope-and-acceptance.md:94` (KEPT, it names that experimental
  package as broken) and `task-3-rust-port.md:233` (KEPT, a record written
  before Task 5). The utils-public-surface follow-up in
  `docs/followups/todo/` also names the experimental package (KEPT).
- **Submodule roster sweep.** `rg -n --hidden '\bsixth\b|\bseventh\b'
  hazma/_core.pyi rust/src/lib.rs` returns `_core.pyi:14-15` and
  `lib.rs:9`, both EDITED to name `axial_vector_mediator` as the seventh;
  `dir(hazma._core)` lists seven domains (`axial`, `axial_vector_mediator`,
  `neutrino`, `photon`, `positron`, `scalar_mediator`, `vector_mediator`)
  plus the probes. `AGENTS.md:118` and `docs/agents/environment.md:71` count
  `test-probes` probe submodules, which Task 5 did not change (KEPT).
- **Line-number citation sweep.** `rg -n
  'axial_vector_mediator\.py:[0-9]+|_model\.py:[0-9]+|axial_rates\.rs:[0-9]+'
  projects/visible-axial-spectra/ docs/` returned no hits.
  `scripts/agents/check_doc_citations.py` over the nine touched Markdown
  documents reported `in-repo citations checked: 0` and `out-of-range or
  ambiguous: NONE`.
- **Forward-looking phrase sweep.** The Task 4 regex over the project and
  `hazma/axial_vector_mediator` hits only the `In Progress` status of
  `PLAN.md:2` and `task-notes/README.md:3` (KEPT, true) and recorded sweep
  files.
- **Count sweep.**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| Verification, 40 model tests | `.venv/bin/python -m pytest -n 0 -q test/axial_vector_mediator` | `40 passed in 1.19s` | KEPT |
| Verification, 9 kernel tests | `grep -c '#\[test\]' rust/src/kernels/axial_rates.rs` | 9 | KEPT |
| `lib.rs` "seven per-domain submodules" | `dir(hazma._core)` | seven | KEPT |
| `AGENTS.md` six probe submodules | `AGENTS.md:118` vs `environment.md:71` | six probes | KEPT |

- **Numerical-impact statement.** Additive: the diff adds a package and a
  `_core` submodule and changes no existing function. `.venv/bin/python
  projects/visible-axial-spectra/references/check_public_impact.py` with
  hazma imported from this worktree (`hazma/__init__.py` and
  `hazma/_core.abi3.so` inside it) reports `"maximum_absolute_difference":
  0.0` and "unchanged on the recorded grid". No public value changes.
- **Exit Criteria to test mapping.**

| Exit criterion | Satisfied by |
| --- | --- |
| Master formula against Proca and explicit spinors | `test_fermion_pairs_match_explicit_spinors` (with `spinor_oracle.py`), `test_zero_width_matches_the_unsplit_spin_one_rate`, `test_gamma_gamma_is_the_note_line_rate` |
| Threshold limits of sigma and sigma v agree | `test_threshold_limits_of_sigma_and_sigma_v_agree` |
| Particle and population factors separate | `test_rates_are_per_pair_and_lines_carry_no_multiplicity` |
| No fabricated zero | `test_rcht_is_declared_but_not_implemented`, `test_three_pions_past_their_support_raise`, `test_totals_refuse_an_open_mediator_pair`, `test_totals_refuse_eta_pi_pi_for_octet_and_singlet_sources`, `test_rates_reject_points_past_the_mixing_guard`, `test_continuum_spectra_are_not_fabricated` |
| Existing outputs bit-identical | `check_public_impact.py`, difference 0.0 |

- **Task-note self-consistency.** `**Status:** Complete` agrees with the
  Tasks row in `task-notes/README.md` (Complete). Every file named under
  Files Changed appears in `git diff --stat origin/master --` with
  `git add -N` applied. No exit criterion is unchecked.

## Handoff to Next Task

- Read ADR-0004 and ADR-0005, then
  `hazma/axial_vector_mediator/_model.py` (`AxialVectorMediator`).
- Task 6 implements `_spectrum_funcs` and `_positron_spectrum_funcs`, which
  raise `NotImplementedError` now, and applies particle multiplicities.
- Tasks 11 and 14 remove their final states from
  `AxialVectorMediator._unmodeled_thresholds` and add their channels to
  `list_annihilation_final_states`; Task 11 also adds vector quark couplings.
- `_core` submodule `hazma._core.axial_vector_mediator` is exempt from the
  parity corpus through `test/parity/cases.py::_CORE_NEW_PHYSICS_MODULES`.
