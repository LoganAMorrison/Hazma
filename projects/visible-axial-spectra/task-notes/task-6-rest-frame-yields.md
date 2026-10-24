# Task 6: Rest-frame stable-particle yields

**Date:** 2026-10-23 (begun 2026-10-18; PR #130 shipped the first part)
**Project:** visible-axial-spectra
**Status:** Complete
**Plan References:** `../PLAN.md` "Task 6 Rest frame stable particle yields",
"Implementation layering", "Numerical impact"
**Related ADRs:** ADR-0002, ADR-0004, ADR-0005 (project-scoped)
**Depends On:** Task 5

## Objective

Convolve signed parent histories with decay kernels to give rest-frame
photon, positron and neutrino yields for every final state of
`AxialVectorMediator`, with the polarized-muon kernel in `π → μν` and in
chiral `μ⁺μ⁻` chains.

## Exit Criteria

- Michel reference multiplicities: one charged lepton per muon and per
  charged pion, and the per-flavor neutrino counts of each parent.
- The public positron output counts e⁺ only, the charge-summed e± injection
  is a distinct result, and the two satisfy the CP identity e± = 2 e⁺ for
  the CP-symmetric final states.
- Neutrinos are ν + ν̄ per flavor with shape (3, N).
- The mean e⁺ energy per π⁺ at rest is 36.94069288635647 MeV with the muon
  spin and 38.430221 MeV without it (note `sec:k22`, research masses).
- Count bins and differential spectra round-trip by bin width.
- Existing public outputs stay bit-identical
  (`references/check_public_impact.py`).

## Inputs Reviewed

- `../PLAN.md` Task 6 and "Implementation layering" sections.
- Research reference `version22/kernels/charged_pion.py`,
  `version22/validation/phase_space_reference.py` (read through a summary).
- `rust/src/kernels/axial_three_pi.rs`, `hazma/axial_vector_mediator/_model.py`,
  `hazma/theory/__init__.py` spectrum and line methods.

## Findings

- The research reference is `version22/kernels/charged_pion.py` (research
  repository HEAD `d0ba4f9`). Its 36.94069288635647 MeV is a closed form with
  PDG 2022 masses (`m_μ = 105.6583755`, `m_e = 0.51099895`) and branching
  fractions 0.999877 and 0.000123, with `π → eν` included. With
  `hazma.parameters`' masses the same mean is 36.94069258356754 MeV, and
  without the spin it is 38.43022075435374 MeV (note masses: 38.43022101929301).
  The closed form uses `J0 = [(1 + r)(1 − r) + 2r ln r]/2`,
  `J1 = (1 − r)³/3` and
  `J_n = [(1 + r)^(n−1)(1 − r)³ + 4(n − 1)r J_(n−2)]/(n + 2)`. It sets
  `I1 = 3(1 + r)J2 − 2J3 − 4rJ1` and
  `I2 = (1 + 3r)(J2 − 4rJ0) − 2(J3 − 4rJ1)`, and then
  `⟨E⟩ = BR_μ[E_μ I1 + p_μ I2/3]/F + BR_e(m_π² + m_e²)/(2m_π)`. Dropping
  the `p_μ I2/3` term turns off the spin.
- Both LO three-pion weights depend on the π⁰ energy alone: transverse
  `∝ |p₀|²` and longitudinal `∝ (s − m²)²`, read off
  `ThreePion::current`. So the π⁰ density is `M(E₀) L(E₀)` and each charged
  pion's is a closed-form primitive of `M` over the Dalitz range.
- Existing models' `"mu mu"` positron spectra are one muon's kernel, so they
  count e⁺ only. That matches this task's public positron convention.
- Hazma's legacy `dnde_photon_neutral_pion` rounds its box to `f32`, so the
  new kernel builds its own double-precision `γγ` box.
- **Performance (closed).** In release builds a pion `dnde` cost about
  15 µs for e⁺ and 70 to 230 µs for neutrinos, because the neutrino
  densities have a layer of width `r = (m_e/m_μ)²` at `q = 1 − x → r` that
  both nested adaptive levels had to resolve; `MUON_EPSREL` 1e-10 or 1e-12
  and `PION_EPSREL` 1e-8 or 1e-10 made no difference. At `e_cm = 500` MeV,
  `rest_frame_yields` over 40 energies took 0.06 s for photons, 0.14 s for
  e⁺ and 2.36 s for neutrinos, and `rest_frame_counts` over 300 bins 17.7 s,
  2.5 s and 24.6 s. With the closed-form muon-level neutrino integrals
  (Decisions) the neutrino figures are 0.22 s and 2.5 s, 11× and 10×
  faster, and the neutrino yields on that grid agree with the quadrature
  build to 8e-15 (densities) and 8e-14 (counts) relative. Photon counts,
  which fold Hazma's `photon_muon` and `photon_pion` kernels bin by bin,
  remain the slowest path at 17.6 s and are outside this task.
- **Closed-form precision.** A first version expanded the whole integrand
  in powers of `q`. It lost 3.5e-9 relative for a muon with `β = 1e-3` at
  `x ≈ 0.01`, where `x²` is reconstructed from cancelling powers of
  `q ≈ 1`, and with the polynomial part in `x` instead it lost 2.8e-8 near
  the endpoint, where the angular factors `x cos θ = β' + αx` have `α` and
  `β'` of order `1/β` whose squares carried coefficients near 1e7
  cancelling to 1e-4. Expanding the polynomial part about each
  subinterval's midpoint keeps every coefficient of the order of the
  values it describes. The closed forms then agree with a 40-digit
  `mpmath` integration of the same densities to 1e-10 or better in eight
  pinned cases, and for a bin of relative width 1e-3 from a muon at
  2000 MeV they are the accurate side: 3e-11 against the quadrature's
  4e-10, since the quadrature forms each edge's `cos θ` before
  differencing them.
- **Pole interval length (PR #131 review, closed).** `Laurent::integral`
  passed `1 − x_to` and `1 − x_from` to `power_integral`, which formed the
  interval length as their difference. That length carries the absolute
  rounding error of two numbers near 1 instead of one relative to the
  window, and after `add_pole_times_power` the pole coefficients are of
  order `r²Δα² ~ r²/β²`, which amplifies it. Against `counts_quadrature`,
  for bins of relative width 1e-6, the reviewer measured a worst relative
  error of 5.6e-3 at `γβ = 1e-3`, 88 at `1e-5` and 5.3e5 at `1e-7`, with
  negative counts, and a `dnde` error of 1.5e-4 at `1e-7`; a 40-digit
  `mpmath` integral agreed with the quadrature (ν_μ, `γβ = 1e-3`,
  P = 1, bin [5.505365311236440e-3, 5.510873430607361e-3] MeV: `mpmath`
  6.798856735042118e-15, the old closed form 6.7987380054813551e-15, a
  relative error of 1.7e-5). `power_integral` now takes the length
  `x_to − x_from` from its caller, which is exact for the polynomial part
  (`u_to − u_from`) and for the poles (`q_to − q_from`), and uses the
  `1 − x` values only for the ratio in `ln_1p` and for the products. Over
  eight energies from 5e-3 to 140 MeV, both neutrinos and
  P ∈ {−1, 0.4, 1}, the worst counts error against `counts_quadrature` for
  bins of relative width 1e-6 went from 1.7e-2, 9.0 and 3.1e5 to 6.6e-11,
  5.6e-8 and 9.2e-5 at `γβ = 1e-3`, `1e-5` and `1e-7` (none of the 48
  counts per `γβ` is negative now, where 0, 3 and 6 were), and the worst
  `dnde` error at `γβ = 1e-7` from 5.0e-6 to 1.7e-9. At rest, ν_e counts
  on [E, 2E] agree with the quadrature to 1.4e-10, 5.9e-13 and 7.5e-15 for
  E = 1e-6, 1e-5 and 1e-4 MeV, where they were off by 2.6e-3, 2.5e-6 and
  2.0e-9 (the
  generalist's absolute floor near 1e-25 from the same cause). With the
  fix, the remaining error scales as `1/(βw)` for a bin of relative width
  `w`: at `γβ = 1e-7` it is 9.2e-5, 8.9e-7 and 1.2e-8 for widths 1e-6, 1e-4
  and 1e-2. The 5e-3 MeV energy is now in
  `assert_closed_forms_match_quadrature`, and a pinned `mpmath` bin
  covers it; with the old length they fail at 4.5e-8 (the slow-muon
  comparison, against a 1e-9 tolerance) and 1.7e-5 (the pin, against 1e-10).
- **Bin sums at threshold.** `rest_frame_counts` for `pi pi pi0` at
  `3m_π⁰ + 1e-6` MeV sums to one e⁺ within 1e-12 over 353 bins of 0.2 MeV
  but to `1 − 5.4e-8` over 704 bins of 0.1 MeV. The loss is a few times
  the fold tolerance `FOLD_EPSREL = 1e-8` and is not understood; see Open
  Questions.

## Decisions and Implementation Notes

- **Mass hybrid (user decision).** The LO currents produce every pion at
  `neutral_pion_mass`. A charged pion decays with its physical mass at its
  production Lorentz factor, so rest maps to rest. The energy budget is
  then off by `γ(m_π± − m_π⁰)` per charged pion.
- **Deferred (user decision).** The `π⁰ → γe⁺e⁻` Dalitz e±, which is the
  only e± source in `3π⁰`, and the polarized radiative-muon photon
  correction are both deferred. Photons use Hazma's unpolarized
  `photon_muon` and `photon_pion` kernels. They are tracked in
  `docs/followups/todo/axial-mediator-pi0-dalitz-positrons.md` and
  `docs/followups/todo/axial-mediator-polarized-radiative-muon-photons.md`.
- **Muon polarization.** The effective polarization `P` is the μ⁻ helicity,
  which equals minus the μ⁺ helicity. It equals `σv_h/σv`, with
  `σv_h = sigma_v(A_h, 0)` and
  `A_h = FermionPair::helicity_density = N β² s g_V g_A/(3π)`. The
  longitudinal contact is unpolarized. For `π → μν`, `P = 1`.
- **Modules.** `rust/src/kernels/polarized_muon.rs` holds `MuonDecay`,
  `PionDecay` and the boost helpers; the masses are fields so the note's
  values are reproducible. `rust/src/kernels/axial_yields.rs` holds the
  three-pion parent densities, the `Probe::{Density, Bin}` responses and
  `Channel::{MuonPair, ThreePion}`. The plan's table named only
  `polarized_muon`; `axial_yields` is the added module.
- **Public API.** `rest_frame_yields(energies, e_cm, *, product)`,
  `rest_frame_counts(edges, e_cm, *, product)` and
  `rest_frame_lines(e_cm, *, product)` cover the products `"photon"`,
  `"positron"` (e⁺ only), `"electron_positron"` (e⁺ + e⁻) and `"neutrino"`
  (shape (3, N)). Each returns yields per annihilation into each open final
  state. `_spectrum_funcs` and `_positron_spectrum_funcs` now delegate to
  `_final_state_yield`.
- **Lines.** The primary lines (`g g`, `e e`, `ν ν`) are declared in Python
  (`_line_multiplicity`) and binned with `searchsorted`; a line exactly at
  the last edge goes in the last bin. Rust returns continua only.
- **Refusal.** `"tau tau"` yields raise `ValueError` when it is open, since
  this model has no tau kernels.
- **Open states.** A final state is open when its own `σv` is positive.
  The yields are per channel, so they do not need the total rate, which
  raises while an unmodeled state such as `"v v"` is open.
- **Count window.** For a massive daughter, a bin edge `low` between `m`
  and `γm` is reached by every box. The window starts at the lower root of
  `boost_range` only above `γm` (`bin_window`). Before this fix, log-spaced
  e⁺ bins from `m_e` lost 1e-7 of a muon's e⁺.
  `bins_starting_below_the_boosted_rest_energy_tile_the_spectrum` fails at
  1.04e-7 with the old rule restored.
- **Threshold stability (PR #130 review).** `ThreePionParents` works in
  kinetic energy `u = E − m` and the excess `q − 3m`. It factors the
  charged pions' cubic primitives through the partner-range length `L`.
  The total-energy form cancelled near `3m`, giving 1.0126 e⁺ per
  annihilation at `1e-6` MeV above threshold and 2.3e5 at `1e-9` MeV. Now
  the deviation from one is −3.9e-11 at `1e-9` MeV, −1.6e-13 at `1e-6` MeV
  and −2.0e-14 at 95 MeV.

- **Neutrino closed forms.** `MuonDecay::dnde` and `counts` take the
  closed form for the two neutrinos and keep the adaptive quadrature for
  the massive charged lepton; `dnde_quadrature` and `counts_quadrature`
  stay as the tests' oracle. The form is the private `Laurent` type: poles
  `q⁻³` to `q⁻¹` integrated in `q`, and a cubic in `u = x − x₀` integrated
  about the interval's midpoint, with each power difference factored
  through the interval length `x_to − x_from`, which `Laurent::integral`
  forms once from the `x` edges and passes to `power_integral`.
  `CosineEdge` holds a bin edge's `cos θ = α + β'/x`, clamped or free per
  subinterval. Both forms know a window's edges only to the machine epsilon
  `ε` of the lab energies, so a bin of relative width `w` from a muon of
  speed `β` carries an error of at most about `ε/(βw)`; the module docs give
  the measured 7e-11, 6e-8 and 9e-5 at `γβ = 1e-3`, `1e-5` and `1e-7` for
  `w = 1e-6`. `MuonDecay::electron_mass` is zero or at least about 1e-5 MeV:
  below that, `r` is comparable to the machine epsilon, `x(max_energy)`
  rounds to 1, and the closed forms return infinity or NaN. Only
  `MuonDecay::PDG` is constructed outside the tests, so the contract is a
  field doc and not a check.
- **Python pins.** The Hazma-mass mean e⁺ energy per π⁺ at rest,
  36.94069258356754 MeV, is pinned in Rust at 1e-11
  (`pion_rest_mean_positron_energy_with_hazma_masses`) and through the
  public `rest_frame_counts` of `pi pi pi0` just above threshold at 1e-8,
  with 0.2 MeV bins centered on the two-body e⁺ line so its energy is
  exact. The polarization `P` is pinned at 1e-12 against
  `spinor_oracle.fermion_polarization`, which resolves the μ⁻ spinors by
  helicity along its momentum and sums the μ⁺ spins without forming the
  helicity density, and the public `mu mu` e⁺ density is checked to equal
  `muon_pair_yields` at the oracle's `P`.

## Files Changed

PR #130:

- `rust/src/kernels/polarized_muon.rs` (new): polarized muon and pion decay.
- `rust/src/kernels/axial_yields.rs` (new): three-pion parents and channel
  assembly.
- `rust/src/kernels/axial_rates.rs`: `FermionPair::helicity_density`.
- `rust/src/kernels.rs`: registers both modules.
- `rust/src/axial_vector_mediator.rs`: `fermion_pair_helicity_density`,
  `muon_pair_yields` and `three_pion_yields`.
- `hazma/axial_vector_mediator/_model.py`: the yield methods.
- `test/axial_vector_mediator/test_axial_vector_mediator.py`: replaces the
  test that expected `NotImplementedError` from `spectra()` with yield
  tests: multiplicities, the e± = 2e⁺ identity, the bin-width round-trip,
  per-flavor shapes and lines, the last-edge line, `spectra()` weighting,
  the tau refusal and input validation.

The second PR:

- `rust/src/kernels/polarized_muon.rs`: `Laurent`, `CosineEdge`,
  `Neutrino`, the closed-form `dnde_neutrino` and `counts_neutrino`, the
  module's "Closed forms for the neutrinos" section, and five tests: the
  oracle comparison from rest to `γ ≈ 19`, the slow-muon comparison, the
  massless-electron limit, the eight `mpmath` reference pins and the
  Hazma-mass pion mean. The review fix passes the interval length into
  `power_integral`, adds 5e-3 MeV to the energies
  `assert_closed_forms_match_quadrature` checks, adds the eighth pin and
  documents the `electron_mass` contract.
- `rust/src/kernels.rs`: the module-doc paragraph for `polarized_muon`
  and `axial_yields`.
- `test/axial_vector_mediator/spinor_oracle.py`: `fermion_polarization`
  and the helicity spinors behind it.
- `test/axial_vector_mediator/test_axial_vector_mediator.py`: the pion
  mean pin, whose first edge a review moved a full bin below `m_e`, and the
  polarization oracle test.
- `CHANGELOG.md`: the yields entry under `Added`, replacing the sentence
  that said the spectra raise.
- `docs/source/axial_vector_mediator.rst`: a "Yields" section.
- `docs/followups/todo/axial-mediator-pi0-dalitz-positrons.md` and
  `axial-mediator-polarized-radiative-muon-photons.md` (new), with their
  rows in `docs/followups/README.md`. The first now gives the `pi pi pi0`
  Dalitz addition as 0.0117 e⁺ per annihilation, about 1.2%.
- `docs/agents/lessons.md` and `docs/agents/lessons-examples.md`: PR #131
  joins the `closed-form-cancels-where-a-series-converges` class.
- This note and `projects/visible-axial-spectra/task-notes/README.md`.

## Verification

PR #130:

- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes -- polarized_muon axial_yields axial_rates`:
  30 passed, covering the note's 36.94069288635647 MeV (1e-11) and
  38.430221 MeV (1e-8), the neutrino rest means, the Michel normalization,
  bin round-trips, the unpolarized limit against `positron_pion` (2e-4), the
  parent weights against `ThreePion::spectral_functions` (1e-10), and the
  channel multiplicities.
- `pytest -n 0 -q test/axial_vector_mediator`: 49 passed.
- `scripts/agents/preflight.sh --paths "hazma/axial_vector_mediator
  test/axial_vector_mediator" --md "<task-6 note, task-notes README>"`:
  `RESULT: PASS`, pytest `2926 passed, 16 skipped, 20 warnings, 127 subtests
  passed`.

The second PR:

- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes -- polarized_muon axial_yields`: 25 passed
  (19 in `polarized_muon`, 6 in `axial_yields`). `cargo clippy
  --all-features --all-targets -- -D warnings` and `cargo fmt --check` are
  clean.
- `.venv/bin/python -m pytest -q test/axial_vector_mediator`: 52 passed
  in 6 s. `black`, `isort` and `ruff` are clean on the two test files and
  the model package.
- `.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py`
  after the rebuild: `"maximum_absolute_difference": 0.0`,
  `"status": "unchanged on the recorded grid"`.
- Timings and the before/after neutrino comparison are in Findings; the
  script is a scratch file, not committed, that builds the test model at
  `e_cm = 500` MeV on 40 log-spaced energies and 300 log-spaced bins from
  1 to 250 MeV.
- `scripts/agents/preflight.sh --paths "hazma/axial_vector_mediator
  test/axial_vector_mediator" --md "<CHANGELOG, followups README, the two
  followups, this note>"` on the final tree, with `.venv/bin` first on
  `PATH` (the script resolves `black`, `ruff`, `pytest` and `python` from
  `PATH`, and the worktree's tools live in `.venv`): `RESULT: PASS`, pytest
  `2932 passed, 16 skipped, 20 warnings, 127 subtests passed in 52.53s`,
  and the black, isort, ruff, cargo fmt, cargo clippy, cargo test, import,
  markdownlint and forbidden-token rows all `PASS`.
  `scripts/agents/check_doc_citations.py` over the six touched documents:
  `out-of-range or ambiguous: NONE`.

The review fixes (PR #131):

- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes -- polarized_muon axial_yields`: 25 passed, the
  same count as before, because the new `mpmath` pin is a row of an
  existing test and the 5e-3 MeV energy an element of an existing loop.
  With `power_integral` forming its own length again,
  `neutrino_bins_match_a_high_precision_reference` fails at a relative
  error of 1.7e-5 and `neutrino_closed_forms_hold_for_a_slow_muon` at
  4.5e-8. `cargo clippy --all-features --all-targets -- -D warnings` and
  `cargo fmt --check` are clean.
- `.venv/bin/python -m pytest -q -n 0 test/axial_vector_mediator`: `52
  passed in 4.56s`, after the pion-mean test's first edge moved a full bin
  below `m_e` and the rebuild with `uv pip install --python .venv/bin/python
  -e . --config-setting build-args="--features test-probes"`.
- `.venv/bin/python projects/visible-axial-spectra/references/check_public_impact.py`
  after the rebuild: `"maximum_absolute_difference": 0.0`. The 500 MeV
  neutrino comparison (`/tmp/hazma-task6-timing.py after3`, against the
  quadrature build) gives maximum relative differences of 8.2e-15 for
  yields and 7.7e-14 for counts.
- `scripts/agents/preflight.sh --paths "hazma/axial_vector_mediator
  test/axial_vector_mediator" --md "<CHANGELOG, the two lessons files, the
  Dalitz follow-up, this note>"` on the final tree with `.venv/bin` first
  on `PATH`: `RESULT: PASS` with pytest `2932 passed, 16 skipped, 20
  warnings, 127 subtests passed`, the same count as before the fixes
  because no test function was added.

## Open Questions

- The two deferred physics items are follow-ups, not open questions of this
  task: the π⁰ Dalitz e± in
  `docs/followups/todo/axial-mediator-pi0-dalitz-positrons.md` and the
  polarized radiative-muon photons in
  `docs/followups/todo/axial-mediator-polarized-radiative-muon-photons.md`.
- Over 704 bins of 0.1 MeV just above the three-pion threshold the
  `pi pi pi0` e⁺ counts sum to `1 − 5.4e-8`, while 353 bins of 0.2 MeV sum
  to one within 1e-12. A fold break point the finer grid steps over, or
  the kink of the pion-rest continuum landing inside a bin, are the
  candidates; the Python pin uses the 0.2 MeV grid. Nothing in the public
  contract depends on it, and Task 7's cascades should check their bin
  sums at threshold before trusting fine grids.
- Both the closed form and the quadrature know a boost window's edges only
  to the machine epsilon of the lab energies, so a narrow bin from a muon
  nearly at rest (a `mu mu` pair within a few parts in 1e6 of threshold)
  gets correspondingly fewer digits: about `ε/(βw)` for a bin of relative
  width `w`, 9e-5 at `γβ = 1e-7` and `w = 1e-6`. The module docs record
  this and no caller needs more.

## Plan Impact

**Impact Level:** Task note only. `axial_yields` is a module the plan's
table did not name; this note and the working-memory README record it. The
two deferred physics items are follow-ups under `docs/followups/todo/`,
not plan changes.

## Stale-state sweep

This block is folded by hand: rows collapse many `rg` hits and carry
dispositions, and each count for this note includes this block. Commands
ran on the final tree, after the two untracked
follow-ups were added with `git add -N`, and after the last prose edit.

- **Identifier sweep.** `rg -c --hidden '<identifier>'
  projects/visible-axial-spectra/ docs/ README.md hazma/ test/ .claude/
  .codex/ rust/src CHANGELOG.md` for each new name:
  - `rest_frame_yields|rest_frame_counts|rest_frame_lines` hits
    `CHANGELOG.md` (1), `docs/source/axial_vector_mediator.rst` (1),
    `hazma/axial_vector_mediator/_model.py` (12), the model test (19),
    `rust/src/kernels/axial_yields.rs` (2), the Dalitz follow-up (2) and
    this note (13), all KEPT as current. `docs/agents/lessons-examples.md:1662`
    cites `rest_frame_counts([0, 400], q, product="positron")` as a record
    of the PR #130 review (KEPT).
  - `fermion_polarization` hits `spinor_oracle.py` (1), the model test (2),
    the task-notes README (1) and this note (6), all KEPT.
  - `axial_yields|polarized_muon` hits `PLAN.md` (1, the layering table
    row, `polarized_muon` only), the task-notes README (6), this note (22),
    both follow-ups, `rust/src/kernels.rs` (3),
    `rust/src/axial_vector_mediator.rs` (2), the two kernels and one
    `lessons-examples.md` record (KEPT).
  - `Laurent|CosineEdge` hits `polarized_muon.rs` (18) and this note (7);
    the one `hazma/gamma_ray_parameters.py` hit is an author's surname in a
    reference (KEPT, unrelated).
  - `dnde_quadrature|counts_quadrature` appear only in `polarized_muon.rs`
    and this note once a `rg` over the generic `dnde_neutrino` is excluded,
    since that name is the public Hazma kernel family (KEPT).
- **Line-number citation sweep.** `rg -n
  'polarized_muon\.rs:[0-9]+|axial_yields\.rs:[0-9]+|_model\.py:[0-9]+|spinor_oracle\.py:[0-9]+|test_axial_vector_mediator\.py:[0-9]+|kernels\.rs:[0-9]+'
  projects/visible-axial-spectra/ docs/` returned no hits.
  `scripts/agents/check_doc_citations.py` over the six touched Markdown
  documents reported `docs scanned: 6`, `in-repo citations checked: 0` and
  `out-of-range or ambiguous: NONE`. (`--changed-vs origin/master` found
  nothing because the changes are uncommitted, so the paths were passed.)
- **Forward-looking phrase sweep.** `rg -n --hidden '(Task [0-9]+ will|will
  be added|still pending|today: ?stub|currently|In Progress)'` over this
  note, the task-notes README, `hazma/axial_vector_mediator`, the two
  kernels, the Sphinx page, the follow-ups and `CHANGELOG.md` hits only
  `task-notes/README.md:3` (`**Status:** In Progress`, KEPT: the project
  has Tasks 7 to 16 open). Before this sweep the Decisions bullet said the
  follow-ups "still need to be written" (EDITED to name them).
- **Docs and follow-up consistency edits made by this sweep.** This note
  said "six tests" for the second PR's Rust tests, where the diff adds
  five (EDITED). The Sphinx page called a line an `(energy, count)` pair
  where `rest_frame_lines` returns a dict with `"energy"` and
  `"multiplicity"` (EDITED). The Dalitz follow-up said
  `hazma.spectra._positron` has kernels for the muon and charged pion
  only, where it also has kaon, eta, omega, rho and eta' kernels and no
  neutral-pion function, and it cited `ThreePionParents::charged_leptons`,
  which is `Channel::charged_leptons` (both EDITED). The radiative-photon
  follow-up routed photons through `muon_response` and `pion_response`,
  which serve only e± and neutrinos, where the photons go through
  `Channel::photons` and `ThreePionParents::charged_pion_photons`
  (EDITED).
- **Review-fix sweep (PR #131).** `rg -n --hidden 'relative precision|
  neither is more accurate|the accurate side|few-percent|nearly at rest'`
  over the project, `docs/source/`, `docs/followups/`, the model package,
  the model tests, the two kernels, `rust/src/kernels.rs` and the
  CHANGELOG found the precision claim in the module docs (EDITED), the
  Dalitz follow-up's "few-percent" (EDITED), the test comment on the
  accurate side (EDITED to the 0.5 MeV case), the Open Questions bullet
  (EDITED) and one record of an earlier follow-up
  (`docs/followups/done/vector-mediator-gev-vv-positron-and-neutrino-are-zero.md`,
  KEPT). Two further hits are the Findings bullet that says the closed
  forms are the accurate side for a 1e-3 bin at 2000 MeV (KEPT, a measured
  record) and this sweep.
- **Count sweep.**

| Claim location | Command | Actual | Status |
| --- | --- | --- | --- |
| Verification, 25 Rust tests (19 + 6) | `cargo test --manifest-path rust/Cargo.toml --no-default-features --features test-probes -- polarized_muon axial_yields` | `running 25 tests`, `25 passed` (rerun after the pole-length fix) | KEPT |
| Same, 19 in `polarized_muon` and 6 in `axial_yields` | `grep -c '#\[test\]' rust/src/kernels/polarized_muon.rs rust/src/kernels/axial_yields.rs` | 19 and 6 | KEPT |
| Files Changed, "five tests" added | `git show origin/master:rust/src/kernels/polarized_muon.rs \| grep -c '#\[test\]'` | 14 on master, 19 now | EDITED from six |
| Files Changed, "eight" reference pins | the `Daughter::` rows of `neutrino_bins_match_a_high_precision_reference` | 8 | EDITED from seven |
| Verification, 52 model tests | `.venv/bin/python -m pytest -q -n 0 test/axial_vector_mediator` | `52 passed in 4.56s` (rerun after the review fixes; the note says 6 s for the earlier run) | KEPT |
| PR #130, 49 model tests | master's 49 plus two new test functions, one with two widths | 49 + 1 + 2 = 52 | KEPT |
| Verification, 2932 passed | `scripts/agents/preflight.sh` on the final tree (recorded above) | `2932 passed, 16 skipped, 20 warnings, 127 subtests passed` | KEPT |
| Findings, neutrino speedup 11x and 10x | 2.36 s / 0.22 s and 24.6 s / 2.5 s | 10.7 and 9.8 | KEPT |
| Exit criterion, 36.94069288635647 MeV | `pion_rest_mean_positron_energy_is_the_note_value` | passes at 1e-11 | KEPT |

- **Numerical-impact statement.** No existing public value changes
  (verified: `.venv/bin/python
  projects/visible-axial-spectra/references/check_public_impact.py`, with
  `hazma.__file__` and `hazma._core.__file__` inside this worktree, reports
  `"maximum_absolute_difference": 0.0` and "unchanged on the recorded
  grid", 164 values). The new neutrino yields of `AxialVectorMediator`
  agree with the previous quadrature build to 8.2e-15 relative on
  densities and 7.7e-14 on counts over 40 energies and 300 bins at
  `e_cm = 500` MeV, measured again after the pole-length fix
  (`/tmp/hazma-task6-timing.py after3`; it moved the counts by 4e-16 from
  the build before it). The fix moves the closed forms of the unreleased
  neutrino kernel for slow muons and low-energy bins toward the quadrature
  by up to the amounts in Findings, and moves no existing public value. The
  yields are new public output, not a change to an existing one.
- **Exit Criteria to test mapping.**

| Exit criterion | Satisfied by |
| --- | --- |
| Michel multiplicities: one charged lepton per muon and charged pion, per-flavor neutrino counts | Rust `channel_multiplicities`, `michel_densities_are_normalized`, `muon_counts_are_normalized_in_flight`; Python `test_rest_frame_counts_carry_each_charged_lepton`, `test_three_pion_counts_stay_normalized_at_threshold` |
| e⁺ only public, charge-summed distinct, e± = 2 e⁺ | `test_rest_frame_counts_carry_each_charged_lepton` (`rtol=1e-14`) |
| Neutrinos are ν + ν̄ per flavor with shape (3, N) | `test_neutrino_yields_are_per_flavor`, `channel_multiplicities` |
| Mean e⁺ energy per π⁺ at rest, 36.94069288635647 MeV with spin and 38.430221 MeV without | `pion_rest_mean_positron_energy_is_the_note_value`, `pion_rest_mean_positron_energy_without_spin`, `pion_rest_mean_positron_energy_with_hazma_masses`, `test_pion_rest_mean_positron_energy_carries_the_muon_spin` |
| Count bins and differential spectra round-trip by bin width | Rust `muon_counts_round_trip_by_bin_width`, `pion_counts_are_normalized_and_round_trip`, `counts_round_trip_by_bin_width`; Python `test_rest_frame_counts_round_trip_by_bin_width` |
| Polarization convention pinned | `test_muon_polarization_matches_helicity_spinors` against `spinor_oracle.fermion_polarization` |
| Closed-form neutrinos agree with their oracle | `neutrino_closed_forms_match_the_quadrature`, `neutrino_closed_forms_hold_for_a_slow_muon`, `neutrino_closed_forms_survive_the_massless_limit`, `neutrino_bins_match_a_high_precision_reference` |
| Existing public outputs bit-identical | `references/check_public_impact.py`, difference 0.0 |

- **Task-note self-consistency.** `**Status:** Complete` agrees with the
  Tasks row of `task-notes/README.md` (Complete). The Exit Criteria
  bullets map one to one onto the table above. Every file named under
  Files Changed is in `git diff --stat origin/master --` or was created by
  `git add -N`: the eleven files of the second PR are the nine listed plus
  this note and the README. Every function and type cited in Decisions
  and Findings (`Laurent`, `CosineEdge`, `dnde_neutrino`,
  `counts_neutrino`, `dnde_quadrature`, `counts_quadrature`,
  `fermion_polarization`) exists in the tree. The follow-up files exist
  under `docs/followups/todo/` with rows in `docs/followups/README.md`.

## Handoff to Next Task

Task 6 is complete with the second PR. Task 7 (polarized cascades) builds
on `rust/src/kernels/polarized_muon.rs`: `MuonDecay` and `PionDecay` take
an effective polarization `P` that is the μ⁻ helicity and minus the μ⁺
helicity, the neutrino paths are closed forms through `Laurent` and
`CosineEdge`, and the charged lepton integrates by quadrature. The
polarization convention is pinned by `spinor_oracle.fermion_polarization`.
The public contract is the "Yields" section of
`docs/source/axial_vector_mediator.rst`. Check bin sums at threshold on
any fine grid (Open Questions), and expect photon counts, not neutrinos,
to be the slow product.
