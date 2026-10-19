# Task 6: Rest-frame stable-particle yields

**Date:** 2026-10-18
**Project:** visible-axial-spectra
**Status:** In Progress
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
- **Performance (open).** In release builds a pion `dnde` costs about 15 µs
  for e⁺ and 70 to 230 µs for neutrinos. The neutrino densities have a layer
  of width `r = (m_e/m_μ)²` at `q = 1 − x → r`, which both nested adaptive
  levels must resolve. The cost is the same with `MUON_EPSREL` 1e-10 or 1e-12
  and `PION_EPSREL` 1e-8 or 1e-10. At `e_cm = 500` MeV the timings are:
  `rest_frame_yields` over 40 energies takes 0.06 s for photons, 0.12 s for
  e⁺ and 3.9 s for neutrinos; `rest_frame_counts` over 300 bins takes 2.2 s,
  16 s and 35 s for e⁺, photons and neutrinos.

## Decisions and Implementation Notes

- **Mass hybrid (user decision).** The LO currents produce every pion at
  `neutral_pion_mass`. A charged pion decays with its physical mass at its
  production Lorentz factor, so rest maps to rest. The energy budget is
  then off by `γ(m_π± − m_π⁰)` per charged pion.
- **Deferred (user decision).** The `π⁰ → γe⁺e⁻` Dalitz e±, which is the
  only e± source in `3π⁰`, and the polarized radiative-muon photon
  correction are both deferred. Photons use Hazma's unpolarized
  `photon_muon` and `photon_pion` kernels. Two `docs/followups/todo/`
  files still need to be written.
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

## Files Changed

- `rust/src/kernels/polarized_muon.rs` (new): polarized muon and pion decay.
- `rust/src/kernels/axial_yields.rs` (new): three-pion parents and channel
  assembly.
- `rust/src/kernels/axial_rates.rs`: `FermionPair::helicity_density`.
- `rust/src/kernels.rs`: registers both modules. The module-doc paragraph
  listing project kernels is not yet updated.
- `rust/src/axial_vector_mediator.rs`: `fermion_pair_helicity_density`,
  `muon_pair_yields` and `three_pion_yields`.
- `hazma/axial_vector_mediator/_model.py`: the yield methods.
- `test/axial_vector_mediator/test_axial_vector_mediator.py`: replaces the
  test that expected `NotImplementedError` from `spectra()` with yield
  tests: multiplicities, the e± = 2e⁺ identity, the bin-width round-trip,
  per-flavor shapes and lines, the last-edge line, `spectra()` weighting,
  the tau refusal and input validation.

## Verification

Done so far:

- `cargo test --manifest-path rust/Cargo.toml --no-default-features
  --features test-probes -- polarized_muon axial_yields axial_rates`:
  29 passed. These tests cover the note's 36.94069288635647 MeV (1e-11) and
  38.430221 MeV (1e-8), the neutrino rest means, the Michel normalization,
  bin round-trips, the unpolarized limit against `positron_pion` (2e-4), the
  parent weights against `ThreePion::spectral_functions` (1e-10), and the
  channel multiplicities.
- `cargo clippy ... -D warnings` and `cargo fmt` are clean. `ruff`, `black`
  and `isort` are clean on `_model.py`.
- A smoke run gave the expected counts per annihilation: e⁺ 1 for `mu mu` and
  `pi pi pi0`; ν `[2, 2, 0]` for `mu mu` and `[2, 4 BR(μν), 0]` for
  `pi pi pi0`; π⁰ photons `6 BR(γγ)` for `3π⁰`.

- `pytest -n 0 -q test/axial_vector_mediator`: 46 passed.
- `scripts/agents/preflight.sh --paths "hazma/axial_vector_mediator
  test/axial_vector_mediator" --md "<task-6 note, task-notes README>"`:
  `RESULT: PASS`, pytest `2926 passed, 16 skipped, 20 warnings, 127 subtests
  passed`.
- `references/check_public_impact.py`: `"maximum_absolute_difference": 0.0`,
  `"status": "unchanged on the recorded grid"`.

Not done:

- A helicity-spinor oracle for `P` at finite mass, and a Python pin of the
  Hazma-mass mean e⁺ energy per π⁺.
- The two followup files.

## Open Questions

- Neutrino performance; see Findings. The proposed fix is a closed-form
  muon-level integral for massless daughters. For them the integrands
  `a/x`, `b/x` and `b/x²` are Laurent polynomials in `q = 1 − x` with powers
  −3 to 3: ν̄_e has `(q − r)²/q = q − 2r + r²/q`, and ν_μ has
  `K A = 2x²(1 − 3r²/q² + 2r³/q³)/F` and
  `K B = x²(q − 3r + 3r²/q − r³/q²)/F`. Their antiderivatives are
  polynomials plus `ln q`. Keep the adaptive form as the test oracle, and
  watch for cancellation in narrow windows (small β). A nonzero
  `epsabs = 1e-13` on the dimensionless muon-level integral alone cut ν_e
  3× but left ν_μ slow.

## Plan Impact

**Impact Level:** Follow-up in learnings. `axial_yields` is a module the
plan's table did not name, and two follow-ups are deferred by user decision.

## Handoff to Next Task

Resume Task 6 in worktree
`.claude/worktrees/visible-axial-spectra/task-6-rest-frame-yields` (branch
`claude/visible-axial-spectra/task-6-rest-frame-yields`). Nothing is
committed. `.venv` is built; rebuild with
`uv pip install --python .venv/bin/python -e . --config-setting
build-args="--features test-probes"` after any `.rs` edit. Remaining work,
in order:

1. Decide on the neutrino performance fix (Open Questions) and record the
   timings.
2. Finish the Python tests. The basic yield tests exist; still needed are
   the mean e⁺ energy per π⁺ with Hazma masses, 36.94069258356754 MeV
   (closed form above), and a helicity-spinor oracle for `P` at finite
   muon mass that extends `spinor_oracle.py`.
3. Write two `docs/followups/todo/` files: the π⁰ Dalitz e± and the
   polarized radiative-muon photons.
4. Update the `rust/src/kernels.rs` module docs; `CHANGELOG.md` and
   `docs/source` per the doc-consistency pass.
5. Rerun `references/check_public_impact.py` and preflight on the final
   tree, then the remaining solo-task Steps 6 to 9.
