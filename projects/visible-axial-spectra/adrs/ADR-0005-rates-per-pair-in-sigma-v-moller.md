# ADR 0005 Compute rates as sigma v_Moller per pair and derive sigma from them

**Date:** 2026-10-17
**Status:** Accepted
**Scope:** Project-scoped within visible-axial-spectra.

## Context

The note's master formula (`eq:ax-master`) gives `sigma v` directly, finite
at threshold for an s-wave, while `TheoryAnn` consumers read the cross
section `sigma` in `MeV^-2`, which diverges there as `1/v`. Two factors are
easy to fold into either by accident: the dark-matter population factor of a
Dirac flux, which `hazma.theory._theory_gamma_ray_limits` applies as `f_dm`,
and the multiplicity of a particle in its final state, such as the two
photons of `"g g"`, which `hazma.parameters` and `hazma.theory._theory_cmb`
apply when they read a line.

## Decision

- **`sigma v` is primary.** Every channel computes `sigma v_Moller` per
  `chi chibar` pair in `MeV^-2` from its densities, in
  `rust/src/kernels/axial_rates.rs`. The cross section is derived once,
  `sigma = sigma v/v_Moller` with the centre-of-mass
  `v_Moller = 2 sqrt(1 - 4 m^2/s)`, so the two cannot disagree at
  threshold. `sigma` is zero below `2 m`, and at exactly `2 m` it is
  infinite for a nonzero s-wave and zero for a zero rate.
- **No population or particle factor in a rate.** Rates and cross sections
  are per pair and per final state. The Dirac population factor and the
  per-final-state particle multiplicities belong to the consumers, which
  already apply them; a line's branching fraction is its channel's share of
  the rate, computed from `sigma v` so that it stays finite at exactly
  `2 m`, where `sigma` is infinite.

## Consequences

- **Positive:** Within the channels' declared ranges, the model plugs into
  the existing thermal-average, limit and CMB code unchanged, and Tasks 6-9
  add multiplicities and population factors in one declared place each.
- **Negative:** A caller who wants `sigma v` from `sigma` must multiply by
  `v_Moller` itself.
- **Mitigation:** The model exposes `annihilation_sigma_v_funcs` and
  `annihilation_sigma_v` beside the cross-section methods.
