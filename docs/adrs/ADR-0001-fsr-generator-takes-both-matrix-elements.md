# ADR 0001: The FSR generator takes both matrix elements

**Date:** 2026-08-04
**Status:** Accepted (the implementing PR,
[#41](https://github.com/LoganAMorrison/Hazma/pull/41), merged
2026-08-05 — acceptance rode on its review)

## Context

ADR-0003 of `projects/cython-to-rust/` removes `hazma.gamma_ray` and
with it `gamma_ray_fsr`, the Monte-Carlo photon spectrum from a
user-supplied squared matrix element. The follow-up
[`msqrd-driven-fsr-generator`](../followups/done/msqrd-driven-fsr-generator.md)
asks for a maintained replacement inside `hazma.spectra`, and names one
design question that must be settled before code is written: the old
signature took the non-radiative rate as a bare float,

```python
gamma_ray_fsr(photon_energies, cme, isp_masses, fsp_masses,
              non_rad, msqrd, nevents=1000)
```

which forces the caller to supply a width or cross section in exactly
the units, spin-averaging, symmetry-factor, and coupling conventions
that the Monte-Carlo numerator happens to use. Nothing checks the two
agree; a mismatch rescales the whole spectrum silently. The same float
is why the old function needed `isp_masses` at all — it had to
reconstruct the flux (annihilation) or `1/2M` (decay) prefactor to make
the numerator's units match the caller's rate.

The physics does not require any of that. The observable is

```text
dN/dE = d\Gamma(X -> F + gamma)/dE / \Gamma(X -> F)
```

and every initial-state factor — flux, 1/(2M), spin averaging,
couplings, propagators at fixed s, symmetry factors of F (the photon is
distinct) — appears identically in numerator and denominator. In the
ratio of bare phase-space integrals

```text
dN/dE = [ dI_rad/dE ] / I_0,     I[|M|^2] = \int |M|^2 d\Phi
```

everything cancels except the two squared matrix elements themselves.

## Decision

The replacement is `hazma.spectra.dnde_photon_fsr`, and its contract is:

1. **Both matrix elements, no rate float.** The caller supplies
   `msqrd` (radiative, photon momentum last) *and* `msqrd_nonrad`
   (non-radiative) as callables. There is no `non_rad` float and no
   `isp_masses` argument; decays and annihilations are the same call.
   Consistency of conventions between numerator and denominator is
   automatic because both come from the same Feynman rules — any common
   constant factor cancels.
2. **No `Theory` coupling.** The function does not accept a model
   object. Models (layer 5) may call `hazma.spectra` (layer 3), never
   the reverse, and the concrete mediator models already ship analytic
   FSR. A convenience wrapper on `Theory` can be added later without
   touching this surface.
3. **Momentum convention.** `msqrd(momenta)` receives a NumPy array of
   shape `(4, n_fsp[, batch])` — rows `(E, px, py, pz)` in MeV, one
   column per final-state particle in the order of
   `final_state_masses`, with the photon appended last for the
   radiative call. Implementations written with `hazma.utils.ldot` /
   `lnorm_sqr` handle both the batched and unbatched forms for free.
4. **Final-state invariants only.** Momenta are generated in a frame
   where the non-photon system is at rest (total invariant mass
   `sqrt(s)` once the photon is included); no initial-state momenta are
   provided. `msqrd` must therefore be a Lorentz-invariant function of
   the final-state momenta alone: decays of unpolarized particles
   always qualify, and annihilation matrix elements must already be
   averaged over the beam direction (exact for s-channel processes).
5. **The statistical error is part of the API.** The return value is a
   `FSRSpectrum` NamedTuple `(dnde, error)`, both in MeV⁻¹ and shaped
   like the input energies. `error` is the one-sigma Monte-Carlo
   estimate (quadrature-error estimate on the deterministic path). A
   caller who ignores it can unpack `dnde` explicitly; limit-setting
   code that would silently fit MC noise now has the noise floor in
   hand.
6. **Deterministic quadrature where possible, RAMBO elsewhere.** For a
   2-body non-radiative final state (the dominant use case) the fixed
   photon energy reduces the radiative integral to one Dalitz-strip
   quadrature and the result carries no MC noise; for three or more
   final-state particles the generator runs RAMBO at the reduced
   invariant mass `s' = s - 2*sqrt(s)*E` with the photon appended —
   the same fixed-photon-energy estimator the removed kernel used
   (evaluating dN/dE exactly at the requested energies, with no
   histogram binning) — seeded via `seed=` for reproducibility.
   `method="auto"` picks between them; both are built on
   `hazma.phase_space`, not a private phase-space implementation.

## Consequences

- **Positive:** the silent-normalization trap is structurally gone; the
  decay/annihilation branch and `isp_masses` disappear; spectra keep
  their layer; MC noise is visible to callers; the 2-body case is
  exact up to quadrature error and needs no seed.
- **Negative:** callers who know the non-radiative width analytically
  must still write `|M_0|^2` as a callable (for a 2-body final state
  this is a one-liner) and pay a cheap quadrature for a number they
  already had; beam-direction-dependent (t-channel annihilation)
  matrix elements are out of scope; polarized initial states are out
  of scope.
- **Mitigation:** the t-channel restriction is documented on the
  function and can be lifted compatibly later (boost the generated
  events to the CM frame and document a beam axis — invariant-only
  `msqrd` implementations are unaffected by that change). Validation
  is pinned in `test/spectra/` against real theoretical calculations:
  exact tree-level `|M|^2` for `x xbar -> V* -> l+ l- gamma`,
  `x xbar -> S* -> l+ l- gamma`, and `x xbar -> V* -> pi+ pi- gamma`
  are integrated by the generator and compared with the closed-form
  spectra the mediator models already ship
  (`dnde_xx_to_v_to_ffg`, `dnde_xx_to_s_to_ffg`,
  `dnde_xx_to_v_to_pipig`), with the soft-photon limit, the
  Altarelli–Parisi collinear limit, and Ward identities as
  corpus-level cross-checks.
