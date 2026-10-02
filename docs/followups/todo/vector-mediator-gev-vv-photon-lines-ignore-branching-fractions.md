# The GeV vector mediator's `v v` photon lines ignore branching fractions

- **Added:** 2026-10-01
- **Source:** measured while resolving
  [`vector-mediator-gev-vv-positron-and-neutrino-are-zero.md`](../done/vector-mediator-gev-vv-positron-and-neutrino-are-zero.md)
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open

## Why

`dnde_photon_v_v` in `hazma/vector_mediator/_gev/spectra.py` adds the
boosted `V → π⁰γ` and `V → ηγ` lines as
`2 * boost_delta_function(...)`, a weight of one photon per decay for
each line, whatever the mediator's branching fractions. Its continuum,
`_dnde_photon_v_v_rest_frame`, weights every channel by its branching
fraction, so the lines are the odd ones out. The `π⁰γ` line also sits at
`(m_V − m_π±² / m_V) / 2`, with the charged pion's mass where the
neutral pion's belongs.

Measured at `m_x = 5` GeV, `m_V = 1` GeV and `e_cm = 10.1` GeV, with
every coupling 1: the `v v` photon spectrum carries 6.09 photons per
annihilation, and the two line boxes carry 4.00 of them. Weighted by
`BR(π⁰γ) = 6e-4` and `BR(ηγ) = 0.0109`, they would carry 0.023. The
line misplacement moves the `π⁰γ` line from 490.89 to 490.26 MeV in the
rest frame.

## What

Weight each line box by `2 BR`, with the branching fractions the
continuum already computes, and use `neutral_pion_mass` for the `π⁰γ`
line. Pin the photon count against twice the mediator's per-decay photon
yield, as `test/vector_mediator/test_gev_v_v_spectra.py` does for
positrons and neutrinos. The continuum is evaluated on the caller's
energies, so a coarse or single-point request degrades it; the
`rest_energies` grid that `dnde_positron_v_v` uses would remove that.

## Entry points

- `hazma/vector_mediator/_gev/spectra.py` — `dnde_photon_v_v`
- `hazma/vector_mediator/_gev/positron.py` — `dnde_positron_v_v`, the
  shape to follow
