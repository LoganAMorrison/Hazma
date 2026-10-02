# The GeV vector mediator's `v v` photon lines ignore branching fractions

- **Added:** 2026-10-01
- **Source:** measured while resolving
  [`vector-mediator-gev-vv-positron-and-neutrino-are-zero.md`](vector-mediator-gev-vv-positron-and-neutrino-are-zero.md)
- **Scope:** cross-cutting (public spectrum values)
- **Status:** done.

> **Resolved.** `dnde_photon_v_v` weights each line by its branching
> fraction, places the `π⁰γ` line at the neutral pion's mass, and returns
> zeros for a mediator with no open decay. The caller-grid continuum is
> [its own follow-up](../todo/vector-mediator-gev-vv-photon-continuum-uses-caller-energies.md).

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
rest frame. The continuum also divides by the mediator's total width, so
a mediator with no open decay gives `NaN` rather than zero, which the
positron and neutrino spectra guard against.

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

## Resolution (measured)

`dnde_photon_v_v` computes the branching fractions once, guards a zero
total width as the positron and neutrino spectra do, and passes the
fractions to `_dnde_photon_v_v_rest_frame`, which no longer recomputes
them. Each line box is weighted by `2 BR`.

**The photon count cannot be pinned directly.** Final-state radiation
makes it diverge as `1/E`, so
`test/vector_mediator/test_gev_v_v_spectra.py` pins two identities that
need no infrared cutoff. A 200 MeV mediator with only quark couplings
decays only to `π⁰γ`, so it yields `1 + 2 BR(π⁰ → γγ)` photons per
decay carrying a known energy; the neutral pion's spectrum counts only
its two-photon mode. For the model above, the `v v` photon energy is
`γ` times twice the branching-weighted sum of each channel's rest-frame
energy and the two line energies.

| Quantity, `e_cm = 10.1` GeV | Before | After | Reference |
| --- | --- | --- | --- |
| hadronic `v v` photon energy, MeV | 9,418 | 972.05 | 972.21 |
| `π⁰γ`-only photon energy, MeV | 9,860 | 10,018 | 10,013 |

The hadronic values come from one model instance; the n-body partial
widths are Monte Carlo phase-space integrals, so the energy moves by
about 0.2% between instances while its residual holds at 1.66e-4.
The residuals, at most 4.2e-4 of the energy at `γ = 5.05` and `20`,
come from tabulating the continuum on the requested energies, where the
pion and eta decay boxes have edges between grid points. The tests
budget 1e-3. A fixed rest-frame grid would remove them, but photons
need an infrared cutoff and different box edges than
`v_v_rest_energies` supplies, so that is
[its own follow-up](../todo/vector-mediator-gev-vv-photon-continuum-uses-caller-energies.md).
No parity corpus array covers `VectorMediatorGeV`, so no corpus value
moves.
