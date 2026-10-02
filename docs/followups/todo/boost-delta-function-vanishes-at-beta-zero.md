# `boost_delta_function` returns zeros at `beta == 0`

- **Added:** 2026-10-01
- **Source:** review of PR #121
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open

## Why

`hazma.spectra.boost.boost_delta_function` fills its output only for
`0 < beta < 1`, so at exactly `cme == 2 m_V`, where `beta == 0`, it
returns zeros. Every monochromatic line boosted through it then vanishes,
while `dnde_boost_array` still returns the rest-frame continuum. The
`π⁰γ` and `ηγ` lines of `dnde_photon_v_v` and the `e e` line of
`dnde_positron_v_v` are affected.

The gap is measured for a 200 MeV mediator that decays only to `π⁰γ`,
`VectorMediatorGeV(5e3, 200.0, 1.0, 1.0, -1.0, 0, 0, 0, 0, 0, 0)`. The
`v v` photon energy, integrating `E dN/dE` over a linear grid of 40,001
points up to `cme / 2`, is 287.65 MeV at `cme = 400` MeV against a
reference of 396.52 MeV. The 108.9 MeV gap is the two line photons, each
carrying 54.45 MeV. At `cme = 400.4` MeV the energy is 397.02 MeV against
396.92 MeV. The behavior predates the lines' branching-fraction fix.

## What

Make `boost_delta_function` return the unboosted delta, or its documented
limit, at `beta == 0`. That fixes every caller at once, rather than each
spectrum special-casing the threshold.

## Entry points

- `hazma/spectra/boost.py`: `boost_delta_function`
- `hazma/vector_mediator/_gev/spectra.py`: `dnde_photon_v_v`
- `hazma/vector_mediator/_gev/positron.py`: `dnde_positron_v_v`

## Risks / open questions

A delta function on a discrete grid has no finite value, so the limit
needs a documented convention, for example a box of the grid's width or
an explicit threshold branch in the callers. Other callers of
`boost_delta_function` may rely on the zero at `beta == 0`.
