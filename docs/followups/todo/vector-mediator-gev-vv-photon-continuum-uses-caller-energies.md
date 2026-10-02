# The GeV `v v` photon continuum is tabulated on the caller's energies

- **Added:** 2026-10-01
- **Source:** split from
  [`vector-mediator-gev-vv-photon-lines-ignore-branching-fractions.md`](../done/vector-mediator-gev-vv-photon-lines-ignore-branching-fractions.md)
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open

## Why

`dnde_photon_v_v` in `hazma/vector_mediator/_gev/spectra.py` evaluates
the rest-frame continuum on the lab energies it was asked for and boosts
that array's linear interpolant with `boost.dnde_boost_array`. The result
therefore depends on the caller's grid: a coarse request misses the
`π⁰` and `η` decay boxes and the final-state radiation's shape, and a
scalar energy raises, because `dnde_boost_array` needs an array when it
is given no `rest_energies`. The positron and neutrino spectra avoid
both through a fixed rest-frame grid, `utils.v_v_rest_energies`.

## What

Tabulate the photon continuum on a fixed rest-frame grid and pass it as
`rest_energies`. `v_v_rest_energies` does not carry over as is. Its
lower end must be an infrared cutoff, since the radiation diverges as
`1/E`, and the photon boxes it must bracket are the `π⁰ → γγ` and
`η → γγ` boxes of the two-body channels (`π⁰ γ`, `η γ`, `π⁰ φ`, `η φ`,
`η ω`), not the charged-pion lines. Pin the result as
`test/vector_mediator/test_gev_v_v_spectra.py` pins the positrons: its
values must not depend on the requested energies, and a scalar energy
must give the array's value.

## Entry points

- `hazma/vector_mediator/_gev/spectra.py` — `dnde_photon_v_v`
- `hazma/vector_mediator/_gev/utils.py` — `v_v_rest_energies`
