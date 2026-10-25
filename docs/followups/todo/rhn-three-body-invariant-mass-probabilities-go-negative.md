# `RHNeutrino`'s `ν ℓ ℓ'` invariant-mass distributions go negative

- **Added:** 2026-10-24
- **Source:** measuring the [AP endpoint repair](../done/altarelli-parisi-fsr-runs-past-its-endpoint.md)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none.

## Why

For `RHNeutrino(300, 1e-3, "e")` the `vm e mu` channel builds
`ThreeBody` invariant-mass distributions with negative probabilities:
the minimum is −0.017 in the `(0, 1)` pair and 16 of its bins are
negative, with 11 in each of the other two pairs. The channel's photon
spectrum is negative at 7 of 160 energies between 0.01 and 150 MeV. At
`RHNeutrino(150, 1e-3, "e")` the same channel is `NaN` at every energy.
Both predate the AP repair. A negative probability means the squared
matrix element it integrates is negative somewhere in the Dalitz region.

## What

Find where `_msqrd_v_l_l` goes negative or `NaN`, and fix it or document
the region where it is valid. Pin the channel's distributions as
nonnegative and its spectrum as finite, at both masses above.

## Entry points

- `hazma/rh_neutrino/_spectra.py` — the three-body channel builders.
- `hazma/rh_neutrino/_widths.py` — `_msqrd_v_l_l` and the dispatcher that
  calls it, which supply the squared matrix element.
