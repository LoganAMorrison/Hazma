# The deprecated Altarelli-Parisi functions keep the old endpoint

- **Added:** 2026-10-24
- **Source:** the [AP endpoint repair](../done/altarelli-parisi-fsr-runs-past-its-endpoint.md)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none.

## Why

`hazma/utils.py::__dnde_altarelli_parisi`, behind the deprecated
`dnde_altarelli_parisi_fermion` and `dnde_altarelli_parisi_scalar`, is a
separate copy of the AP spectrum. It still zeroes the spectrum only past
`x = 1 − e m²/s`, so for an equal pair it runs past the photon endpoint
`x = 1 − 4m²/s`, as `hazma.spectra.altarelli_parisi` did before its repair.
It is not dead: `hazma/rh_neutrino/_rh_neutrino_fsr.py` calls it for the
`ℓ ℓ ν`, `ℓ π` and `π π` FSR of its right-handed neutrino class.

## What

Point `_rh_neutrino_fsr.py` at `hazma.spectra.dnde_photon_ap_fermion` and
`dnde_photon_ap_scalar`, with each radiator's `partner_mass`, and with
twice the spectrum for a pair since those carry `α/2π` per leg. Then
decide whether the deprecated wrappers delegate to them, which moves
their values, or stay frozen until removed. Measure and record the
change in `CHANGELOG.md`.

## Entry points

- `hazma/utils.py` — `__dnde_altarelli_parisi`.
- `hazma/rh_neutrino/_rh_neutrino_fsr.py` — its four call sites.
