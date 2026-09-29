# The rho neutrino tables are swapped, and `rho0` drops its leptons

- **Added:** 2026-09-29
- **Source:** measuring the RHN `ℓ ρ±` channel repair
  ([`rhn-charged-rho-channel-evaluates-the-kaon.md`](../done/rhn-charged-rho-channel-evaluates-the-kaon.md))
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open

## Why

`hazma/spectra/_neutrino/__init__.py` loads `charged_rho_neutrino_*.csv`
for `dnde_neutrino_charged_rho` and `neutral_rho_neutrino_*.csv` for
`dnde_neutrino_neutral_rho`, but the tables hold each other's spectra.
Integrated over energy at `E_ρ = 1.5 m_ρ`, the charged rho yields 2.000
electron-flavor and 3.996 muon-flavor neutrinos, and the neutral rho
yields 1.000 and 1.998. The charged pion yields 1.000 and 2.000, counting
neutrinos and antineutrinos together. `ρ± → π± π⁰` holds one charged pion
and `ρ⁰ → π⁺π⁻` holds two, so the charged rho should give the pion's
yield and the neutral rho twice it. The generator scripts,
`notebooks/decay_spectra/charged_rho.py` (`π π⁰`) and `neutral_rho.py`
(`π π`), agree on which decay is which.

Separately, `hazma/spectra/_nbody.py` routes the `"rho0"` final state to
`_dnde_zero` for positrons and `_dnde_zero_nu` for neutrinos, although
`dnde_positron_neutral_rho` and `dnde_neutrino_neutral_rho` exist. Every
N-body spectrum with a `ρ⁰` in its final state therefore drops the `ρ⁰`'s
positrons and neutrinos. That includes `RHNeutrino`'s `ν ρ⁰` channel.

## What

Load each rho's own table in `hazma/spectra/_neutrino/__init__.py`,
then route `"rho0"` to `dnde_positron_neutral_rho` and
`dnde_neutrino_neutral_rho` in `_nbody.py`. Wiring `"rho0"` before the
swap is fixed would give the `ν ρ⁰` channel the charged rho's yield, so
land both together. Pin each rho's integrated neutrino yield per flavor
against the charged pion's: the charged rho's equals one pion's to the
`π⁰ γ` branching fraction, and the neutral rho's is twice it, less the
`ρ⁰` modes without a charged pion pair.

Measure the change in `dnde_neutrino_charged_rho`,
`dnde_neutrino_neutral_rho`, `hazma.spectra.dnde_positron` and
`dnde_neutrino` with a `"rho0"` final state, and every `RHNeutrino`
spectrum summing `ν ρ⁰` or `ℓ ρ`. Record it in `CHANGELOG.md`. The
parity corpus pins neither rho's neutrino or positron spectrum, so no
roster label is needed; confirm that in `test/parity/cases.py` first.

## Entry points

- `hazma/spectra/_neutrino/__init__.py`, the four `_*_rho_integrand_interp_*`
  loads.
- `hazma/spectra/_nbody.py`, the `"rho0"` entries of
  `_dnde_positron_dict` and `_dnde_neutrino_dict`.
- `hazma/spectra/_neutrino/data/*_rho_neutrino_*.csv`.

## Risks / open questions

Check whether the swap is in the loads or in the CSV files themselves;
renaming the files and swapping the loads give the same values, but the
file names should match their contents. `dnde_positron_charged_rho`
returns `dnde_positron_neutral_rho`, which is right for `ρ⁺` only because
each decay holds one `π⁺`; it is not part of this item.
