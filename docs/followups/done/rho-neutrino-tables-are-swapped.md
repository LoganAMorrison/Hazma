# The rho neutrino tables are swapped, and `rho0` drops its leptons

- **Added:** 2026-09-29
- **Source:** measuring the RHN `ℓ ρ±` channel repair
  ([`rhn-charged-rho-channel-evaluates-the-kaon.md`](rhn-charged-rho-channel-evaluates-the-kaon.md))
- **Scope:** cross-cutting (public spectrum values)
- **Status:** done. No corpus value moves, so no roster label is issued.

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

## Resolution (measured)

**The swap was in the file names.** Each CSV's header names the decay it
tabulates: the two `charged_rho_neutrino_*.csv` files carried a `pi_pi`
column and the two `neutral_rho_neutrino_*.csv` files a `pi_pi0` column.
The four files are renamed so that each name matches its contents, and
the loads in `hazma/spectra/_neutrino/__init__.py` are unchanged. The
`"rho0"` entries of `_dnde_positron_dict` and `_dnde_neutrino_dict` in
`hazma/spectra/_nbody.py` now name `dnde_positron_neutral_rho` and
`dnde_neutrino_neutral_rho`.

**The pin.** `test/spectra/test_rho_neutrino_yields.py` integrates each
rho's neutrino spectrum at `γ = 1.5` and `5` and holds it to the charged
pion's count of 1.000 electron-flavor and 1.9998 muon-flavor neutrinos:
once for the charged rho and twice for the neutral rho, at `rtol = 2e-3`.
The tables carry only the two-pion modes, so the `π⁰ γ` and non-pion
`ρ⁰` modes do not enter, and the residual of at most 9.4e-4 is the
tables' interpolation error. A third test holds `dnde_positron` and
`dnde_neutrino` for the `("rho0", "pi0")` final state to the neutral
rho's own spectra at its two-body energy. All five tests fail on the
previous tables and dispatch.

**The measurement.** Integrated over energy at `E_ρ = 1.5 m_ρ`, the
charged rho's `e`- and `μ`-flavor neutrinos go from 2.000 and 3.996 to
1.000 and 1.998, and the neutral rho's go the other way. The
`("rho0", "pi0")` N-body spectra at 1.5 times threshold go from zero to
2.000 and 3.996 neutrinos and 0.9997 positrons. For
`RHNeutrino(1000, 1e-3, "e")`, the `ν ρ⁰` channel (branching fraction
0.0199) goes from zero to the same counts, and the `e ρ` channel
(0.246) from 4.000 and 7.991 neutrinos to 2.000 and 3.996. Per
heavy-neutrino decay, the `e`- and `μ`-flavor neutrinos fall from 2.956
and 4.585 to 2.503 and 3.680, the `τ`-flavor count stays 0.788, and the
positrons rise from 1.440 to 1.460. Photon spectra do not move.

**The parity corpus** pins only the rho photon spectra and reaches
neither `RHNeutrino` nor the N-body dispatch, so no corpus value moves.
