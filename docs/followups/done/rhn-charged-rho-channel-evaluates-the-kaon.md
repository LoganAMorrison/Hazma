# The RHN `ℓ ρ±` channel evaluates the `ℓ K` final state

- **Added:** 2026-09-26
- **Source:** measuring the downstream reach of the rho photon boost repair
  (`C3`,
  [`rho-photon-outer-boost-misses-support.md`](rho-photon-outer-boost-misses-support.md))
- **Scope:** cross-cutting (public spectrum values)
- **Status:** done. No corpus value moves, so no roster label is issued.

## Why

`hazma/rh_neutrino/_spectra.py`'s `dnde_l_rho` documents the decay of a
right-handed neutrino into a charged rho and a lepton, but it returns
`_dnde_two_body(model, product_energies, product, (ell, "k"))`, the body
of `dnde_l_k` two functions above it. `hazma/rh_neutrino/_configure.py`
binds it to the `"{ll} rho"` final state with `width_l_rho` and the
rho's mass, so `RHNeutrino`'s `ℓ ρ` channel is weighted by the rho's
width and shaped like the kaon's spectrum. The charged rho's photon,
positron and neutrino spectra never reach the model.

It showed up because the `C3` repair moved `dnde_v_rho` and left
`dnde_l_rho` bit-identical at `m_N = 1`, `2` and `5` GeV, for both lepton
flavors, although the charged rho's photon spectrum moved more than the
neutral one's.

## What

Pass `(ell, "rho")`. Measure the change in every `RHNeutrino` spectrum
that sums this channel, for all three products. The kaon and rho
kinematics differ, so the channel's shape and endpoint both move.
Record the change in `CHANGELOG.md`. The parity corpus has no
`rh_neutrino` case, so no declaration is needed, but a pinned test in
`test/rh_neutrino/` should check that the channel equals the boosted
charged rho spectrum at the two-body energy.

## Entry points

- `hazma/rh_neutrino/_spectra.py`, `dnde_l_rho`.
- `hazma/rh_neutrino/_configure.py`, the `"{ll} rho"` entry.

## Risks / open questions

Check whether the `"rho"` final state is wired for all three products in
`hazma/spectra/_nbody.py`. The photon and positron tables name it and the
neutrino table names `dnde_neutrino_charged_rho`, but the neutral
entries differ (`_dnde_zero` for the positron).

## Resolution (measured)

**The fix.** `dnde_l_rho` passes `(ell, "rho")`. The `"rho"` final state
is wired for all three products in `hazma/spectra/_nbody.py`, to
`dnde_photon_charged_rho`, `dnde_positron_charged_rho` and
`dnde_neutrino_charged_rho`, so the channel now reaches the charged rho's
own spectra. The `"rho0"` entries route the positron and the neutrino
to zero, which is a separate defect, filed with the swap below as
[`rho-neutrino-tables-are-swapped.md`](rho-neutrino-tables-are-swapped.md).

**The pin.** `test/rh_neutrino/test_rh_neutrino_two_body_channels.py`
holds the `e ρ` channel to the charged rho boosted to
`E_ρ = (m_N² + m_ρ² − m_e²) / (2 m_N)`, at `m_N = 1`, `2` and `5` GeV and
for all three products. The photon expectation adds the electron's
Altarelli-Parisi FSR at `s = m_N²`. Every expectation is built from public
`hazma.spectra` kernels, not from `hazma.spectra.dnde_photon`'s `(e, rho)`
final state, and agrees to 1e-12. All nine cases fail against the kaon.

**The channel.** Measured before and after on this tree, per `ℓ ρ` decay
of one charge state, with trapezoidal integrals over 40,000 log-spaced
energies from `1e-6 m_N` to `m_N`. The photon count is omitted because
the lepton's FSR makes it depend on the grid's lower edge.

| flavor | `m_N` (GeV) | photon energy (MeV) | positrons | positron energy (MeV) | neutrinos | neutrino energy (MeV) |
| --- | --- | --- | --- | --- | --- | --- |
| e | 1 | 104.2 → 406.3 | 1.111 → 1.000 | 111.6 → 110.2 | 2.598 → 5.996 | 229.2 → 579.5 |
| e | 2 | 184.3 → 591.7 | 1.111 → 1.000 | 190.4 → 158.3 | 2.598 → 5.996 | 391.0 → 832.6 |
| e | 5 | 453.7 → 1330.4 | 1.111 → 1.000 | 453.0 → 352.4 | 2.598 → 5.996 | 930.4 → 1853.2 |
| μ | 1 | 96.6 → 396.1 | 2.111 → 2.000 | 244.9 → 181.2 | 4.598 → 7.996 | 476.5 → 708.7 |
| μ | 2 | 171.0 → 577.2 | 2.111 → 2.000 | 519.6 → 456.4 | 4.598 → 7.996 | 1002.2 → 1384.7 |
| μ | 5 | 422.0 → 1298.1 | 2.111 → 2.000 | 1319.8 → 1206.7 | 4.598 → 7.996 | 2539.7 → 3438.9 |

The photon energy roughly quadruples at 1 GeV, because the charged rho's
`π⁰ → γγ` box replaces the kaon's mostly photon-free `K → μ ν` mode. The
rho is heavier than the kaon, so it carries less momentum and its
daughters' spectra end lower: at `m_N = 1` GeV with `ℓ = e` the last
nonzero positron and neutrino energy on a 400-point log grid moves from
491.7 to 475.0 MeV.

The neutrino count, 6.0 per decay, is twice what `ρ± → π± π⁰` can make.
The charged and neutral rho neutrino tables are swapped, so this channel
now carries the neutral rho's `π⁺π⁻` yield. That is
[`rho-neutrino-tables-are-swapped.md`](rho-neutrino-tables-are-swapped.md).

**The model.** Only `m_N = 1` GeV can be measured through the model's
totals. At 2 and 5 GeV a channel with a tau is open, and every
`RHNeutrino` spectrum raises
([`rhn-spectra-raise-when-a-tau-channel-opens.md`](../todo/rhn-spectra-raise-when-a-tau-channel-opens.md)).
At 1 GeV the `ℓ ρ` channel's branching fraction is 0.2464 for `ℓ = e` and
0.2290 for `ℓ = μ`, and 359 or 360 of 400 log-spaced energies move in
every total.

| flavor | photon energy (MeV) | positrons | positron energy (MeV) | neutrinos |
| --- | --- | --- | --- | --- |
| e | 244.1 → 393.0 | 1.468 → 1.440 | 262.3 → 262.0 | 6.65 → 8.33 |
| μ | 218.5 → 355.7 | 1.763 → 1.737 | 261.0 → 246.4 | 8.93 → 10.49 |

These are per heavy-neutrino decay, summed over both charge states.
Pointwise, `RHNeutrino(1000, 1e-3, "e").total_spectrum` at 300 MeV goes
from 2.390e-3 to 4.536e-3 MeV⁻¹.

The parity corpus has no `rh_neutrino` case, so no corpus value moves and
no roster label is issued. The line tables beside this channel,
`_gamma_ray_line_energies` and `_positron_line_energies`, name channels
the model does not have and raise `KeyError`. That is
[`rhn-line-tables-name-missing-channels.md`](../todo/rhn-line-tables-name-missing-channels.md).
