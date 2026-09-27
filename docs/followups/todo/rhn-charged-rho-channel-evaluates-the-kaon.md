# The RHN `ℓ ρ±` channel evaluates the `ℓ K` final state

- **Added:** 2026-09-26
- **Source:** measuring the downstream reach of the rho photon boost repair
  (`C3`,
  [`rho-photon-outer-boost-misses-support.md`](../done/rho-photon-outer-boost-misses-support.md))
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open

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
