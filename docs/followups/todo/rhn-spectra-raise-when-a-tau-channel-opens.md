# `RHNeutrino` spectra raise once a channel with a tau opens

- **Added:** 2026-09-29
- **Source:** measuring the RHN `ℓ ρ±` channel repair
  ([`rhn-charged-rho-channel-evaluates-the-kaon.md`](../done/rhn-charged-rho-channel-evaluates-the-kaon.md))
- **Scope:** cross-cutting (public API)
- **Status:** open

## Why

`hazma/spectra/_nbody.py`'s product tables have no `"tau"` entry, and
`_conv_dnde_dist` asserts that every state of a three-body final state is
in them. `RHNeutrino` sums every channel, so once a three-body channel
with a tau opens, `total_spectrum`, `spectra`, `positron_spectra` and
`total_positron_spectrum` raise `AssertionError: Invalid final state:
tau.`, although that channel's branching fraction starts at zero. That
happens at `m_N > m_τ + m_e = 1777.4` MeV for the `e` and `τ` flavors,
through `ν_τ e τ` and `ν_e e τ`, and at `m_N > m_τ + m_μ = 1882.5` MeV for
the `μ` flavor. The model's two-body tau channels do not raise, since
`_dnde_two_body` skips a state with no table entry, so `τ π` and `τ K`
silently contribute nothing from the tau's decay.

`test/rh_neutrino/test_rh_neutrino_integration.py` stops at 1,450.99 MeV,
below every tau threshold, so nothing catches this.

## What

Decide what a tau contributes. Hazma has no tau decay spectra, so the
honest options are a tau spectrum (tabulated like the mesons', from the
tau's measured branching fractions) or an explicit zero entry with a
documented omission. Either way the three-body path must stop raising,
and the two-body path's silent skip should become the same explicit
choice. Extend the integration test past the tau thresholds.

## Entry points

- `hazma/spectra/_nbody.py`, `_spectra_dict` and the assert in
  `_conv_dnde_dist`.
- `hazma/rh_neutrino/_configure.py`, the channels with a tau.
- `test/rh_neutrino/test_rh_neutrino_integration.py`.

## Risks / open questions

A zero entry changes a crash into a spectrum that omits the tau's
daughters, which is a silent under-count unless `CHANGELOG.md` and the
docstrings say so.
