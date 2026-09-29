# The GeV vector mediator's `v v` positron and neutrino spectra are zero

- **Added:** 2026-09-29
- **Source:** the "Risks" note in
  [`mediator-decay-angular-windows-miss-their-support.md`](../done/mediator-decay-angular-windows-miss-their-support.md),
  checked while resolving it
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open

## Why

`VectorMediatorGeV`'s `v v` positron and neutrino spectra are exactly
`0.0` wherever `χχ → VV` is open. Measured at `m_x = 5` GeV, `m_V = 1`
GeV and `cme = 10.1` GeV, with every coupling 1: over 30 energies from
1 MeV to 5 GeV the largest `|dN/dE|` of
`positron_spectrum_funcs()["v v"]` and of both flavors'
`neutrino_spectrum_funcs(flavor)["v v"]` is `0.0`. The photon `v v`
spectrum at the same point is non-zero, peaking at 8.5e-3 MeV⁻¹.

The photon path boosts correctly: `spectra.py`'s `dnde_photon_v_v` takes
`γ = (cme / 2) / m_V` and a rest-frame spectrum evaluated at `m_V`. The
positron and neutrino paths do neither. `positron.py:874` and
`neutrino.py:977` set `gamma = 2.0 * self.mv / cme`, the inverse, and
return zeros when `gamma < 1`, which is exactly where the channel is open.
Behind that guard, the rest-frame spectrum they hand to
`make_boost_function` is evaluated at `cme` rather than at `m_V`, which a
corrected `gamma` would expose next.

## What

Follow the photon path in both: take `γ = cme / (2 m_V)` and evaluate
the rest-frame `V` decay spectrum at `m_V`. Pin the result against the
boost's own identities: the `v v` spectrum must carry twice the `V`'s
rest-frame positron and neutrino numbers at any `γ`. `make_boost_function`
integrates over an unclipped energy window, so measure whether it needs
the support clip the `hazma._core` mediator kernels gained under `C6`.

## Entry points

- `hazma/vector_mediator/_gev/positron.py:874` — `dnde_positron_v_v`
- `hazma/vector_mediator/_gev/neutrino.py:977` — `dnde_neutrino_v_v`
- `hazma/vector_mediator/_gev/spectra.py` — `dnde_photon_v_v`, the path
  that is right
- `hazma/spectra/boost.py` — `make_boost_function`

## Risks / open questions

- The parity corpus pins no `_gev` spectrum, so nothing gates the values
  this moves; the identities above have to be the evidence.
