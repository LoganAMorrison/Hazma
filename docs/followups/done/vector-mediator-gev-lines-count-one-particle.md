# The GeV vector mediator's lepton lines count one particle

- **Added:** 2026-10-01
- **Source:** measured while resolving
  [`vector-mediator-gev-vv-positron-and-neutrino-are-zero.md`](../done/vector-mediator-gev-vv-positron-and-neutrino-are-zero.md)
- **Scope:** cross-cutting (public spectrum values)
- **Status:** done.

> **Resolved.** The model counts positrons and neutrinos only, as
> `ScalarMediator` and `VectorMediator` do, so the lines stand and the
> continua halve. See "Resolution (measured)" below.

## Why

Every continuum in `hazma.vector_mediator._gev` counts particles and
antiparticles together. `dnde_positron_mu_mu` integrates to two per muon
pair, one electron and one positron, and `dnde_neutrino_pi_pi` to two
`ν_e`-flavored and four `ν_μ`-flavored neutrinos per pion pair.
`test/test_mediator_positron_channels.py` records the positron
convention.

The lines do not follow it. `VectorMediatorGeV._positron_line_energies`
and `_neutrino_line_energies` feed `TheoryAnn.positron_lines` and
`neutrino_lines`, which weight each line by its branching fraction
alone. So `χχ → e⁺e⁻` contributes one lepton per annihilation where the
continua would count two, and `χχ → ν ν̄` likewise. The `v v` spectra
count two per mediator decay, in line with the continua.

## What

Decide which convention the model's lines follow and make it explicit.
Either the lines carry a multiplicity of two, or the continua are halved
to count one species; the second moves every positron and neutrino
continuum of the model by a factor of two. Measure the change and record
it in `CHANGELOG.md`.

## Entry points

- `hazma/vector_mediator/_gev/model.py` — `_positron_line_energies`,
  `_neutrino_line_energies`, `neutrino_lines`
- `hazma/theory/__init__.py` — `positron_lines`, and the convolved
  spectra that consume it

## Risks / open questions

- Other models may count only positrons in their continua, as
  `ScalarMediator` and `VectorMediator` do. A cross-model convention may
  be the better fix.

## Resolution (measured)

The channel functions in `hazma/vector_mediator/_gev/positron.py` and
`neutrino.py` keep counting particles and antiparticles together, and
`dnde_positron_spectrum_fns` and `dnde_neutrino_spectrum_fns` halve each
one. Every final state is its own charge conjugate, so half the count is
the particle count exactly. The lines, weighted by branching fraction
alone, already counted one particle and are unchanged. Neutrinos follow
the positrons, so a flavor's spectrum counts `ν` and not `ν̄`.

The `k k` channel was the one function that counted a single kaon's
leptons. That is 1.111 e± per K⁺, which is `1 + 2 BR(K⁺ → 3π)`, and by
CP it is also the positron count of a `K⁺K⁻` pair. It now doubles the
kernel like every other channel, so its model-level spectrum is
unchanged. The `v v` rest-frame sum calls `k k` directly, so `v v` falls
by less than half where `V → K⁺K⁻` is open. At `m_x = 5` GeV,
`m_V = 1` GeV, `e_cm = 10.1` GeV and every coupling 1, it carries 1.49
positrons and 1.31, 1.78 and 0.187 `e`-, `μ`- and `τ`-flavored
neutrinos per annihilation. Halving alone would give 1.32, 1.15, 1.56
and 0.187.

`test/test_mediator_positron_channels.py` now pins `VectorMediatorGeV`'s
`mu mu` and `pi pi` channels to one positron per annihilation beside the
MeV models, and `test/vector_mediator/test_gev_neutrino_channels.py`
pins the neutrino counts of the same channels. Both fail at twice their
value against the former code. No parity-corpus array pins a `_gev`
spectrum, so no repair label applies. Recorded in `CHANGELOG.md`.
