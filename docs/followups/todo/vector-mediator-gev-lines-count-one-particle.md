# The GeV vector mediator's lepton lines count one particle

- **Added:** 2026-10-01
- **Source:** measured while resolving
  [`vector-mediator-gev-vv-positron-and-neutrino-are-zero.md`](../done/vector-mediator-gev-vv-positron-and-neutrino-are-zero.md)
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open

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
