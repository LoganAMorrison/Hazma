# `RHNeutrino`'s line tables name channels the model does not have

- **Added:** 2026-09-29
- **Source:** measuring the RHN `ℓ ρ±` channel repair
  ([`rhn-charged-rho-channel-evaluates-the-kaon.md`](../done/rhn-charged-rho-channel-evaluates-the-kaon.md))
- **Scope:** cross-cutting (public API)
- **Status:** open

## Why

`TheoryDec.gamma_ray_lines` and `positron_lines` look each line's key up
in `decay_branching_fractions()`, whose keys are the final-state strings
of `hazma/rh_neutrino/_configure.py`, such as `"ve a"` and `"e pi"`.
`RHNeutrino._gamma_ray_line_energies` returns `"nu g"`, and
`_positron_line_energies` returns `"pi l"` and `"k l"` for the `e`
flavor. So `RHNeutrino(1000, 1e-3, "e").gamma_ray_lines()` raises
`KeyError: 'nu g'`, `positron_lines()` raises `KeyError: 'pi l'`, and
`total_conv_spectrum_fn` and `total_conv_positron_spectrum_fn` raise with
them.

The tables are also incomplete or wrong where the keys would match. The
`e ρ` channel has an electron line at `(m_N² + m_e² − m_ρ²) / (2 m_N)`
that `_positron_line_energies` omits. `_neutrino_line_energies` writes
the `ν η` line's energy into the `ν π⁰` key, overwriting the pion's.

## What

Key every line by the `_configure.py` final-state string it belongs to.
List one positron line per open two-body channel with an electron, and
one neutrino line per open two-body channel with a neutrino. Pin each
table against the two-body energy formula and against the branching
fraction it returns, and test that `gamma_ray_lines`, `positron_lines`
and the convolved spectra evaluate.

## Entry points

- `hazma/rh_neutrino/_model.py`, `_gamma_ray_line_energies`,
  `_positron_line_energies` and `_neutrino_line_energies`.
- `hazma/theory/__init__.py`, `TheoryDec.gamma_ray_lines` and
  `positron_lines`.

## Risks / open questions

Line energies are only meaningful where the channel is open; decide
whether a closed channel's line is omitted or carried with a zero
branching fraction, and match the annihilation models' convention.
