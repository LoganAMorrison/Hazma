# `SingleChannelAnn` line-energy tables are nested dicts

- **Added:** 2026-10-25
- **Source:** PR #136 review (reviewers A and E)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none

## Why

`SingleChannelAnn._positron_line_energies` returns `{fs: {fs: energy}}`, and
`_gamma_ray_line_energies` has the same shape for its final states, while
`TheoryAnn` consumers expect `{fs: energy}`. As a result
`SingleChannelAnn("e e").f_eff_ep()` raises `TypeError` in
`hazma/theory/_theory_cmb.py`, so the CMB efficiency of a single-channel model
with an e± line cannot be computed. The defect predates and lies outside the
axial-mediator observation adapters that exposed it.

## What

Return flat `{final_state: energy}` dicts from both methods, matching the
`Dict[str, float]` annotations and the base-class contract in
`hazma/theory/__init__.py`, then add a regression test that evaluates
`f_eff_ep()` and `f_eff_g()` for a single-channel model with a line. Check the
gamma-ray line path that reads the same table for the same nesting.

## Entry points

- `hazma/single_channel.py:109` (`_positron_line_energies`) and `:87`
  (`_gamma_ray_line_energies`)
- `hazma/theory/_theory_cmb.py` (`_f_eff_helper`)
- `hazma/theory/__init__.py` (`_gamma_ray_line_energies`,
  `_positron_line_energies`)
