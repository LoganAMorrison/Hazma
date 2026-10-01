# The two thermal kernels disagree above `x = 300`

- **Added:** 2026-09-29
- **Source:** "Risks" in
  [`vector-thermal-kernel-fixed-floor-loses-accuracy-at-large-x.md`](../done/vector-thermal-kernel-fixed-floor-loses-accuracy-at-large-x.md)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. It moves parity-pinned values, so it
  takes the next `C<n>` label under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).

## Why

Above `x = m_x / T = 300` the scalar kernel's `thermal_cross_section`
(`rust/src/kernels/scalar_xs.rs`) returns exactly `0.0`, while the
vector kernel's (`rust/src/kernels/vector_xs.rs`) clips `x` to 300 and
keeps returning the value there. Both rules come from the `.pyx`, and
neither is physics: ⟨σv⟩ tends to its s-wave limit as `x` grows, and
it vanishes only when nothing is s-wave. The pure-Python fallback in
`hazma.relic_density._thermal_functions.thermal_cross_section` follows
the scalar's rule, so a model's abundance depends on which path
computes its ⟨σv⟩.

The resolved follow-up above measured the consequence on 2026-09-25:
at the two `KineticMixing` corpus points it checked, the fallback's
`relic_density` sits 8% off the vector kernel's.

## What

Decide one large-`x` rule and apply it at all three sites. Candidates
are integrating at the true `x`, which needs the prefactor
`x / (2 K₂(x))²` and `K₁(x z)` computed without overflow, for example by
factoring out `exp(−x)`; or clipping as the vector does. Then re-derive
the corpus positions above `x = 300` in both
`cross_sections.*.thermal_cross_section` cases and declare them as the
next `C<n>` in `test/parity/deltas.py`, with a `CHANGELOG.md` entry.

## Entry points

- `rust/src/kernels/scalar_xs.rs` — `if x > 300.0 { return Ok(0.0) }`
  in `thermal_cross_section`, and
  `the_thermal_average_cuts_off_above_three_hundred`.
- `rust/src/kernels/vector_xs.rs` — the `xnew` clip, and
  `the_thermal_average_saturates_above_three_hundred`.
- `hazma/relic_density/_thermal_functions.py` — the fallback's cutoff.
- `test/parity/thermal_reference.py` — `_clip` mirrors both rules.
- `test/parity/cases.py::_thermal_blocks` — the anchors at `x = 300`.

## Risks / open questions

- `relic_density` integrates past `x = 300` for any model that freezes
  out late, so this changes abundances as well as the corpus.
