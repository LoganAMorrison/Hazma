# The two thermal kernels disagree above `x = 300`

- **Added:** 2026-09-29
- **Source:** "Risks" in
  [`vector-thermal-kernel-fixed-floor-loses-accuracy-at-large-x.md`](../done/vector-thermal-kernel-fixed-floor-loses-accuracy-at-large-x.md)
- **Scope:** cross-cutting
- **Status:** done. Repaired as parity roster entry `C8`, the eighth
  label issued under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).
- **Triggers / blockers:** none.

> **Resolved.** All four sites integrate at the true `x` with
> exponentially scaled Bessel factors, rather than clipping, up to a
> supported maximum of `x = 1e7`; past it they raise `ValueError`. See
> "Resolution (measured)" below.

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

## Resolution (measured)

**The rule.** Every site integrates at the true `x`, and none clips or
cuts off. The prefactor `x / (2 K₂(x))²` and the kernel `K₁(x z)` are
evaluated as `exp(-x)`-scaled factors, so the product that reaches the
integrand never overflows. The Rust side adds `bessel_k1e` and
`bessel_kne` in `rust/src/special.rs`, and `prefactor` and
`boltzmann_weight` in `rust/src/kernels/thermal_window.rs`; the Python
side adds `thermal_average` in
`hazma/relic_density/_thermal_functions.py`. Clipping was rejected
because it holds ⟨σv⟩ at its `x = 300` value instead of letting it
approach the `v → 0` limit.

**Supported domain.** The average is supported for `0 < x ≤ 1e7`
(`X_MAX` in `rust/src/kernels/thermal_window.rs`, mirrored by
`_THERMAL_X_MAX` in `hazma/relic_density/_thermal_functions.py`).
Measured, the kernels returned `0.0`, a negative value
(`HiggsPortal(300, 200, 1e-2)` gave −1.52e-6 at `x = 1e12`), a
`TypeError` (vector) or NaN from roughly `x = 5e8`, so every entry point
raises `ValueError` past `X_MAX` instead. Master's vector clip had
returned a finite value there; no in-tree default reaches it, since
`relic_density`'s Boltzmann path stops near `x ≈ 1.1e3`.

**Four sites, not three.** The follow-up named the two Rust kernels and
the generic fallback. The fourth is
`hazma/vector_mediator/_gev/thermal_cross_section.py`, which also
returned `0.0` above 300. It and the fallback now share
`thermal_average`, and `thermal_cross_section_integrand` is gone.

**Corpus.** The repair is parity roster entry `C8`, and the six
thermal arrays are declared as `B6+C7+C8`. The 60 corpus positions
above `x = 300` move. At `x = 1000` the scalar average goes from `0.0`
to 0.29 to 0.30 of its `x = 300` value, and the vector average falls by
up to 9.4e-3 from the held value. The composed prediction holds to
1.2e-11 above `x = 300` and to 1.8e-8 over all positions.

**Downstream.** The relic densities of the six parity model points move.
`HiggsPortal` falls by up to 1.26%, at `mx=300, ms=200`, and
`KineticMixing` rises by 1.4e-4 to 6.2e-4. The fallback and both kernels
now agree in `relic_density` to at most 8.5e-8 at four model points,
where the fallback and the vector kernel had sat 8% apart.

**Testing.** Each site has its own check past `x = 300`. The kernels and
the generic fallback agree from `x = 300` to `1e7`, and the GeV closure
matches a converged reference in the exponentially scaled form to at
most 1.7e-9 over the same range. All four raise past `x = 1e7`. The
scalar kernel's 6.6e-8 offset from the fallback is their electron
masses, legacy `0.510998928` against `0.5109989461`, and belongs to
[the constants consolidation](../todo/consolidate-the-two-constants-tables.md).
