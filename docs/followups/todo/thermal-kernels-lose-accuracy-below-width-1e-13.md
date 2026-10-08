# Thermal kernels lose accuracy for mediator widths below 1e-13 of `m_x`

- **Added:** 2026-10-01
- **Source:** PR #122 review
- **Scope:** cross-cutting
- **Status:** open

## Why

The width ladder in `rust/src/kernels/thermal_window.rs` resolves a
resonance only while its width `g` (in units of `m_x`) is well above the
ulp of the peak position `z_r`. Near `g ≈ 1e-13` the propagator's
`s − m²` cancels to about `ulp(z_r)/g` relative, which no partition of the
integration interval recovers. The kernels return a confident value and
discard QUADPACK's `ier` (`scalar_xs.rs` near line 993, `vector_xs.rs`
near line 507), so the loss is silent.

Measured against a ratio-2 reference, the thermal average is low by
1.8e-3 for `HiggsPortal(mx=200, ms=550, gsxx=1e-6, stheta=1e-8)`
(`g = 3.5e-14`) at every `x` from 0.1 to 20, and by 3.7e-3 for
`KineticMixing(gvxx=1e-6, eps=1e-8)`. References built with ladder ratios
1.5, 2, and 4 disagree with each other at 1e-3, and scipy returns
`ier = 4` on the same partition.

## What

Pick one or both of two fixes.

- **Reformulate the propagator near the pole**, for example by
  integrating in a variable centered on the peak so that `s − m²` is
  formed without cancellation.
- **Surface QUADPACK's `ier`**, so a caller learns when the integral did
  not converge instead of receiving a value that looks converged.

Either fix moves numbers only for widths below the floor, and the
`thermal_window` module docs should drop the caveat once it lands.

## Entry points

- `rust/src/kernels/thermal_window.rs`: resonance bullet of the module docs.
- `rust/src/kernels/scalar_xs.rs` and `rust/src/kernels/vector_xs.rs`:
  the thermal-average integrations that discard `ier`.
- Related: `docs/followups/done/thermal-kernels-miss-narrow-mediator-resonances.md`

## Risks / open questions

Whether a peak-centered variable keeps the existing parity-corpus values
for wide resonances unchanged is untested.
