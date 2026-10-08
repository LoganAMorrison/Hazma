# Scalar thermal kernel sits 6.6e-8 below its reference at a resonance

- **Added:** 2026-10-07
- **Source:** PR #124 review (numerics reviewer)
- **Scope:** commit
- **Status:** open
- **Triggers / blockers:** none.

## Why

For `HiggsPortal(mx=100, ms=300, gsxx=1, stheta=0.1)`, whose scalar
resonance is open, the scalar Rust thermal kernel sits a constant
relative −6.61e-8 below both the pure-Python fallback and an independent
mpmath reference (40 digits, unscaled `K1`/`K2`). The offset is the same
at `x = 300`, `1e3`, `1e4` and `1e6`. That is about four times the
kernel's `epsrel` of 1.49e-8.

The offset is present at `x = 300`, where the formulation change in PR
\#124 moves the result by only about 1e-14, so it predates that PR and
comes from the resonance handling or the quadrature budget, not from the
scaled Bessel factors.

## What

Find the source of the offset, for example by comparing the kernel's
window and break-point ladder against the mpmath integrand at the
resonance, and either repair it or document the larger error budget
beside the kernel's `epsrel`. A repair moves parity-pinned values, so it
takes the next `C<n>` label in `test/parity/deltas.py` per
[ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).

## Entry points

- `rust/src/kernels/scalar_xs.rs` — `thermal_cross_section`.
- `rust/src/kernels/thermal_window.rs` — the window and break points.
- `hazma/relic_density/_thermal_functions.py` — the fallback used as the
  comparison.

## Risks / open questions

- The quadrature budget and the resonance ladder may both contribute, so
  the offset may not be a single constant error.
