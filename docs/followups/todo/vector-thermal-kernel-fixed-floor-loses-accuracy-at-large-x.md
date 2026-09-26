# The vector thermal kernel loses 1e-4 of ⟨σv⟩ above `x = 200`

- **Added:** 2026-09-25
- **Source:** resolving
  `docs/followups/done/thermal-fallback-upper-limit-collapses-at-x-25.md`
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. It moves parity-pinned values, so it
  takes the next `C<n>` label under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).

## Why

`rust/src/kernels/vector_xs.rs` integrates ⟨σv⟩ over
`[2, max(50/x, 150)]`. The integrand decays as `e^{−x z}`, so at large
`x` all of it sits within a few `1/x` of `z = 2`. On a fixed interval
`[2, 150]`, QUADPACK's first Gauss–Kronrod nodes land in the tail and its
error estimate misjudges the peak.

Measured against the pure-Python fallback, which integrates to
`2 + 50/x` and agrees with a threshold-split `epsrel = 1e-12` reference
to 1e-13 there, `KineticMixing(mx=300, mv=200, gvxx=1, eps=1e-2)` gives:

```text
x          100      150      200      250      300
kernel/ref-1  +2e-16  -6e-14  +1.0e-4  +1.4e-4  +1.9e-4
```

scipy's QUADPACK on the same interval reproduces the error, so the port
is faithful and the interval is at fault. Above `x = 300` the kernel
clips `x` to 300, so this error is carried into every larger `x` too.
`KineticMixing(mx=100, mv=300, ...)` is unaffected (≤ 5e-11). Its
resonance break point at `z = 3` shortens the first QAGP piece, which is
the likely reason, but that has not been checked.

The scalar kernel's `max(50/x, 100)` did not show this at the two
`HiggsPortal` points (≤ 1e-7), but scipy on `[2, 100]` loses 1e-4 at
`x = 300` for the vector points, so the scalar kernel is exposed to the
same failure at other parameters.

## What

Change both kernels' upper limit to `2 + 50/x`, the limit the
pure-Python sites use; its derivation is in the docstring of
`hazma.relic_density._thermal_functions.thermal_cross_section_upper_limit`.
Then re-derive the moved corpus positions and declare them as the next
`C<n>` in `test/parity/deltas.py`, with a `CHANGELOG.md` entry.

## Entry points

- `rust/src/kernels/vector_xs.rs` — `max(50/x, 150)`, near the
  `THERMAL_LIMIT` call.
- `rust/src/kernels/scalar_xs.rs` — `max(50/x, 100)` in
  `thermal_cross_section`.
- `test/parity/cases.py::_thermal_blocks` — pins `x = 0.5` and `x = 1/3`
  specifically because they are the floors' switch points. Those anchors
  lose their meaning once the floors go.

## Risks / open questions

- The above-300 divergence (the scalar returns `0.0`, the vector clips
  and saturates) is a separate defect on the same path. It puts the
  pure-Python fallback's `relic_density` 8% off the vector kernel's at
  the two `KineticMixing` corpus points. Deciding it in the same change
  would save a second corpus re-derivation.
