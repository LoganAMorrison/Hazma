# The pure-Python thermal averages return `0.0` for every `x >= 25`

- **Added:** 2026-09-06
- **Source:** PR #91 review round 1, while writing regression coverage for
  the two pure-Python `thermal_cross_section` sites
- **Scope:** cross-cutting
- **Status:** done — both sites integrate to
  `hazma.relic_density._thermal_functions.thermal_cross_section_upper_limit`.
- **Triggers / blockers:** none. Independent of the quadrature-tolerance
  repair that surfaced it (roster entry `B6`), which is already landed.

> **Resolved.** Both pure-Python sites now integrate to `2 + 50/x`, not
> to either Rust kernel's `max(50/x, floor)`. The open question below,
> what floor to use, came out as "none": the limit should scale with the
> decay length `1/x` rather than sit at a constant.
>
> **Why not a floor.** The kernel `z² (z² − 4) K₁(x z)` decays as
> `e^{−x z}`, so `2 + 50/x` cuts it where the Bessel argument is 50 past
> its threshold value `2x`. Measured at 40 digits for `x` from 0.01 to
> 300, the dropped tail is at most 1.4e-17 of the kernel's integral. A
> constant limit is instead too long at large `x`: QUADPACK's first
> Gauss–Kronrod nodes on `[2, 100]` or `[2, 150]` sit in the tail, and
> against a split `epsrel = 1e-12` reference over the four mediator
> points `max(50/x, 150)` is up to 1.9e-4 off for `x ≥ 200`, and
> `max(50/x, 100)` is 1.0e-4 off at `x = 300`. `2 + 50/x` stays within
> 8.6e-9 of a reference split at every channel threshold, across `x`
> from 1 to 300. The vector kernel inherits that
> error; it is
> [its own follow-up](../todo/vector-thermal-kernel-fixed-floor-loses-accuracy-at-large-x.md).
>
> **What moved.** Above `x = 25` both sites returned zero. They were also
> already truncating below it: at `x = 20` the `50/x` cut lost 19% of
> ⟨σv⟩ at `HiggsPortal(mx=100, ms=300, gsxx=1, stheta=0.1)`, and at
> `x = 24` it lost 32% to 100% across the four mediator points and 24%
> for the GeV model. Through the generic fallback, `relic_density` at
> that point moves from 27.19 to 26.67 semi-analytically and from 35.64
> to 34.42 by the Boltzmann solve. At the other three mediator points it
> falls by three to six decades: 4.2e-3 to 9.36e-8, 1.41e-3 to 6.64e-7
> and 4.1e-3 to 6.25e-9 semi-analytically, and 4.4e-3 to 9.83e-8,
> 1.54e-3 to 6.89e-7 and 4.1e-3 to 6.47e-9 by the Boltzmann solve. It
> now matches the Rust scalar kernel to 7.6e-8 and 3e-14 at the two
> `HiggsPortal` points.
>
> **The GeV `nan` was fully this defect.** With `50/x < 2` the integrand
> is evaluated below threshold, where the GeV cross sections are `nan`,
> not zero. The model above now gives `relic_density` = 1.654e-4
> semi-analytically and 1.721e-4 by the Boltzmann solve.
>
> The third divergence named in "Risks" is still open: above `x = 300`
> the fallback and the scalar kernel return `0.0`, while the vector
> kernel holds its `x = 300` value. That alone puts the fallback's
> `relic_density` 8% off the vector kernel's at the two
> `KineticMixing` points, although the two agree on ⟨σv⟩ to 1e-7 below
> `x = 300`.

## Why

Both pure-Python thermal averages integrate from `2` to `50 / x`:

- `hazma/relic_density/_thermal_functions.py:490`
- `hazma/vector_mediator/_gev/thermal_cross_section.py:150`

`50 / x` falls to `2` at `x = 25` and below it thereafter, so the
interval closes and then inverts. QUADPACK is given `b < a`, returns the
negated integral over the reversed interval, and the answer is a signed
zero:

```text
x=    24  upper=50/x= 2.083  fallback=  8.48957e-22
x=    25  upper=50/x= 2.000  fallback=  0.00000e+00
x=    26  upper=50/x= 1.923  fallback= -0.00000e+00
x=    50  upper=50/x= 1.000  fallback= -0.00000e+00
x=   100  upper=50/x= 0.500  fallback= -0.00000e+00
```

(measured with a `HiggsPortal(mx=100, ms=300, gsxx=1, stheta=1e-1)`
wrapped so that it exposes `annihilation_cross_sections` but no
`thermal_cross_section`, which is what routes it through the generic
fallback.)

**Freeze-out sits at `x ~ 20`–`30`**, so this silently zeroes ⟨σv⟩ across
most of the region the quantity exists to describe. Relic abundance goes
as `1 / <sigma v>`, so a zero there does not merely perturb the answer —
it makes it singular. That is the immediate cause of

```python
VectorMediatorGeV(mx=5e3, mv=2e3, gvxx=1.0, gvuu=3.0, gvdd=1.0, gvss=-1.0,
                  gvee=0.0, gvmumu=0.0, gvveve=0.0, gvvmvm=0.0, gvvtvt=0.0
).relic_density(semi_analytic=True, three_body=False, four_body=False)
```

returning `nan`, which it does both before and after `B6` — the two are
independent defects that happen to live on the same two lines.

The two Rust kernels do **not** have this: they integrate to
`max(50 / x, floor)` with `floor` 100 (scalar) and 150 (vector), so their
interval never closes. The `.pyx` those were ported from carried the same
floor. The pure-Python pair simply never had one.

## What

Give both Python sites the floor their Rust counterparts already have,
then re-derive whatever moves.

The value of the floor is the open question, not whether there should be
one. The Rust kernels disagree with each other (100 vs 150) and neither
number is derived anywhere; both are inherited from the `.pyx`. The
integrand decays like `e^{-x z}`, so any floor comfortably past the decay
length is numerically equivalent — but "comfortably past" needs stating
with a measurement rather than copying a magic number a third time.

Fixing this **moves published numbers** for every model that reaches
either site, and it is the more consequential of the two defects on these
lines: `B6` corrected ⟨σv⟩ by up to 100%, whereas this one replaces a
hard zero. Expect a `minor` bump and a `CHANGELOG.md` entry, and note
that the affected surface is *not* the `ScalarMediator` /
`VectorMediator` families — those define their own
`thermal_cross_section` and short-circuit the fallback. It is the GeV
vector model plus any model that supplies `annihilation_cross_sections`
and no `thermal_cross_section`.

## Entry points

- `hazma/relic_density/_thermal_functions.py:490` — the generic fallback.
- `hazma/vector_mediator/_gev/thermal_cross_section.py:150` — the GeV
  vector model's own.
- `rust/src/kernels/scalar_xs.rs`, `rust/src/kernels/vector_xs.rs` — the
  `max(50/x, floor)` the Python pair is missing, and the 100-vs-150
  divergence.
- `test/test_relic_density.py::TestThermalQuadratureConverges` — pins
  both Python sites, deliberately below `x = 25` because above it there
  is no integral to check. Widen its grid when this is fixed.
- Related: `docs/followups/done/thermal-cross-section-quadrature-never-converges.md`
  (`B6`), the tolerance defect on the same two lines.

## Risks / open questions

- The scalar and vector Rust kernels also diverge above `x = 300` (the
  scalar returns `0.0`, the vector clips and saturates). That is a third,
  separate divergence on this code path, recorded in `B6`'s notes and
  untouched by either.
- Whether the GeV `nan` is *fully* explained by this or whether the
  solver has its own issue should be confirmed by re-running the model
  above once a floor is in place, rather than assumed.
