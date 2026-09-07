# The pure-Python thermal averages return `0.0` for every `x >= 25`

- **Added:** 2026-09-06
- **Source:** PR #91 review round 1, while writing regression coverage for
  the two pure-Python `thermal_cross_section` sites
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. Independent of the quadrature-tolerance
  repair that surfaced it (roster entry `B6`), which is already landed.

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
