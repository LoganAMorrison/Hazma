# `thermal_cross_section` returns its integrator's *initial estimate*

- **Added:** 2026-08-20
- **Source:** cython-to-rust Task 5.1
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. Task 5.3 ran the sweep (see
  "Downstream size" below); this is now free-standing work whose only
  constraint is that it moves published numbers.

## Why

Both mediator models' `thermal_cross_section` compute

```text
<sigma v>(x) = x / (2 K2(x))^2 * quad(integrand, 2, max(50/x, floor),
                                      points=[2, m/mx, 2 m/mx])[0]
```

and pass `scipy.integrate.quad` **neither** `epsabs` nor `epsrel`, so it
runs at scipy's defaults — `epsabs = 1.49e-8`, `epsrel = 1.49e-8`, and
the two are combined disjunctively: QUADPACK stops as soon as *either*
is met. The integrand's integral is of order `1e-27` for an ordinary
`KineticMixing` point, twenty decades below the absolute criterion. So
the absolute test is satisfied by the very first Gauss–Kronrod pass,
QUADPACK returns on its initial three-interval partition, and no
subdivision ever happens. `test/test_core_quad.py` already records the
partition (`neval = 63`, `last = 3`) from the other side; what had not
been measured is what it costs.

Measured on `mx = 100 MeV`, `mv = 300 MeV`, `gvxx = 1`, against the same
integrand and the same integrator at a convergent tolerance
(`epsabs = 0`, `epsrel = 1e-11`):

| `x = mx/T` | shipped | converged | relative error |
| --- | --- | --- | --- |
| 0.5 | 2.821030e-07 | 2.821031e-07 | 2.1e-07 |
| 1 | 1.061436e-06 | 1.061437e-06 | 1.3e-06 |
| 5 | 1.224095e-06 | 1.200201e-06 | **2.0e-02** |
| 20 | 1.597609e-08 | 1.610258e-08 | **7.9e-03** |
| 100 | 1.306081e-08 | 1.299293e-08 | **5.2e-03** |
| 300 | 1.344296e-08 | 1.281258e-08 | **4.9e-02** |

Below `x ~ 1` the integrand is large enough that the relative criterion
binds and the answer is good to six or seven digits. Above it — which is
the whole freeze-out region, `x ~ 20` — the answer is a **0.5% to 5%**
approximation, and it degrades as `x` grows. Freeze-out abundance goes
roughly as `1/<sigma v>`, so this propagates more or less linearly into
the relic density of every model that supplies one of the affected
`thermal_cross_section` implementations — see "Which models" below — and
into every coupling a user solves for by matching one.

This is shipped 2.1.0 behavior, unchanged by the Rust port: the port
reproduces the same quadrature settings and lands within 2.1e-14 of the
Cython at every corpus point.

## What

Pass tolerances that bind. The one-line version is `epsabs=0.0` at both
call sites, which makes `epsrel = 1.49e-8` the operative criterion and
costs a few hundred extra integrand evaluations per call — an integrand
that is now native Rust rather than a Python callback, so the cost is
smaller than it was when the code was written.

Two things to check before doing it:

- `epsabs = 0` makes QUADPACK's "tolerance unachievable" input check
  bind on `epsrel` alone (`epsrel >= max(50*eps, 0.5e-28)`), which
  `1.49e-8` satisfies comfortably. `rust/src/quad.rs` already
  implements that check.
- The relic-density solver calls this inside an ODE right-hand side, so
  a large cost increase would show up as wall-clock. Measure it; the
  benchmark Task 5.3 records is the natural place.

This moves published numbers by up to ~5%, so it is a `minor` bump at
least and needs a `CHANGELOG.md` entry naming the magnitude and the
affected surface (`relic_density`, and any limit that goes through a
thermally averaged cross section).

## Entry points

- `rust/src/kernels/vector_xs.rs` — `thermal_cross_section`, and
  `tests::the_thermal_integral_matches_a_composite_rule`, which measures
  the gap and is where the numbers above come from.
- `rust/src/kernels/scalar_xs.rs` — `thermal_cross_section`, the scalar
  twin, with the same inherited defaults. Ported by Task 5.2 on
  2026-08-21, so **both** mediator paths now carry this defect in Rust.
- `hazma/relic_density/_thermal_functions.py` — the live consumer, and
  at line 476 a **third** instance of the same defect (see
  "Which models").
- `hazma/vector_mediator/_gev/thermal_cross_section.py:143` — a
  **fourth**, in the GeV vector model. Neither pure-Python site was
  ported, so neither is covered by a Rust-side fix.
- `test/test_core_quad.py` — `TestLiveIntegrandShapes::test_thermal_cross_section_site`,
  which pins the initial-partition behavior this would change.
- Related project: `projects/cython-to-rust/` (Phase 05 Task 5.3)

## Downstream size (measured)

cython-to-rust Task 5.3 swept `relic_density` through both mediators on
six model points. The relevant fact is the shape of the dependence, not
a single number: relic abundance goes as `1 / <sigma v>`, so a 0.5%–5%
error on ⟨σv⟩ across the freeze-out region propagates roughly linearly
into the relic density of every affected model. It is by a wide margin
the largest of the project's known-wrong entries — four orders of
magnitude above the
1e-9-and-below drifts the port itself introduced — and it predates the
port entirely.

### Which models

`relic_density` accepts any model, and `_thermal_functions.
thermal_cross_section` short-circuits to `model.thermal_cross_section`
when the model defines one — so the blast radius is "models that supply
an affected implementation", not "everything".

- **Affected via the ported kernels:** the `ScalarMediator` and
  `VectorMediator` families — `HiggsPortal`, `HeavyQuark`,
  `KineticMixing`, `QuarksOnly`, and the base classes. These are what
  Task 5.3 measured.
- **Affected via pure Python, on paths this project never ported** —
  found while scoping this section, and the reason the defect is a
  four-site class rather than the two-site one the Entry points above
  imply: `hazma/relic_density/_thermal_functions.py:476` (the generic
  fallback used when a model defines no `thermal_cross_section`) and
  `hazma/vector_mediator/_gev/thermal_cross_section.py:143` both call
  `quad` with neither `epsabs` nor `epsrel`, exactly as the two `.pyx`
  did. A fix that touches only the Rust kernels leaves these two behind.
- **Not affected:** any model supplying its own unrelated
  `thermal_cross_section`, such as `test/test_relic_density.py`'s
  `ToyModel`.

Fixing it is therefore a `minor`-at-least numerical change with a
CHANGELOG line, not a silent repair, and
`test/test_relic_density.py::TestMediatorRelicDensity`'s twelve pinned
values move with it. Those pins are pre-port Cython values captured at
`14f1c66`; re-derive them from the corrected kernel rather than
loosening the tolerances.

## Risks / open questions

- Tightening the tolerance changes `last` and `neval` at the site, so
  `test_thermal_cross_section_site`'s `expected_last` values move with
  it. That test is about the break-point *filtering*, which is
  unaffected; re-derive the numbers rather than deleting the assertion.
- The `x > 300` clip means the saturated value moves too, and the
  scalar model short-circuits to `0.0` there instead — a separate
  divergence between the two models that this follow-up does not touch.
