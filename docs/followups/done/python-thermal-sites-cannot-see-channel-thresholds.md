# The pure-Python thermal averages cannot see channel thresholds

- **Added:** 2026-09-29
- **Source:** PR #112 review, round 1
- **Scope:** cross-cutting
- **Status:** done — both sites build their interval and break points
  with
  `hazma.relic_density._thermal_functions.thermal_cross_section_partition`
  from `TheoryAnn.annihilation_thresholds()` and
  `annihilation_resonances()`.
- **Triggers / blockers:** none.

> **Resolved.** `TheoryAnn.annihilation_thresholds()` maps each final
> state to the center-of-mass energy at which it opens, empty by default,
> and `ScalarMediator`, `VectorMediator` and `VectorMediatorGeV` supply
> theirs. `thermal_cross_section_partition` mirrors
> `thermal_window::partition`: the integral runs to 100 decay lengths past
> the last threshold or resonance, with break points at each threshold
> and at 1, 4, 16 and 50 decay lengths past it. Resonances keep the width
> ladder `z_r ± (Γ/m_x) 4^k` rather than the kernels' single point at the
> peak. The GeV site takes only the thresholds of the channels it sums.
>
> **The three cases.** At `x = 30`, wrapped as `NoThermalCrossSection`,
> `HiggsPortal(mx=200, ms=550, gsxx=1)` now agrees with
> `converged_thermal_cross_section` to 1.3e-14 at `stheta = 0` and
> 2.3e-11 at `stheta = 1e-20`. The resonance case had already been
> repaired by the width ladder before this change; with the splits the
> fallback is within 3.5e-9 of that reference on the 400-point grid over
> `x` from 0.1 to 300, for `stheta` of 0, 1e-20 and 1e-4.
> `test/test_relic_density.py` pins all three.
>
> **The default.** A model that defines neither hook integrates over
> `[2, 2 + 100/x]`, split at decay lengths past threshold. That drops any
> channel opening past the window, as the `annihilation_thresholds`
> docstring states. The splits alone still improve it: at
> `HiggsPortal(mx=200, ms=550, gsxx=1, stheta=1e-4)` with both hooks
> hidden, the worst error on a 120-point grid falls from 2.6e-5 to
> 8.3e-8.

## Why

`hazma.relic_density._thermal_functions.thermal_cross_section`, the
generic fallback for a model without its own `thermal_cross_section`,
and the `VectorMediatorGeV.relic_density` closure in
`hazma/vector_mediator/_gev/thermal_cross_section.py` integrate ⟨σv⟩
over `[2, 2 + 100/x]` with `points=[2.0]`. The Rust kernels instead
build their interval and break points from each channel's threshold and
the mediator resonance (`rust/src/kernels/thermal_window.rs`). The
Python sites see only `annihilation_cross_sections(e_cm)`, so they
cannot.

Two failures follow, both measured on 2026-09-29 against
`test/parity/thermal_reference.py::converged_thermal_cross_section`
with `HiggsPortal(mx=200, ms=550, gsxx=1)` wrapped as
`NoThermalCrossSection` (`test/test_relic_density.py`):

- **A channel opening past the window is dropped.** At `x = 30`,
  `2 + 100/x = 5.33` lies below the `S S` threshold `z = 5.5`. With
  `stheta = 0` the fallback returns `0.0` against a reference
  `2.566e-52` MeV⁻². With `stheta = 1e-20` it returns the average 99.0%
  low. `2 + 50/x`, which `master` shipped before PR #112, fails the
  same way from `x = 14.3`.
- **A narrow resonance is missed at isolated `x`.** With no break point
  at `m_s/m_x`, QUADPACK's error estimate misses the peak. At
  `stheta = 1e-4` the fallback is 3.9e-4 off at `x = 0.891` and above
  1e-6 at `x = 0.223`, on a 400-point grid over `x` in 0.1 to 300.
  Everywhere else it sits near its `epsrel` of 1.49e-8.

## What

Give the Python sites the features the kernels use. Options include an
optional model hook that returns the `z` at which each channel opens,
with the mediator models supplying theirs; or a partition helper in
`_thermal_functions.py` mirroring `thermal_window::partition`, fed from
that hook. Keep the generic default for models that supply nothing.
Pin the three cases above in `test/test_relic_density.py`, and state in
`CHANGELOG.md` how far values move.

## Entry points

- `hazma/relic_density/_thermal_functions.py` —
  `thermal_cross_section_upper_limit` and `thermal_cross_section`.
- `hazma/vector_mediator/_gev/thermal_cross_section.py`.
- `rust/src/kernels/thermal_window.rs` — the rule to mirror.

## Risks / open questions

- The mediator models never reach the generic fallback, because they
  define `thermal_cross_section`. A hook helps user models only if they
  implement it, so the generic default still needs a documented limit.
