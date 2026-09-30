# The vector thermal kernel loses 1e-4 of ⟨σv⟩ above `x = 200`

- **Added:** 2026-09-25
- **Source:** resolving
  `docs/followups/done/thermal-fallback-upper-limit-collapses-at-x-25.md`
- **Scope:** cross-cutting
- **Status:** done. Repaired as parity roster entry `C7`, the seventh
  label issued under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).
- **Triggers / blockers:** none.

> **Resolved.** Both kernels and both pure-Python sites integrate to
> `2 + 100/x`, not the `2 + 50/x` "What" names: at one corpus point that
> limit truncates a real 1.2e-7 of the integral. See "Resolution
> (measured)" below.

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

## Resolution (measured)

**Why not `2 + 50/x`.** The docstring's tail bound, 1.4e-17 of the
Bessel kernel, only holds for a cross section that grows by fewer than
eight decades across the tail. At the corpus's scalar
`narrow_resonance` point, `HiggsPortal(mx=200, ms=550, gsxx=1,
stheta=1e-4)`, the `S S` channel opens at `z = 5.5` about fifteen
decades above the `stheta`-suppressed channels. Near `x = 12.6` the cut
`2 + 50/x = 5.96` falls just past that threshold and drops 1.17e-7 of
⟨σv⟩, beyond the kernels' `epsrel`. Scipy at `epsrel = 1e-12` on
`[2, 2 + 50/x]` reproduces the loss, and it is the same whatever break
points are passed, so it is truncation rather than quadrature error.
The kernels' fixed floors had covered it.

At `2 + 100/x` the Bessel kernel's tail is at most 3.0e-38 of its
integral, measured at 30 digits for `x` from 0.01 to 300, so a cross
section would have to grow by thirty decades to reach `epsrel`. All
four sites share the limit through
`hazma.relic_density._thermal_functions.thermal_cross_section_upper_limit`,
whose docstring carries the derivation.

**Schemes compared.** Each was run with scipy at the kernels' settings
(`epsrel = 1.49e-8`, `epsabs = 0`, `limit = 100`) against a converged
reference. The table gives the worst relative error per block:

| block | `max(floor, 50/x)` | `2 + 50/x` | `2 + 100/x` |
| --- | --- | --- | --- |
| scalar open_resonance | 1.8e-8 | 2.7e-8 | 1.8e-8 |
| scalar narrow_resonance | 6.2e-11 | 1.2e-7 | 7.9e-11 |
| scalar closed_resonance | 8.4e-12 | 8.5e-12 | 8.5e-12 |
| vector open_resonance | 3.5e-9 | 2.4e-9 | 3.5e-9 |
| vector narrow_resonance | 3.0e-10 | 3.0e-10 | 3.0e-10 |
| vector closed_resonance | 1.9e-4 | 3.9e-10 | 3.9e-10 |

The reference is split at `2 + k/x` for `k` in 0, 1, 4, 16, 50, 100 and
200, and at every channel threshold and mediator feature, each piece to
`epsrel = 1e-12`. At the scalar `narrow_resonance` point it agrees with
a variant that also brackets the resonance at `±k Γ/m_x` to 2e-15.
Keeping the fixed floor and adding break points at `2 + k/x` also
works, but only with `k` reaching 50: with `k` in 1 and 10 the piece
past the last point is still long, and scalar `closed_resonance` loses
1.2e-3. The limit is the one-line change.

**The vector's `closed_resonance` point, why it alone failed.** Its
break points `m_v/m_x = 0.67` and `1.33` fall below threshold, so QUADPACK
sees one interval `[2, 150]`. The `open_resonance` point's break point
at `z = 3` shortens the first piece to `[2, 3]`, which is why the
followup measured it unaffected.

**Corpus.** C7 changes the partition behind all 540 positions the two
thermal cases integrate, and the value at 518 of them. Only 16 move by
more than B6's 1e-7 budget, all in vector `closed_resonance`, from
`x = 209.9` up: the B6 value sat 1.09e-4 high there, 1.77e-4 at
`x = 286.8`, and 1.89e-4 at the clip and beyond. Everywhere else the
change is at most 1.8e-8 relative.

B6's reference integrated the kernels' own interval, so it shared the
defect: it too was 1.9e-4 high above `x = 200`. `thermal_reference.py`
keeps that integral as B6's base and adds `interval_term`, the
decay-length-split integral minus it, as C7's `Additive`. All six arrays
are declared as `B6+C7`, whose prediction moves 540 of their 570
positions, all but the 30 scalar zeros above `x = 300`. The rebuilt
kernels sit within 1.8e-8 of it everywhere, worst at scalar
`open_resonance`, `x = 14.8`; the budget stays B6's 1e-7.

**Oracles and invariants.** `vector_xs.rs`'s
`the_thermal_average_resolves_its_peak_at_large_x` holds the kernel at
`x = 300` and the `closed_resonance` couplings to a decay-length-split
quad at 1e-7. It fails at 1.9e-4 with the old floor.
`scalar_xs.rs`'s `the_thermal_average_keeps_a_channel_far_above_threshold`
sets the Standard Model couplings to 1e-6, where `2 + 50/x` loses 1.6e-4
at `x = 12`. It fails with that limit and lands 6.2e-12 from its oracle
at `2 + 100/x`. The vector's Simpson test keeps its 4.2e-8 residual on
the new interval. Neither kernel flags an unconverged position at
`THERMAL_LIMIT = 100` any more, down from 16; at 50, 15 would.

**Downstream.** `TestMediatorRelicDensity`'s `vector.closed_resonance`
pins rise by 2.65e-5 (semi-analytic) and 2.50e-5 (Boltzmann) and were
re-derived. The other five pins move by at most 8.2e-9, inside their
budgets. Moving the pure-Python sites from `2 + 50/x` to `2 + 100/x`
changes the generic thermal average by at most 1.3e-7 at the six
corpus points, and its `relic_density` by at most 9.3e-9;
`VectorMediatorGeV.relic_density` does not move.

**Not decided here.** The above-300 divergence under "Risks" is
unchanged, and is
[its own follow-up](../todo/thermal-kernels-disagree-above-x-300.md).
