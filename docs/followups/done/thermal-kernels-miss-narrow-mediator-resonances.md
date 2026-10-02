# The Rust thermal kernels miss narrow mediator resonances

- **Added:** 2026-09-29
- **Source:** bracketing the resonance in the pure-Python thermal
  averages (now `thermal_cross_section_partition`)
- **Scope:** cross-cutting
- **Status:** done — `rust/src/kernels/thermal_window.rs::partition`
  takes each kernel's resonance as `(m/m_x, Γ/m_x)` and brackets it with
  the ladder `z_r ± (Γ/m_x) 4^k` instead of a break point at the peak,
  and each kernel's `limit` is `THERMAL_LIMIT` plus the break-point
  count. Both narrow points are pinned by
  `the_thermal_average_resolves_a_narrow_resonance` in `scalar_xs.rs`
  and `vector_xs.rs`. The repair takes no `C<n>` label, because it moves
  no corpus position past an existing declaration's budget.
- **Triggers / blockers:** none.

## Why

`rust/src/kernels/scalar_xs.rs` and `vector_xs.rs` integrate ⟨σv⟩ with
QAGP break points at `[2, m/m_x, 2 m/m_x]`. That puts the mediator peak
exactly on a subinterval endpoint. Gauss–Kronrod nodes are interior, so
once the resonance is narrower than the node spacing no node samples it,
and QUADPACK reports convergence on a partition that never saw the peak.

Measured against a reference split at `z_r ± (Γ/m_x) 2^k` and at decay
lengths, integrated to `epsrel = 1e-12`:

```text
model                                               Γ/m_x    x=0.5    x=1      x=5.818  x=20
HiggsPortal(mx=200, ms=550, gsxx=1e-2, stheta=1e-3)  3.5e-6  -2e-11   -4e-10   -1.0     -1.0
KineticMixing(mx=200, mv=550, gvxx=1e-2, eps=1e-3)   6.3e-6  -1.0     -1.0     -1.0     -0.97
```

The pure-Python sites used to share the endpoint placement's failure.
They now bracket each resonance with the ladder
`z_r ± (Γ/m_x) 4^k` from
`hazma.relic_density._thermal_functions.thermal_cross_section_partition`,
which holds both rows above to 8e-11 or better. Wide resonances of a
few percent are unaffected either way.

## What

Replace the kernels' `[2, ratio, 2 ratio]` with the same ladder. Both
kernels already receive the mediator width (`width_s`, `width_v`). The
ladder can add up to 70 points at `Γ/m_x ~ 1e-17` and `x = 0.01` (a
width below the floor recorded in
[`thermal-kernels-lose-accuracy-below-width-1e-13.md`](../todo/thermal-kernels-lose-accuracy-below-width-1e-13.md)),
so
`THERMAL_LIMIT` must grow with the point count, as the Python sites'
`limit=50 + len(points)` does. Keep `2 ratio` as well, since the kernels
open `xx → ss` and `xx → vv` there. Then re-derive the moved corpus
positions against `test/parity/thermal_reference.py`, which also places
its break point at the peak and needs the ladder first. Declare them as
the next `C<n>` in `test/parity/deltas.py`, with a `CHANGELOG.md` entry.

## Entry points

- `rust/src/kernels/scalar_xs.rs` — `let points = [2.0, ratio, 2.0 * ratio];`
  in `thermal_cross_section`, and `THERMAL_LIMIT`.
- `rust/src/kernels/vector_xs.rs` — the same two.
- `test/parity/thermal_reference.py::thermal_cross_section` — the
  reference's `points`.
- `hazma/relic_density/_thermal_functions.py::thermal_cross_section_partition`
  — the Python ladder and the measurements behind its ratio.

## Resolution

The kernels had moved to `thermal_window::partition` since this item was
filed, so the ladder lives there rather than at the two call sites the
entry points name. Against an independent reference split at decay
lengths and at `z_r ± (Γ/m_x) 2^k`, at `x` from 0.1 to 300:

- **`HiggsPortal(mx=200, ms=550, gsxx=1e-2, stheta=1e-3)`** was
  1.5e-4 low at `x = 0.1`, 1.2% low at `x = 0.5` and 87% low at
  `x = 2`. It is now within 3.0e-9 everywhere.
- **`KineticMixing(mx=200, mv=550, gvxx=1e-2, eps=1e-3)`** was 85% low
  at `x = 0.1`, kept 2.9e-3 of its value at `x = 0.5` and 2.6e-4 at
  `x = 1`, and was 2.1e-8 low at `x = 50`. It is now within 1.9e-9
  everywhere.
- **`HiggsPortal(mx=100, ms=300, gsxx=1, stheta=0.1)`**, whose width is
  5% of `m_x`, moves by at most 7.0e-9.

The failures fall at isolated `x`, not on the contiguous ranges the table
above reports, which came from a different grid.

Across the 570 positions of the two corpus thermal cases the kernels
move by at most 5.7e-10, and they stay within 1.8e-8 of the `B6+C7`
reference, the figure `C7` already records. `converged_thermal_cross_section`
in `test/parity/thermal_reference.py` gains the same ladder at ratio 2,
which moves it by at most 1.2e-14 there. No corpus block sits at a
narrow resonance, so the kernel tests above are what pin the repair.

## Risks / open questions

- The corpus's thermal blocks are at wide resonances, so the ladder will
  move them only at the 1e-8 level. A narrow-resonance block would pin
  the repair itself.
