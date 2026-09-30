# The Rust thermal kernels miss narrow mediator resonances

- **Added:** 2026-09-29
- **Source:** bracketing the resonance in the pure-Python thermal
  averages (`thermal_cross_section_break_points`)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none. It moves parity-pinned values, so it
  takes the next `C<n>` label under
  [ADR-0003](../../adrs/ADR-0003-corpus-repairs-are-declared-deltas.md).

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
`hazma.relic_density._thermal_functions.thermal_cross_section_break_points`,
which holds both rows above to 1.5e-6 or better. Wide resonances of a
few percent are unaffected either way.

## What

Replace the kernels' `[2, ratio, 2 ratio]` with the same ladder. Both
kernels already receive the mediator width (`width_s`, `width_v`). The
ladder can add up to about 60 points at `Γ/m_x ~ 1e-17`, so
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
- `hazma/relic_density/_thermal_functions.py::thermal_cross_section_break_points`
  — the Python ladder and the measurements behind its ratio.

## Risks / open questions

- The corpus's thermal blocks are at wide resonances, so the ladder will
  move them only at the 1e-8 level. A narrow-resonance block would pin
  the repair itself.
