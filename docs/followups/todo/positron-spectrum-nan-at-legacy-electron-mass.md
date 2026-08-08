# The mediator positron spectra return `nan` at exactly the legacy `MASS_E`

- **Added:** 2026-08-08
- **Source:** `projects/cython-to-rust/task-notes/phase-01/task-1.4-legacy-npy.md`
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** ripens **before Phase 05/06** of `cython-to-rust`,
  which port the mediator spectrum kernels. `rules.md` rule 4 (constants keep
  bit-parity through the port) decides how the port must treat this; deciding
  after the swap costs a second declared numerical change.

## Why

`hazma/_utils/legacy_parameters.pxd:18` carries `MASS_E = 0.510998928` (the
pre-2014 PDG value), while `hazma/_utils/constants.pxd:5` and
`hazma/parameters.py:50` both carry `0.5109989461`. That divergence is already
on the record — `projects/cython-to-rust/references/cython-inventory.md` §Bugs
item 3, governed by the bit-parity rule in
`projects/cython-to-rust/rules.md`. What was not on the record is that it has
an observable consequence, found while mapping the deleted `.npy` suites onto
corpus coverage in Task 1.4:

**Evaluated at exactly `0.510998928` MeV, the mediator positron spectrum
returns `nan`.** One point, not a window. Measured on `master` at `7a81ce4`:

```python
from hazma.scalar_mediator.scalar_mediator_positron_spec import dnde_decay_s
import numpy as np
pw = np.array([1.0, 0.0, 0.0])                 # e e only
dnde_decay_s(np.array([0.510998928]), 250.0, 125.0, pw, "total")   # -> nan
dnde_decay_s(np.array([0.5109989]),   250.0, 125.0, pw, "total")   # -> 0.0
dnde_decay_s(np.array([0.5109989461]),250.0, 125.0, pw, "total")   # -> 0.0
```

A 2,000,001-point sweep of `[0.5109988, 0.5109990]` finds exactly one `nan`,
at `0.510998928` — the kernel's own electron mass, i.e. its exact kinematic
threshold, where the boost integrand goes `0/0`. Away from that one value the
kernel returns `0.0` on both sides, so no ordinary grid hits it.

Two reasons this matters more than a one-point edge case usually would:

1. **The parity corpus does not pin it.** All three positron cases
   (`spectra.positron.muon`, `spectra.positron.charged_pion`,
   `mediator_spectra.scalar.positron.dnde_decay_s`) contain zero `nan` across
   19,610 pinned values — their grids straddle the electron mass but miss this
   point. A Rust port that lands anywhere else here passes the corpus.
2. **Users do hit it.** The deleted `test/scalar_mediator/data/sm_1/e_ps.npy`
   grid started at `0.510998928`, because it was generated with
   `np.geomspace(me, ...)` back when `hazma.parameters.electron_mass` *was*
   that value. Any saved grid or notebook from that era reproduces the `nan`.

## What

Decide which of the two constants the positron kernels should use, then act
once:

- **Consolidate** (`legacy_parameters.pxd` adopts `0.5109989461`) — moves
  published positron spectra by the mass ratio near threshold and relocates
  the singular point. This is the "separate, declared numerical change" that
  `rules.md` rule 4 reserves for after the port, so it should be sequenced
  against `docs/followups/todo/` items in the same family rather than folded
  into a swap PR.
- **Or keep bit-parity and fix the singularity** — return `0.0` at the
  threshold instead of `0/0`, which is the limit from both sides and changes
  no value anywhere else.

Either way, add the point to the parity corpus (`test/parity/cases.py`,
positron cases) so the port has something to reproduce, and record the outcome
in the project's "Numerical impact so far".

## Entry points

- `hazma/_utils/legacy_parameters.pxd:18` — `MASS_E = 0.510998928`
- `hazma/_utils/constants.pxd:5` — `MASS_E = 0.5109989461`
- `hazma/parameters.py:50` — `electron_mass = 0.5109989461`
- `hazma/scalar_mediator/scalar_mediator_positron_spec.pyx`,
  `hazma/vector_mediator/vector_mediator_positron_spec.pyx` — the kernels
- `test/parity/cases.py` — where the point would be pinned
- `projects/cython-to-rust/references/cython-inventory.md` §Bugs item 3 — the
  divergence this consequence belongs to
- `projects/cython-to-rust/rules.md` — Constants rule 1 (bit-parity first)

## Risks / open questions

- The vector kernel behaves identically —
  `dnde_decay_v(np.array([0.510998928]), 125.0, 125.0, pw, "total")` is `nan`
  and both neighbours are `0.0` — so the mechanism really is the shared
  constant, not one kernel's arithmetic. Other `(e_med, m_med, pw)`
  combinations were not swept; the fixing PR should confirm rather than
  assume the point does not move.
- Consolidating `MASS_E` almost certainly disturbs `BR_PI_TO_ENU`'s three
  spellings too (same inventory item); scope that deliberately.
