# The boost integral mis-covers its window at both ends

- **Added:** 2026-08-10
- **Source:** cython-to-rust Task 3.4 (the interp + boost port)
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** must land **after** Phase 06 Task 6.4 (the
  last Cython deletion). Repairing it before then would put the Rust and
  the Cython on different answers while both are alive, and the parity
  corpus — which pins the *current* values — would have to be regenerated
  in the same change, which `projects/cython-to-rust/rules.md` rule 2
  forbids for a tree with ported kernels.

## Why

`boost_integrate_linear_interp` sums whole interior cells with

```python
np.trapezoid(yy[ilow:ihigh], x=x[ilow:ihigh])
```

(`hazma/_utils/boost.pyx:216`). The slice is exclusive at the top, so the
sum covers the cells ending at `x[ihigh - 1]` and stops. The upper
partial-cell term that follows starts at `x[ihigh]`. Nothing covers
`[x[ihigh - 1], x[ihigh]]`.

The sharpest form: when the boosted window reaches past the table, `ub`
is clamped to `x[-1]` and `ihigh` becomes the last index, so the upper
partial-cell term is skipped entirely and **the table's final row
contributes to no term at all**. Replacing it with a value six orders of
magnitude larger leaves the answer bit-identical — checked against the
live Cython, `test/test_core_boost.py::TestDroppedInteriorCell`.

This is a real error in a published number, not a rounding artifact. On a
hand-computable case (`x = y = [1, 2, 3, 4]`, `beta = 0.6`, `E = 2.2`)
the routine returns `1.9 / (2γβ)` where the region it claims to integrate
is worth `2.9 / (2γβ)` — 34% low.

The same off-by-one read from the other side is far worse. When both
bounds land inside **one** cell, `ilow` is the node above `lb` and
`ihigh = ilow - 1` is the node below `ub`, so the two partial-cell terms
integrate `[lb, x[ilow]]` and `[x[ihigh], ub]` — which **overlap**, and
between them cover about two whole cells instead of the sliver between
the bounds. The over-count is the ratio of the cell width to the window
width, and the window width is `2Eγβ`, so it **diverges as the parent
slows down**. Measured against the live Cython on `x = y = [1..6]`,
`beta = 0.01`, `E = 3.5`: 53.497 returned against 3.500 intended, a
factor of 15.3, predicted exactly by the overlap arithmetic.

That regime is not hypothetical — it is the threshold region every model
spectrum passes through. All seven public tabulated photon spectra
diverge instead of converging to their own rest-frame spectrum as the
parent approaches rest. At `E_γ = m/10` and a parent one part in 1e12
above rest, against the same function evaluated exactly at rest:

| channel | at rest | one part in 1e12 above rest | ratio |
| --- | --- | --- | --- |
| `dnde_photon_eta` | 0.02313 | 767.2 | 33,000 |
| `dnde_photon_eta_prime` | 0.020022 | 130.35 | 6,500 |
| `dnde_photon_charged_kaon` | 0.0039628 | 38.625 | 9,700 |
| `dnde_photon_long_kaon` | 0.015173 | 148.6 | 9,800 |
| `dnde_photon_short_kaon` | 0.0060399 | 59.174 | 9,800 |
| `dnde_photon_omega` | 0.0089247 | 87.403 | 9,800 |
| `dnde_photon_phi` | 0.011033 | 71.716 | 6,500 |

(MeV⁻¹; `hazma` 2.1.0, this worktree, 2026-08-10.) The exact-rest column
is right because the callers short-circuit at `E − M < DBL_EPSILON` and
return the rest-frame spectrum directly; one ulp above that short circuit
the integral takes over and is wrong by three to four orders of
magnitude. Away from threshold the same defect shrinks to the gap case
above — one cell out of a wide window, systematically low.

The parity corpus pins these values, faithfully: its `rest_plus_eps`
block sits exactly in the divergent regime. That is the corpus doing its
job (it records what the Cython returns, not what is correct), and it is
why the repair has to regenerate the corpus in the same change.

Note `references/cython-inventory.md` already lists "off-by-one index
pairing in `boost_integrate_linear_interp_massive`" under *dead* code.
This is the same class in the **live** routine, which that audit did not
flag.

## What

Change the interior sum to include the cell ending at `ihigh` — most
directly by slicing `[ilow : ihigh + 1]` — and then re-derive the upper
partial-cell term so the two do not overlap. Note the lower end is
already contiguous (`ilow`'s partial cell ends exactly where the sum
begins), so only the upper end needs the work.

Fix the overlapping single-cell case in the same change: when
`ilow > ihigh`, neither partial-cell term should run whole — the answer
is the integral of one linear interpolant over `[lb, ub]`, which is a
third closed form rather than a repair of the other two. This is the
larger of the two errors by far and the reason the near-threshold limit
is wrong; both are the same off-by-one read from different sides and
neither should be fixed alone.

A cheap regression to add alongside: for each of the seven channels,
`dnde_photon_X(E, m * (1 + 1e-12))` must approach `dnde_photon_X(E, m)`.
That identity holds for no channel today and would hold for all seven
after the repair, so it is the natural acceptance test.

The change moves published numbers for the seven tabulated photon
spectra — `dnde_photon_{eta, eta_prime, charged_kaon, long_kaon,
short_kaon, omega, phi}` — and therefore for every model spectrum that
sums them. Quantify the shift on the corpus grids, state it in the PR
body and in `CHANGELOG.md`, and regenerate the parity corpus in the same
change (a deliberate, declared regeneration, which is the only kind rule 2
allows).

## Entry points

- `hazma/_utils/boost.pyx:206-241` — the interior sum and both partial
  cells.
- `rust/src/boost.rs` — `boost_integrate_linear_interp`, where the
  behavior is reproduced with the reasoning in its
  `# Faithfulness notes`.
- `test/test_core_boost.py::TestDroppedInteriorCell` — the pin to invert
  when this is fixed.
- `projects/cython-to-rust/task-notes/phase-03/task-3.4-interp-boost.md`
  — how it was found and the numbers above.
- `test/parity/tolerances.py` — the `TABULATED` budget class, which is
  what these seven cases are graded against.

## Risks / open questions

- **How big is it away from threshold?** Measured only as a ratio against
  an independent reference (the linear interpolant integrated on a dense
  grid), which is not a repair and does not reproduce every branch. Over
  a sweep of nine boost regimes and 300 energies per table, the returned
  value lands between 0.02× and 161× the reference, with the extremes at
  the smallest boosts and the well-boosted regimes close to 1. Redo this
  properly against the actual repair before quoting a figure in a
  CHANGELOG.
- **How far downstream does it reach?** The tabulated spectra feed
  branching-fraction-weighted sums in `hazma/theory/`, so every model
  `total_spectrum` inherits whatever these do near threshold — and
  threshold is exactly where an indirect-detection spectrum is
  interesting. Check whether any published figure, limit, or notebook in
  `docs/source/` or `notebooks/` sits in the affected region.
- **Does anything depend on the current behavior?** The parity corpus
  does, by construction, and it is regenerated as part of this work. Look
  for anything else pinned to a near-threshold tabulated spectrum before
  assuming the corpus is the only consumer.
