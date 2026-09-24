# The boost integral mis-covers its window at both ends

- **Added:** 2026-08-10
- **Source:** cython-to-rust Task 3.4 (the interp + boost port)
- **Scope:** cross-cutting
- **Status:** done — repaired as roster entry A1 of
  `projects/parity-pinned-defect-repair`, Task 4
  ([PR #93](https://github.com/LoganAMorrison/Hazma/pull/93), merge
  `46ec629`, 2026-09-07); moved to `done/` at that project's close (Task 12,
  2026-09-24). Measurement:
  `projects/parity-pinned-defect-repair/task-notes/task-4-boost-window.md`.
- **Triggers / blockers:** none remain. The deadline was on the oracle,
  not on the fix: the parity corpus pins the shipped values, and
  `projects/cython-to-rust/rules.md` rule 2 forbids regenerating them
  from a tree with ported kernels, so the corrected reference values had
  to come from the Cython twin `hazma/_utils/boost.pyx` before
  `cython-to-rust` Task 6.4 deleted it (`f479b231`). They did:
  `projects/parity-pinned-defect-repair` Task 2 patched that `.pyx` in a
  scratch build, drove it through its `__pyx_capi__` capsules, and
  committed the result as `test/parity/oracles/data/A1.npz`. Task 4 then
  repaired `rust/src/boost.rs` and declared the move against the
  committed corpus arrays rather than regenerating them, as
  [`projects/parity-pinned-defect-repair/PLAN.md`](../../../projects/parity-pinned-defect-repair/PLAN.md)
  sequences it.

## Why

`boost_integrate_linear_interp` sums whole interior cells with

```python
np.trapezoid(yy[ilow:ihigh], x=x[ilow:ihigh])
```

(`hazma/_utils/boost.pyx:216` at `f479b231^`, the last revision before
`cython-to-rust` Task 6.4 deleted it). The slice is exclusive at the top, so the
sum covers the cells ending at `x[ihigh - 1]` and stops. The upper
partial-cell term that follows starts at `x[ihigh]`. Nothing covers
`[x[ihigh - 1], x[ihigh]]`.

The sharpest form: when the boosted window reaches past the table, `ub`
is clamped to `x[-1]` and `ihigh` becomes the last index, so the upper
partial-cell term is skipped entirely and **the table's final row
contributes to no term at all**. Replacing it with a value six orders of
magnitude larger leaves the answer bit-identical — checked against the
live Cython, and now inverted in
`test/test_core_boost.py::TestWindowCoverage`.

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
job (it records what the Cython returns, not what is correct). The repair
therefore landed as a declared delta against those arrays rather than as
a regeneration — see
[`projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md`](../../../projects/parity-pinned-defect-repair/adrs/ADR-0001-corpus-repairs-are-declared-deltas.md).

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
body and in `CHANGELOG.md`, and declare it in `test/parity/deltas.py` in
the same change. The corpus itself is not regenerated: `test/parity/generate.py`
refuses to run once `hazma._core` serves a kernel, and ADR-0001 keeps the
committed arrays as the record of what 2.1.0 shipped.

## Entry points

- `hazma/_utils/boost.pyx:206-241` at `f479b231^` — the interior sum and
  both partial cells in the pre-port source, which `cython-to-rust`
  Task 6.4 deleted in `f479b231`.
- `rust/src/boost.rs` — `boost_integrate_linear_interp`, where the
  behavior is reproduced with the reasoning in its
  `# Faithfulness notes`.
- `test/test_core_boost.py::TestWindowCoverage` — the pin, inverted: it
  now asserts both hand-computable cases at their correct values.
- `projects/cython-to-rust/task-notes/phase-03/task-3.4-interp-boost.md`
  — how it was found and the numbers above.
- `test/parity/tolerances.py` — the `TABULATED` budget class, which is
  what these seven cases are graded against.
- `test/test_core_photon_tables.py::TestPhysics::test_a_barely_moving_parent_converges_to_its_rest_frame_spectrum`
  — the acceptance test this file proposed, over all seven channels. It
  pinned the divergence through a public entry point when the seven
  tabulated spectra moved to Rust (Task 4.2), and now pins the limit.
- Sibling defects of the same class:
  [`positron-muon-spectrum-normalization-inverted.md`](positron-muon-spectrum-normalization-inverted.md),
  [`eta-prime-two-photon-line-missing-factor-two.md`](eta-prime-two-photon-line-missing-factor-two.md),
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](phi-photon-lines-use-the-daughter-meson-energy.md).
  Each was repaired as its own declared delta; none needed a corpus
  regeneration, and after Phase 04 Task 4.1 none could have had one.

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
  does, by construction; it keeps the shipped values and the repair is
  declared against them in `test/parity/deltas.py`. Look
  for anything else pinned to a near-threshold tabulated spectrum before
  assuming the corpus is the only consumer.
