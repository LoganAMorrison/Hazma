# The charged pion's `pi -> e nu` neutrino line is added twice

- **Added:** 2026-08-20
- **Source:** `projects/cython-to-rust/task-notes/phase-04/task-4.6-positron-pion-neutrino.md`
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open
- **Triggers / blockers:** **corpus re-pinning only** — no ordering
  constraint against cython-to-rust Phase 06. The Cython twin
  (`hazma/spectra/_neutrino/_pion.pyx`) is already gone, deleted in
  Task 4.6 in the same PR as the swap, so no later phase takes away
  anything this repair could have used. The corrected values need no
  Cython oracle either: the excess is a closed-form plateau
  (`BR_e / (2 γ β E_ν^rf)`) over a computable window, so the expected
  delta is a closed-form transform of the committed array. Sequenced
  alongside the other pinned defects in
  [`projects/parity-pinned-defect-repair/PLAN.md`](../../../projects/parity-pinned-defect-repair/PLAN.md).

## Why

`hazma.spectra.dnde_neutrino_charged_pion` sums two contributions that
were meant to partition the pion's decay modes:

```python
# hazma/spectra/_neutrino/_pion.pyx:196-200 (deleted in Task 4.6)
mu_nu = c_dnde_mu_numu_point(enu, epi)
e_nu = c_dnde_e_nue_point(enu, epi)
result.electron = mu_nu.electron + e_nu.electron
result.muon = mu_nu.muon + e_nu.muon
```

`c_dnde_e_nue_point` is the `pi -> e nu_e` line and nothing else. But
`c_dnde_mu_numu_point`, despite its name, **also** adds it:

```python
# hazma/spectra/_neutrino/_pion.pyx:112-114 (deleted in Task 4.6)
# Contribution from pi -> nu_e + e
enu_rf = two_body_energy(MASS_PI, 0.0, MASS_E)
delta_e = BR_PI_TO_E_NUE * boost_delta_function(enu_rf, enu, 0.0, beta)
```

so the electron-neutrino row carries `2 BR(pi -> e nu)` where physics
wants one. `rust/src/kernels/neutrino_pion.rs` reproduces it under
`projects/cython-to-rust/rules.md` rule 1.

The muon row is unaffected: `c_dnde_e_nue_point` writes nothing there, so
the `pi -> mu nu_mu` line appears exactly once. That asymmetry is what
makes the defect a transcription slip rather than a convention.

Measured on this tree (Task 4.6) at `E_pi = 400` MeV, by subtracting the
muon-decay continuum recomputed with `scipy.integrate.quad` over the
ported muon kernel:

| `E_nu` (MeV) | excess / one line's height | expected |
| --- | --- | --- |
| 20 | 2.0000 | 1 |
| 30 | 2.0000 | 1 |
| 50 | 2.0000 | 1 |

and the muon line's ratio is 1.0000 at the same points.

## What

Delete the `delta_e` term from the `pi -> mu nu_mu` half — in
`rust/src/kernels/neutrino_pion.rs` that is the `delta_e` binding in
`dnde_mu_numu` and its use in the returned `electron` field — leaving
`dnde_e_nue` as the sole source of the line. Then update the docs, the
two Rust unit tests and the Python test that currently pin the defect.

**Size of the change to published numbers.** Integrated over energy the
electron-neutrino yield falls from `BR_mu + 2 BR_e` to `BR_mu + BR_e`,
i.e. by `1.23e-4` per pion — 0.0123% of the row. Locally it is larger
where it matters: on the plateau the line occupies, the electron-neutrino
spectrum falls by 0.06% at `E_pi = 200` MeV and 0.036% at 1000 MeV
(measured). That is a *shape* change on a narrow band rather than a
normalization shift, which is the kind a limit calculation notices.

## Entry points

- `rust/src/kernels/neutrino_pion.rs` — `dnde_mu_numu`'s `delta_e`, and
  the module docs' "counted twice" paragraph
- `rust/src/kernels/neutrino_pion.rs` tests —
  `the_electron_line_is_counted_by_both_halves`,
  `the_boost_conserves_neutrino_number_per_flavor`
- `test/test_core_neutrino.py` —
  `TestPhysics.test_the_electron_line_is_counted_twice_and_the_muon_line_once`,
  `test_the_pion_yields_one_muon_neutrino_from_each_of_two_sources`, and
  the module docstring's "Two declared defects" section
- `test/parity/data/` — the boosted blocks of
  `spectra.neutrino.charged_pion`
- Downstream: every `hazma.spectra.dnde_neutrino_*` built on the charged
  pion, and `hazma/spectra/_nbody.py`'s neutrino path

## Risks / open questions

- **Is the doubled line the only place this pattern appears?** The
  positron sibling `hazma/spectra/_positron/_pion.pyx` adds its `pi -> e
  nu` line exactly once, in one place, so it does not share the defect —
  checked at Task 4.6. Worth re-checking the two mediator positron
  spectrum modules when Phase 06 ports them.
- **The repair moves a published number**, so it needs a `CHANGELOG.md`
  entry stating the 0.0123% integrated / 0.06% local figures, and is
  `minor` at least under `docs/versioning.md`.
- A pion **at rest** loses *both* prompt lines instead (the
  `E - m < DBL_EPSILON` branch returns only the muon-decay continuum).
  That is a separate rest-frame-branch question, of the same family as
  [`rho-rest-frame-branch-returns-the-integrand.md`](rho-rest-frame-branch-returns-the-integrand.md),
  and is not filed separately because a delta function has no rest-frame
  representation in this API at all — deciding what it *should* return is
  a design question, not a transcription fix.
