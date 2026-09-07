# The charged pion's `pi -> e nu` neutrino line is added twice

- **Added:** 2026-08-20
- **Source:** `projects/cython-to-rust/task-notes/phase-04/task-4.6-positron-pion-neutrino.md`
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open — **repaired** 2026-09-06 as roster entry B5,
  `projects/parity-pinned-defect-repair` Task 10a. The file stays here
  until that project's close (Task 12) moves all eight of its follow-ups
  to `done/` in one sweep, so the inbound references are repointed once.
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
# hazma/spectra/_neutrino/_pion.pyx:196-200 at ed1fa20 (deleted in Task 4.6)
mu_nu = c_dnde_mu_numu_point(enu, epi)
e_nu = c_dnde_e_nue_point(enu, epi)
result.electron = mu_nu.electron + e_nu.electron
result.muon = mu_nu.muon + e_nu.muon
```

`c_dnde_e_nue_point` is the `pi -> e nu_e` line and nothing else. But
`c_dnde_mu_numu_point`, despite its name, **also** adds it:

```python
# hazma/spectra/_neutrino/_pion.pyx:112-114 at ed1fa20 (deleted in Task 4.6)
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
ported muon kernel. The repair turns every `2.0000` below into `1.0000`,
which `test/test_core_neutrino.py`'s
`TestPhysics::test_each_prompt_line_is_counted_exactly_once` now asserts
at the same three energies:

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

**Done, 2026-09-06.** The kernel carries one line per decay mode; the
module docs carry a `# The prompt electron-neutrino line is added once,
not twice` section in the shape `scalar_decay_photon.rs` uses for the
same kind of deliberate divergence from the `.pyx`; and
`test/parity/deltas.py` declares the 215 corpus positions it moves as
roster entry `B5`, leaving `test/parity/data/` untouched.

**Size of the change to published numbers.** Integrated over energy the
electron-neutrino yield falls from `BR_mu + 2 BR_e` to `BR_mu + BR_e`,
i.e. by `1.23e-4` per pion — 0.0123% of the row. That figure held when
Task 10a measured it.

The local figure this paragraph used to give did not, and it was three
orders of magnitude too small: it said 0.06% at `E_pi = 200` MeV and
0.036% at 1000 MeV, from points on the low-energy plateau. Above the
muon-decay continuum's support the doubled line is the *entire*
spectrum, so the repair halves it exactly — a relative drop of
`0.500000000000`, at 6 of the 174 moved points of a 2001-point
`geomspace(1e-4, 1e5)` sweep at `E_pi = 200` MeV and at 241 of 824 at
`E_pi = 5000` MeV. The median drop over the moved points is 4.7e-5 to
1.1e-4. So it is a shape change on a band, as this paragraph said, but a
much sharper one than it claimed.

Those factor-of-two positions turned out to expose a second, larger and
wholly separate defect — the continuum's quadrature loses its support and
returns a hard zero, which is why the line was all that was left there.
Filed as
[`neutrino-pion-continuum-loses-its-quadrature-support.md`](neutrino-pion-continuum-loses-its-quadrature-support.md).

## Entry points

Named as they stand after the repair; the pre-repair names are in
`projects/parity-pinned-defect-repair/task-notes/task-10a-neutrino-pion-line.md`.

- `rust/src/kernels/neutrino_pion.rs` — `dnde_mu_numu`, and the module
  docs' "added once, not twice" section
- `rust/src/kernels/neutrino_pion.rs` tests —
  `the_electron_line_comes_from_one_half_only` (renamed from
  `the_electron_line_is_counted_by_both_halves`),
  `the_boost_conserves_neutrino_number_per_flavor`
- `test/test_core_neutrino.py` —
  `TestPhysics.test_each_prompt_line_is_counted_exactly_once` (renamed
  from `test_the_electron_line_is_counted_twice_and_the_muon_line_once`),
  `test_the_pion_yields_one_muon_neutrino_from_each_of_two_sources`,
  `reference_dnde_neutrino_charged_pion`, and the module docstring's
  declared-defect section
- `test/parity/deltas.py` — the `B5` declaration and its closed-form term
- `test/parity/data/` — the boosted blocks of
  `spectra.neutrino.charged_pion`, unchanged and staying that way
- Downstream: only `hazma.spectra.dnde_neutrino_charged_pion` moves, plus
  `hazma/spectra/_nbody.py`'s `dnde_neutrino` for any final state
  containing one. The other `dnde_neutrino_*` are interpolated off
  committed tables rather than composed from the pion at runtime, and
  were verified identical.

## Risks / open questions

- **Is the doubled line the only place this pattern appears?** The
  positron sibling `hazma/spectra/_positron/_pion.pyx` adds its `pi -> e
  nu` line exactly once, in one place, so it does not share the defect —
  checked at Task 4.6. Worth re-checking the two mediator positron
  spectrum modules when Phase 06 ports them.
- **The repair moves a published number**, so it needed a `CHANGELOG.md`
  entry and is `minor` at least under `docs/versioning.md`. Written under
  `[Unreleased]` by Task 10a, with the 0.0123% integrated figure and the
  measured local one — a factor of two, not the 0.06% this file
  originally predicted.
- A pion **at rest** loses *both* prompt lines instead (the
  `E - m < DBL_EPSILON` branch returns only the muon-decay continuum).
  That is a separate rest-frame-branch question, of the same family as
  [`rho-rest-frame-branch-returns-the-integrand.md`](rho-rest-frame-branch-returns-the-integrand.md),
  and is not filed separately because a delta function has no rest-frame
  representation in this API at all — deciding what it *should* return is
  a design question, not a transcription fix.
