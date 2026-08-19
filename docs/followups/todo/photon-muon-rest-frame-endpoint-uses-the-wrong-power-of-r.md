# The muon photon spectrum's rest-frame branch stops 0.25 MeV short of the endpoint

- **Added:** 2026-08-16
- **Source:** `projects/cython-to-rust/` Phase 04 Task 4.3 — porting
  `hazma/spectra/_photon/_muon.pyx` to Rust and writing the boost-integral
  identity the original never asserted
  (`rust/src/kernels/photon_muon.rs`,
  `the_in_flight_form_is_the_boost_integral_of_the_rest_frame_form`)
- **Scope:** commit (one guard in one `.pyx`, plus the corpus block it pins)
- **Status:** open
- **Triggers / blockers:** **fix BEFORE cython-to-rust Phase 06
  Task 6.4** — the constraint is a deadline, not a wait. The parity
  corpus does pin the truncated values, so the repair needs corrected
  reference values before it can pass the gate that governs the remaining
  kernel swaps (`projects/cython-to-rust/rules.md` rules 1–2). But
  Task 6.4 is where `hazma/spectra/_photon/_muon.pyx` is **deleted**, and
  that twin is the only independent implementation a corrected corpus case
  can be re-pinned from: fix the `.pyx`, drive it through its
  `__pyx_capi__` capsules, and the corrected
  values come from a compiler and a source tree that both predate the
  Rust port. After Task 6.4 the only remaining source of corrected values
  is the fixed Rust itself, which pins the port against its own answer —
  exactly the vacuous gate `projects/cython-to-rust/rules.md` rule 2
  exists to prevent. The window is open today and closes at Task 6.4.
  Sequenced in
  [`projects/parity-pinned-defect-repair/PLAN.md`](../../../projects/parity-pinned-defect-repair/PLAN.md);
  where a later section of this file still reads "after Task 6.4", that
  wording is superseded and the plan is authoritative.

## Why

`hazma/spectra/_photon/_muon.pyx:41` guards the muon-rest-frame branch
with

```python
if y <= 0.0 or y >= 1.0 - MASS_E / MASS_MU:
    return 0.0
```

in the scaled variable `y = 2 E_γ / m_μ`. The kinematic endpoint of the
radiative decay is `E_γ,max = (m_μ² − m_e²)/(2 m_μ)`, i.e. `y = 1 − r`
with `r = (m_e/m_μ)²` — the very `r` the same function defines two lines
above (`hazma/spectra/_photon/_muon.pyx:39`). The guard uses `1 − √r`
instead.

Three independent statements inside hazma agree that `1 − r` is the right
edge and the guard is the outlier:

- `dnde_photon_muon_point` in the *same file*
  (`hazma/spectra/_photon/_muon.pyx:88`) tests
  `x >= (1.0 - r) / (1.0 - beta)`, so the in-flight branch already uses
  `1 − r`;
- `hazma/spectra/_photon/_pion.pyx:16` hard-codes
  `ENG_GAM_MAX_MURF = 52.82795006985128`, commented as the maximum photon
  energy in the muon rest frame and equal to `m_μ(1 − r)/2` over the
  legacy mass table;
- the in-flight closed form is exactly the boost integral of the
  rest-frame distribution taken over `(0, 1 − r)`. Task 4.3 measured that:
  with the endpoint at `1 − r` the two agree to **machine precision**
  (relative difference 1e-15 or exactly 0 wherever the boost window is not
  truncated, `emu` from 110 to 1500 MeV); with the shipped `1 − √r` cut
  they disagree by 3.2e-6.

The consequence is a step, not a taper. Over
`52.5736877769 MeV < E_γ < 52.8279515698 MeV` — 0.2543 MeV, or 0.48% of
the endpoint — `dnde_photon_muon(E, m_μ)` returns exactly `0.0` while the
spectrum there runs from `5.34e-7 MeV⁻¹` down to zero. A muon
infinitesimally off rest takes the other branch and returns
`5.335612532537976e-07` at the cut where the rest-frame branch returns
`0.0`, so the published spectrum is discontinuous in the parent energy at
`E_μ = m_μ`.

Size of the error, for whoever prices the fix: `5.45e-8` photons per decay
are lost, which is `1.1e-6` of the photons above 1 MeV, `4.2e-6` of those
above 10 MeV, and `7.2e-6` of the radiated energy above 1 MeV (all by
`scipy.integrate.quad` over the corrected rest-frame form). Small, and a
hard zero in a region where the spectrum is not zero is still wrong — and
it is the *discontinuity* rather than the yield that would bite a user
scanning `E_μ` through the muon mass.

## What

Change the guard at `hazma/spectra/_photon/_muon.pyx:41` to
`y >= 1.0 - r` and drop the corresponding `Y_MAX` constant from
`rust/src/kernels/photon_muon.rs` in favour of the `ONE_MINUS_R` already
beside it. The Rust module carries a test
(`the_two_branches_disagree_about_the_rest_frame_endpoint`) written
specifically to be *inverted* by this fix: it asserts the shipped gap and
pins its size, so repairing the guard turns it red and the repair has a
ready-made regression test in
`rest_frame_to_the_true_endpoint`, which is the corrected form already
transcribed in that module's test block.

Ordering, because the `.pyx` is a Phase 06 casualty:

1. If this lands **before** Task 6.4, fix both the `.pyx` and the Rust
   guard, and regenerate the `spectra.photon.muon` corpus case — which
   rule 2 forbids from a tree where any kernel runs on Rust, so in
   practice it cannot.
2. If it lands **after**, only the Rust guard exists and the corpus
   regeneration is the single blocking step, shared with the four other
   defects below.

Also on the same visit: `test/test_core_photon_muon.py`'s
`test_a_muon_at_rest_stops_at_the_shipped_cut` and
`test_the_rest_frame_cut_is_short_of_the_kinematic_endpoint` both encode
the defect and must flip together with it.

## Entry points

- `hazma/spectra/_photon/_muon.pyx:39,41` — the guard and the `r` it
  should use.
- `hazma/spectra/_photon/_muon.pyx:88` — the in-flight branch that already
  uses `1 − r`.
- `hazma/spectra/_photon/_pion.pyx:16` — `ENG_GAM_MAX_MURF`, the same
  endpoint hard-coded.
- `rust/src/kernels/photon_muon.rs` — `Y_MAX`, `ONE_MINUS_R`, and the two
  tests that pin the gap.
- `test/test_core_photon_muon.py` — the Python half of the same pins.
- `test/parity/tolerances.py`, `test/parity/data/` — the corpus block that
  must be regenerated.
- `projects/cython-to-rust/phases/phase-06-mediator-spectra.md` — Task 6.4,
  the blocker.

## Risks / open questions

- **Five blocked defects now share one corpus regeneration**: the boost
  integral's dropped cell
  ([`boost-integral-drops-last-interior-cell.md`](boost-integral-drops-last-interior-cell.md)),
  the positron muon normalization
  ([`positron-muon-spectrum-normalization-inverted.md`](positron-muon-spectrum-normalization-inverted.md)),
  the η′ line weight
  ([`eta-prime-two-photon-line-missing-factor-two.md`](eta-prime-two-photon-line-missing-factor-two.md)),
  the φ line energies
  ([`phi-photon-lines-use-the-daughter-meson-energy.md`](phi-photon-lines-use-the-daughter-meson-energy.md)),
  and this one. They should be priced and scheduled together rather than
  one at a time.
- **This one reaches further than the rest-frame branch alone.** The
  charged-pion photon spectrum (`hazma/spectra/_photon/_pion.pyx`)
  integrates the *point*
  function over the boost cone, and the neutral-rho spectrum integrates
  that in turn, so the rest-frame branch is reachable from them wherever a
  muon lands within one `DBL_EPSILON` MeV of rest. Whether any live grid
  hits that is not settled here.
- **Is `1 − √r` a deliberate approximation from the source paper?**
  arXiv:hep-ph/9909265 is the citation in the `.pyx` docstring; the
  boost-integral evidence above says the *in-flight* formula was derived
  with `1 − r`, which makes a deliberate `1 − √r` in the rest frame
  internally inconsistent rather than merely approximate. Worth one read
  of the paper before the fix, not before filing.
