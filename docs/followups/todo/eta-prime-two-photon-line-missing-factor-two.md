# The η′ two-photon line carries one photon per decay instead of two

- **Added:** 2026-08-12
- **Source:** `projects/cython-to-rust/` Phase 04 Task 4.2 — porting the
  five tabulated photon `.pyx` files to Rust; writing them as one
  parameterised implementation is what put the five line weights side by
  side and made the odd one out visible
  (`projects/cython-to-rust/task-notes/phase-04/task-4.2-photon-table-family.md`)
- **Scope:** cross-cutting (a published number is wrong; the repair is
  gated by the `cython-to-rust` corpus)
- **Status:** open
- **Triggers / blockers:** **blocked until after `cython-to-rust` Phase 06
  Task 6.4**, for the same reason as
  [`positron-muon-spectrum-normalization-inverted.md`](positron-muon-spectrum-normalization-inverted.md)
  and
  [`boost-integral-drops-last-interior-cell.md`](boost-integral-drops-last-interior-cell.md):
  the parity corpus pins the shipped value by construction
  (`projects/cython-to-rust/rules.md` rule 2 forbids regenerating it from
  a tree where any kernel runs on Rust), so a repair landed during the
  port would fail the gate that governs every remaining swap. Cheapest to
  fix in one declared regeneration together with its two siblings and
  with
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](phi-photon-lines-use-the-daughter-meson-energy.md).

## Why

Five of hazma's tabulated photon spectra add a monochromatic `X → γγ`
line on top of their CSV continuum, because the CSVs reserve an `a_a`
column that is identically zero and the line is put in analytically. A
two-photon final state contributes **two** photons per decay, so the
line's weight is `2 · BR(X → γγ)`. Four of the five write exactly that:

```text
hazma/spectra/_photon/_eta.pyx:99    res += 2.0 * BR_ETA_TO_A_A  * boost_delta_function(MASS_ETA  / 2.0, ...)
hazma/spectra/_photon/_kaon.pyx:300  res += 2   * BR_KL_TO_A_A   * boost_delta_function(MASS_K0   / 2.0, ...)
hazma/spectra/_photon/_kaon.pyx:407  res += 2   * BR_KS_TO_A_A   * boost_delta_function(MASS_K0   / 2.0, ...)
hazma/spectra/_photon/_eta_prime.pyx:107
                                     res +=      BR_ETAP_TO_A_A  * boost_delta_function(MASS_ETAP / 2.0, ...)
```

(Quoted from the pre-port sources, which Task 4.2 deleted in the same
PR as the swap — `git show 665aed5:<path>` recovers them. The
expressions themselves live on unchanged in
`rust/src/kernels/photon_tables.rs`.)

The η′ has no factor of two. It is the code and not a reading of it: the
shipped `_eta_prime.cpython-312-darwin.so` loads the immediate
`0x3f979fa97e132b56` = `2.307e-2` = `BR_ETAP_TO_A_A` into its single
`fmadd`, where `_eta.so` loads `0x3fe938ef34d6a162` = `0.7882` =
`2 × 0.3941`.

Measured, by integrating the line term alone (the full spectrum minus the
boosted continuum, `scipy.integrate.quad`, `E_parent = 2 M`):

```text
η  line term integral = 0.78819993 ± 1.0e-08     (2 · BR = 0.7882)
η′ line term integral = 0.02306998 ± 1.3e-08     (    BR = 0.02307; 2 · BR = 0.04614)
```

A boosted δ-function integrates to its own weight at any boost, so those
integrals *are* the photon count each mode contributes per decay. The η′
is short 0.02307 photons per decay out of a corrected total of 3.6403
(continuum 3.5942 from `eta_prime_photon.csv` plus 0.04614), i.e.
**0.63% of the η′ photon yield is missing**, concentrated entirely in a
line at `M_η′/2 = 478.89` MeV.

**The ω and φ weights are correctly un-doubled and are not part of this.**
Their lines are `ω → π⁰γ`, `ω → ηγ`, `φ → ηγ` and `φ → η′γ` — one photon
each, so a bare `BR` is right there. The comparison that matters is
against the four `X → γγ` siblings, and the η′ is the only one of those
that differs. (The φ's lines have a separate defect of their own; see
[`phi-photon-lines-use-the-daughter-meson-energy.md`](phi-photon-lines-use-the-daughter-meson-energy.md).)

## What

1. Change the weight to `2 · BR(η′ → γγ)` in the Rust port, which is
   where the kernel lives from Task 4.2 on:
   `rust/src/kernels/photon_tables.rs`, `ETAP_TO_A_A_WEIGHT`.
2. Regenerate the affected parity corpus case
   (`spectra.photon.eta_prime`), or re-pin it, under whatever mechanism
   the port has by then. This is a **declared numerical change**:
   `projects/cython-to-rust/rules.md` rule 3 and `docs/versioning.md`
   make a moved published spectrum a `minor` bump at least, and it needs
   a `CHANGELOG.md` entry stating the 0.63%.
3. Check the downstream consumers of `dnde_photon_eta_prime` — the η′
   final state in `hazma.theory`'s channel dicts and any limit that
   opens it — but note the shift is confined to one line, so anything
   integrating over a band that excludes `M_η′/2` in the parent's rest
   frame does not move at all.

The port's tests already encode the correct statement alongside the
shipped one, so the repair mostly means flipping which is asserted:
`the_eta_prime_line_is_missing_its_factor_of_two` in
`rust/src/kernels/photon_tables.rs`, and
`TestPhysics::test_the_eta_prime_line_carries_half_the_photons_it_should`
in `test/test_core_photon_tables.py`.

## Entry points

- `rust/src/kernels/photon_tables.rs` — `ETAP_TO_A_A_WEIGHT` and the
  three sibling weights beside it.
- `test/test_core_photon_tables.py` —
  `TestPhysics::test_each_line_carries_the_photon_count_its_weight_declares`
  is the measurement, generalised over all seven spectra.
- `test/parity/tolerances.py` — `spectra.photon.eta_prime` is
  `TABULATED` (`rtol = 1e-12`), five decades tighter than this shift, so
  nothing absorbs it quietly.
- `hazma/_utils/constants.pxd:220` — `BR_ETAP_TO_A_A = 2.307e-2`, the
  branching ratio itself, which is right.
- Sibling defects, same class and same blocker:
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](phi-photon-lines-use-the-daughter-meson-energy.md),
  [`positron-muon-spectrum-normalization-inverted.md`](positron-muon-spectrum-normalization-inverted.md),
  [`boost-integral-drops-last-interior-cell.md`](boost-integral-drops-last-interior-cell.md).

## Risks / open questions

- **The `a_a` columns are confirmed empty, so doubling the line cannot
  double-count.** `eta_prime_photon.csv`'s `a_a` column is identically
  zero across all 500 rows, and `eta_photon.csv`'s across all 100
  (checked). The two kaon tables carry no `a_a` column at all and say so
  in their header comments (`# missing: a_a`). The analytic line is the
  whole `X → γγ` contribution in every one of the four.
- Sequencing against the corpus is the real cost, as with its siblings:
  one declared regeneration after Phase 06 Task 6.4 covering all four
  defects is cheaper than four.
