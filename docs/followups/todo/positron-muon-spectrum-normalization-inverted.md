# The muon positron spectrum divides by its normalization instead of multiplying

- **Added:** 2026-08-11
- **Source:** `projects/cython-to-rust/` Phase 04 Task 4.1 — porting
  `hazma/spectra/_positron/_muon.pyx` to Rust; the port's analytic
  normalization test measured it
  (`projects/cython-to-rust/task-notes/phase-04/task-4.1-positron-muon.md`)
- **Scope:** cross-cutting (a published number is wrong; the repair is
  gated by the `cython-to-rust` corpus)
- **Status:** open
- **Triggers / blockers:** **fix BEFORE `cython-to-rust` Phase 06
  Task 6.4** — the constraint is a deadline, not a wait, and this one
  also affects published numbers today. The parity corpus does pin the
  wrong values by construction, so the repair needs corrected reference
  values before it can pass the gate that governs every remaining swap.
  But Task 6.4 is where `hazma/spectra/_positron/_muon.pyx` is
  **deleted**, and that twin is the only independent implementation a
  corrected corpus case can be re-pinned from: fix the `.pyx`, drive it
  through the `__pyx_capi__` capsules this file's "Entry points" section
  already names, and the corrected
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

`hazma/spectra/_positron/_muon.pyx` computes Michel's spectrum for
`μ → e ν̄ν` and normalizes it with the module constant

```text
# 1 / (1 - 8 r^2 + 8 r^6 - r^8 - 12 r^2 ln(r^2))
DEF R_FACTOR = 1.0001870858234163
```

at `r = m_e/m_μ`. (The comment's `12 r^2` is itself a typo for `12 r^4`;
only the quartic reproduces the digits, which cython-to-rust Task 3.1
established and `test/test_core_constants.py` pins. The *value* is
right.) So the comment is right about what the constant is: the
reciprocal of the un-normalized spectrum's integral. Both the rest-frame
and the in-flight expressions then **divide** by it, where normalizing
requires **multiplying**.

Measured rather than argued. Integrating the un-normalized polynomial
over its support with `scipy.integrate.quad`:

```text
∫_{2r}^{1+r²} -2√(x²-4r²)(4r² + x(-3-3r²+2x)) dx = 0.999812949171142
1 - 8r² + 8r⁶ - r⁸ - 12 r⁴ ln(r²)               = 0.9998129491711419
1 / R_FACTOR                                     = 0.999812949171142
```

The three agree to the last digit, so the raw integral is exactly
`1/R_FACTOR` and the normalized spectrum should be `raw × R_FACTOR`.
Dividing instead leaves every value low by `1/R_FACTOR²`:

```text
∫ dN/dE dE  (dnde_positron_muon, m_μ at rest, 4e5-point trapezoid) = 0.9996236
∫ dN/dE dE  (dnde_positron_muon, E_μ = 500 MeV)                    = 0.9996259
1 / R_FACTOR²                                                       = 0.9996259
```

**0.0374% low, uniformly** — it is a constant factor, not a shape
distortion, so no ratio or spectral-index result is affected and every
absolute rate is.

**The sibling kernel settles that this is a mistake and not an unstated
convention.** `hazma/spectra/_neutrino/_muon.pyx` declares the identical
constant (`:23`, same digits, same comment) and *multiplies* by it in
both of its expressions — `common = R_FACTOR * x**2 * ...` (`:58`) and
`pre = R_FACTOR * e_to_x / (2.0 * beta)` (`:114`). Two files, one
constant, opposite operators; the neutrino one matches the closed form
above. `hazma/spectra/_photon/_muon.pyx` declares no `R_FACTOR` at all
(it is the radiative spectrum) and is unaffected.

It propagates. `dnde_positron_charged_pion` boosts this spectrum and
inherits the factor exactly (`∫ = 0.999623` at `E_π = 500 MeV`), as do
both mediator positron spectra and therefore every positron-based limit
in `hazma/limits/`.

## What

1. Change the two divisions to multiplications — in the Rust port, which
   is where the kernel lives from Task 4.1 on:
   `rust/src/kernels/positron_muon.rs`, `dndx_rest_frame`'s
   `/ R_FACTOR` and `dndx`'s `/ ((beta + beta) * R_FACTOR)`. The second
   becomes `* R_FACTOR / (beta + beta)`; note the operation order is
   itself a numerical choice once the corpus no longer pins it.
2. Regenerate the affected parity corpus cases, or re-pin them, under
   whatever mechanism the port has by then. This is a **declared
   numerical change**: `projects/cython-to-rust/rules.md` rule 3 and
   `docs/versioning.md` make a moved published spectrum a `minor` bump at
   least, and it needs a `CHANGELOG.md` entry stating the 0.0374%.
3. Re-run `test/test_theory_aggregation.py` and the positron-limit
   notebooks; the shift is uniform, so anything that normalizes its own
   spectra will not move.

The port's own tests already encode the *correct* statement alongside the
shipped one, so the repair mostly means flipping which of the two is
asserted:
`rest_frame_spectrum_carries_the_inverted_normalization` in
`rust/src/kernels/positron_muon.rs`, and
`TestPhysics::test_the_integral_is_the_shipped_inverted_normalization`
in `test/test_core_positron_muon.py`.

## Entry points

- `rust/src/kernels/positron_muon.rs` — `dndx_rest_frame`, `dndx`.
- `hazma/spectra/_positron/_muon.pyx` — the pre-port source, kept for its
  `__pyx_capi__` capsules until `cython-to-rust` Phase 06 Task 6.4.
- `rust/src/constants.rs` — `constants::derived::positron_muon::R_FACTOR`,
  whose doc comment records that the `.pyx` comment's `12 r²` exponent is
  itself a typo for `12 r⁴` (Task 3.1). The constant's *value* is right;
  only the operator using it is wrong.
- `hazma/spectra/_neutrino/_muon.pyx:23,58,114` — the sibling that
  applies the same constant the other way round, and so is the evidence
  that this is an inversion rather than a convention.
- `test/parity/tolerances.py` — `spectra.positron.muon` is `EXACT`
  (`rtol = 0`), so nothing absorbs this quietly.
- Sibling defects, same class and same blocker:
  [`boost-integral-drops-last-interior-cell.md`](boost-integral-drops-last-interior-cell.md),
  and the two the tabulated photon port surfaced in Task 4.2 —
  [`eta-prime-two-photon-line-missing-factor-two.md`](eta-prime-two-photon-line-missing-factor-two.md)
  and
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](phi-photon-lines-use-the-daughter-meson-energy.md).
  All four want one declared corpus regeneration after Phase 06
  Task 6.4, not four.

## Risks / open questions

- **Anything computed with a released hazma moves by 0.0374%.** Not
  checked here against any specific published figure; the point is only
  that this is a changed number in a released library rather than an
  internal refactor, which is what makes it a declared change with a
  version bump rather than a patch.
- **The in-flight expression's operation order** is unpinned once the
  corpus no longer holds it: `raw * R_FACTOR / (2β)` and
  `raw / (2β) * R_FACTOR` differ in the last ulp. Pick one and say why.
- Sequencing against the corpus is the real cost. Repairing this and
  `boost-integral-drops-last-interior-cell.md` in one declared
  regeneration after Phase 06 is cheaper than two.
