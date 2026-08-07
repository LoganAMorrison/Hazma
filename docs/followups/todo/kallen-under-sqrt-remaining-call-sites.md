# The remaining `sqrt(kallen_lambda(...))` call sites lose accuracy at their thresholds

- **Added:** 2026-08-05
- **Source:** carved out of
  [`cross-section-prefactor-threshold-cancellation`](../done/cross-section-prefactor-threshold-cancellation.md)
  while fixing the two-body-momentum sites
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** none technically. Sequencing against
  `projects/cython-to-rust/` matters — see "Risks".

## Why

`kallen_lambda(a, b, c) = a² + b² + c² − 2ab − 2ac − 2bc` cancels to zero
whenever the triangle degenerates. Under a square root that is a
conditioning bug: near the degenerate point the four terms are each far
larger than their sum, so the result is dominated by roundoff.

The two-body-momentum spelling of that pattern,
`sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)`, was fixed by
`hazma.utils.two_body_momentum` — the factored form with the heavier mass
subtracted first, which holds ≤4.4e-16 relative error all the way to
threshold, where the Källén form reaches 4.0e-2. The measurements and the
reasoning are in that follow-up.

The same defect is still present at roughly 25 other call sites. They
were left out deliberately: each is a distinct physics function whose
published numbers would move, which is a much larger declared numerical
change than one helper. But the defect is identical, and the fix is now
sitting in `hazma.utils` waiting to be applied.

## What

Repoint each site below onto `hazma.utils.two_body_momentum` where the
quantity really is a two-body momentum, and onto a factored form
otherwise. Note that a dimensionless `kallen_lambda(1, x**2, y**2)` is
the same case — it is `λ(1², x², y²)`, so
`sqrt(kallen_lambda(1, x**2, y**2)) == 2 * two_body_momentum(1.0, x, y)`.

Do it in **separate, individually declared commits grouped by physics
area**, not as one sweep: each group moves different published numbers
and wants its own CHANGELOG magnitude and its own before/after
measurement. The follow-up above is the worked template for how to
measure (exact-rational reference via `fractions.Fraction`, a grid of
distances from threshold, a table of max relative error).

Two sites are worth calling out as *not* mechanical:

- `hazma/phase_space/_utils.py:98` (`two_body_phase_space_prefactor`)
  clips λ at zero, so it currently returns 0 rather than NaN below
  threshold. Rewriting it as `p / (4 π cme)` changes that below-threshold
  behavior unless the clip is kept — decide deliberately, and note that
  the library's stance elsewhere is that NaN beats a plausible zero.
- The `_three_body.py` and `form_factors/vector/_three_body.py` sites
  multiply *two* Källén factors under one square root. Both need
  factoring, and the arguments there are Mandelstam variables rather
  than `cme²`, so the "heavier mass first" ordering has to be re-derived
  per site rather than copied.

## Entry points

Grouped as suggested commits:

- **Phase space:** `hazma/phase_space/_utils.py:98`,
  `hazma/phase_space/_three_body.py:251`
- **Vector form factors:** `hazma/form_factors/vector/_pi_omega.py:107`,
  `hazma/form_factors/vector/_three_body.py:512-513,794-795,845-846,904-912,926-927`
- **Legacy vector-mediator form factors:**
  `hazma/vector_mediator/form_factors/omega_pi.py:41`,
  `hazma/vector_mediator/form_factors/widths.py:67,87`,
  `hazma/vector_mediator/form_factors/pi_pi_eta.py:93-94,142-143`,
  `hazma/vector_mediator/form_factors/pi_pi_etap.py:73-74,101-110`,
  `hazma/vector_mediator/form_factors/pi_pi_omega.py:45-46,104-105`,
  `hazma/vector_mediator/form_factors/pi_pi_omega.py:123-124`,
  `hazma/vector_mediator/form_factors/utils.py:486`
- **RH neutrino widths:** `hazma/rh_neutrino/_widths.py:125,166,323,476`
- **Scalar-mediator constraints:**
  `hazma/scalar_mediator/_scalar_mediator_constraints.py:59,160,212`
- **Dead / removed by cython-to-rust, skip:**
  `hazma/_utils/kinematics.pxd:15` (nothing `cimport`s
  `two_body_three_momentum`), and `hazma/_gamma_ray/gamma_ray_fsr.pyx`
  line 36 **as of `c6991a6`** — that module was removed by ADR-0003 in
  Task 0.2, so the call site no longer exists.
- Prior art and measurement recipe:
  [`docs/followups/done/cross-section-prefactor-threshold-cancellation.md`](../done/cross-section-prefactor-threshold-cancellation.md)

## Risks / open questions

- **Every group is a declared numerical change** under
  `docs/versioning.md` — CHANGELOG line with a magnitude, per group.
  Most of these feed decay widths and cross sections that go straight
  into published limit plots, so this is more visible than the two-body
  fix was.
- Sequencing against `projects/cython-to-rust/`: doing this **before**
  the Phase 01 parity corpus is generated means the corpus captures the
  fixed values and the Rust port must reproduce them; doing it after
  means each group is a clean isolated change. The two-body fix landed
  before the corpus, so the precedent is set, but these sites touch far
  more of the surface the corpus covers — landing them mid-migration
  would invalidate corpus data already generated.
- Open question: is it worth a shared `_factored_kallen_sqrt` helper for
  the sites that are not two-body momenta, or does each get its own
  inline factoring with a comment? A helper that takes `(a, b, c)`
  cannot know which argument is the "energy", so it cannot choose the
  stable subtraction order — that argues for inline.
