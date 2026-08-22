# Four scalar elastic cross sections cancel away every significant bit

- **Added:** 2026-08-18
- **Source:** closing
  [`parity-corpus-pins-ill-conditioned-points.md`](../done/parity-corpus-pins-ill-conditioned-points.md)
  — establishing which pinned values were rounding residue required
  knowing the right answer, which is what identified the defect
- **Scope:** cross-cutting (a published number moves)
- **Status:** open (Phase 05 declined it in scope — see
  **Triggers / blockers**)
- **Triggers / blockers:** nothing blocks it. **The "do it during
  Phase 05" window closed on 2026-08-21**: cython-to-rust Task 5.2
  ported all four kernels faithfully, defect included, and deleted the
  `.pyx`. That was the deliberate call, not an oversight — see the
  Status note below — so the remaining work is now a standalone,
  declared numerical change against the Rust, and the "moving the
  numbers twice" cost this entry warned about has been incurred once.
  It has not grown: the Rust expressions are the same closed forms in
  the same order, so the rewrite described under **What** applies to
  them unchanged.

## Why

`hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx` computes
four elastic cross sections — `sigma_xl_to_xl` (l. 265–292),
`sigma_xpi_to_xpi` (293–391), `sigma_xpi0_to_xpi0` (392–489) and
`sigma_xg_to_xg` (490–515) — with the same shape:

```text
P * atan(ms / width_s)  -  P * atan((ms**2 - 4 mx**2 + s) / (ms * width_s))
```

the *same* prefactor `P` on both terms. Two regimes destroy it in double
precision:

- **`e_cm -> 2 mx`.** The two arguments become equal, so the difference
  goes to zero while each `atan` stays O(1). Every significant bit
  cancels.
- **`width_s -> 0`.** Both arguments exceed ~9e15, each `atan` rounds to
  the double nearest `pi/2`, and the difference is 0 or ±1 ulp
  regardless of the physics.

Dividing that residue by the `4 mx**2 - s` in the denominator then
manufactures a pole where the function is smooth. Evaluated at 60 digits
(`test/parity/reference.py`), `sigma_xl_to_xl` at `mx = 300`,
`ms = 200`, `width_s = 3.7e-15`, muon target:

```text
e_cm              the library returns   the formula is worth
595.380695738     -1.381932e-06         +6.775521e-07
599.9994          -1.504081e-02         +6.198557e-07
599.9999994       -1.504133e+01         +6.198489e-07
600.0             -inf                  +6.198489e-07
600.0000006       +1.504133e+01         +6.198489e-07
```

Wrong sign, seven orders of magnitude, and a fabricated singularity —
from correct source, on a correctly-built library. The `-inf` at
`e_cm = 2 mx` is 0/0: both the `atan` difference and the whole log tail
vanish there with the denominator, so the limit is finite and the
expression simply cannot reach it.

This is the same class of defect as
[`kallen-under-sqrt-remaining-call-sites.md`](kallen-under-sqrt-remaining-call-sites.md)
and the closed
[`cross-section-prefactor-threshold-cancellation.md`](../done/cross-section-prefactor-threshold-cancellation.md):
an algebraically correct closed form evaluated in an order that cancels.

## What

Rewrite the `atan` difference with the addition identity, which is exact
and well conditioned. With `u = ms / width_s` and
`v = (ms**2 - 4 mx**2 + s) / (ms * width_s)`, both positive so `uv > -1`:

```text
atan(u) - atan(v) = atan((u - v) / (1 + u v))
u - v             = (4 mx**2 - s) / (ms * width_s)          # exact, no cancellation
```

so the whole term becomes

```text
P * atan((4 mx**2 - s) / (ms * width_s * (1 + u v)))
```

and the `4 mx**2 - s` in it cancels analytically against the same factor
in the denominator, removing the fabricated pole as well. This is now a
change to `rust/src/kernels/scalar_xs.rs` rather than to a `.pyx`: Task
5.2 ported the four kernels unchanged, because
`projects/cython-to-rust/rules.md` rule 1 gates every swap on the parity
corpus and rule 2 forbids regenerating that corpus, so a stabilised
kernel could not have shipped inside the swap PR. `PLAN.md`'s **Scope**
excludes "any physics change" and its **Numerical impact** section
declares exactly two exceptions, neither of them this one. Landing the
fix therefore needs its own PR and its own declaration, which is what
this entry is.

Three things follow and must land with it:

1. **It is a numerical change to a published API.**
   `docs/versioning.md` makes that `minor` at least; the values move by
   O(1) in the affected region. `projects/cython-to-rust/rules.md`
   rule 3 requires the function, grid and magnitude in the PR body and
   in the project's "Numerical impact so far".
2. **The corpus mask shrinks.** `test/parity/stability.py` currently
   masks 494 stored positions because the *pinned* values are residue.
   A stabilised Rust kernel would not reproduce them — correctly — so
   the mask must be re-derived against the new implementation, and the
   affected cases move from "pinned against pre-port Cython" to
   "pinned against the reference". That is a genuine change to what the
   corpus means for those four cases and wants saying out loud, probably
   in a project ADR.
3. **`sigma_xg_to_xg`'s `e_cm == 2 mx` guard becomes wrong.** It returns
   `0.0` there ("complete destructive interference"); with the
   stabilised form the limit is finite and non-zero, so the guard is
   either a real physics statement that the identity contradicts, or a
   workaround for the 0/0 that no longer applies. Settle which.
   `rust/src/kernels/scalar_xs.rs`'s
   `the_photon_elastic_cross_section_is_zero_exactly_at_two_mx` pins the
   current behavior and cites this entry, so it is the test to revisit.
   `sigma_xs_to_xs` is the counter-example worth reading first: that
   kernel's `.pyx` supplied the finite limit at `e_cm == 2 mx` instead
   of zero, and
   `the_mediator_elastic_limit_is_the_general_expressions_limit`
   measures how far the general expression converges toward it before
   the same cancellation stops it (1.96e-7 at a 1e-7 offset; the gap
   grows again below 1e-8).

## Entry points

- `rust/src/kernels/scalar_xs.rs` — `sigma_xl_to_xl`,
  `sigma_xpi_to_xpi`, `sigma_xpi0_to_xpi0` and `sigma_xg_to_xg`, each
  with the same `atan(ms/width_s)` and
  `atan((ms**2 - 4 mx**2 + s)/(ms*width_s))` pair on a shared
  prefactor. The line numbers below are into the deleted
  `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx` and are
  provenance, not live locations: `__sigma_xl_to_xl` (`atan` at l. 278,
  281), `__sigma_xpi_to_xpi` (320, 341), `__sigma_xpi0_to_xpi0` (420,
  441), `__sigma_xg_to_xg` (502, 505).
- `test/parity/reference.py` — the 60-digit evaluation that measures the
  error, and the oracle a stabilised implementation should be checked
  against.
- `test/parity/stability.py` — `AFFECTED_CASES` and the mask that would
  shrink.
- `projects/cython-to-rust/phases/phase-05-mediator-cross-sections.md` —
  where the port of these four is planned.
- `projects/cython-to-rust/task-notes/phase-01/followup-parity-corpus-stability.md`
  — the full measurement and the reasoning behind the mask.

## Risks / open questions

- **Is the published formula itself right?** This item assumes it is and
  fixes only the evaluation. Nothing here checks the physics; the
  reference is a verbatim copy of the same expression, so it inherits any
  error in the derivation. A cross-check against an independent
  calculation is a separate, larger question.
- **`closed_resonance` may be a bad corpus sample regardless.** With
  `width_s = 3.7e-15` the model point is degenerate: the `s`-channel
  propagator is a delta function in all but name. Even a perfectly
  stabilised kernel is being asked for a number of doubtful meaning
  there. Worth deciding alongside, though changing `cases.py` moves
  abscissae and rule 2 then makes it expensive.
- **How much moves in the sub-GeV domain the library is actually for?**
  The measurements above are at `mx = 300 MeV`, inside it. The
  `open_resonance` and `narrow_resonance` points are affected only within
  ~1e-7 of `e_cm = 2 mx`, so a user scanning a coarse grid may never see
  it — which is an argument about *urgency*, not about correctness.
