# The charged-pion positron clip turns a `NaN` energy into a spectrum

- **Added:** 2026-09-25
- **Source:** PR #105 review, which found the same defect in the neutrino
  pion's boost window and repaired it there
- **Scope:** commit
- **Status:** open

## Why

`rust/src/kernels/positron_pion.rs` clips its boost window with
`(gamma * beta.mul_add(k, e)).min(EMAX_PI_RF)`. `f64::min` discards a
`NaN` operand, so a `NaN` positron energy gets the finite upper limit
`EMAX_PI_RF`, the lower limit is floored to `ME` by `.max(ME)`, and the
quadrature integrates the whole support. The kernel is unchanged since
`dfd62eea`, where `dnde_positron_charged_pion(nan, 400.0)` returns
`0.005797088917702574` rather than `NaN`. A `NaN` pion energy still
propagates, because `gamma` and `beta` are then `NaN` too.

A caller who passes a `NaN` energy, for instance from a masked grid,
gets a plausible-looking finite number back, which is the silent wrong
number the repo's numerical-correctness rules rank below a crash.

## What

Replace both clips with comparisons that keep `NaN`, as
`neutrino_pion::boost_window` does: `if upper > EMAX_PI_RF { EMAX_PI_RF }
else { upper }` for the upper limit, and the analogous comparison for
the `ME` floor. Pin scalar and array `NaN` inputs in
`test/test_core_positron_pion.py`, and a kernel unit test beside the
existing ones in `positron_pion.rs`. No finite value moves, so the parity
corpus is untouched.

## Entry points

- `rust/src/kernels/positron_pion.rs:163-164`
- `rust/src/kernels/neutrino_pion.rs`, `boost_window` and its test
  `a_nan_input_stays_nan_through_the_clip`, for the repaired sibling
