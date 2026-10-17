# ADR 0002 Rust kernels behind Python validation, pinned by a NumPy oracle

**Date:** 2026-10-16
**Status:** Accepted
**Scope:** Project-scoped within visible-axial-spectra.

## Context

The plan's implementation layering puts every per-sample loop, contraction
and integral in `hazma._core` and keeps cards, validation, result objects
and messages in Python. Tasks 1 and 2 shipped the axial tensors and the LO
three-pion currents in NumPy, so Task 3 had to fix how the two layers meet:
the batch layout, where each check runs, and what pins the kernels, since
they port no Cython and the parity corpus has no case for them.

## Decision

The wrappers in `hazma/form_factors/axial/` validate and broadcast their
arguments in NumPy, reshape each batch to a C-contiguous
`(components, N)` array, and call one `hazma._core.axial` entry point,
which maps a per-sample kernel in `rust/src/kernels/axial_*.rs` over the
batch and returns `(components, N)` or `(N,)`. That is NumPy's own layout
for a `(4, ...)` array, so neither side transposes. Checks of the
arguments (shapes, finiteness, timelike q, Hermiticity, on-shell momenta,
support) stay in Python. Checks of the numbers stay with the arithmetic in
Rust: the tensor projection's roundoff budget, negative-density test and
conditioning guard, which raise the public messages, and which a batch
failing in several ways reports with the precedence the NumPy code had.

The NumPy implementation is kept, unchanged, as
`test/form_factors/axial/axial_oracle.py`, and nothing under `hazma/`
imports it. `test_oracle.py` compares every ported function against it in
units in the last place of the scale both round against, and
`test/parity/cases.py` exempts `hazma._core.axial` from the corpus's
ported-kernel count through `_CORE_NEW_PHYSICS_MODULES`, naming that
oracle. Later kernels in the project's table follow the same split.

## Consequences

- **Positive:** One validation path and one set of messages; kernels stay
  PyO3-free and are unit-tested under `cargo test`; the oracle catches a
  kernel that drifts from the implementation Tasks 1 and 2 validated.
- **Negative:** The oracle and the kernels are not bit-identical where
  NumPy fuses complex multiplies (arm64), divides complex by real as a
  multiply by the reciprocal, or sums in another order, so the comparison
  carries measured ulp budgets rather than equality. The spectral
  functions differ more, because the Rust Newton rule and NumPy's
  `leggauss` are different quadrature rules (weights good to 3e-12 and
  2e-10 relative at order 511), so they carry a relative tolerance.
  Gauss-Legendre nodes are computed in Rust on every call, which costs
  about 2.3 ms at order 512 where NumPy cached them.
- **Mitigation:** Each budget in `test_oracle.py` states its scale and the
  measured worst case. Cache the quadrature rule if a caller ever needs
  order 512 in a tight loop.
