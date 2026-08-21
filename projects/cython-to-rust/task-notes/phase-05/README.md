# Working Memory: Phase 05 — Mediator cross sections

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 05
**Status:** In Progress — Task 5.1 complete (2026-08-20)
**Plan References:** `../../phases/phase-05-mediator-cross-sections.md`
**Related ADRs:** ADR-0002
**Depends On:** Phase 03 complete (may run parallel to Phase 04 — no
shared files)

## Objective

Track live per-task status and phase-scoped findings for the
cross-section ports.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 5.1 | Vector cross sections (template) | — | **Complete (2026-08-20)** | [task-5.1-vector-xs.md](task-5.1-vector-xs.md) |
| 5.2 | Scalar cross sections | 5.1 | Not started | [task-5.2-scalar-xs.md](task-5.2-scalar-xs.md) |
| 5.3 | Thermal ⟨σv⟩ validation sweep | 5.1, 5.2 | Not started | [task-5.3-thermal-sweep.md](task-5.3-thermal-sweep.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-05-mediator-cross-sections.md`.

## Inputs Reviewed

- `../../phases/phase-05-mediator-cross-sections.md`; `../README.md`;
  `../../references/cython-inventory.md` (three-tier structure, bug §2).

## Findings

- **The `**` operator was not real arithmetic.** Cython 3's default
  `cpow` semantics compile a `double ** double` — and the *whole
  expression around it* — in `double _Complex`, so the two vector
  kernels with a `** 1.5` reach `cpow` and compiler-rt's `__divdc3`
  rather than `pow` and `/`. Neither agrees with its real spelling (up
  to 9.0e-15 and 4.0e-16 relative), so both had to be reproduced:
  `cpow(t+0i, 1.5+0i)` is bit-for-bit `exp(1.5·ln t)`, and `__divdc3` is
  C99 Annex G's scaled quotient. **Check `grep -c SoftComplexToDouble`
  on the generated C before porting any `.pyx` with a fractional
  exponent** — the scalar module has none, but Phase 06's four might.
- **Three of the corpus's pinned records are `raises`, not values**, all
  `TypeError` at `e_cm = 2 m_x`, and `test_parity.py` replays them
  rather than skipping. That forced `dispatch::map_unary_try`, the
  fourth live entry-point shape — a kernel that can fail per element,
  taking the whole array down as the Cython's `__vec_*` loop does.
- **`pip install -e .` gives you a debug build of `hazma._core`**
  (`setuptools_rust` infers `debug = self.inplace or self.debug`). Any
  benchmark taken from an editable tree is ~20× pessimistic and points
  the wrong way: in debug the port looks like a 20× regression, in
  release it is a 1.1×–3.2× win. Filed, not fixed —
  `[profile.release]`'s LTO makes a release rebuild 64 s.
- **The thermal integrals never converge.** Both `.pyx` pass `quad`
  neither `epsabs` nor `epsrel`, so scipy's default absolute tolerance
  (1.49e-8) is met by the first Kronrod pass on an integrand whose
  integral is ~1e-27. The shipped answer is 0.5%–5% off the true
  integral for every `x ≳ 5`. Filed; Task 5.3 should measure the relic
  consequence.
- **`pow(x, 2.0)` folds to `x·x`, `pow(x, 3.0)` and `pow(x, 4.0)` do
  not** — `_pow` is a live libm import of the shipped object. Writing
  `x·x·x` is a different number.

## Decisions and Implementation Notes

- Task 5.1: reproduce `cpow` and `__divdc3` rather than widen two
  `EXACT` budgets — the stronger gate turned out to be the cheaper one
  (ten lines), and all five closed forms came back bit-equal.
- Task 5.1: the reproduced `TypeError` keeps its type and drops Cython's
  wording, which advises a compiler directive that will not exist after
  Phase 07. The corpus records only the type.
- Task 5.1: module-local `cdef double` constants live in the kernel
  module, not `constants::derived`, which
  `test/test_core_constants.py` scores against *surviving* `.pyx`.
  Phase 04's §5 established the pattern.
- Task 5.1: the corpus cases point at the pure-Python wrapper under the
  kernels' canonical names, because this wrapper defines no function of
  its own (its surface is a mixin class).

## Files Changed

### Task 5.1

`rust/src/kernels/vector_xs.rs` (new), `rust/src/vector_mediator.rs`,
`rust/src/{kernels,dispatch,quad}.rs`,
`hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx` (deleted),
`hazma/vector_mediator/_vector_mediator_cross_sections.py`, `setup.py`,
`hazma/_core.pyi`, `test/test_core_vector_xs.py` (new),
`test/parity/{cases,tolerances}.py`, `test/test_core_quad.py`, three
`docs/followups/todo/` entries + index.

## Verification

- Corpus over the mediator parameter grid; relic-density end-to-end
  check (Task 5.3); benchmark per rules.md rule 12.
- Task 5.1: `pytest -q` → `2013 passed, 15 skipped`;
  `pytest test/parity -q` → `658 passed, 1 skipped`;
  `cargo test --no-default-features` → `186 passed`;
  `pytest test/test_theory_aggregation.py -q` → `69 passed`.

## Open Questions

- Three follow-ups opened by Task 5.1, none blocking:
  [the `2 m_x` raise](../../../../docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md),
  [the unconverged thermal quadrature](../../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md),
  [the debug editable build](../../../../docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md).
- `test/test_core_quad.py`'s scalar `sigma_xx_to_all` oracle dies with
  the scalar `.pyx` in Task 5.2; the vector branch shows the
  replacement shape.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 05 (Task 5.2):** read
`../../PLAN.md`, `../README.md`, this file, then the phase file, then
[`task-5.1-vector-xs.md`](task-5.1-vector-xs.md)'s Findings and
`rust/src/kernels/vector_xs.rs`'s module docs — Task 5.2 is the same
shape at four times the size. Transliterate the Mathematica dumps
mechanically; never retype a 90-line expression.

**Currently safe to assume:**

- Both `_c_*` modules cimport nothing from hazma — they are
  self-contained above the foundation. The vector one is gone; nothing
  cimported it and it exported no capsules, so it went whole.
- The layout is settled: kernels in `crate::kernels::<name>_xs`, PyO3
  registration in `crate::<model>_mediator`, module-local constants
  beside the kernels.
- `dispatch::map_unary_try` exists for a kernel that raises, and
  `crate::quad`'s `qagp` plus `crate::special`'s Bessels are now on a
  live path rather than only under their probes.

**Currently risky / unknown:**

- Near-resonance corpus points are the most drift-sensitive for ⟨σv⟩
  (the integrand peaks at the `points=` breakpoints). Task 5.1 measured
  2.06e-14 there; **re-derive rather than inherit** — Phase 04's §1 is
  emphatic that per-kernel drift is not predictable from shape.
- The scalar file's two 90-line expressions are the phase's real
  transliteration risk, and
  `docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`
  says four of its kernels cancel away every significant bit — read
  `test/parity/stability.py` before trusting a comparison there.
- The scalar model **short-circuits to `0.0`** above `x = 300` where the
  vector saturates. The corpus pins both; do not unify them.
