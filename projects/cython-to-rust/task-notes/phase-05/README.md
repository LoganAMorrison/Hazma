# Working Memory: Phase 05 — Mediator cross sections

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 05
**Status:** In Progress — Tasks 5.1 and 5.2 complete (2026-08-21)
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
| 5.2 | Scalar cross sections | 5.1 | **Complete (2026-08-21)** | [task-5.2-scalar-xs.md](task-5.2-scalar-xs.md) |
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
- **Where clang fuses is one syntactic rule, not a list of cases.**
  `EmitFMulAdd` contracts `A + B` when `A` is a multiply, else when `B`
  is — on the *C* tree Cython emits, where `x ** n` is a `pow` **call**
  and `-x**n` is an `FNeg`. That is why `-4*mx**2 + e_cm**2` fuses,
  `ms**2 - e_cm**2` does not, and `-mpi0**2 + e_cm**2` does not. Task
  5.2 implemented the rule and reproduced all 138 FMA sites in eleven
  kernels from it, without reading a disassembly per site.
- **One Python-level call boxes everything above it.** `np.log(4)` at
  `_c_scalar_mediator_cross_sections.pyx:283` makes Cython evaluate the
  whole path to the root through `PyNumber_*` on `PyFloat`s, so nothing
  there contracts while the pure-C operands still fuse internally. Same
  observable as Phase 04's `_photon/_rho.pyx`, different cause.

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
- Task 5.2: the **scalar** wrapper does define one per kernel, so its
  cases point at the wrapper's short aliases through a new
  `Case.attribute` field. `hazma._core.<sub>` cannot be named directly —
  a PyO3 submodule has no `__file__` and
  `assert_module_is_repo_tree` rejects it.
- Task 5.2: the two complex-arithmetic shims moved from `vector_xs.rs`
  into `crate::kernels::soft_complex` rather than being copied, once the
  scalar module turned out to need the identical pair.
- Task 5.2: the four kernels whose `atan` difference cancels were ported
  **with the defect**, not stabilised. Rule 1 gates the swap on a corpus
  rule 2 forbids regenerating, so a stabilised kernel cannot pass the
  gate that would let it ship;
  [the follow-up](../../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md)
  is updated to say the "do it during Phase 05" window closed and what
  the standalone change now looks like.

## Files Changed

### Task 5.2

`rust/src/kernels/{scalar_xs,soft_complex}.rs` (both new),
`rust/src/scalar_mediator.rs`,
`rust/src/{kernels,vector_mediator}.rs`,
`rust/src/kernels/vector_xs.rs`,
`hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx` (deleted,
1,606 lines), `hazma/scalar_mediator/_scalar_mediator_cross_sections.py`,
`setup.py`, `hazma/_core.pyi`, `test/test_core_scalar_xs.py` (new),
`test/parity/{cases,tolerances}.py`,
`test/test_core_{quad,dispatch}.py`, one `docs/followups/todo/` entry.

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
- Task 5.2: `pytest -q` → `2091 passed, 15 skipped`;
  `pytest test/parity -q` → `658 passed, 1 skipped`;
  `cargo test --no-default-features` → `201 passed`;
  `pytest test/test_theory_aggregation.py -q` → `69 passed`.

## Open Questions

- Three follow-ups opened by Task 5.1, none blocking:
  [the `2 m_x` raise](../../../../docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md),
  [the unconverged thermal quadrature](../../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md),
  [the debug editable build](../../../../docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md).
- Task 5.2 rebuilt `test/test_core_quad.py`'s scalar `sigma_xx_to_all`
  oracle from the ported kernels, the way Task 5.1 did the vector one,
  and retired the two `test_core_dispatch.py` cases whose oracle was the
  deleted module.
- Four scalar kernels carry the `atan`-cancellation defect into Rust
  unchanged, and `sigma_xg_to_xg`'s `e_cm = 2 mx` guard disagrees with
  `sigma_xs_to_xs`'s treatment of the same 0/0. Both are
  [the same follow-up](../../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md).

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 05 (Task 5.3):** read
`../../PLAN.md`, `../README.md`, this file, then the phase file, then
[`task-5.2-scalar-xs.md`](task-5.2-scalar-xs.md)'s Handoff. Both thermal
cross sections are Rust now, so 5.3's relic-density sweep and benchmark
run against a fully ported path — and the benchmark must come from a
**release** build, not an editable install.

**Currently safe to assume:**

- Both `_c_*` cross-section modules are gone — neither had a `.pxd` and
  neither exported capsules, so both went whole. Nothing in the phase
  cimports anything from hazma any more.
- The layout is settled: kernels in `crate::kernels::<name>_xs`, PyO3
  registration in `crate::<model>_mediator`, module-local constants
  beside the kernels, and the shared `**`-operator shims in
  `crate::kernels::soft_complex`.
- `dispatch::map_unary_try` exists for a kernel that raises, and
  `crate::quad`'s `qagp` plus `crate::special`'s Bessels are on the live
  path of **both** thermal averages now.
- Every closed-form cross section in the phase is bit-equal to its
  Cython. The two `thermal_cross_section` entry points are the only ones
  that moved (2.06e-14 vector, 3.12e-15 scalar) and both budgets are
  tightened to `PORTED_QUAD_RTOL`.

**Currently risky / unknown:**

- **Task 5.3 owns both of the phase's deferred items**: the
  relic-density end-to-end check through
  `hazma/relic_density/_thermal_functions.py`, and the benchmark. The
  benchmark must be taken from a **release** build —
  [an editable install builds the extension in debug](../../../../docs/followups/todo/editable-installs-build-the-rust-extension-in-debug.md)
  and reads ~20× pessimistic.
- The two models disagree above `x = 300`: the scalar returns exactly
  `0.0`, the vector saturates. The corpus pins both and 5.3's sweep
  crosses that boundary; do not unify them.
- [The thermal quadrature never converges](../../../../docs/followups/todo/thermal-cross-section-quadrature-never-converges.md)
  on either model — the shipped answer is 0.5%–5% off the true integral
  for `x ≳ 5`. Reproduced, not fixed; its relic-density consequence is
  what Task 5.3 measures.
- Four scalar elastic kernels carry
  [the `atan`-cancellation defect](../../../../docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md)
  into Rust unchanged, and read `test/parity/stability.py` before
  trusting any comparison near `e_cm = 2 mx`.
