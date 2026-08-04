---
phase: 03
title: Numerics foundation
status: Not started
---

<!-- markdownlint-disable-file MD025 -- frontmatter title is the schema -->

# Phase 03: Numerics foundation

## Goal

The shared numeric substrate every kernel port stands on: constants,
special functions, the QUADPACK port, `np.interp`-exact interpolation,
the boost integrals, and the scalar/array dispatch layer. Everything
here is unit-tested against scipy/analytic references before any kernel
uses it.

## Prerequisites

- Phase 02 complete; **ADR-0002 accepted** (gates Tasks 3.2/3.3 —
  confirm status before starting them).
- Read `../references/numerics-replacements.md` in full.

## Tasks

### Task 3.1: Constants module

**Task note:** [`../task-notes/phase-03/task-3.1-constants.md`](../task-notes/phase-03/task-3.1-constants.md)
**Depends on:** —

**Exit criteria:**

- `rust/src/constants.rs` carries every `DEF` from
  `_utils/constants.pxd` and every value from
  `_utils/legacy_parameters.pxd`, in **two distinct namespaces**
  preserving the known divergences verbatim (rules.md rule 4).
- A test extracts values from the `.pxd` sources (script or generated
  fixture) and asserts bit-equality — no hand-transcription trust.
- Derived module-local `DEF`s (e.g. `eng_mu_pi_rf`) become `const fn`
  or literal consts with the same float semantics.

### Task 3.2: Special functions

**Task note:** [`../task-notes/phase-03/task-3.2-specfun.md`](../task-notes/phase-03/task-3.2-specfun.md)
**Depends on:** —

**Exit criteria:**

- `spence`, `bessel_k1`, `bessel_kn` exposed via a thin
  `rust/src/special.rs` over `spec_math` (or in-tree cephes translation
  on any gap — ADR-0002 fallback).
- Convention pinned: Rust `spence`-wrapper matches
  `scipy.special.spence` (Li₂(1−z) convention) on a grid covering
  (0,1), [1,∞), z→0⁺, z=1, z=2 — rtol ≤ 1e-13.
- `k1`/`kn` swept vs scipy over the thermal domain incl. large-argument
  underflow region — rtol ≤ 1e-13.

### Task 3.3: QUADPACK port (qk15, qk21, qelg, qags, qagp)

**Task note:** [`../task-notes/phase-03/task-3.3-quadpack.md`](../task-notes/phase-03/task-3.3-quadpack.md)
**Depends on:** —

**Exit criteria:**

- Finite-interval `qags` and `qagp` in `rust/src/quad.rs`, translated
  from netlib QUADPACK Fortran (provenance header per rules.md rule 5),
  closure-based API carrying `epsabs`/`epsrel`/`limit`/breakpoints.
- **Breakpoint preprocessing contract, pinned empirically against
  scipy** (do not design it from the QUADPACK docs alone): determine
  by experiment what `scipy.integrate.quad(points=...)` does with
  unsorted lists, duplicates, points coinciding with the endpoints,
  and points outside `[a, b]` — then replicate that behavior
  (including any raised errors) exactly. Both degenerate cases occur
  live: the spectra calls pass `points=[-1, 1]` on the interval
  `[-1, 1]` (all breakpoints endpoint-coincident), and the thermal
  ⟨σv⟩ calls pass `[2, m_med/mx, 2·m_med/mx]` where the mediator
  points can exceed the upper bound `max(50/x, 100|150)` for heavy
  mediators (out-of-interval). Tests cover three parameter regimes per
  thermal call: breakpoints interior (resonance active), breakpoints
  at/near threshold, and breakpoints outside the interval (inactive).
- Unit tests: QUADPACK's own reference problems, plus every live
  integrand *shape* from the call-site table in
  `../references/numerics-replacements.md`, compared against
  `scipy.integrate.quad` at matching settings — agreement within 10×
  the requested tolerance, and within 1e-12 rel on smooth cases.
- Error/abnormal-termination behavior mapped (roundoff, max
  subdivisions, invalid breakpoints) — returns a Result, never panics
  across FFI.

### Task 3.4: Interpolation + boost kernels

**Task note:** [`../task-notes/phase-03/task-3.4-interp-boost.md`](../task-notes/phase-03/task-3.4-interp-boost.md)
**Depends on:** Task 3.1

**Exit criteria:**

- `interp` replicating `np.interp` exactly (ascending grid, edge
  clamping, node hits) — property-tested against numpy over random
  grids.
- `boost_beta`/`boost_gamma`/`boost_delta_function` +
  `boost_integrate_linear_interp` ported with per-branch unit tests
  (interior, both partial edge cells, below-table `1/E` tail,
  above-table clamp, β→0 guard) pinned against the Cython originals
  via dedicated micro-fixtures captured in Phase 01.

### Task 3.5: Dispatch and error layer

**Task note:** [`../task-notes/phase-03/task-3.5-dispatch.md`](../task-notes/phase-03/task-3.5-dispatch.md)
**Depends on:** —

**Exit criteria:**

- One generic helper implementing the scalar-or-1D contract
  (`../references/numerics-replacements.md`, dispatch section),
  including the neutrino tuple/`(3,N)` variant; kernel crates stay
  PyO3-free (rules.md rule 8).
- Error messages byte-match the Cython ones the tests assert on.

## Exit Criteria

- All tasks complete; `cargo test` covers the foundation GIL-free; the
  scipy-comparison suite passes in CI.
- No kernel swapped yet — `hazma._core` still unreferenced by wrappers.
- Phase learnings written to `../learnings/phase-03-numerics-foundation.md`.
