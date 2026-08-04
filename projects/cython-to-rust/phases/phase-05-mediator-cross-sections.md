---
phase: 05
title: Mediator cross sections
status: Not started
---

<!-- markdownlint-disable-file MD025 -- frontmatter title is the schema -->

# Phase 05: Mediator cross sections

## Goal

Port `_c_vector_mediator_cross_sections` (6 kernels) and
`_c_scalar_mediator_cross_sections` (13 kernels) plus both
`thermal_cross_section`s to `hazma._core`. These are self-contained
(no hazma cimports — only `libc.math` + `k1`/`kn`), so they depend
only on the Phase 03 foundation and can run in parallel with Phase 04
if staffing allows (they share no files with it).

## Prerequisites

- Phase 03 complete (specfun + qags for the thermal integrals; the
  dispatch helper).
- `../references/cython-inventory.md` (structure: three-tier layout,
  Mathematica-dump characteristics) and `../rules.md` rules 1–3, 12.

## Tasks

### Task 5.1: Vector cross sections (template)

**Task note:** [`../task-notes/phase-05/task-5.1-vector-xs.md`](../task-notes/phase-05/task-5.1-vector-xs.md)
**Depends on:** —

**Exit criteria:**

- The 6 vector kernels transliterated **mechanically** (scripted or
  expression-by-expression copy, never retyped — rules.md; the
  expressions are Mathematica dumps where a dropped digit is silent);
  tiers 2–3 replaced by the generic dispatch helper.
- `thermal_cross_section` on Rust `qagp` (breakpoints `[2, mv/mx,
  2mv/mx]`) + `bessel_k1`/`bessel_kn`.
- Corpus green across the parameter grid incl. near-resonance; the
  unused `sigma_xx_to_all` export dropped from the Python wrapper
  surface only if truly unimported (verify at execution time).
- Wrapper `_vector_mediator_cross_sections.py` swapped; Cython twin
  deleted.

### Task 5.2: Scalar cross sections

**Task note:** [`../task-notes/phase-05/task-5.2-scalar-xs.md`](../task-notes/phase-05/task-5.2-scalar-xs.md)
**Depends on:** Task 5.1 (reuses its layout and dispatch pattern)

**Exit criteria:**

- All 13 scalar kernels ported, incl. the two 90-line expressions
  (`__sigma_xpi_to_xpi`, `__sigma_xpi0_to_xpi0`) — factor the 8×
  repeated subexpression into a named local (identical arithmetic
  order; confirm no value shift).
- `np.log(4)` at line 283 becomes the constant `LN_4` (value change:
  none — record in task note).
- Corpus green; wrapper swapped; twin deleted.

### Task 5.3: Thermal ⟨σv⟩ validation sweep

**Task note:** [`../task-notes/phase-05/task-5.3-thermal-sweep.md`](../task-notes/phase-05/task-5.3-thermal-sweep.md)
**Depends on:** Tasks 5.1, 5.2

**Exit criteria:**

- End-to-end check through the live consumer:
  `hazma/relic_density/_thermal_functions.py` paths that call
  `thermal_cross_section` produce relic densities matching pre-port
  values within the corpus budget (extend the corpus with one
  relic-density scenario per mediator if Phase 01 didn't).
- Benchmark recorded (rules.md rule 12) — this path re-entered Python
  per Bessel evaluation and per quad node before; the speedup is the
  headline side effect.

## Exit Criteria

- 19 cross-section kernels + 2 thermal functions on Rust; both
  `_c_*` `.pyx` files deleted; corpus + relic-density checks green.
- Drift table updated in `../task-notes/README.md`.
- Phase learnings written to `../learnings/phase-05-mediator-cross-sections.md`.
