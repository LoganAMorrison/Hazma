---
phase: 05
title: Mediator cross sections
status: In Progress
---

# Phase 05: Mediator cross sections

## Goal

Port the **18 consumed public defs** of the two cross-section modules
to `hazma._core` and drop the 2 unconsumed exports. Exact accounting
(source-verified 2026-08-03; re-verify imports at execution time):

- `_c_scalar_mediator_cross_sections` — 13 defs, **12 consumed**:
  `sigma_xx_to_s_to_{ff,gg,pi0pi0,pipi}`, `sigma_xx_to_ss`,
  `sigma_ss_to_xx`, `sigma_x{l,pi,pi0,g,s}_to_x{l,pi,pi0,g,s}` (11
  cross-section kernels) + `thermal_cross_section`. The 13th def,
  `sigma_xx_to_all`, is imported by nothing → dropped, not ported.
- `_c_vector_mediator_cross_sections` — 7 defs, **6 consumed**:
  `sigma_xx_to_v_to_{ff,pipi,pi0g,pi0v}`, `sigma_xx_to_vv` (5
  cross-section kernels) + `thermal_cross_section`. Its
  `sigma_xx_to_all` is likewise unimported → dropped.

Total ported: 16 cross-section kernels + 2 thermal functions = 18
consumed defs; with the 16 spectra and 7 mediator-spectrum entry
points this reproduces the project's 43-defs / 41-consumed tally.
These modules are self-contained (no hazma cimports — only
`libc.math` + `k1`/`kn`), so they depend only on the Phase 03
foundation and can run in parallel with Phase 04 if staffing allows
(they share no files with it).

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

- The 5 consumed vector kernels transliterated **mechanically**
  (scripted or expression-by-expression copy, never retyped —
  rules.md; the expressions are Mathematica dumps where a dropped
  digit is silent); tiers 2–3 replaced by the generic dispatch helper.
- `thermal_cross_section` on Rust `qagp` (breakpoints `[2, mv/mx,
  2mv/mx]`, incl. the out-of-interval regime — Task 3.3 contract) +
  `bessel_k1`/`bessel_kn`.
- `sigma_xx_to_all` dropped after re-running the importer check
  (`rg sigma_xx_to_all` outside the `_c_*` modules must be empty) and
  quoting it in the PR body.
- Corpus green across the parameter grid incl. near-resonance.
- Wrapper `_vector_mediator_cross_sections.py` swapped; Cython twin
  deleted.

### Task 5.2: Scalar cross sections

**Task note:** [`../task-notes/phase-05/task-5.2-scalar-xs.md`](../task-notes/phase-05/task-5.2-scalar-xs.md)
**Depends on:** Task 5.1 (reuses its layout and dispatch pattern)

**Exit criteria:**

- All 11 consumed scalar kernels ported, incl. the two 90-line
  expressions (`__sigma_xpi_to_xpi`, `__sigma_xpi0_to_xpi0`) — factor
  the 8× repeated subexpression into a named local (identical
  arithmetic order; confirm no value shift). `sigma_xx_to_all`
  dropped under the same importer-check rule as Task 5.1.
- `thermal_cross_section` on Rust `qagp` (breakpoints
  `[2, ms/mx, 2ms/mx]`).
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

- 16 cross-section kernels + 2 thermal functions (18 consumed defs)
  on Rust; the 2 `sigma_xx_to_all` exports dropped; both `_c_*`
  `.pyx` files deleted; corpus + relic-density checks green.
- Drift table updated in `../task-notes/README.md`.
- Phase learnings written to `../learnings/phase-05-mediator-cross-sections.md`.
