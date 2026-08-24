# Working Memory: Phase 06 — Mediator spectra

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 06
**Status:** In Progress
**Plan References:** `../../phases/phase-06-mediator-spectra.md`
**Related ADRs:** ADR-0001, ADR-0002
**Depends On:** Phases 04 and 05 complete

## Objective

Track live per-task status and phase-scoped findings for the mediator
spectrum redesign — the last Cython.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 6.1 | Spectrum-table struct design | — | **Complete (2026-08-23)** | [task-6.1-table-struct.md](task-6.1-table-struct.md) |
| 6.2 | Decay spectrum pair | 6.1 | Not started | [task-6.2-decay-spectra.md](task-6.2-decay-spectra.md) |
| 6.3 | Positron spectrum pair | 6.1 | Not started | [task-6.3-positron-spectra.md](task-6.3-positron-spectra.md) |
| 6.4 | Retire capi survivors + `_utils` headers | 6.2, 6.3 | Not started | [task-6.4-retire-survivors.md](task-6.4-retire-survivors.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`;
  `find hazma -name "*.pyx" -o -name "*.pxd"` empty.
- Phase learnings at `../../learnings/phase-06-mediator-spectra.md`.

## Inputs Reviewed

- `../../phases/phase-06-mediator-spectra.md`; `../README.md`; both
  references (8-symbol cimport list; dead-cache bug; dispatch
  contract).

## Findings

Phase-scoped, from Task 6.1; the full evidence is in its note.

- **The four `.pyx` differ in exactly four ways and all four are data:**
  grid start (`10⁻¹` MeV vs legacy `m_e`), below-grid policy (`1/E` tail
  vs `np.interp`'s clamp), which tables are built, and the selector type
  (`list[str]` → bitflag for the scalar decay module, one `str` for the
  other six entry points). One parameterised implementation each, as the
  phase file predicted.
- **`grep -c SoftComplexToDouble` on the generated C: 6 / 0 / 6 / 0** for
  scalar-decay / scalar-positron / vector-decay / vector-positron — the
  check the Phase 05 learnings record as unrun for this phase. Five of
  each six are proto/definition lines; the one live site each is the
  `** 1.5` in an FSR coefficient
  (`scalar_mediator_decay_spectrum.pyx:113`,
  `vector_mediator_decay_spectrum.pyx:73`), both already covered by
  `crate::kernels::soft_complex`. **Neither positron module needs it.**
- **An unrecognised mode string returns `0.0`, not an error** — verified
  against the shipped extensions, reproduced under rule 1, filed as
  [`../../../../docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`](../../../../docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md).
- **The dead cache is two different bugs.** The decay modules have no
  cache; the positron modules have the predicate and never assign to
  what it reads. Both rebuild a 500-point quadrature-backed table per
  call.
- **`__set_spectra` reads only the mass**, so the tables are a pure
  function of it and the Cython's declared width-bearing key is wider
  than its inputs.

## Decisions and Implementation Notes

- **Task 6.1 amended two of its own exit criteria in the phase file**
  (cache key, and "error text"). See its `## Plan Impact`. No ADR.
- **`crate::kernels::mediator_tables`** is the shared foundation, a
  documented naming exception beside `soft_complex`;
  **`hazma._core.mediator_tables`** is a sixth `_CORE_TEST_ONLY_MODULES`
  probe, because every oracle (`numpy.logspace`, `numpy.interp`, the
  Phase 04 entry points, the four live twins) is in Python.
- **`cargo` gates the grid algorithm, Python gates its agreement with
  NumPy** — captured NumPy bits in a cargo test would turn a Linux CI
  job red for a libm difference (Phase 04 learnings §4).

## Files Changed

### Task 6.1

`rust/src/kernels/mediator_tables.rs` (new), `rust/src/mediator_tables_probe.rs`
(new), `rust/src/kernels.rs`, `rust/src/lib.rs`,
`test/test_core_mediator_tables.py` (new), `test/parity/cases.py`,
`../../phases/phase-06-mediator-spectra.md`,
`docs/followups/{todo/mediator-spectra-accept-unknown-mode-strings.md,README.md}`.

## Verification

- Corpus (quad budgets); benchmark vs pre-swap Cython (dead-cache fix
  is the expected headline); import smoke after 6.4.
- **After Task 6.1:** `cargo test --no-default-features` → `222 passed`
  (201 at Phase 05's close); `pytest -q` → `2158 passed, 15 skipped, 12
  subtests passed`; `pytest test/parity -q` → `658 passed, 1 skipped`;
  `pytest test/test_theory_aggregation.py -q` → `69 passed`. No public
  value moved — nothing appended to `../numerical-impact.md`.

## Open Questions

- **Does the charged pion's forward-cone defect reach the mediator
  spectra?** Still open — carried in from Phase 04 and owed by Task 6.2,
  which builds the boost integral over exactly the affected kernel.
- **Is one cache slot enough, and should the shared photon table set stop
  building the muon table the scalar module ignores?** Both are
  performance questions Task 6.2's benchmark answers.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 06:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file. This is a redesign —
review Task 6.1's design against all four modules before writing
kernel code.

**Currently safe to assume:** the Phase 04 kernel `fn`s are natively
callable from Rust (rules.md rule 8 kept them PyO3-free).

**Currently risky / unknown:** memoization keying (mediator mass +
partial widths) must match how model classes mutate parameters —
verify against `hazma/scalar_mediator/__init__.py` setter behavior
before trusting the cache.
