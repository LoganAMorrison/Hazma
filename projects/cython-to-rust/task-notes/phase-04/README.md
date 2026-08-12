# Working Memory: Phase 04 — Spectra kernels

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 04
**Status:** In Progress
**Plan References:** `../../phases/phase-04-spectra-kernels.md`
**Related ADRs:** ADR-0001, ADR-0002
**Depends On:** Phase 03 complete

## Objective

Track live per-task status and phase-scoped findings for the spectra
kernel ports.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 4.1 | `_positron/_muon` (template swap) | — | **Complete (2026-08-11)** | [task-4.1-positron-muon.md](task-4.1-positron-muon.md) |
| 4.2 | Photon table family (kaon + eta/omega/eta′/phi) | 4.1 | Not started | [task-4.2-photon-table-family.md](task-4.2-photon-table-family.md) |
| 4.3 | `_photon/_muon` (spence) | 4.1 | Not started | [task-4.3-photon-muon.md](task-4.3-photon-muon.md) |
| 4.4 | `_photon/_pion` | 4.3 | Not started | [task-4.4-photon-pion.md](task-4.4-photon-pion.md) |
| 4.5 | `_photon/_rho` (nested quad) | 4.4 | Not started | [task-4.5-photon-rho.md](task-4.5-photon-rho.md) |
| 4.6 | `_positron/_pion` + neutrino pair | 4.1, 4.3 | Not started | [task-4.6-positron-pion-neutrino.md](task-4.6-positron-pion-neutrino.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-04-spectra-kernels.md`.

## Inputs Reviewed

- `../../phases/phase-04-spectra-kernels.md` (incl. the capi-survivor
  exception in its Goal); `../README.md`;
  `../../references/cython-inventory.md` (cimport DAG).

## Findings

- **The port surfaced a second live 2.1.0 numerical defect** (Task 4.1).
  `hazma/spectra/_positron/_muon.pyx` **divides** by the Michel
  normalization `R_FACTOR` where it should multiply, so every positron
  spectrum is low by `1/R_FACTOR²` — **0.0374%**, uniformly, propagating
  through `dnde_positron_charged_pion` and both mediator positron
  modules. The sibling `hazma/spectra/_neutrino/_muon.pyx` declares the
  same constant and multiplies by it, which is what makes this an
  inversion rather than a convention. Reproduced per rule 1 and filed as
  [`positron-muon-spectrum-normalization-inverted.md`](../../../../docs/followups/todo/positron-muon-spectrum-normalization-inverted.md),
  blocked behind Phase 06 Task 6.4. **Found the same way Task 3.4 found
  the boost-integral defect:** by writing an analytic test the original
  never had. Every task in this phase should write one.
- **Disassemble before porting** (Task 4.1). `objdump -d` the shipped
  `.so` and read the `fmadd`/`fmsub` sites; `_positron/_muon` has nine,
  and three expressions that look fusable are not (`x² − 4r²`,
  `1 − β²`, and any sum whose operand went through a division). Written
  from the map, the port was bit-equal on the first build — no
  bisection round, unlike Task 3.4.
- **Scope a bit-equality-against-Cython class to the corpus's capturing
  platform, never to a "does this compiler contract" probe** (Task 4.1,
  learned from two CI failures after two green macOS runs). The probe
  asks the wrong question: a compiler contracting a *different* set of
  expressions, or a libm rounding one call differently, breaks the
  comparison just as thoroughly, and no probe over one mechanism sees the
  others. `test/test_core_positron_muon.py` now reads the platform out of
  `test/parity/data/manifest.json`, which is the mechanism `test/parity`
  and `ci.yml` already use. **Copy that, not a probe.**
- **The capturing platform cannot see a bug in its own skip logic.** On
  macOS the probe answered True whether or not it was right, so every
  local run was green and no test in the module could tell a working
  guard from a broken one. Expect to learn this class from CI, and read
  a Linux failure in a bit-equality test as "the scope is wrong" before
  "the port is wrong".
- **A fused Python reference (correctly-rounded `fma` via `Fraction`)
  reproduces the shipped macOS Cython bit-for-bit** — 0 mismatches in
  21,000 points for `_positron/_muon`, against 11,713 for the unfused
  form. A cheap second confirmation of an FMA map, independent of the
  disassembly. It says nothing about other platforms, which is exactly
  why the scope above is a platform.
- **Repointing the corpus case is part of the swap, not bookkeeping**
  (Task 4.1). `cases.py` names the `.pyx` module; leave it and the gate
  keeps calling the twin while the wrapper calls Rust — green and
  vacuous. `PORTED_ENTRY_POINTS` records the origin so
  `assert_full_coverage` still balances, and now also fails if a ported
  entry point's `.pyx` still exports its `def`.
- **A `NaN` energy does not propagate through a kernel that clips with
  `fmax`/`fmin`** (Task 4.1), in either language: both limits collapse
  onto the rest-frame support and a finite number comes back. The corpus
  samples no `NaN`, so only a hand-written test catches a port that
  differs. Expect the same shape in every boosted kernel.

## Decisions and Implementation Notes

- **The per-kernel swap recipe now lives in the phase file's Goal**
  (Task 4.1), so it is canonical rather than inferred from one task
  note. Eight steps, of which "map the FMAs first" and "repoint the
  corpus case" are the two that a reader would otherwise skip.
- **A capi survivor loses its `def`, not its file** (Task 4.1) — the
  `cdef`s and their `__pyx_capi__` capsules stay, so the mediator
  modules keep importing while no Python caller can reach the replaced
  implementation. Tasks 4.3 and 4.4 (the other two survivors) do the
  same.
- **Per-kernel test modules do not copy `test/test_core_dispatch.py`**
  (Task 4.1), reversing Task 2.3's instruction: since Task 3.5 the
  dispatch layer is three shared helpers, so those 118 tests cover code
  every kernel routes through unchanged. `test/test_core_positron_muon.py`
  is the shape to copy — 45 tests, one per contract branch plus
  bit-equality against the twin (17, scoped to the capturing platform)
  plus physics.

## Files Changed

### Task 4.1

- New: `rust/src/kernels/positron_muon.rs`,
  `test/test_core_positron_muon.py`,
  `docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`.
- Changed: `rust/src/{kernels,positron}.rs`,
  `hazma/spectra/_positron/{__init__.py,_muon.pyx}`, `hazma/_core.pyi`,
  `test/parity/{cases,test_parity}.py`, `docs/followups/README.md`,
  `../../phases/phase-04-spectra-kernels.md`.
- Deleted: `hazma/spectra/_positron/_muon.pyi`.

## Verification

- Per task: corpus suite for the swapped entry points + full pytest +
  import smoke (mediator modules must stay importable — capi survivors
  intact).

## Open Questions

- Run Phase 05 in parallel once 4.1 lands? (Project-level question.)
- **Should the corpus's mode switch become per-case?** Task 4.1 measured
  the cost of the global one: 22 of 41 cases now run at their declared
  budget rather than `rtol = 0`, though the 19 `EXACT`-class cases lose
  nothing. Belongs with
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md),
  not beside a kernel swap.
- **Task 4.2 is the first task that meets one of the six
  ill-conditioned corpus blocks** (`spectra.photon.eta`). Resolve or
  explicitly waive that follow-up before starting it.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 04:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file — its Goal now carries
the eight-step swap recipe *and* the capi-survivor exception. Deleting
the `_photon/_muon`, `_photon/_pion`, `_positron/_muon` or
`_positron/_pion` extensions here breaks the mediator imports; deleting
their Python `def` is what the swap does instead.

**Currently safe to assume:**

- The foundation (interp, boost, quad, dispatch, constants) is
  unit-tested against scipy and NumPy, and Task 4.1 has now exercised
  `constants::derived`, `boost::boost_beta` and `dispatch::map_unary`
  through a real kernel end to end.
- `hazma._core.positron.dnde_positron_muon` is bit-equal to the `cdef`
  the mediator modules still cimport (126,182 points, 0 mismatches), so
  Task 4.6 has a verified Rust dependency to call natively.
- The corpus is in budget mode from Task 4.1 and **cannot be
  regenerated**. `EXACT`-class cases are still `rtol = 0`.

**Currently risky / unknown:**

- **Task 4.2** meets `spectra.photon.eta` — one of the six
  ill-conditioned blocks *and* the first `TABULATED` swap, now budgeted
  at 1e-12 against a Task 3.4 measurement that says unfused arithmetic
  misses by up to 3.6e-12. Read that follow-up and Task 3.4's note
  together first.
- Nested-ρ drift (Task 4.5) is the project's numerical stress test —
  measure before adjusting any budget.
- The positron normalization defect will be reproduced again by Task 4.6
  and by Phase 06. Do not "fix" it in passing.
