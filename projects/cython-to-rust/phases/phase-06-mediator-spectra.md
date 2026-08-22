---
phase: 06
title: Mediator spectra
status: Not started
---

# Phase 06: Mediator spectra

## Goal

Replace the four mediator decay/positron spectrum modules — the last
Cython — with a Rust redesign, then delete the four capi-survivor
spectra twins and the `_utils` headers Phase 04 left behind. This is a
**redesign, not a transliteration**: the Cython versions use mutable
module-global interp tables with a broken memo-cache, string-keyed mode
dispatch inside the integrand, and cross-extension cimports (the 8 cdef
symbols) — none of which survives contact with Rust.

## Prerequisites

- Phases 04 and 05 complete.
- `../references/cython-inventory.md` (the 8-symbol cimport list, the
  dead-cache bug) and `../references/numerics-replacements.md`.

## Tasks

### Task 6.1: Spectrum-table struct design

**Task note:** [`../task-notes/phase-06/task-6.1-table-struct.md`](../task-notes/phase-06/task-6.1-table-struct.md)
**Depends on:** —

**Exit criteria:**

- A Rust struct owning the precomputed 500-point log-spaced rest-frame
  tables (built once per parameter set by calling the Phase 04 kernel
  `fn`s natively — no Python round trips), with genuine memoization
  keyed on the mediator mass + partial widths (fixing the dead-cache
  bug; same numbers, declared as performance-only per rules.md 3, 12).
- Mode dispatch (`"total"`, `"e e g"`, `"pi pi g"`, …) becomes an enum
  at the PyO3 boundary; string parsing happens once per call, not per
  quadrature node. Accepted strings and error text byte-match today's.
- Design reviewed against both decay + both positron modules before
  implementation (they are two clone-pairs — one parameterized
  implementation each).

### Task 6.2: Decay spectrum pair (`scalar`, `vector`)

**Task note:** [`../task-notes/phase-06/task-6.2-decay-spectra.md`](../task-notes/phase-06/task-6.2-decay-spectra.md)
**Depends on:** Task 6.1

**Exit criteria:**

- `scalar_mediator_decay_spectrum`, `dnde_decay_v`/`dnde_decay_v_pt`
  on Rust; corpus green (quad budget); wrappers swapped; both Cython
  twins deleted.
- Benchmark vs pre-swap Cython recorded (expected large win — the old
  path rebuilt two quad-backed tables per call).

### Task 6.3: Positron spectrum pair

**Task note:** [`../task-notes/phase-06/task-6.3-positron-spectra.md`](../task-notes/phase-06/task-6.3-positron-spectra.md)
**Depends on:** Task 6.1

**Exit criteria:**

- `dnde_decay_s`/`dnde_decay_s_pt` (scalar) and the vector clone on
  Rust; corpus green; wrappers swapped; twins deleted.

### Task 6.4: Retire the capi survivors and `_utils` headers

**Task note:** [`../task-notes/phase-06/task-6.4-retire-survivors.md`](../task-notes/phase-06/task-6.4-retire-survivors.md)
**Depends on:** Tasks 6.2, 6.3

**Exit criteria:**

- `rg "cimport|__pyx_capi__|\.pxd"` over `hazma/` confirms zero
  consumers; then delete the four capi-survivor extensions
  (`_photon/_muon`, `_photon/_pion`, `_positron/_muon`,
  `_positron/_pion` `.pyx`+`.pxd`), `hazma/_utils/boost.{pyx,pxd}`,
  `constants.pxd`, `kinematics.pxd`, `legacy_parameters.pxd`, and the
  `spectra/_neutrino/_neutrino` struct module.
- `find hazma -name "*.pyx" -o -name "*.pxd"` returns **nothing**;
  `setup.py` builds only `hazma._core`; full suite + corpus green.

## Exit Criteria

- Zero Cython in the tree; all 41 consumed entry points on
  `hazma._core`.
- Drift table complete in `../task-notes/numerical-impact.md` — this
  is the input to Phase 07's CHANGELOG aggregation.
- Phase learnings written to `../learnings/phase-06-mediator-spectra.md`.
