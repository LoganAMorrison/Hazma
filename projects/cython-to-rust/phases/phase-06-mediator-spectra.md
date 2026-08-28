---
phase: 06
title: Mediator spectra
status: Complete
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
  keyed on the mediator mass (fixing the dead-cache bug; same numbers,
  declared as performance-only per rules.md 3, 12).

  **Amended by Task 6.1 (2026-08-23):** this bullet read "keyed on the
  mediator mass + partial widths", copying the key the Cython's dead
  predicate *declares*. `__set_spectra` takes only the mass and reads no
  width, so the tables are a pure function of the mass; keying on the
  widths too would be equally correct and strictly slower, rebuilding
  both tables whenever a caller varies a coupling at fixed mass — the
  sweep the cache exists to make cheap.
- Mode dispatch (`"total"`, `"e e g"`, `"pi pi g"`, …) becomes an enum
  at the PyO3 boundary; string parsing happens once per call, not per
  quadrature node. Accepted strings byte-match today's.

  **Amended by Task 6.1 (2026-08-23):** this bullet read "Accepted
  strings and error text byte-match today's". There is no error text to
  match: an unrecognised mode raises nothing today, it returns `0.0` —
  every `cdef double` integrand ends in an `if`-chain with no `else` and
  a C function that falls off its end returns zero. Reproduced under
  rule 1 and filed as
  [`../../../docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`](../../../docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md).
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
  `constants.pxd`, `kinematics.pxd` and `legacy_parameters.pxd`.

  **Amended by Task 6.4 (2026-08-27):** this bullet also listed "the
  `spectra/_neutrino/_neutrino` struct module". Phase 04 Task 4.6 had
  already deleted it, together with `_neutrino/{_muon,_pion}.pyx`,
  because nothing outside that package cimported any of the three — so
  there was no capi survivor to retire here. Corrected rather than left
  naming a file this task cannot find. The count is therefore **14
  tracked files**, not thirteen: `hazma/_utils/kinematics.pyx.bak`, a
  tracked backup shadowing the header beside it, went with them.
- `find hazma -name "*.pyx" -o -name "*.pxd"` returns **nothing**;
  `setup.py` builds only `hazma._core`; full suite + corpus green.

  **Extended by Task 6.4 (2026-08-27):** the tree-wide half of this is
  asserted in `test/test_no_cython_remains.py` rather than left to a
  command someone remembers to run — no Cython source anywhere, no
  Cython in `setup.py`'s declarations, no Cython toolchain in
  `[build-system] requires`, and no transpiler-output glob in
  `MANIFEST.in`. Deleting the sources also required repairing the six
  test modules whose oracle was one of them; that work is not implied by
  the wording above and is recorded in the task note.

## Exit Criteria

- Zero Cython in the tree; all 41 consumed entry points on
  `hazma._core`.
- Drift table complete in `../task-notes/numerical-impact.md` — this
  is the input to Phase 07's CHANGELOG aggregation.
- Phase learnings written to `../learnings/phase-06-mediator-spectra.md`.
