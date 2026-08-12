---
phase: 04
title: Spectra kernels
status: In Progress
---

# Phase 04: Spectra kernels

## Goal

Port all 13 `hazma/spectra/_{photon,positron,neutrino}` kernels to
`hazma._core` and swap the three wrapper `__init__.py`s. 16 public
entry points move; every swap gates on the corpus.

**Scoped exception to rules.md rule 1 (delete-twin-same-PR):** the four
extensions the mediator spectrum `.pyx` files cimport C symbols from —
`_photon/_muon`, `_photon/_pion`, `_positron/_muon`, `_positron/_pion` —
stay **built but Python-unreferenced** through this phase (their
`__pyx_capi__` capsules keep the still-Cython mediator modules
importable). They, the other spectra twins' shared `.pxd` headers, and
`hazma/_utils/{boost,constants,kinematics}` are deleted in Phase 06
Task 6.4 once the last cimporter is gone. All other twins (rho, kaon,
eta family, neutrino pair) delete in their swap PR as usual.

"Python-unreferenced" is literal, and Task 4.1 settled how: a survivor's
top-level `def` is **deleted in the swap PR** while its `cdef`s stay, so
no Python caller can reach the implementation the swap replaced. That is
as close to rule 1 as the exception allows.

## The per-kernel swap recipe

Established by Task 4.1 and copied by every later task in this phase and
in Phases 05–06. Steps 1 and 5 are the two most often skipped.

1. **Map the FMAs before writing any Rust.** `objdump -d` the shipped
   `.so` for `fmadd`/`fmsub`; clang contracts `a*b + c` by default and
   the corpus was captured from a build that does. Which expressions
   contract is a per-expression fact — Task 4.1 found three that look
   fusable and are not — so read it, do not pattern-match. Task 3.4's
   note has the method; `rust/src/boost.rs` has the rationale.
2. **Port into `rust/src/kernels/<pyx name>.rs`**, PyO3-free, one
   submodule per ported `.pyx`. Fold the same compile-time constants the
   generated C folds and pin them against the disassembled immediates.
3. **Register in the per-domain module** (`photon`, `positron`, …)
   through `dispatch::map_unary` / `map_flavors`, passing the quantity
   wording the twin's `assert` used.
4. **Repoint the wrapper** in `hazma/spectra/**/__init__.py`.
5. **Repoint the corpus case** in `test/parity/cases.py` from the `.pyx`
   module to the wrapper, and add a `PORTED_ENTRY_POINTS` row recording
   the origin. Without this the corpus keeps calling the twin and the
   gate is vacuous. `assert_full_coverage` then also fails if the origin
   still exports its `def`.
6. **Delete the twin** — the whole `.pyx` where it is not a capi
   survivor, its `def` where it is — plus any `.pyi` stub for it.
7. **Add `test/test_core_<kernel>.py`**, shaped like
   `test/test_core_positron_muon.py`: the dispatch contract with this
   kernel's wording (one assertion per branch — the branch-by-branch
   argument stays in `test/test_core_dispatch.py`, which every kernel
   now shares rather than copies), the twin as a two-mode oracle
   (bit-for-bit on the platform the parity corpus was captured on, a
   budget scaled to the spectrum's peak elsewhere), and physics that
   outlives the Cython.
8. **Record the drift** in the task note and in `task-notes/README.md`'s
   "Numerical impact so far", per rules 2–3.

## Prerequisites

- Phase 03 complete. Read the cimport DAG in
  `../references/cython-inventory.md` — order within this phase follows
  it (muon before pion before rho; struct module before neutrino pair).
- `../rules.md` rules 1–3 (parity discipline) and 9 (edge guards).

## Parts

### Part 1: Closed-form kernels (no quadrature)

Cheapest first — exercises constants + dispatch on pure math.

### Part 2: Table-driven family (interp + boost integral)

### Part 3: Quadrature-backed kernels (the drift-sensitive ones)

## Tasks

### Task 4.1: `_positron/_muon` (walking-skeleton kernel)

**Task note:** [`../task-notes/phase-04/task-4.1-positron-muon.md`](../task-notes/phase-04/task-4.1-positron-muon.md)
**Depends on:** —

**Exit criteria:**

- `dnde_positron_muon` on Rust; corpus exact-or-≤1e-13; wrapper
  swapped. (Twin is a Phase 06 capi survivor — see Goal.)
- Establishes the per-kernel PR template (port → corpus diff → swap →
  delete-or-defer → drift note) later tasks copy.

### Task 4.2: Photon table family (`_kaon`, `_eta`, `_omega`, `_eta_prime`, `_phi`)

**Task note:** [`../task-notes/phase-04/task-4.2-photon-table-family.md`](../task-notes/phase-04/task-4.2-photon-table-family.md)
**Depends on:** Task 4.1

**Exit criteria:**

- One shared Rust implementation parameterized by (embedded table,
  mass, delta terms); the 7 CSVs under `spectra/_photon/data/` embedded
  via `include_str!` parsed once at init (CSVs stay in-repo as source
  of truth).
- 7 entry points swapped, corpus-green; Cython twins + their ~170 lines
  of commented-out dead code gone.
- Import-time file I/O for these modules eliminated (note the
  package-data globs to retire in Phase 07).

### Task 4.3: `_photon/_muon` (spence)

**Task note:** [`../task-notes/phase-04/task-4.3-photon-muon.md`](../task-notes/phase-04/task-4.3-photon-muon.md)
**Depends on:** Task 4.1

**Exit criteria:**

- `dnde_photon` (radiative muon decay, hep-ph/9909265 spectrum incl.
  the `spence(xm) - spence(xp)` term) corpus-green at ≤1e-12 rel.
- This kernel stays cimport-compatible-in-spirit: its Rust `fn` is
  callable natively by Phase 06 (mediator spectra) — keep it in the
  PyO3-free kernel layer.

### Task 4.4: `_photon/_pion`

**Task note:** [`../task-notes/phase-04/task-4.4-photon-pion.md`](../task-notes/phase-04/task-4.4-photon-pion.md)
**Depends on:** Task 4.3 (cimports muon point function)

**Exit criteria:**

- Both entry points (charged: π→ℓνγ radiative + boosted-μ `qagp` over
  cosθ; neutral: π⁰→γγ box) corpus-green within the quad budget.
- First real `qagp` consumer — record measured drift vs the corpus in
  the task note and tighten the budget if warranted.

### Task 4.5: `_photon/_rho` (nested quadrature)

**Task note:** [`../task-notes/phase-04/task-4.5-photon-rho.md`](../task-notes/phase-04/task-4.5-photon-rho.md)
**Depends on:** Task 4.4

**Exit criteria:**

- Both ρ entry points corpus-green; the nested integral (ρ quad over
  `_pion`, which quads over `_muon`) gets a dedicated drift analysis in
  the task note (this is the project's numerical stress test).
- The Cython version's untyped `cdef` locals (Python-boxed) are ported
  as plain f64 — confirm no value shift beyond budget.

### Task 4.6: `_positron/_pion` + neutrino pair (`_muon`, `_pion`, struct)

**Task note:** [`../task-notes/phase-04/task-4.6-positron-pion-neutrino.md`](../task-notes/phase-04/task-4.6-positron-pion-neutrino.md)
**Depends on:** Tasks 4.1, 4.3 (neutrino `_pion` boosts the muon spectrum)

**Exit criteria:**

- `dnde_positron_charged_pion`, `dnde_neutrino_muon`,
  `dnde_neutrino_charged_pion` corpus-green; the `NeutrinoSpectrumPoint`
  struct becomes a plain Rust struct; tuple/`(3,N)` return contract
  verified against existing wrapper tests.
- Neutrino/rho/kaon/eta-family twins deleted; only the four capi
  survivors (see Goal) plus `_utils` headers remain as `.pyx`/`.pxd`
  under `spectra/` + `_utils/`.

## Exit Criteria

- All 16 spectra entry points served by `hazma._core`; corpus green in
  CI; cumulative drift table recorded in
  `../task-notes/README.md` ("Numerical impact so far").
- Remaining Cython under `hazma/spectra/` + `hazma/_utils/` is exactly
  the four capi survivors and their headers, each still built and
  importable (mediator modules unbroken — CI import smoke proves it).
- Phase learnings written to `../learnings/phase-04-spectra-kernels.md`.
