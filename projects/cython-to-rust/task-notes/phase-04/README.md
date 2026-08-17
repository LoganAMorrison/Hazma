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
| 4.2 | Photon table family (kaon + eta/omega/eta′/phi) | 4.1 | **Complete (2026-08-12)** | [task-4.2-photon-table-family.md](task-4.2-photon-table-family.md) |
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

- **Five near-copies port to one implementation, and that is what
  surfaces the defects** (Task 4.2). The five tabulated photon `.pyx`
  differed only in table, parent mass and line terms; written as one
  `dnde` over seven `Spectrum` values, the five line-weight expressions
  sit in one column and two of them are visibly wrong. Neither is
  findable one file at a time, which is the general shape: **a
  parameterised port is a diff between siblings.**
- **The port has now surfaced three live 2.1.0 defects, all by writing a
  statement the original never made.** Task 4.2's two are both in the
  line terms: `_eta_prime.pyx:107` weights its `η′ → γγ` line with `BR`
  where four siblings use `2·BR` (0.02307 photons per decay instead of
  0.04614 — 0.63% of the η′ yield), and `_phi.pyx:111,113` place both
  photon lines at the **daughter meson's** energy (656.94 MeV where
  362.52 is right; 959.65 where 59.82 is right, a factor of 16). Both
  reproduced per rule 1, filed as
  [`eta-prime-two-photon-line-missing-factor-two.md`](../../../../docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md)
  and
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](../../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md),
  both blocked behind Phase 06 Task 6.4. **Four blocked defects now share
  one eventual corpus regeneration.**
- **`numpy.sum(axis=0)` is pairwise above eight terms, and exactly one
  live table is wide enough to care** (Task 4.2). The φ CSV has ten decay-mode
  columns; the other six have 2–7, where NumPy's reduction degenerates to
  a sequential fold. Reusing `boost::pairwise_sum` is what makes the
  embedded parse bit-equal — a mutation to a sequential fold fails six
  tests, all φ, and none on the other six tables.
- **Deleting an extension strands whatever read its module *globals*, not
  just whatever imported it** (Task 4.2). `test/test_core_interp.py` and
  `test/test_core_boost.py` built their seven-table fixtures from
  `_eta.eta_data_energies` and friends, and both failed at *collection* —
  so the whole suite reported two errors and ran nothing, which reads
  like a broken build rather than a stranded dependent. Repaired by
  loading the CSVs the way the deleted modules did, which also makes
  those oracles independent of the Rust that now consumes them.
- **A monkeypatch that shadows a real submodule stops measuring a delta**
  (Task 4.2). `test/parity/test_parity.py`'s served-kernel meta-tests
  patched `hazma._core.photon` with a one-kernel fake and asserted
  `baseline + 1`; once `photon` held seven real kernels the fake replaced
  seven and added one. Repointed at `hazma._core.not_a_real_domain`.
  **Any later task filling `neutrino`, `scalar_mediator` or
  `vector_mediator` would have hit the same thing.**
- **A `NaN` energy had no faithful answer, and the honest move was to
  declare a change** (Task 4.2). The Cython raised `IndexError` out of
  `np.flatnonzero(lb <= x)[0]`; the Rust panicked at an `.expect`.
  `dispatch::map_unary` has no per-element error channel, so neither type
  survives an element-wise map — the port returns `NaN`, which is what
  the same kernels' rest-frame branch already did. Declared in
  `rust/src/boost.rs`, in the numerical-impact log, and by test.
  **Expect this shape wherever a kernel's error path is per-element.**
- **The Rust and Python halves of a kernel port do not accept the same
  physics notation** (Task 4.2). `rust/src/kernels/photon_tables.rs`
  writes `η′ → γγ` and `(M² − m²)/(2M)` freely; the same strings in a
  Python docstring produce 22 `RUF002` "ambiguous unicode" findings,
  because ruff reads `γ` as a Latin `y`, `′` as a backtick, `−` as a
  hyphen and `×` as an `x`. Every other `test/test_core_*.py` is clean, so
  this is a rule the suite already follows silently and a new module has
  to learn: **spell final states the way hazma's own CSV headers do**
  (`a` for a photon — `a_a`, `pi0_a`, `eta_a`), and use ASCII `-`, `x` and
  `'`. `η`, `φ`, `ω`, `β`, `δ`, `→`, `·` and superscripts are *not*
  flagged, so the notation stays readable. Three `PLR2004` magic-value
  comparisons and one missing return annotation came with it — all four
  worth fixing rather than silencing.

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
  is the shape to copy — 47 tests, one per contract branch plus the twin
  as a two-mode oracle (bit-for-bit on the capturing platform, a
  peak-scaled budget elsewhere, nothing skipped) plus physics.
- **The ill-conditioned-corpus follow-up is waived for the tabulated
  photon family, not resolved** (Task 4.2). The waiver rests on a
  measurement: the port is bit-equal to the Cython at all 336,000 sampled
  points, so on the capturing platform there is nothing for a
  conditioning budget to absorb, and off it the parity suite does not run
  (`ci.yml` passes `--ignore=test/parity`). The follow-up's prediction
  that "every affected block will produce a false failure the moment a
  Rust implementation lands" is **refuted for `spectra.photon.eta`**; it
  stays open for the five cross-section blocks.
- **The `TABULATED` budget class is kept rather than tightened to
  `EXACT`** (Task 4.2), even though the port would pass `EXACT` today.
  Unlike `spectra.positron.muon`, bit-equality here rests on reproducing
  *NumPy's summation order* — an implementation detail a future NumPy may
  change — so `EXACT` would be the wrong contract rather than a tighter
  one.
- **One module may serve several `.pyx`** (Task 4.2), which is the stated
  exception to `kernels.rs`'s one-submodule-per-`.pyx` convention rather
  than a silent violation of it. `photon_tables` serves five.

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

### Task 4.2

- New: `rust/src/kernels/photon_tables.rs`,
  `test/test_core_photon_tables.py`,
  `docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md`,
  `docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md`.
- Changed: `rust/src/{kernels,photon,boost,interp}.rs`,
  `hazma/spectra/_photon/__init__.py`, `hazma/_core.pyi`, `setup.py`,
  `test/parity/{cases,tolerances,test_parity}.py`,
  `test/test_core_{boost,interp}.py`, `docs/followups/README.md`,
  `docs/followups/todo/{boost-integral-drops-last-interior-cell,positron-muon-spectrum-normalization-inverted}.md`.
- Deleted: `hazma/spectra/_photon/{_eta,_eta_prime,_kaon,_omega,_phi}.{pyx,pxd,pyi}`
  and `hazma/spectra/_photon/path.py` — 16 files, 1,020 lines.

## Verification

- Per task: corpus suite for the swapped entry points + full pytest +
  import smoke (mediator modules must stay importable — capi survivors
  intact).
- **After Task 4.2 (2026-08-12):** bare `pytest -q` →
  `1628 passed, 15 skipped in 587.90s`; collection 1458 → 1643 against
  `origin/master`, all +185 in `test/test_core_photon_tables.py`.
  `pytest -q test/parity` → `629 passed, 1 skipped`, all seven tabulated
  photon cases green at `TABULATED`. `cargo test --no-default-features`
  → `96 passed` (15 new). `python test/parity/generate.py --check` →
  `corpus OK: 41 cases / 1580 arrays`. `scripts/agents/preflight.sh` **RESULT: PASS**.

## Open Questions

- Run Phase 05 in parallel? (Project-level question; nothing in Phase 04
  has blocked on it so far.)
- **Should the corpus's mode switch become per-case?** Task 4.1 measured
  the cost of the global one: 22 of 41 cases now run at their declared
  budget rather than `rtol = 0`, though the 19 `EXACT`-class cases lose
  nothing. Belongs with
  [`parity-corpus-pins-ill-conditioned-points.md`](../../../../docs/followups/todo/parity-corpus-pins-ill-conditioned-points.md),
  not beside a kernel swap.
- ~~**Task 4.2 is the first task that meets one of the six
  ill-conditioned corpus blocks** (`spectra.photon.eta`). Resolve or
  explicitly waive that follow-up before starting it.~~ **Waived by Task
  4.2 on evidence**: the port is bit-equal at 336,000 sampled points, so
  the block held. The follow-up stays open for its five cross-section
  blocks; nothing in Phase 04 is expected to meet those.
- **Does the φ omit a `φ → π⁰γ` line entirely?** `BR_PHI_TO_PI0_A` is
  defined and read by nothing, and the ω adds the analogous line. Not
  settled by Task 4.2 — recorded on
  [`phi-photon-lines-use-the-daughter-meson-energy.md`](../../../../docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md)
  for whoever repairs the line energies.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 04:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file — its Goal carries the
eight-step swap recipe *and* the capi-survivor exception. Task 4.3
(`_photon/_muon`, spence) is next and **is** a capi survivor: delete its
`def`, not its file. Deleting the `_photon/_muon`, `_photon/_pion`,
`_positron/_muon` or `_positron/_pion` extensions breaks the mediator
imports.

**Currently safe to assume:**

- The foundation (interp, boost, quad, dispatch, constants) is
  unit-tested against scipy and NumPy, and Tasks 4.1–4.2 have exercised
  `constants::{pdg,derived}`, all four `boost::*`, `interp::interp` and
  `dispatch::map_unary` through eight real kernels end to end.
- `hazma._core.positron.dnde_positron_muon` is bit-equal to the `cdef`
  the mediator modules still cimport (126,182 points, 0 mismatches), so
  Task 4.6 has a verified Rust dependency to call natively.
- `hazma._core.photon` serves seven kernels; `rust/src/photon.rs` shows
  the registration shape for an entry point with a fixed second argument
  and a guard that must fail before any element is mapped.
- `boost::pairwise_sum` is `pub(crate)` and reproduces
  `numpy.sum(axis=0)` for any column count;
  `boost::boost_integrate_linear_interp` is total (a `NaN` window returns
  `NaN` rather than panicking).
- `test/test_core_{boost,interp}.py` load their photon tables from the
  CSVs, so deleting further `.pyx` will not strand them again.
- The corpus is in budget mode from Task 4.1 and **cannot be
  regenerated**. `EXACT`-class cases are still `rtol = 0`.

**Currently risky / unknown:**

- **Four blocked defects now share one eventual corpus regeneration** —
  the positron normalization (4.1), the boost integral (3.4), the η′ line
  weight and the φ line energies (both 4.2). Do not "fix" any of them in
  passing; each fails the gate that governs the remaining swaps.
- Nested-ρ drift (Task 4.5) is the project's numerical stress test —
  measure before adjusting any budget.
- Task 4.3's `spence` is the first `SPECFUN`-class swap in this phase
  (1e-13), and Task 3.2's finding stands: the plan's model of a
  third-party library is a hypothesis, and only the sweep can refute it.
