# Working Memory: Phase 03 — Numerics foundation

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 03
**Status:** In Progress (Task 3.1 complete 2026-08-09)
**Plan References:** `../../phases/phase-03-numerics-foundation.md`
**Related ADRs:** ADR-0002 (Accepted 2026-08-04 — governs the
provenance of Tasks 3.2/3.3, no longer gates them)
**Depends On:** Phase 02 complete

## Objective

Track live per-task status and phase-scoped findings for the numerics
foundation.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 3.1 | Constants module | — | **Complete (2026-08-09)** | [task-3.1-constants.md](task-3.1-constants.md) |
| 3.2 | Special functions | — (ADR-0002 accepted) | Not started | [task-3.2-specfun.md](task-3.2-specfun.md) |
| 3.3 | QUADPACK port (qk15/qk21/qelg/qags/qagp) | — (ADR-0002 accepted) | Not started | [task-3.3-quadpack.md](task-3.3-quadpack.md) |
| 3.4 | Interpolation + boost kernels | 3.1 | Not started | [task-3.4-interp-boost.md](task-3.4-interp-boost.md) |
| 3.5 | Dispatch and error layer | — | Not started | [task-3.5-dispatch.md](task-3.5-dispatch.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-03-numerics-foundation.md`.

## Inputs Reviewed

- `../../phases/phase-03-numerics-foundation.md`; `../README.md`;
  `../../references/numerics-replacements.md` (full).

## Findings

- **`hazma/spectra/_photon/_pion.pyx` reads from both constant tables at
  once** (Task 3.1). It `include`s `constants.pxd`, so its `MPI` / `ME` /
  `MMU` aliases are PDG values — but its five hard-coded kinematic
  literals (`ENG_MU_PIRF`, `GAMMA_MU_PIRF`, `BETA_MU_PIRF`,
  `ENG_GAM_MAX_MURF`, `ENG_GAM_MAX_PIRG`) reproduce **bit-exactly** from
  the *legacy* masses and from no other table. Recomputing them from the
  header the file includes moves `ENG_MU_PIRF` by 4.7e-5 MeV and every
  charged-pion photon spectrum with it. Preserved as-is per rule 4 and
  pinned in both languages; **Phase 04 must not consolidate it.**
- **The two `.pxd` share 19 names and disagree on 12** (Task 3.1): ten
  masses plus `ALPHA_EM` and `RATIO_E_MU_MASS_SQ`. The seven that agree
  are the form factors and decay constants. The partition is a literal
  roster in `test/test_core_constants.py`, so a silent consolidation
  fails there even though every file-by-file bit-equality check would
  still pass.
- **`R_FACTOR`'s Cython comment has an exponent typo** (Task 3.1): both
  muon kernels annotate `1.0001870858234163` with a `12 r^2 ln(r^2)` log
  term, but only `r^4` reproduces the digits. The number is right and
  the comment is wrong; the `.pyx` is left untouched and the correct
  formula is pinned in the test.
- **Two different twelves live in this phase's docs, and conflating them
  cost a review round** (Task 3.1, PR #58). `constants.pxd` is `include`d
  by **12 spectra extensions** and declares **14 masses** — but only 12
  *distinct* mass values, since `MASS_K0` / `MASS_KL` / `MASS_KS` all
  carry 497.611. The task note first claimed "all twelve masses are
  bit-equal to `hazma/parameters.py`'s"; the check behind that claim ran
  a real script over a hand-typed 12-pair list that omitted `MASS_KL`
  and `MASS_KS`, and the wrong answer matched the (correct) extension
  count two paragraphs away, so nothing looked inconsistent. All 14 do
  agree — re-derived by enumerating `^DEF MASS_` from the header and
  matching on bit pattern rather than on a typed name pairing. **Any
  count in this phase gets its population enumerated from source**, not
  from a list someone wrote out; see
  `docs/agents/lessons.md` `[hand-written-population-in-a-derived-check]`.
- **Pin `numpy==2.5.1` when building an env for this phase** (Task 3.1).
  A fresh `uv pip install -e .` resolves 2.5.2, which puts the parity
  corpus into budget mode (`exact: False`, detail
  `numpy '2.5.1' -> '2.5.2'`) and turns the bare suite's skip count from
  13 into 14. The kernel digest and the served-kernel predicate were both
  clean — it is purely the dependency. Recipe and the one-second
  provenance check are in [../README.md](../README.md) under Findings.
- **`clippy::excessive_precision` is on by default** (Task 3.1) and fires
  on any literal transcribed verbatim with trailing significant zeros
  (`0.9998770`). `constants.rs` carries a module-level `allow` with the
  reason. Later verbatim transcriptions in this phase will hit the same
  lint.

## Decisions and Implementation Notes

- **Three namespaces, mapped to sources** (Task 3.1):
  `constants::pdg` ← `hazma/_utils/constants.pxd` (151 values),
  `constants::legacy` ← `hazma/_utils/legacy_parameters.pxd` (48), and
  `constants::derived::<source_pyx>` ← the module-local `DEF`s of the
  five `.pyx` that declare any (25). A Phase 04 kernel names the table
  its `.pyx` `include`s.
- **Every module-local `DEF` is carried, including pure aliases**
  (Task 3.1), which is what lets the coverage check rescan the tree and
  be total rather than a transcribed list.
- **`pub mod constants;` while its neighbours in `lib.rs` are private**
  (Task 3.1) — nothing reads the tables yet and a private module of 224
  unread `const`s is a wall of `dead_code` under `-D warnings`.
- **The bit-equality gate parses text on both sides** (Task 3.1) rather
  than importing `hazma._core`: no build, 0.03s, platform-independent,
  and sound because both CPython and rustc parse decimal literals
  correctly-rounded. The compiled side is covered by five `cargo test`
  units in `constants.rs`.

## Files Changed

### Task 3.1

- `rust/src/constants.rs` — **new**, 224 `pub const`s in three
  namespaces plus five unit tests and a `# Sources` provenance header.
- `rust/src/lib.rs` — `pub mod constants;` and the paragraph on why.
- `test/test_core_constants.py` — **new**, 25 tests.

## Verification

- **Task 3.1 (2026-08-09):**
  `pytest test/test_core_constants.py -q` → `25 passed in 0.03s`;
  `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  `7 passed` (5 new); clippy and fmt clean; bare `pytest -q` →
  `1088 passed, 13 skipped` with the parity corpus in bit-equality mode
  (skip count unchanged at 13). Thirteen mutations, each caught by the
  test whose name claims it — table in the task note.
- Remaining tasks: `cargo test` (foundation units); scipy-comparison
  pytest suite green in CI.

## Open Questions

- `spec_math::li2` convention vs `scipy.special.spence` — Task 3.2
  resolves.
- **Which PDG edition each `constants.pxd` value came from is recorded
  nowhere** (Task 3.1). The `± uncertainty` annotations are the only
  provenance; some entries predate the current edition (α⁻¹ is
  pre-CODATA-2022). `constants.rs` cites the PDG review index for the
  tables rather than claiming an edition per value. Not blocking, and
  not to be resolved by re-sourcing values — rule 4 forbids that.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**For the next agent working in Phase 03:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file. Every task in this
phase is unblocked — ADR-0002 was accepted 2026-08-04 — but 3.2/3.3
must honor its provenance rule: cephes lineage and netlib QUADPACK
only, nothing GSL-derived in the tree or the dependency graph.

**Currently safe to assume:**

- Every live integral is finite-interval — `qagi` is out of scope.
- **Task 3.1 is done, so `hazma_core::constants` exists and Task 3.4 is
  unblocked.** `constants::{pdg, legacy}` are the two Cython tables and
  `constants::derived::<source_pyx>` the module-local `DEF`s, all
  bit-equal to the Cython and held there by
  `test/test_core_constants.py` (25 tests, 0.03s, platform-independent)
  and five `cargo test` units. Name the table the `.pyx` you are porting
  `include`s: `pdg` for everything under `hazma/spectra/**`, `legacy`
  for the four mediator spectrum extensions.

**Currently risky / unknown:**

- `qelg` Fortran→Rust translation is the fiddliest item in the project;
  budget review time accordingly.
- **`derived::photon_pion` deliberately mixes both tables** — PDG
  aliases, legacy-frozen literals. Read that module's doc comment before
  touching the charged-pion photon kernel in Phase 04.
- `derived::positron_pion::ENG_MU_PI_RF` and
  `derived::photon_pion::ENG_MU_PIRF` are the same physical quantity,
  different numbers, one underscore apart.
