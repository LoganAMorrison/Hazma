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
| 6.2 | Decay spectrum pair | 6.1 | **Complete (2026-08-23)** | [task-6.2-decay-spectra.md](task-6.2-decay-spectra.md) |
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
- **The drift these swaps carry is the *integrator's*, and that is
  measured rather than argued** (Task 6.2). Setting `eng_s == ms` makes
  the boost integrand a constant, and every channel of both entry points
  then agrees with the Cython to within one ulp — so the 5.3e-12 the
  corpus sees is `crate::quad` against scipy's QUADPACK, not the
  transliteration. A constant integrand is not reproduced exactly
  either: `∫₋₁¹ c dcl` lands one ulp off the exact `2c` on **both** sides,
  at different `c`. Expect the same floor for the positron pair.
- **`pws` is read lazily, and a length check would break a working
  call** (Task 6.2). The last partial width is read *only* inside the
  boosted line window, so a short buffer legitimately succeeds outside it
  and raises `IndexError` inside. `PartialWidths::get` carries
  `boundscheck(True)` for exactly this.
- **The two clone-pairs differ in laziness and it is observable** (Task
  6.2). The scalar decay integrand guards each channel with a bitflag
  `if`; the vector one computes all six components and then selects, so a
  single-channel call still raises where any component would. Read the
  `.pyx`'s structure, not just its formulae.
- **Fourteen of Task 6.2's thirty-seven fused sites survive their
  mutation *by construction*** — the coefficient is a power of two, so
  the product is exact and fusing cannot change the sum (zero
  disagreements over 40,002 masses per shape, exact rational
  arithmetic). Two more survived only because the grid never reached
  `2 m_μ`, above which `2 − 12μ²` disagrees at 8.5% of masses. **A
  mutation survivor here is a statement about the coefficient or about
  the grid; decide which before writing it off.**
- **A mutation harness that does not force a rebuild will lag its own
  mutations** (Task 6.2 — the second of three campaign runs lagged by two
  iterations). `rm -f hazma/_core.abi3.so` before each install, `test -f`
  after.
- **`numpy.logspace` and `f64::powf` are not the same code.** The grid
  is bit-equal to NumPy's on macOS/arm64 and one ulp off it at ~5% of
  points on Linux/x86-64 (CI run 32681245809; every measured
  disagreement exactly one ulp, worst relative 2.16e-16). So
  "bit-equal to the Cython" is a **capturing-platform statement** for
  anything reading these tables. No gate changes — the corpus already
  runs in budget mode off macOS.

## Decisions and Implementation Notes

- **Task 6.1 amended two of its own exit criteria in the phase file**
  (cache key, and "error text"). See its `## Plan Impact`. No ADR.
- **`crate::kernels::mediator_tables`** is the shared foundation, a
  documented naming exception beside `soft_complex`;
  **`hazma._core.mediator_tables`** is a sixth `_CORE_TEST_ONLY_MODULES`
  probe, because every oracle (`numpy.logspace`, `numpy.interp`, the
  Phase 04 entry points, the four then-live twins) is in Python.
- **Task 6.2 lifted `pyproject.toml`'s `cython<3.3` cap**, because the
  only file the cap protected was
  `scalar_mediator_decay_spectrum.pyx`, which that task deleted.
  Evidence taken before removing it: the seven surviving `.pyx` compile
  under cython 3.3.0 and a tree whose extensions are *built* by it runs
  the suite at the same counts as 3.2.9, corpus and Cython-twin
  bit-equality included.
- **Task 6.2 retired `test_core_dispatch.py`'s two Cython-oracle
  classes**, because it deleted the last `.pyx` that spells a dispatch
  message. `cython_dispatch_messages()` survives as the guard that the
  tree stays silent; the roster the port emits is frozen there with
  per-message provenance. From Task 6.3 on, "the port's messages are the
  Cython's" is transcription, not execution.
- **One test module per clone-pair, not per entry point** (Task 6.2:
  `test/test_core_mediator_decay_photon.py` covers all three photon entry
  points). The independent reference is one function parameterised by
  pair, so splitting it would mean the same 300-line reference twice.
- **`cargo` gates the grid algorithm, Python gates its agreement with
  NumPy, scoped by platform** — captured NumPy bits in a cargo test
  would turn a Linux CI job red for a libm difference (Phase 04
  learnings §4), and the first CI round proved the same is true of an
  unscoped Python comparison. The mode is declared from
  `ON_THE_CAPTURING_PLATFORM`, never probed, and the two off-platform
  comparators are separate functions so they are exercised everywhere.

## Files Changed

### Task 6.1

`rust/src/kernels/mediator_tables.rs` (new), `rust/src/mediator_tables_probe.rs`
(new), `rust/src/kernels.rs`, `rust/src/lib.rs`,
`test/test_core_mediator_tables.py` (new), `test/parity/cases.py`,
`../../phases/phase-06-mediator-spectra.md`,
`docs/followups/{todo/mediator-spectra-accept-unknown-mode-strings.md,README.md}`.

### Task 6.2

`rust/src/kernels/{scalar,vector}_decay_photon.rs` (new),
`rust/src/kernels/mediator_tables.rs`, `rust/src/kernels.rs`,
`rust/src/{scalar,vector}_mediator.rs`,
`hazma/{scalar,vector}_mediator/_*_mediator_spectra.py`,
`hazma/{scalar,vector}_mediator/*_mediator_decay_spectrum.pyx` (deleted),
`hazma/vector_mediator/vector_mediator_decay_spectrum.pyi` (deleted),
`setup.py`, `pyproject.toml`,
`test/test_core_mediator_decay_photon.py` (new),
`test/test_core_{dispatch,mediator_tables,scalar_xs}.py`,
`test/parity/{cases,tolerances}.py`,
`test/parity/oracles/{defects,entry_points}.py`,
`docs/followups/{todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md,README.md}`,
`docs/agents/lessons.md`.

## Verification

- Corpus (quad budgets); benchmark vs pre-swap Cython (dead-cache fix
  is the expected headline); import smoke after 6.4.
- **After Task 6.1:** `cargo test --no-default-features` → `222 passed`
  (201 at Phase 05's close); `pytest -q` → `2163 passed, 15 skipped, 12
  subtests passed`; `pytest test/parity -q` → `657 passed, 1 skipped`;
  `pytest test/test_theory_aggregation.py -q` → `69 passed`. No public
  value moved — nothing appended to `../numerical-impact.md`.
- **After Task 6.2:** `cargo test --no-default-features` → `249 passed`;
  `pytest -q` → `2262 passed, 15 skipped, 12 subtests passed`;
  `pytest test/parity -q` → `658 passed, 1 skipped`;
  `pytest test/test_theory_aggregation.py -q` → `69 passed`;
  `find hazma -name "*.pyx"` → **7** (was 9). All three swapped entry
  points moved — see `../numerical-impact.md`; worst 5.3327e-12, three
  budgets tightened from `NESTED_RTOL` to `PORTED_NESTED_RTOL`.

## Open Questions

- ~~**Does the charged pion's forward-cone defect reach the mediator
  spectra?**~~ — **closed by Task 6.2: yes.** No new measurement was
  needed; `test/parity/oracles/data/manifest.json` already holds defect
  A3's corrected-value capture over exactly the three photon corpus
  cases, and repairing the forward cone moves **1,032 of 8,610** scalar
  values by up to **1.63e-06** relative and **2,013 of 29,295** vector
  values by up to **7.77** relative (a factor of 8.8, at an absolute
  7.3e-10). In the vector case it changes the *shape* of the low-energy
  tail, not a total. **Task 6.3 owes the same question for the positron
  pair**, where the relevant defect is A4 and the same manifest already
  answers it.
- ~~**Is one cache slot enough, and should the shared photon table set
  stop building the muon table the scalar module ignores?**~~ — **closed
  by Task 6.2's benchmark: yes and no.** One slot is enough because every
  consumer sweeps a whole energy grid at one mass; the wasted muon table
  is 500 evaluations of a closed-form kernel per *distinct mass*, against
  the 4.2x win on the table build and the 4,180x win on a fixed-mass
  parameter sweep. Splitting the cache would mean two `LazyLock`s
  differing by a field, for no measurable gain.
- **The oracle roster has no restore revision for the two `.pyx` Task 6.2
  deleted**, because `RESTORED_SOURCES` records literal SHAs and a task
  cannot know its own commit's. Filed as
  [`../../../../docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md`](../../../../docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md);
  **Task 6.3 should discharge it for both pairs at once**, after which
  the roster is complete. Not blocking — the `pytest` gate does not read
  that dict.
- **`test/parity/oracles` re-capture closes at Task 6.4.** Two of the four
  defect patches (A3, A4) reach the mediator spectra, and the arrays are
  committed; nothing needs recapturing unless a patch changes. Task 6.4
  should say so explicitly when it deletes the last `.pyx`.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**Task 6.3 (the positron pair) is next**, then 6.4 and Phase 07. Read
`../../PLAN.md`, `../README.md`, this file, the phase file, then
[`task-6.2-decay-spectra.md`](task-6.2-decay-spectra.md) — its
`## Findings`, `## Decisions` and `## Handoff` are the direct template,
because the positron modules are the *other* clone-pair and 6.2 already
paid for four of the five things that would otherwise cost a build each.
[`task-6.1-table-struct.md`](task-6.1-table-struct.md) still carries the
foundation's own measurements.

**Now safe to assume** (Tasks 6.1 and 6.2 delivered all of it):

- **`crate::kernels::mediator_tables` is complete for 6.3.**
  `positron_tables(mass)` returns a memoized `Arc<PositronTables>` whose
  columns are the Phase 04 kernels on a `numpy.logspace`-identical grid
  starting at the **legacy** `m_e`, with `BelowGrid::Clamp`; plus
  `PositronMode`, `PartialWidths` (carrying `boundscheck(True)`) and
  `SpectrumError`. 6.3 adds no foundation.
- **The module layout, naming and test shape are settled.** One kernel
  module per `.pyx`, named `<model>_decay_<product>`; one shared test
  module per clone-pair; `kernels.rs` already documents the naming
  exception. Both PyO3 submodules already own `spectrum_error`,
  `PARTIAL_WIDTHS` and `OUT_OF_BOUNDS_MESSAGE`.
- **The fallible-integrand shape**: capture the first `SpectrumError` in
  an `Option`, return `NaN` from the closure, raise after `quad` returns.
- **`require_vector` for the array-only entry points** (`dnde_decay_s`,
  `dnde_decay_v`), a plain `f64` for the `_pt` twins, with the two
  divergences 6.2 declared — a scalar energy raises `ValueError` rather
  than `TypeError`, and a `list` is accepted.
- **Neither positron module needs `soft_complex`**
  (`grep -c SoftComplexToDouble` → `0 / 0`, Task 6.1), so
  `SpectrumError::NonReal` is unreachable from them and their
  `spectrum_point` can carry only `OutOfBounds`.
- **The cython cap is gone** and the seven surviving `.pyx` are known to
  compile under cython 3.3.0 with the suite at the same counts.

**Still risky / unknown for Task 6.3:**

- **The `nan` at the legacy `m_e` lands on the grid's first abscissa.**
  The positron grid *starts* at `0.510998928` MeV, which is exactly where
  [`../../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md`](../../../../docs/followups/todo/positron-spectrum-nan-at-legacy-electron-mass.md)
  says the positron spectra return `nan`. The project handoff has been
  asking for that follow-up to be met "before Phases 05/06"; 6.3 is where
  it actually bites.
- **`dnde_decay_s` short-circuits `fs == "e e"` before the integral**
  (`scalar_mediator_positron_spec.pyx:207`), returning the line term
  alone — a fifth structural difference from the decay pair, and one that
  changes which `pws` indices are read at all. Read the `.pyx`'s read
  *order* before writing the integrand; 6.2's `pws[4]`-inside-the-window
  finding is the warning.
- **Take the benchmark before deleting the twins, from a release build of
  both sides in one interpreter, and run it from outside the repo.** The
  editable install is ~20x pessimistic and inverts the comparison; and a
  suite or benchmark run from the repo root imports `hazma` from the
  worktree rather than site-packages, which silently invalidated the
  first cython-3.3 measurement Task 6.2 took. The recipe that worked:
  `git worktree add --detach <dir> origin/master`, then a non-editable
  `uv pip install .` of each side into one scratch venv.
- **`test/parity/cases.py` needs three edits per swap, not one** — the
  `Case.module` moves to the wrapper, a `PORTED_ENTRY_POINTS` row is
  added, and the `_CORE_TEST_ONLY_MODULES` comment counting live mediator
  `.pyx` needs re-deriving. Missing the second turns
  `test_the_served_roster_is_exactly_the_ported_entry_points` red with a
  set-difference message that does not name the cause.
- **A mutation survivor is a statement about the coefficient or about the
  grid.** Task 6.2's fourteen survivors were provably identity-equivalent
  (power-of-two coefficients); two more were alive only because the grid
  never reached `2 m_μ`. Decide which before writing one off — and force
  a rebuild between mutations, or the harness will measure the previous
  one.
- **Task 6.4's ground is visible now.** With both decay modules gone,
  `hazma/spectra/_photon/{_muon,_pion}.pyx` have no consumer outside
  their own pair, and the `_positron` pair is read only by the two `.pyx`
  Task 6.3 deletes — so 6.4's `rg` sweep should come back empty for all
  four the moment 6.3 lands, and `pyproject.toml`'s Cython *requirement*
  goes with them.
