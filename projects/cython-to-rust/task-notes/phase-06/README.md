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
| 6.3 | Positron spectrum pair | 6.1 | **Complete (2026-08-27)** | [task-6.3-positron-spectra.md](task-6.3-positron-spectra.md) |
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

- **The two positron `.pyx` were not a clone-pair but the *same
  implementation twice*** (Task 6.3). Normalised for the model's name,
  `diff` reports only those substitutions and the order of two `import`
  lines — unlike the decay pair, which differed in four ways. One Rust
  kernel serves all four entry points, and the scalar and vector spectra
  are bit-for-bit equal at equal arguments by construction.
- **The `nan` at the legacy `m_e` was a clang FMA contraction, not a
  constants divergence** (Task 6.3). `sqrt(E² − m_e²)` compiles to an FMA
  that subtracts the *rounded* `m_e²` from an exactly-computed square, and
  that rounding is upward by 1.45e-17. Fixed by clamping the radicand;
  the two `MASS_E` tables are untouched and their consolidation is still
  a separate declared change under rule 4.
- **A Python replica of a `cdef` is unfused and the shipped C is not**
  (Task 6.3). Four rounds of transliterating the integrand into Python
  returned `0.0` where the extension returned `NaN`. What settled it was
  a temporary `def` in the `.pyx` returning the intermediates.
  **Instrument the extension before trusting a replica.**
- **A mutation campaign can overturn the contraction rule, and did**
  (Task 6.3). `head − coef·m_e·m_e` ends in a syntactic multiply, so the
  rule predicts a fusion; fusing it *loses* bit-equal values. The rule
  predicts, the campaign decides.
- **The shipped `e⁺e⁻` line is low by the positron's rest-frame velocity**
  (Task 6.3): the box's edges carry `r` and its height does not, so it
  integrates to `pw_ee · r`. Reproduced under rule 1, filed as
  [the missing electron velocity](../../../../docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md).
- **Task 6.2 left an attribution bug in the oracle roster** (found by
  Task 6.3): its rename flipped the two
  `mediator_spectra.vector.positron.*` rows to `restored` alongside the
  three photon rows it was really deleting. Latent — nothing in `pytest`
  reads that dict — and corrected in 6.3.

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
- **Task 6.3 named the `_core` positron entry points
  `dnde_positron_decay_{s,v}`** rather than the Cython's
  `dnde_decay_{s,v}`, because Task 6.2 had already taken the vector pair's
  spelling for the *photon* spectrum in the same PyO3 submodule. The
  scalar half follows for symmetry; both wrappers re-export under the
  Cython names, and `test/parity/cases.py`'s new `CORE_RENAMES` declares
  the mapping for the served-roster test alone — the corpus still calls
  the wrapper.
- **Task 6.3 emptied `NESTED_RTOL`**, the last opening budget any corpus
  case held. Fourteen budgets tightened across the project, none widened.
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

### Task 6.3

`rust/src/kernels/mediator_decay_positron.rs` (new — one kernel for both
models), `rust/src/kernels.rs`, `rust/src/{scalar,vector}_mediator.rs`,
`hazma/{scalar,vector}_mediator/_*_mediator_positron_spectra.py`,
`hazma/{scalar,vector}_mediator/*_mediator_positron_spec.pyx` (deleted),
`hazma/vector_mediator/vector_mediator_positron_spec.pyi` (deleted),
`hazma/spectra/{_photon,_positron}/{_muon,_pion}.pyx` (survivor comments),
`setup.py`, `test/test_core_mediator_positron.py` (new),
`test/test_core_{mediator_tables,positron_pion,scalar_xs}.py`,
`test/test_theory_aggregation.py`,
`test/parity/{cases,tolerances,test_parity}.py`,
`test/parity/oracles/entry_points.py`,
`docs/followups/{done/positron-spectrum-nan-at-legacy-electron-mass.md,todo/mediator-positron-line-misses-the-electron-velocity.md,todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md,README.md}`.

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
- **After Task 6.3:** `cargo test --no-default-features` → `258 passed`;
  `pytest -q` → `2389 passed, 15 skipped, 12 subtests passed`;
  `pytest test/parity -q` → `658 passed, 1 skipped`;
  `pytest test/test_theory_aggregation.py -q` → `69 passed`;
  `find hazma -name "*.pyx" -o -name "*.pxd"` → **13** (was 15), and
  neither mediator package holds one. All four swapped entry points
  moved — see `../numerical-impact.md`; worst 2.3319e-12, four budgets
  tightened, and one value moved `NaN → 0.0` at the legacy `m_e`.
  Benchmark 32x–43x from release builds of both sides.

## Open Questions

- ~~**Does the charged pion's forward-cone defect reach the mediator
  spectra?**~~ — **closed by Task 6.2: yes.** No new measurement was
  needed; `test/parity/oracles/data/manifest.json` already holds defect
  A3's corrected-value capture over exactly the three photon corpus
  cases, and repairing the forward cone moves **1,032 of 8,610** scalar
  values by up to **1.63e-06** relative and **2,013 of 29,295** vector
  values by up to **7.77** relative (a factor of 8.8, at an absolute
  7.3e-10). In the vector case it changes the *shape* of the low-energy
  tail, not a total. **Task 6.3 answered the same question for the
  positron pair from the same manifest** (defect A4): 5,237 of 16,740
  values in each of the four cases, all moving *up*, by up to
  **3.7421e-04** relative — which agrees with `R_FACTOR**2 - 1` to eight
  digits and is identical across all four, so a pure normalization rather
  than a change of shape.
- ~~**Is one cache slot enough, and should the shared photon table set
  stop building the muon table the scalar module ignores?**~~ — **closed
  by Task 6.2's benchmark: yes and no.** One slot is enough because every
  consumer sweeps a whole energy grid at one mass; the wasted muon table
  is 500 evaluations of a closed-form kernel per *distinct mass*, against
  the 4.2x win on the table build and the 4,180x win on a fixed-mass
  parameter sweep. Splitting the cache would mean two `LazyLock`s
  differing by a field, for no measurable gain.
- **The oracle roster still has no restore revision for the four `.pyx`
  Tasks 6.2 and 6.3 deleted**, because `RESTORED_SOURCES` records literal
  SHAs and a task cannot know its own commit's. The handoff asked Task
  6.3 to discharge it for both pairs; it could not, for exactly the
  reason 6.2 could not, and instead widened the follow-up to name all
  four —
  [`oracle-restore-revisions-for-the-mediator-decay-pyx`](../../../../docs/followups/todo/oracle-restore-revisions-for-the-mediator-decay-pyx.md).
  **Task 6.4 can**, because by then both deleting commits are merged, and
  6.4 is also where re-capture closes and the item becomes moot — so
  decide there rather than defer again. Not blocking; the `pytest` gate
  does not read that dict.
- **`test/parity/oracles` re-capture closes at Task 6.4.** Two of the four
  defect patches (A3, A4) reach the mediator spectra, and the arrays are
  committed; nothing needs recapturing unless a patch changes. Task 6.4
  should say so explicitly when it deletes the last `.pyx`.
- **The line term's missing `1/r` is a post-6.4 item.** It moves
  published numbers well above the budgets the four positron cases now
  hold, so it needs a corpus re-capture or a declared exception, and
  neither belongs inside a swap —
  [the missing electron velocity](../../../../docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md).

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).

## Handoff to Next Task

**Task 6.4 (retire the capi survivors and the `_utils` headers) is
next**, and it closes the phase. Read `../../PLAN.md`, `../README.md`,
this file, then the phase file's Task 6.4 block;
[`task-6.3-positron-spectra.md`](task-6.3-positron-spectra.md)'s
`## Handoff` is the direct brief, and
[`task-6.2-decay-spectra.md`](task-6.2-decay-spectra.md) still carries
the FMA-campaign method both swaps used.

**Now safe to assume** (Tasks 6.1–6.3 delivered all of it):

- **6.4's `rg` sweep is already empty.** The four capi survivors cimport
  only each other and `hazma/_utils/boost`; the two mediator decay
  modules that read them went in 6.2 and the two positron ones in 6.3.
  Every stale comment claiming otherwise — in all four `.pyx`, in
  `test/parity/oracles/entry_points.py`, in `test/test_core_positron_pion.py` —
  was corrected in 6.3, so what 6.4 reads is current.
- **Both mediator packages are Cython-free**, and `setup.py` builds no
  extension for either. Thirteen `.pyx`/`.pxd` remain and all are 6.4's:
  the four survivors with their `.pxd`, `_utils/boost.{pyx,pxd}`,
  `constants.pxd`, `kinematics.pxd`, `legacy_parameters.pxd`.
- **All 41 consumed entry points are on `hazma._core`.** Nothing is left
  to swap; 6.4 is deletion and build plumbing only.
- **Every mode string and dispatch message in the tree is the port's.**
  No `.pyx` spells one, so `test_core_mediator_tables.py` and
  `cython_dispatch_messages()` hold the provenance instead.
- **No corpus or oracle re-capture is pending.** The committed defect
  arrays cover A3 and A4 and nothing changes unless a patch does.

**Still risky / unknown for Task 6.4:**

- **Deleting a `.pyx` does not make its module unimportable**, and 6.4
  deletes four at once. The built `.so` and generated `.c` sit beside
  each source, are gitignored, and survive `git rm` — Task 6.3 relied on
  that deliberately, keeping both twins callable after `git rm` to take
  the drift measurement, then removed them by hand. Assert on the source
  files and the `setup.py` entry, never with `pytest.raises(ImportError)`.
- **`pyproject.toml`'s Cython *requirement* goes with them**, and so does
  `setuptools-rust`'s reason to coexist with it. Check
  `[build-system] requires`, the `lint`/`dev` groups, `MANIFEST.in` and
  CI's toolchain steps in the same pass — a clean wheel is not evidence
  of a clean sdist (`docs/agents/environment.md`).
- **Decide the restore-revision follow-up rather than defer it.** 6.4 is
  the first task that *can* resolve it and also the task that makes it
  moot; leaving it open past the phase leaves a dangling item nobody
  else will own.
- **The `_utils` headers are read by more than the survivors.** Sweep
  `include "` as well as `cimport` — the mediator `.pyx` reached
  `legacy_parameters.pxd` through `include`, which no `cimport` grep
  would have found.
