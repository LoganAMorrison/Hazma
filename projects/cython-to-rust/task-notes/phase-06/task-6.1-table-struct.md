# Task 6.1: Spectrum-table struct design

**Date:** 2026-08-23
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-06-mediator-spectra.md` (Task
6.1), `../../PLAN.md` §Numerical impact, `../../rules.md` rules 1–4,
6–9, 12
**Related ADRs:** ADR-0001, ADR-0002
**Depends On:** Phases 04 and 05 complete

## Objective

Give Phase 06 the shared Rust foundation its four mediator-spectrum
`.pyx` need: one struct owning the 500-point log-spaced rest-frame
interpolation tables, a real memo cache to replace the Cython's dead
one, and the mode selectors as enums parsed once per call rather than
per quadrature node. No entry point is swapped here — Tasks 6.2 and 6.3
consume this.

## Exit Criteria

Copied from the phase file's Task 6.1 block, with the two amendments
this task made to it (see `## Plan Impact`):

- A Rust struct owning the precomputed 500-point log-spaced rest-frame
  tables, built once per parameter set by calling the Phase 04 kernel
  `fn`s natively (no Python round trips), with genuine memoization
  keyed on the mediator mass (fixing the dead-cache bug; same numbers,
  declared as performance-only per rules.md 3, 12). — **met**
- Mode dispatch (`"total"`, `"e e g"`, `"pi pi g"`, …) becomes an enum
  at the PyO3 boundary; string parsing happens once per call, not per
  quadrature node. Accepted strings byte-match today's. — **met**
- Design reviewed against both decay + both positron modules before
  implementation (they are two clone-pairs — one parameterized
  implementation each). — **met**; all four `.pyx` were read in full
  first and the differences between them are the module's parameters
  ([`Findings`](#findings) rows 1–3).

## Inputs Reviewed

- `../../PLAN.md` (all sections); `../README.md`; `README.md` (phase);
  `../../rules.md`.
- `../../phases/phase-06-mediator-spectra.md` — Tasks 6.1–6.4.
- `../../learnings/phase-04-spectra-kernels.md`,
  `../../learnings/phase-05-mediator-cross-sections.md`.
- The four `.pyx` in full:
  `hazma/{scalar,vector}_mediator/*_{decay_spectrum,positron_spec}.pyx`.
- `rust/src/kernels.rs`, `rust/src/lib.rs`,
  `rust/src/kernels/photon_tables.rs` (the house style for a
  table-owning kernel struct), `rust/src/kernels/soft_complex.rs`,
  `rust/src/interp.rs`, `rust/src/constants.rs`.
- `test/parity/cases.py` (mediator-spectrum blocks),
  `test/parity/tolerances.py` (the seven `NESTED_RTOL` budgets).

## Findings

**The four modules differ in exactly four ways**, and every one is data:

| | photon (`*_decay_spectrum.pyx`) | positron (`*_positron_spec.pyx`) |
| --- | --- | --- |
| grid start | `10⁻¹` MeV (literal exponent `-1.0`) | `m_e` (legacy, 0.510998928) |
| below grid | `dnde[0]·e[0]/E` tail below `10**-1` | `np.interp`'s clamp, no guard |
| tables | scalar: charged pion only; vector: + muon | both: charged pion + muon |
| selector | scalar: `list[str]` → bitflag; vector: one `str` | one `str` |

- **The dead cache is deader than the phase file's "broken memo-cache"
  suggests, and in two different ways.** The two *decay* modules have no
  cache at all — `__set_spectra` runs unconditionally at the top of every
  entry point. The two *positron* modules have the predicate
  (`__recompute_rf_spectra`) but **no line anywhere assigns to `cache_ms`
  / `cache_mv` or `cache_pws`**, so the `-1.0` sentinels never change and
  it always returns 1. Both spellings rebuild a 500-point,
  quadrature-backed table on every single call.
- **The Cython's declared cache key is wider than the tables' actual
  inputs.** `__set_spectra` takes the mediator mass and reads no partial
  width; the tables are a pure function of `m/2`. Keying the port on the
  widths as well would be slower for no correctness gain, so the cache
  keys on the mass. This amends the phase file (see `## Plan Impact`).
- **An unrecognised mode string raises nothing today — it returns
  `0.0`.** Every `cdef double` integrand ends in `if mode == …: return …`
  with no `else`, and a C function that falls off its end returns zero,
  so a typo'd mode integrates a zero integrand; the enclosing
  `__dnde_decay_*` adds a line term only for modes it *does* recognise.
  Verified against the shipped extensions, not inferred:
  `dnde_decay_v_pt(30, 600, 550, pws, "pi0g")` and
  `dnde_decay_s_pt(30, 600, 550, pws, "e e g")` both return exactly
  `0.0`. Reproduced under rule 1 and filed.
- **Both photon modules carry one live complex-`pow` site; neither
  positron module carries any.** `grep -c SoftComplexToDouble` on the
  generated C — the check the Phase 05 learnings say must be run and
  records as unrun for Phase 06 — returns **6 / 0 / 6 / 0** for
  scalar-decay / scalar-positron / vector-decay / vector-positron, and
  five of each six are the proto and definition lines. The one call site
  each is the `** 1.5` in an FSR coefficient:
  `scalar_mediator_decay_spectrum.pyx:113`
  (`qe**2 / (16.·(1 − 4 mul²)**1.5·π²)`, in `dnde_fsr_l_srf`) and
  `vector_mediator_decay_spectrum.pyx:73`
  (`qe**2 / (4.·(1 − 4 mupi²)**1.5·π²)`, in `__dnde_fsr_cp_vrf`). Both
  are the shape `crate::kernels::soft_complex` already covers —
  `soft_complex_pow_1_5` plus `complex_quotient_real_denominator` — so
  Task 6.2 inherits the pair rather than deriving new ones. The sibling
  FSR functions in each file spell the same factor `sqrt(1 − 4μ²)` and
  stay real.
- **`np.log10` and libc `log10` agree here.** The scalar decay module
  takes its upper grid endpoint from `np.log10` and the other three from
  `libc.math.log10`. Measured equal at every mass tried on the capturing
  platform, so the port takes one `log10`.
- **`numpy.logspace`'s final-point substitution is not cosmetic.** NumPy
  overwrites the last exponent with `stop` rather than continuing the
  step arithmetic; the two differ by one ulp at **732 of 8,008**
  (mass, start) pairs over `np.linspace(1, 2000, 4001)` MeV × the two
  grid starts — and at **none** of the corpus's three masses, so a
  three-mass check would have missed it.

## Decisions and Implementation Notes

- **One shared module, not one per clone-pair.**
  `rust/src/kernels/mediator_tables.rs` is a documented exception to
  `kernels.rs`'s one-submodule-per-`.pyx` rule, alongside
  `soft_complex` — it is the part all four repeat, and the four
  differences above are its parameters (`BelowGrid`, the grid start, the
  table set, the selector type).
- **The cache is one slot per table set, keyed on the mass's bit
  pattern.** One slot because the Cython had one set of module globals
  and consumers sweep a whole grid at one mass. Bits rather than `==`
  so a `NaN` mass is a hit rather than an unbounded rebuild loop; the
  one place that is *stricter* than `==` is `0.0` vs `-0.0`, asserted
  deliberately rather than discovered later. The value is an `Arc` so a
  caller drops the lock before integrating.
- **The parsers return `Option`, and `None` means the Cython's `0.0`.**
  Tightening it into a raise would be a behaviour change the corpus
  cannot gate (it samples valid modes only) — filed instead.
- **`cargo` gates the grid *algorithm*; Python gates its agreement with
  NumPy.** Hard-coding the capturing platform's NumPy bits into a cargo
  test would turn a Linux CI job red for a libm difference rather than a
  defect — the failure Phase 04 learnings §4 records twice. So
  `cargo test` asserts the unfused step, the substituted endpoint and
  monotonicity, and `test/test_core_mediator_tables.py` compares against
  live `numpy.logspace` on whatever platform runs.
- **A sixth probe submodule, `hazma._core.mediator_tables`.** Added to
  `_CORE_TEST_ONLY_MODULES` with the rest, and for the same reason the
  five before it were: the oracles (`numpy.logspace`, `numpy.interp`,
  the Phase 04 entry points, and the four *live* Cython twins) are all
  in Python. No module under `hazma/` imports it, which
  `test_test_only_core_submodules_have_no_importer` asserts.
- **`ScalarPhotonModes::contains` was written and then removed.** It had
  no non-test caller, so `clippy -D warnings` rejected it; Task 6.2 adds
  it back with the integrand that branches on it rather than carrying a
  `dead_code` allowance through the phase.
- **The shared photon table set builds the muon table the scalar decay
  module will not read.** One 500-point evaluation per *distinct mass*
  against the two-per-*call* the dead cache costs today; Task 6.2
  measures it either way.

## Files Changed

- `rust/src/kernels/mediator_tables.rs` — **new.** `logspace`,
  `RestFrameTable` + `BelowGrid`, `TableCache`, `PhotonTables` /
  `PositronTables` and their memoized constructors, `PhotonMode` /
  `PositronMode` / `ScalarPhotonModes`. 22 unit tests.
- `rust/src/mediator_tables_probe.rs` — **new.** `hazma._core.mediator_tables`:
  `logspace`, `photon_tables`, `positron_tables`, `lookup`,
  `photon_mode`, `positron_mode`, `scalar_photon_mode_bits`.
- `rust/src/kernels.rs` — register the module; document why it is the
  fifth naming exception.
- `rust/src/lib.rs` — register the probe; extend the probe paragraph.
- `test/test_core_mediator_tables.py` — **new.** 65 tests over six
  classes.
- `test/parity/cases.py` — `hazma._core.mediator_tables` added to
  `_CORE_TEST_ONLY_MODULES`, with its rationale in the existing comment.
- `projects/cython-to-rust/phases/phase-06-mediator-spectra.md` — Task
  6.1's first two exit criteria amended (see `## Plan Impact`);
  frontmatter `status: In Progress`, as Task 5.1 did for Phase 05.
- `docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`
  and `docs/followups/README.md` — **new** follow-up and its index row.
- `projects/cython-to-rust/task-notes/phase-06/README.md`,
  `projects/cython-to-rust/task-notes/README.md` — status, findings,
  handoff.

## Verification

- `cargo fmt --manifest-path rust/Cargo.toml --check` — clean.
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
  — clean.
- `cargo test --manifest-path rust/Cargo.toml --no-default-features` —
  `222 passed; 0 failed`, from 201 at Phase 05's close. The 22 new tests
  cover: the `logspace` algorithm (the unfused step *and* that `logspace`
  is the unfused one, the substituted endpoint, endpoints, monotonicity);
  `RestFrameTable` (values are the kernel at the nodes, exact node hits,
  the `1/E` tail's continuity at the threshold, that the tail opens at
  the threshold and **not above it**, the clamp, the upper clamp under
  both policies, `NaN`); `TableCache` (build-once, re-key, one slot,
  `NaN` hit, `±0.0` miss); the two table sets (memoized, endpoints,
  columns are the Phase 04 kernels, legacy `m_e`); and all three
  selectors (accepted set, rejected set, line-carrying modes, bit values,
  the default list, repeated/unknown names).
- `pytest test/test_core_mediator_tables.py -q -n 0` — `65 passed`.
  Covers: `logspace` vs `numpy.logspace` at the corpus masses and over a
  4,001-mass sweep, the last point, the `num < 2` raise; both table sets'
  grids and columns against the public Rust entry points; the legacy vs
  PDG electron mass; `lookup` vs `np.interp` and vs the `.pyx` tail
  branch over 6,000+ probes, both clamps, `NaN`, both error paths; and
  every selector's accepted and rejected sets — the rejected sets
  **against the live Cython twins**, which return `0.0` for them.
- `pytest test/parity -q` — `658 passed, 1 skipped`; all 41 entry points
  unchanged.
- `pytest test/test_theory_aggregation.py -q` — `69 passed` (the
  model-layer gate the corpus cannot be).
- `pytest -q` (full suite) — `2158 passed, 15 skipped, 12 subtests
  passed`. `git diff origin/master --name-only -- test/` names only
  `test/parity/cases.py` (a comment and one `frozenset` entry) and the
  new module, so the 65 new tests are the whole delta.
- `scripts/agents/preflight.sh --paths "…"` — see the sweep block.
- **Mutation campaign, 8/8 killed** — the Phase 04 discipline, run
  against `cargo test` because that is the gate a future edit meets
  first:

  | # | Mutation | Result |
  | --- | --- | --- |
  | M1 | `i * step + start` → `mul_add(step, start)` | killed |
  | M2 | drop `grid[num - 1] = stop` | killed (2 tests) |
  | M3 | cache computes the key and never stores it (the Cython's bug) | killed (3 tests) |
  | M4 | tail threshold `0.1` → `0.2` | **survived**, then killed |
  | M5 | tables built at `mass` rather than `mass / 2` | killed |
  | M6 | `has_line` also true for `"pi pi"` | killed |
  | M7 | positron grid starts at `pdg::MASS_E` | killed |
  | M8 | the `1/E` tail branch never taken | killed |

  M1 and M4 are the two the campaign paid for. M1 survived the first
  version of `logspace_does_not_fuse_the_step_arithmetic`, which asserted
  only that the fused and unfused exponents *differ* somewhere and never
  that `logspace` used the unfused one — a test that would have passed
  against either spelling. M4 survived because every tail probe sat far
  from the threshold; a new test now pins that at and above `10**-1` the
  table interpolates, so moving the constant up cannot silently convert
  the grid's first decade into extrapolation. Both were caught by
  `test/test_core_mediator_tables.py` (M1: 13 failures) — the campaign's
  value was making `cargo` catch them too.
- **Test validity.** Most new assertions pin behaviour this task
  created, so there is no production change to stash; the mutation
  campaign above is the substitute, and it is stronger — every mutation
  was applied to the real source, built, and run.
- **Deferred:** no benchmark. `rules.md` rule 12 wants one for a
  performance claim, and this task makes none — the cache has no caller
  until Task 6.2, which the phase file already charges with the
  measurement ("benchmark vs pre-swap Cython recorded").

## Numerical impact

**No public value changes.** No file under `hazma/` is touched
(`git diff origin/master --name-only -- hazma/` is empty), no existing
Rust function is modified, and the whole diff outside `projects/` and
`docs/` is two new Rust modules, one new test module, two registration
lines and a `frozenset` entry in a test helper. Verified rather than
argued: `pytest test/parity -q` → `658 passed, 1 skipped` evaluates all
41 pinned entry points against the stored corpus, and
`pytest test/test_theory_aggregation.py -q` → `69 passed` is the
model-layer gate. Nothing is appended to
[`../numerical-impact.md`](../numerical-impact.md).

## Open Questions

- **Does the charged pion's forward-cone defect reach the mediator
  spectra?** Carried forward from Phase 04 unanswered — the project
  README asks it of Phase 06 specifically, and it is a question about
  the boost integral, not about the tables. Task 6.2 owns it: the
  charged-pion photon table this task builds is exactly the kernel the
  defect lives in, so the measurement is one table lookup away.
- **Is one cache slot enough?** The Cython had one and consumers sweep a
  whole grid per mass, but nothing measured a real caller's access
  pattern. If Task 6.2's benchmark shows thrashing on a mass scan, the
  fix is a small LRU and nothing else changes.
- **Should the shared photon table set stop building the muon table the
  scalar module ignores?** Only worth answering with Task 6.2's numbers.

## Plan Impact

**Impact Level:** Update phase file.

Two of Task 6.1's three exit criteria were factually wrong about the
code they described, and both are patched in
[`../../phases/phase-06-mediator-spectra.md`](../../phases/phase-06-mediator-spectra.md)
with a dated amendment rather than deferred:

1. "keyed on the mediator mass + partial widths" → "keyed on the
   mediator mass". `__set_spectra` reads no partial width; the wider key
   was the Cython's *declared* one, and honouring it would rebuild both
   tables on every coupling change at fixed mass.
2. "Accepted strings and error text byte-match today's" → "Accepted
   strings byte-match today's". There is no error text: an unrecognised
   mode returns `0.0` today.

No ADR. Neither amendment changes an architecture, an interface or a
public number — the first is a performance decision inside one struct
and the second is a correction of a factual claim about the source. The
behaviour the second one preserves is filed as a follow-up so the
project does not lose it.

`rules.md` is unchanged: rule 9 ("Cython `assert`s become unconditional
error returns") does not reach the silent-`0.0` fall-through, because
there is no `assert` involved.

## Stale-state sweep

Every command below was run on this branch
(`claude/cython-to-rust/task-6.1-spectrum-table-struct-design`) against
an editable install in this worktree — `hazma.__file__` and
`hazma._core.__file__` both resolve inside it.

| Check | Command | Result |
| --- | --- | --- |
| Task 6.1 status, all copies | `rg -n "6\.1" projects/cython-to-rust/task-notes/{README.md,phase-06/README.md}` | phase README cell `**Complete (2026-08-23)**`; project README phase row `**In Progress** — Task 6.1 complete (2026-08-23)`; note header `**Status:** Complete`. Agree. |
| Phase 06 frontmatter | `head -5 projects/cython-to-rust/phases/phase-06-mediator-spectra.md` | `status: In Progress` — flipped by this task, as Task 5.1 did for Phase 05. `PLAN.md`'s Phases row carries a Delivers cell, not a status, and is unchanged. |
| Every `.pyx:NNN` citation added | `grep -rhoE '\.pyx:[0-9]+(-[0-9]+)?'` over the six touched files, then `sed -n` on each line | 20 distinct citations, **all re-derived from the live sources**. Ten were wrong on the first pass — both `__set_spectra` call sites, both cache predicates, both tail branches, both mode chains, the line-term line and the grid line — and are corrected. |
| Other citations | `sed -n '1218,1222p' test/parity/cases.py` | the mediator-spectrum registrations run 1147–1219, ending before `by_name = {…}` at 1221; the follow-up cites that range. |
| Corpus mode / served kernels | `python -c "import cases; print(len(cases.rust_core_kernels()))"` | **34**, unchanged — the probe sits in `_CORE_TEST_ONLY_MODULES`, now six entries. |
| No `hazma/` source touched | `git diff origin/master --name-only -- hazma/` | no occurrences. |
| Forbidden tokens in new files | `rg -n "TODO\|FIXME\|breakpoint()\|import pdb\|print("` over the three new source files | no occurrences; preflight's own gate reports `none added`. |
| Follow-up not a duplicate | `rg` over `docs/followups/{todo,done}/`; `gh pr list --state open` | no existing entry, no open PRs. New stub plus index row. |
| The two prose lists of probe submodules | `rg -n "_CORE_TEST_ONLY_MODULES" test/parity/cases.py`; `rust/src/lib.rs` module docs | both name all six submodules and their oracles, and agree with the `frozenset`. |
| Numerical-impact statement | `pytest test/parity -q`; `pytest test/test_theory_aggregation.py -q` | `658 passed, 1 skipped` and `69 passed`. **No public value changes**; nothing appended to `../numerical-impact.md`. |
| Preflight | `env PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh --paths … --md …` | `RESULT: PASS` — eleven rows, one `SKIP` (version bump, not a closing PR). |

## Handoff to Next Task

**Read this note's `## Findings` before writing any Task 6.2 kernel
code.** Three of its rows are answers to questions the phase and project
handoffs were still asking, and re-deriving them costs a build:

1. **`grep -c SoftComplexToDouble` has been run** — 6 / 0 / 6 / 0 for
   scalar-decay / scalar-positron / vector-decay / vector-positron, one
   live site each in the photon pair and **none** in either positron
   module. Both live sites are the `** 1.5` in an FSR coefficient and
   both are the shape `crate::kernels::soft_complex` already covers, so
   Task 6.2 inherits `soft_complex_pow_1_5` and
   `complex_quotient_real_denominator` rather than deriving new shims.
   This does **not** discharge the FMA reading: `grep -c` on the
   generated C answers the complex-`pow` question only, and the
   disassembly still has to be read per kernel (Phase 04 learnings §2,
   step 1).
2. **An unrecognised mode returns `0.0`**, so `PhotonMode::parse` /
   `PositronMode::parse` return `Option` and the entry points owe that
   `0.0` on `None` — do not turn it into a raise; the tightening is
   [filed](../../../../docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md).
3. **The cache keys on the mediator mass alone**, and the phase file's
   exit criterion is amended to say so.

**Safe to assume:**

- `crate::kernels::mediator_tables` gives Task 6.2 the whole rest-frame
  half: `photon_tables(mass)` / `positron_tables(mass)` return memoized
  `Arc`s whose columns are the Phase 04 kernels evaluated natively on a
  `numpy.logspace`-identical grid, and `RestFrameTable::lookup` carries
  each clone-pair's below-grid policy. What is left for 6.2/6.3 is the
  FSR functions, the boost integrand, the `qagp` call, the line terms
  and the PyO3 boundary.
- `hazma._core.mediator_tables` is a test surface only, already in
  `_CORE_TEST_ONLY_MODULES`, so it does not move the corpus's mode.
  `cases.rust_core_kernels()` is still **34**.
- `ScalarPhotonModes::contains` does **not** exist — it was written,
  found to have no non-test caller, and removed rather than carrying a
  `dead_code` allowance. Add it back with the integrand that branches
  on it.

**Still risky / unknown:**

- **Whether the charged pion's forward-cone defect reaches the mediator
  spectra is still open**, and Task 6.2 is where it gets measured — the
  charged-pion photon table this task builds *is* the affected kernel,
  so the measurement is one lookup away.
- **No benchmark exists yet.** The cache has no caller until 6.2, which
  the phase file already charges with the comparison against pre-swap
  Cython. Establishing that baseline costs a build from a git commit
  only *after* the twins are deleted — while they are alive, 6.2 can
  measure both in one interpreter, which is much cheaper. **Take the
  benchmark before the deletion, not after.**
- **One cache slot, and the shared photon set builds a muon table the
  scalar decay module ignores.** Both are deliberate and both are
  cheap to change; 6.2's numbers decide.
