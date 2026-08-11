# Working Memory: Phase 03 — Numerics foundation

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 03
**Status:** In Progress (Tasks 3.1 and 3.2 complete 2026-08-09; Tasks 3.3
and 3.4 complete 2026-08-10)
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
| 3.2 | Special functions | — (ADR-0002 accepted) | **Complete (2026-08-09)** | [task-3.2-specfun.md](task-3.2-specfun.md) |
| 3.3 | QUADPACK port (qk15/qk21/qelg/qags/qagp) | — (ADR-0002 accepted) | **Complete (2026-08-10)** | [task-3.3-quadpack.md](task-3.3-quadpack.md) |
| 3.4 | Interpolation + boost kernels | 3.1 | **Complete (2026-08-10)** | [task-3.4-interp-boost.md](task-3.4-interp-boost.md) |
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
- **`scipy.special.kn` is not cephes `kn`** (Task 3.2). scipy dispatches
  integer orders to `kv`; only `k0`/`k1` are still cephes there. So
  `spec_math`'s faithful cephes `kn` — and equally the plan's "vendor
  the cephes routine" fallback — misses `scipy.special.kn(2, ·)` by up
  to **5.1e-9** relative over `x ∈ [1e-8, 300]`, worst at `x = 9.531`
  on the low side of its own `x = 9.55` branch switch. That enters
  `thermal_cross_section` **squared** (prefactor `x/(2·kn(2,x))²`),
  right at the corpus's 1e-8 budget for it. `Kₙ` is built instead from
  the upward recurrence `K_{m+1} = K_{m-1} + (2m/x)·K_m` on cephes
  `k0`/`k1` seeds: ≤ 3.4e-15 vs scipy for n = 0..5. **Phase 05 must not
  "simplify" it back.**
- **`spec_math::Polylog::li2` *is* `scipy.special.spence`** (Task 3.2) —
  its body is `cephes64::spence`, so the convention is `Li₂(1−z)`. The
  name is the trap, not the function. `crate::special::spence` re-exports
  it under scipy's name so no kernel has to remember which one it got;
  the muon photon kernel wants the `.pyx`'s arguments unreflected.
- **`spence` and `k1` need no such care** (Task 3.2): same cephes
  routine on both sides, agreeing to 2.4e-15 and 1.2e-15 respectively
  over the swept domains.
- **The `cython_special` C symbols equal the `scipy.special` ufuncs bit
  for bit** for all three (Task 3.2, checked through `__pyx_capi__`), so
  a test may use the ufunc as the oracle for what the `.pyx` calls. Two
  of the three (`spence`, `kn`) are *fused-type* exports mangled
  `__pyx_fuse_<i><name>`, where `i` tracks declaration order — resolve
  them by the capsule's signature string, never by a hardcoded index.
- **Negative zero is a zero argument, and a recurrence can lose that**
  (Task 3.2, PR #59 review). `-0.0 < 0.0` is false, so IEEE routes `-0.0`
  to a routine's *zero* branch — cephes and scipy both return `+∞` for
  `kn(n, -0.0)`. A recurrence seeded on `k0`/`k1` cannot: the seeds are
  `+∞` while `2m/x` is `-∞`, and `∞ + -∞` is `NaN`. Every order from 2
  up returned `NaN`. Fixed with an `x == 0.0` short-circuit.
  **The general shape for Phases 04–06: any kernel that divides by its
  argument inherits a signed-zero case the underlying cephes routine
  does not have**, and `+0.0` passing says nothing about `-0.0`. Sweep
  both signs of zero against the oracle at every order or branch, not
  just the one a reviewer names.
- **The quadrature break-point contract belongs to scipy, not to
  QUADPACK** (Task 3.3). `scipy.integrate.quad` filters `points` in
  Python before `qagpe` sees them — `np.unique`, then strictly interior —
  so the QUADPACK rule a doc-driven port would implement (sort, and
  `ier = 6` unless the extremes equal `a` and `b`) is unreachable. Read
  the wrong way round, the five `points=[-1, 1]` call sites would have
  **errored**, because QUADPACK rejects a break point equal to an
  endpoint and scipy silently drops it. Both live degeneracies are
  discards: `points=[-1, 1]` on `[-1, 1]` leaves nothing, and the heavy
  mediator's `m/mx`, `2 m/mx` fall outside `max(50/x, 100|150)`.
- **`points is None` selects `qagse` — "no break point survived" does
  not** (Task 3.3). scipy dispatches before it filters, so five of the
  twelve live call sites run `qagpe` with an *empty* list. The two
  routines are nearly indistinguishable (identical value, `neval` and
  `last` across 3,776 random combinations that converged; differing on 45,
  all of which exhausted `limit`), which is a trap for the test as much as
  for the port: the obvious singular integrands cannot tell them apart, so
  a test written on one would pass against either.
- **The port tracks scipy wherever QUADPACK converges, and only there**
  (Task 3.3). Over 11,274 random (integrand, tolerance, limit, points)
  combinations: the 4,461 converged runs reproduced scipy's `neval` and
  `last` on all but **5** (0.11%) and landed within 3.6e-2 of the
  requested tolerance (8.2e-11 relative worst case); the 6,813 that
  exhausted `limit` can separate without bound (4.5e-5 there, 11% on a
  hand-picked case), because Wynn's ε-algorithm is chaotic on a
  non-converging sequence. Termination flags agreed on all 11,274.
  **Phases 04–06: no live shape reaches the second regime** — each
  returns `ier = 0` and `test/test_core_quad.py` asserts it.
  **A narrower sweep is what made this look cleaner than it is:** an
  earlier 6,000-combination design using at most two break points found
  *zero* subdivision mismatches among converged runs, and the mismatches
  only appeared once 9- and 39-point grids went into the draw. A sweep's
  parameter space is part of its result.
- **Only `qk21` is on the live path** (Task 3.3): `qagse` and `qagpe` both
  evaluate with the 21-point rule and nothing else, so `qk15` — named in
  the exit criteria — is reachable from no hazma call site. Kept as an
  independent second rule for the cross-checks, not as production code.
- **A mutation harness can poison its own baseline** (Task 3.3). Two
  copies of the campaign ran concurrently after the first was wrongly
  read as failed to start, so the second's "pristine" source already
  carried the first's mutation and every result was measured against a
  wrong Gauss–Kronrod table. The tell was easy to rationalise — mutating a
  `qk15` weight reported `qk21` tests failing — and what settled it was a
  check owing nothing to the crate: re-parsing the Fortran `data`
  statements and comparing f64 bit patterns. **Assert a green baseline
  before a campaign and again after**, and hold a lock. Two smaller
  siblings: `cargo test`'s default parallelism interleaves
  `test NAME ... FAILED` lines so a scraped failure list names the wrong
  tests (`-- --test-threads=1`), and a background job reported as failed
  may still be running.
- **The compiled Cython contracts `a*b + c` into fused multiply-adds, and
  the port has to as well** (Task 3.4). Clang's default is
  `-ffp-contract=on`, and the corpus's capturing platform (macOS/arm64)
  contracts eight distinct expressions across `boost.pyx` — plus
  `slope·(x − xp[j]) + fp[j]` inside NumPy's own `arr_interp`. Written
  the obvious unfused way the port misses the corpus by up to **3.6e-12**
  relative *on the corpus's own grids* for the seven tabulated photon
  spectra, past the 1e-12 `TABULATED` budget; with `f64::mul_add` at
  those sites it is bit-equal at every one of those points. The sites
  were established twice — `fmsub`/`fmadd` in the disassembled `.so`, and
  a 16-combination bisection against the live kernel in which only
  all-on reaches zero mismatches. **The converse is the trap:**
  `boost_beta` spells its square as `(mass/energy) ** 2`, whose rounded
  product completes before the subtraction, and **none** of its ten
  inlining call sites contract it. Contraction is a per-expression fact,
  not a per-file one, and Phases 04–06 must measure each kernel rather
  than adopting a house style.
- **`np.trapezoid` reduces pairwise, and `np.interp` has quirks the
  reference does not list** (Task 3.4). `ndarray.sum` runs eight
  accumulators over 128-element blocks and recurses; a sequential sum is
  a different number (1.8e-15 relative on the 500-row tables), so
  `rust/src/boost.rs` mirrors the blocking. `np.interp`: a one-point grid
  answers *everything* with `fp[0]` including NaN (the NaN check lives on
  the multi-point path only), and duplicate abscissae resolve to the
  **last** matching node. Also: the live tables are rows of a transposed
  `np.loadtxt` result, so they are **strided views** and
  `PyReadonlyArray1::as_slice` refuses them.
- **`boost_integrate_linear_interp` mis-covers its window at both ends,
  and near threshold it is wrong by four orders of magnitude** (Task
  3.4). The interior sum's slice is exclusive at the top while the upper
  partial-cell term starts at `x[ihigh]`, so one cell is covered by
  nothing — and with both bounds inside a single cell the two partial-cell
  terms **overlap**, over-counting by (cell width)/(window width), which
  diverges as `β → 0`. Consequence on the public API: all seven tabulated
  photon spectra blow up rather than converging to their own rest-frame
  spectrum as the parent approaches rest (6,500× to 33,000× one part in
  1e12 above rest). Reproduced per rule 1, pinned in both languages,
  filed as
  [`../../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md).
  The inventory's §Bugs lists the same class in the *dead*
  `boost_integrate_linear_interp_massive`; the live twin was not flagged.
  **Phase 04 inherits it unchanged** — the corpus pins these values, so a
  swap that "fixes" it fails the gate.
- **A `cdef` with a `.pxd` declaration is callable from Python** (Task
  3.4), which is what replaced the micro-fixtures the phase file's Task
  3.4 criterion named and that Phase 01 never captured (the corpus only
  sees top-level `def`s). Cython exports declared `cdef`s through
  `__pyx_capi__` as capsules; `ctypes` calls them. Two constraints:
  `PYFUNCTYPE`, not `CFUNCTYPE` — the latter releases the GIL and
  anything calling back into NumPy segfaults (exit 139, no Python error);
  and the capsule's *name* is its C signature string, so the argument
  list is checkable rather than a silent stack corruption. **Any later
  task needing a `cdef` oracle should reach for this instead of adding a
  temporary shim to a `.pyx`.**
- **A Python-visible test surface on `hazma._core` reads as a started
  port** (Task 3.2). Registering `hazma._core.special` flipped the
  parity corpus straight out of bit-equality mode
  (`exact=False, detail='hazma._core serves 3 kernel(s)'`) for the rest
  of the project, with nothing turning red — the class Task 2.1 already
  fixed once. `cases._CORE_TEST_ONLY_MODULES` now exempts a *submodule*
  (not a name, which would exempt a future real kernel too), and
  `test_test_only_core_submodules_have_no_importer` makes the exemption
  conditional on nothing under `hazma/` importing it. **Any later task
  that puts a non-kernel on the extension inherits this mechanism, and
  must not widen the exemption to quiet a red mode check.**

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
- **`special.rs` is PyO3-free; `special_probe.rs` is the Python half**
  (Task 3.2). Rule 8 keeps the math GIL-free, and the probe registers
  `hazma._core.special` only because the oracle (scipy) lives in Python.
  All three bindings route through `dispatch::map_unary`, so the sweeps
  run as arrays rather than as 25k-iteration Python loops — the kind of
  test that otherwise gets trimmed to a dozen points later.
- **The QUADPACK translation is deliberately literal** (Task 3.3):
  1-based indexing kept by giving every array a dead element 0, every
  `go to` a labelled `break` carrying its Fortran statement number, same
  variable names, same magic constants. Idiomatic Rust would read better
  and be much harder to check against the source. The cost is three
  module-level clippy `allow`s (`needless_range_loop`,
  `explicit_counter_loop`, `int_plus_one`), each with its reason written
  down.
- **`quad` is the entry point Phases 04–06 call, not `qagse`/`qagpe`**
  (Task 3.3) — it is the one that reproduces scipy's limit ordering and
  break-point filtering, so twelve call sites do not each re-derive them.
  `ier` rides along inside `Ok`; only the inputs scipy raises
  `ValueError` for are `Err`, because hazma's call sites read
  `quad(...)[0]` and never see scipy's warning.
- **The Gauss–Kronrod tables were extracted by script, not typed**
  (Task 3.3), and are pinned by **degree of exactness** (22 for `qk15`,
  31 for `qk21`) plus a complement test that the next even degree is not
  exact — a wrong digit breaks exactness, where a spot check against one
  integral could be passed by a rule that is merely close.
- **`quad_probe` takes a Python callable on purpose** (Task 3.3). A menu
  of Rust integrands would compare a Rust integrand against a Python one
  and blame the quadrature for the difference; with a callback the
  integrand is byte-identical on both sides. Same shape as `special_probe`
  and it inherits the same `_CORE_TEST_ONLY_MODULES` exemption and
  importer guard rather than widening them.
- **`interp.rs` and `boost.rs` are PyO3-free; `interp_probe.rs` and
  `boost_probe.rs` are the Python halves** (Task 3.4) — the shape Tasks
  3.2 and 3.3 set, with two probe modules rather than one because the two
  oracles are different (NumPy versus the Cython capsules). Both join
  `cases._CORE_TEST_ONLY_MODULES` under Task 3.2's importer guard; the
  exemption mechanism is reused, not widened.
- **The QUADPACK-style literal translation was *not* carried into
  `boost.rs`** (Task 3.4). Ninety lines of ordinary arithmetic do not
  need 1-based indexing and labelled `break`s to stay checkable, so the
  Rust reads as Rust (`Option` for the `-1` index sentinel, a closure for
  the `y / x` column the Cython materialises) while every branch,
  tolerance and ordering is preserved verbatim. The literal posture is
  for Fortran with `go to`, not for a policy.
- **`interp` asserts its preconditions; the boost integral returns
  `Result`** (Task 3.4). The Cython's two `assert`s become error returns
  per rule 9, plus an `EmptyTable` variant for the case it leaves
  undefined; `interp` keeps a plain `f64` return because the probe raises
  `ValueError` with NumPy's own wording first, so the assert is
  unreachable from Python and no Rust call site has to unwrap.
- **Only `n = 2` is live, but `bessel_kn` carries general `n`**
  (Task 3.2), because the recurrence is general and the order factor
  `2m/x` is *invisible* at n = 2 (it is `2/x` there). Both the ν = 2
  Wronskian in `cargo test` and the n = 0..5 Python sweep exist for that
  one mutation; the ν = 2 case was added after a first pass where
  `cargo test` alone missed it.

## Files Changed

### Task 3.4

- `rust/src/interp.rs` — **new**, `np.interp` with NumPy's full contract,
  a `# Sources and licensing` header, the `mul_add` rationale and 11 unit
  tests.
- `rust/src/boost.rs` — **new**, the four kernels plus `trapezoid` /
  `pairwise_sum`, `BoostError`, the contracted-site rationale, the
  `# Faithfulness notes` on the four preserved defects and 13 unit tests.
- `rust/src/interp_probe.rs`, `rust/src/boost_probe.rs` — **new**,
  registration-only `hazma._core.interp` and `hazma._core.boost`.
- `rust/src/lib.rs` — four `mod` lines, two `add_submodule` calls, and the
  reconciled paragraphs on the foundation modules and their probes.
- `test/test_core_interp.py`, `test/test_core_boost.py` — **new** (the
  latter carries the `__pyx_capi__` shim).
- `test/parity/cases.py`, `test/parity/test_parity.py`,
  `test/parity/README.md` — `hazma._core.{interp,boost}` added to
  `_CORE_TEST_ONLY_MODULES` and the three prose sites reconciled.
- `hazma/_core.pyi` — the unstubbed-submodule comment now covers all four
  probes (the only change under `hazma/`, non-executable).
- `docs/followups/todo/boost-integral-drops-last-interior-cell.md` +
  `docs/followups/README.md` — **new**, the preserved defect.
- **Two canonical patches:** `../../phases/phase-03-numerics-foundation.md`
  (five Task 3.4 criteria added during execution) and
  `../../references/numerics-replacements.md` (the measured block).

### Task 3.1

- `rust/src/constants.rs` — **new**, 224 `pub const`s in three
  namespaces plus five unit tests and a `# Sources` provenance header.
- `rust/src/lib.rs` — `pub mod constants;` and the paragraph on why.
- `test/test_core_constants.py` — **new**, 25 tests.

### Task 3.2

- `rust/src/special.rs` — **new**, `spence` / `bessel_k1` / `bessel_kn`
  over `spec_math`, a `# Sources and licensing` provenance header, the
  `kn` deviation rationale, the `x == 0.0` guard, and 9 unit tests.
- `rust/src/special_probe.rs` — **new**, registration-only
  `hazma._core.special` for the scipy sweep.
- `rust/src/lib.rs` — `pub mod special;` + `mod special_probe;`, the
  submodule registration, and the paragraph on why the probe is the
  exception to "registration-only means per-domain".
- `rust/Cargo.toml` / `rust/Cargo.lock` — `spec_math = "0.1.6"`.
- `test/test_core_special.py` — **new**, 65 tests in 9 classes.
- `test/parity/cases.py`, `test/parity/test_parity.py`,
  `test/parity/README.md` — `_CORE_TEST_ONLY_MODULES`, its importer
  guard test, and the reconciled prose.
- `hazma/_core.pyi` — comment recording the unstubbed `special`
  submodule and why (the only change under `hazma/`, non-executable).
- Canonical patches: `../../phases/phase-03-numerics-foundation.md`
  (three Task 3.2 criteria added),
  `../../references/numerics-replacements.md` (the measured block).

### Task 3.3

- `rust/src/quad.rs` — **new**, 1,972 lines: `qk15`, `qk21`, `qelg`,
  `qpsrt`, `qagse`, `qagpe`, the scipy-shaped `quad` driver and
  `filter_points`, a `# Sources and licensing` provenance header, the
  call-site table, and 24 unit tests.
- `rust/src/quad_probe.rs` — **new**, registration-only
  `hazma._core.quad` (a Python callable in, so scipy and the port see the
  same integrand).
- `rust/src/lib.rs` — `pub mod quad;` + `mod quad_probe;`, the submodule
  registration, and the reconciled paragraphs on the two probe modules.
- `test/test_core_quad.py` — **new**, 58 tests in 8 classes.
- `test/parity/cases.py`, `test/parity/test_parity.py`,
  `test/parity/README.md` — `hazma._core.quad` added to
  `_CORE_TEST_ONLY_MODULES`, and the three places naming `special` alone
  reconciled.
- `hazma/_core.pyi` — the unstubbed-submodule comment now covers both
  probes (the only change under `hazma/`, non-executable).
- **Two canonical patches:** `../../phases/phase-03-numerics-foundation.md`
  (four Task 3.3 criteria added during execution) and
  `../../references/numerics-replacements.md` (the measured break-point
  contract).

## Verification

- **Task 3.4 (2026-08-10):** bare `pytest -q` →
  `1314 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13, and
  `tolerances.provenance` → `exact=True` checked directly; +102 on Task
  3.3's 1212, all of them this task's new tests).
  `pytest test/test_core_interp.py -q` → `33 passed in 0.46s` (6
  classes); `pytest test/test_core_boost.py -q` → `69 passed in 0.91s`
  (9 classes);
  `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  `67 passed` (24 new); clippy, fmt and `markdownlint --dot` clean;
  `scripts/agents/preflight.sh` RESULT: PASS. Twenty-one mutations
  against `interp.rs` and `boost.rs`, sequential behind a lock with a
  green baseline before and after — **17 of the first 20 caught**, and
  the three survivors are exactly what
  `test_an_infinite_node_returns_its_own_value` and
  `test_the_window_edges_sit_on_the_same_double_as_the_cython` were
  written for; a third round confirmed all 21 caught. The survivors'
  shared shape is worth carrying: each moved a *branch boundary* by one
  double without touching any returned value, so no grid sample could
  see it. Tables in the task note.
- **Task 3.3 (2026-08-10):** bare `pytest -q` →
  `1212 passed, 13 skipped` on the capturing environment, parity suite
  included and in bit-equality mode (skip count unchanged at 13, which is
  what proves the mode; +58 on Task 3.2's 1154, all of them this task's
  new tests). `pytest test/test_core_quad.py -q` → `58 passed in 5.10s`
  (8 classes, population derived by `--collect-only`);
  `cargo test --manifest-path rust/Cargo.toml --no-default-features` →
  `43 passed` (27 new); clippy and fmt clean;
  `scripts/agents/preflight.sh` RESULT: PASS. Seventeen mutations against
  `quad.rs`, each from a green baseline and reverted after — 15 caught on
  the first pass, and the two that were not (`ndin`, the roundoff
  threshold) are what `TestAdaptiveHeuristics` was written for, with both
  re-run against the final tree. The Gauss–Kronrod literals are checked
  against the netlib Fortran as f64 bit patterns (47 values,
  `MISMATCHES: 0`) by a script independent of the crate — which is what
  caught a poisoned mutation baseline. Tables in the task note.
- **Task 3.2 (2026-08-09; PR #59 review round 1, 2026-08-10):** bare
  `pytest -q` → `1154 passed, 13 skipped` on the capturing environment,
  parity suite included and in bit-equality mode (skip count unchanged
  at 13, which is what proves the mode; +66 on Task 3.1's 1088, all of
  them this task's new tests). `pytest test/test_core_special.py -q` →
  `65 passed in 0.50s`; `cargo test --manifest-path rust/Cargo.toml
  --no-default-features` → `16 passed` (9 new); clippy and fmt clean;
  `scripts/agents/preflight.sh` RESULT: PASS. Eleven mutations — nine
  against `special.rs`, two against the corpus guard — each caught by
  the test whose name claims it (tables in the task note). Measured
  agreement with scipy: `spence` 2.425e-15, `k1` 1.215e-15 (3.078e-16
  through the underflow tail), `kn(2, ·)` 9.786e-16 over `x ≤ 300`,
  4.007e-15 worst across orders 0–5 — against 5.1e-9 for the cephes
  `kn` that was rejected. **Round 1 figures supersede the pre-review
  ones** (`1142 / 53 / 15`), which were partly typed rather than
  derived; every count here now comes from a command quoted in the task
  note.
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

- ~~`spec_math::li2` convention vs `scipy.special.spence` — Task 3.2
  resolves.~~ **Resolved (Task 3.2, 2026-08-09):** the same convention,
  `Li₂(1−z)`, because `li2`'s body is `cephes64::spence`. Verified
  against scipy on the exit-criterion grid at 2.425e-15, and
  distinguished from `Li₂(z)` by an independent series at z = 0.25/0.75
  rather than only by agreement.
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
`../README.md`, this file, then the phase file — whose Tasks 3.2, 3.3
and 3.4 blocks all now carry criteria added during execution. **Only
Task 3.5 (dispatch and error layer) remains**, and it closes the phase:
synthesize `../../learnings/phase-03-numerics-foundation.md`, flip the
phase file's frontmatter to `status: Complete`, and update the `PLAN.md`
Phases row.

**Currently safe to assume:**

- Every live integral is finite-interval — `qagi` is out of scope.
- **Task 3.4 is done, so `hazma_core::{interp, boost}` exist.**
  `interp::interp` is `np.interp` with NumPy's full contract;
  `boost::{boost_beta, boost_gamma, boost_delta_function,
  boost_integrate_linear_interp}` are the four live routines of
  `hazma/_utils/boost.pyx`. Both are bit-equal to what they replace on
  the capturing platform, PyO3-free, and already route through
  `dispatch::map_unary` in their probes — so whatever Task 3.5 settles
  about the four live dispatch behaviors applies to them for free.
- **Task 3.1 is done, so `hazma_core::constants` exists and Task 3.4 is
  unblocked.** `constants::{pdg, legacy}` are the two Cython tables and
  `constants::derived::<source_pyx>` the module-local `DEF`s, all
  bit-equal to the Cython and held there by
  `test/test_core_constants.py` (25 tests, 0.03s, platform-independent)
  and five `cargo test` units. Name the table the `.pyx` you are porting
  `include`s: `pdg` for everything under `hazma/spectra/**`, `legacy`
  for the four mediator spectrum extensions.
- **Task 3.2 is done, so `hazma_core::special` exists.** `spence` (in
  scipy's `Li₂(1−z)` convention), `bessel_k1` and `bessel_kn`, PyO3-free
  and tracking `scipy.special` to ≤ 4.0e-15 over every domain hazma
  reaches. Phase 04's muon photon kernel and Phase 05's thermal ⟨σv⟩
  call these **directly in Rust**; `hazma._core.special` is a test
  surface and must stay importer-free, or the parity corpus leaves
  bit-equality mode.

**Currently risky / unknown:**

- **Do not "clean up" a `mul_add` in `boost.rs` or `interp.rs`, and do
  not add one where they do not have it** (Task 3.4). Unfusing costs up
  to 3.6e-12 against the corpus — past the 1e-12 `TABULATED` budget —
  and fusing `boost_beta`, which the Cython does *not* contract at any
  of its ten call sites, moves every boosted spectrum. Every Phase 04
  kernel needs its own disassembly or bisection; there is no house style
  to apply.
- **`boost_integrate_linear_interp` is wrong near threshold and stays
  wrong for now** (Task 3.4): all seven tabulated photon spectra diverge
  instead of converging to their rest-frame values as the parent slows.
  The corpus pins those values, so a Phase 04 swap that repairs the
  coverage **fails the gate**. Repair blocked until after Phase 06
  Task 6.4 —
  [`../../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md).
- **A test that compares against a *locally compiled* oracle is scoped to
  that build** (Task 3.4, PR #61 review round 1). Nineteen bit-equality
  assertions against the Cython and NumPy passed on macOS/arm64 and failed
  on Linux/x86-64, where neither contracts — the port was right and the
  tests over-claimed. `CYTHON_CONTRACTS` / `NUMPY_CONTRACTS` are measured
  at import and the comparisons skip where false, the same scoping
  `test/parity` already has. **Phases 04–06 inherit this**: any kernel test
  whose oracle is the Cython twin needs the same guard, and loosening to a
  tolerance is the wrong fix — the worst relative gap sits at a
  cancellation point where any admitting tolerance would hide real defects.
- **An edge that only decides a branch needs a bisection test, not a
  grid** (Task 3.4). Three mutations survived a 40,000-draw sweep because
  they moved a support boundary by one double without touching any
  returned value. Any Phase 04–06 kernel with a window, threshold or
  piecewise switch inherits that — and check the sweep's *parameter*
  space reaches the branch at all (with `m = 0` the fused and unfused
  momenta are bit-identical, so only massive draws could see it).
- `qelg` Fortran→Rust translation is the fiddliest item in the project;
  budget review time accordingly.
- **Task 3.3 inherits Task 3.2's central lesson**: do not design against
  the documentation of what scipy is *supposed* to do — measure what it
  does. Task 3.2 found `scipy.special.kn` is `kv` rather than cephes
  only by running the sweep, and 3.3's breakpoint-preprocessing
  criterion already says the same thing about `quad(points=...)`.
  **Task 3.4 makes it three for three**, and widens it past scipy: the
  compiled artifact's own arithmetic is a hypothesis too.
- **Do not "simplify" `special::bessel_kn` to `spec_math`'s** — a
  5.1e-9 miss that the corpus's 1e-8 `thermal_cross_section` budget
  would absorb. `test_beats_cephes_kn_at_its_worst_argument` names the
  exact substitution it guards against.
- **`derived::photon_pion` deliberately mixes both tables** — PDG
  aliases, legacy-frozen literals. Read that module's doc comment before
  touching the charged-pion photon kernel in Phase 04.
- `derived::positron_pion::ENG_MU_PI_RF` and
  `derived::photon_pion::ENG_MU_PIRF` are the same physical quantity,
  different numbers, one underscore apart.
