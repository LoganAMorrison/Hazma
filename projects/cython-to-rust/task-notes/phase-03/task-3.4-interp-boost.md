# Task 3.4: Interpolation + boost kernels

**Date:** 2026-08-10
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-03-numerics-foundation.md` (Task
3.4); `../../references/numerics-replacements.md` (§`np.interp`
semantics, §`boost_integrate_linear_interp`); `../../rules.md` rules 1, 3,
8, 9
**Related ADRs:** none (nothing revises ADR-0001 or ADR-0002; the
provenance here is original work plus NumPy's BSD-3-Clause behavior)
**Depends On:** Task 3.1 (`hazma_core::constants`)

## Objective

Port `np.interp` and the four live routines of `hazma/_utils/boost.pyx`
to PyO3-free Rust, with per-branch tests pinned against the
implementations they replace, so Phase 04's tabulated photon spectra have
a foundation that reproduces the parity corpus.

## Exit Criteria

From the phase file's Task 3.4 block, plus the five criteria this task
added there during execution:

- `interp` replicating `np.interp` exactly (ascending grid, edge
  clamping, node hits) — property-tested against NumPy over random grids.
- `boost_beta` / `boost_gamma` / `boost_delta_function` /
  `boost_integrate_linear_interp` ported with per-branch unit tests
  (interior, both partial edge cells, below-table `1/E` tail, above-table
  clamp, β→0 guard) pinned against the Cython originals.
- The oracle is the live Cython through `__pyx_capi__`, not the
  "micro-fixtures captured in Phase 01" the criterion names — they do not
  exist and could not (see Findings).
- The port reproduces the shipped Cython's fused multiply-adds where they
  occur and **not** where they do not, held to bit-equality rather than a
  tolerance.
- The dropped interior cell is reproduced, not repaired, and filed.
- `interp` carries NumPy's quirks (one-point-grid NaN, duplicate-node
  tie-break) as well as its documented contract.
- The parity corpus's served-kernel predicate stays sound.
- The bit-equality comparisons are scoped to a platform whose references
  contract, and the suite is green where they do not (added in review).

## Inputs Reviewed

- `../../PLAN.md`; `../README.md`; `README.md` (phase working memory);
  `../../phases/phase-03-numerics-foundation.md`; `../../rules.md`.
- `../../references/numerics-replacements.md` §§ `np.interp`,
  `boost_integrate_linear_interp`; `../../references/cython-inventory.md`
  §Bugs.
- `hazma/_utils/boost.pyx`, `hazma/_utils/boost.pxd`, and the twelve
  `.pyx` call sites (`rg` over `hazma/`).
- `rust/src/{lib,dispatch,special,special_probe,quad,quad_probe}.rs` for
  the conventions this task follows.
- `test/parity/{cases,tolerances,test_parity}.py`, `test/parity/README.md`.
- The generated `hazma/_utils/boost.c` and the compiled
  `hazma/_utils/boost.cpython-312-darwin.so` (disassembly).
- `docs/agents/lessons.md`, `docs/agents/environment.md`.

## Findings

- **The oracle the exit criterion names does not exist, and a better one
  does.** Phase 01's corpus enumerates top-level `def`s in the surviving
  `.pyx`; every routine here is `cdef`, so no micro-fixture was captured
  and none could have been. But `boost.pxd` *declares* the `cdef`s, which
  makes Cython export them through `hazma._utils.boost.__pyx_capi__` as
  capsules — so `ctypes` can call the live kernel at whatever arguments a
  test picks, which is strictly stronger than a frozen sample. Two
  mechanical constraints, both hit: the shim must use
  `ctypes.PYFUNCTYPE`, since `CFUNCTYPE` releases the GIL and
  `boost_integrate_linear_interp` calls `np.trapezoid` (`CFUNCTYPE`
  segfaults, exit 139, with no Python-level error); and the capsule's
  *name* is its C signature string, so a changed argument list is
  checkable rather than a silent stack corruption.
- **The shipped Cython's arithmetic is fused, and reproducing that is
  what makes the port pass the corpus.** Clang defaults to
  `-ffp-contract=on`. Written the obvious unfused way, the port misses
  the corpus by up to **3.6e-12** relative *on the corpus's own grids*
  for the seven tabulated photon spectra — past the 1e-12 `TABULATED`
  budget in `test/parity/tolerances.py`, so the Phase 04 swap would have
  failed its own gate and the only alternatives would have been widening
  a budget by three decades or shipping a declared drift. With
  `f64::mul_add` at the contracted sites the port is **bit-equal at every
  one of those points** (0.000e+00, all seven tables).
- **Which sites contract is a per-expression fact, established twice.**
  Disassembling `hazma/_utils/boost.cpython-312-darwin.so` shows
  `fmsub`/`fmadd` at `1 - β·β` (both routines), `e·e - m·m` and
  `e0·e0 - m·m` and `e ∓ β·k` (the line), and `y1 - m·x1`,
  `0.5·m·(x2 + lb) + b` and the accumulation itself (each partial cell).
  Independently, bisecting all 16 on/off combinations of the integral's
  four sites against the live kernel over 2,462 corpus-grid points found
  exactly one combination at zero mismatches — all four on; the next best
  left 115. NumPy's `arr_interp` contracts too: unfused, `interp` misses
  `np.interp` at 1,549 of 20,204 eta-table points by up to 1.1e-13.
- **`boost_beta` must *not* be fused, and that is not symmetry-breaking
  for its own sake.** The Cython spells it `(mass / energy) ** 2`, whose
  rounded product completes before the subtraction; disassembling the
  inlined copies in `_eta`, `_kaon` and `_positron/_pion` shows
  `fdiv / fmul / fsub / fsqrt` with no contraction at any of them.
  Fusing it would move every boosted spectrum for no reason. "The
  compiler contracts" is a claim about an expression, not about a file.
- **`np.trapezoid` reduces pairwise, not sequentially.** `ndarray.sum`
  runs eight accumulators over 128-element blocks and recurses above
  that; a left-to-right sum is a different number, up to 1.8e-15 relative
  on the 500-row tables. Reproducing the blocking is the only way the
  interior sum is bit-equal. A Python transcription of the blocking
  matched `np.sum` on 3 random arrays at each of 306 lengths from 0 to
  5,000 with zero mismatches, which is what justified writing it in Rust.
- **The live tables are strided views, not contiguous buffers.** They are
  rows of a transposed `np.loadtxt` result, so `PyReadonlyArray1::as_slice`
  raises `TypeError: The given array is not contiguous or is misaligned`.
  Both probes copy with `as_array().to_vec()`. Phase 04 kernels own their
  tables in Rust and never see the stride, but any future probe taking a
  live table inherits this.
- **`boost_integrate_linear_interp` mis-covers its window at both ends,
  and near threshold it is wrong by four orders of magnitude.** The
  interior sum's slice `yy[ilow:ihigh]` is exclusive at the top while the
  upper partial-cell term starts at `x[ihigh]`, so
  `[x[ihigh-1], x[ihigh]]` is covered by nothing — and when the window is
  clamped, the table's final row contributes to no term at all. The
  mirror-image case is worse: with both bounds inside one cell,
  `ihigh = ilow - 1` and the two partial-cell terms **overlap**, covering
  ~two whole cells instead of the sliver between the bounds, over-counting
  by (cell width)/(window width) — which diverges as `β → 0`. Consequence
  on the public API: all seven tabulated photon spectra blow up instead of
  converging to their own rest-frame spectrum as the parent approaches
  rest (ratios 6,500× to 33,000× one part in 1e12 above rest; table in the
  follow-up). Reproduced per rule 1, pinned in both languages, filed as
  [`../../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md).
  `../../references/cython-inventory.md` §Bugs lists the same class in the
  *dead* `boost_integrate_linear_interp_massive`; the live twin was not
  flagged.
- **`np.interp` has two behaviors the reference's three-bullet contract
  omits.** A one-point grid answers *everything* with `fp[0]`, NaN
  included, because NumPy's NaN check sits on the multi-point path only;
  and duplicate abscissae resolve to the **last** matching node, because
  its bisection walks past equal keys. A port written from the reference
  alone would have got both backwards.
- **`boost_jac` and `boost_eng` are exported and called by nothing.**
  `rg` over the tree at execution time finds no consumer; they are out of
  this port's scope and Phase 06 Task 6.4 deletes the extension. They are
  not dead weight for the tests, though — `boost_eng(ep, mp, 1, 0, 0)`
  reduces to `γ` exactly and is used as the Cython-side oracle for
  `boost_gamma`.

## Decisions and Implementation Notes

- **`f64::mul_add` at exactly the contracted sites, and nowhere else.**
  The alternative — plain arithmetic plus a widened budget — was rejected
  because it costs three decades of resolution on the class of error the
  corpus exists to catch, and because the fused form is *more* portable
  rather than less: the Rust returns the same numbers on x86-64 and
  arm64, while the Cython's answer depends on whether its target has FMA.
  Stated as a consequence rather than hidden: on a platform whose C
  compiler does not contract, hazma today returns the unfused values, and
  the Phase 04 swap will move them by up to that same 3.6e-12.
- **`pairwise_sum` mirrors a NumPy implementation detail, deliberately.**
  Mimicking a documented contract is faithfulness; mimicking a
  compiler's instruction selection would not be — except that here the
  compiler's choice is what the corpus records, which is why both end up
  in. The cost is a coupling to NumPy's reduction, and the mitigation is
  that `TestTrapezoidSummation` compares against the live `np.trapezoid`,
  so a future NumPy change is a red test rather than a silent drift.
- **Two probe submodules, not one.** `hazma._core.interp` and
  `hazma._core.boost` are separate because their oracles are (NumPy
  versus the Cython capsules), and both join
  `cases._CORE_TEST_ONLY_MODULES` under the importer guard Task 3.2
  built — the same mechanism, not a widened exemption. Third instance of
  `docs/agents/lessons.md` `[gate-disabled-stays-green]` in this project.
- **`interp` is scalar-in, scalar-out; the array form lives in the
  probe.** Every Cython call site passes a scalar. The probe routes the
  abscissa through `dispatch::map_unary` so a 20,000-point sweep against
  NumPy is one call rather than a Python loop — the kind of test that
  otherwise gets trimmed later.
- **`boost_integrate_linear_interp` returns `Result`, and the probe
  validates once.** Rule 9 turns the Cython's two `assert`s into error
  returns, and adds `EmptyTable` for the case the Cython leaves undefined
  (`x[npts - 1]` with `wraparound(False)` and no bounds check). The guards
  depend only on `beta` and the table, so the probe checks them once
  before mapping rather than deciding what value a failed element takes.
- **`interp` asserts its preconditions rather than returning `Result`.**
  The probe raises `ValueError` with NumPy's own wording first, so the
  assert is unreachable from Python and every Rust call site keeps a plain
  `f64` return. Panicking across FFI is what Task 3.3 ruled out; this
  cannot reach FFI.
- **The cross-implementation comparisons are scoped to a contracting
  platform** (added in PR #61 review round 1, after CI caught it). Whether
  the local Cython and NumPy fuse their multiply-adds is a property of the
  compiler that built *them*, so on Linux/x86-64 the references compute the
  unfused values and 19 bit-equality assertions failed — the port was
  right, the tests over-claimed. `CYTHON_CONTRACTS` / `NUMPY_CONTRACTS` are
  now measured at import and those comparisons skip where false, which is
  the scoping the parity corpus already has (CI runs
  `pytest --ignore=test/parity` off macOS for the same reason). Loosening
  to a tolerance was rejected: the worst *relative* gap between the two
  forms sits at a catastrophic-cancellation point (the eta tail, where the
  interpolant is 2.4e-26 against a table of scale 0.2 — an absolute gap of
  1.4e-30), so a tolerance admitting it would hide real defects. The
  per-branch tests keep their platform-independent halves running
  everywhere and route only the bit-equality claim through
  `assert_matches_cython`, which skips mid-test *after* those have run.
- **The QUADPACK-style literal-translation posture does not apply here.**
  `boost.pyx` is 90 lines of ordinary arithmetic, so the Rust reads as
  Rust (`Option` for the `-1` index sentinel, a closure for the `y / x`
  column the Cython materialises) while every branch, tolerance and
  ordering is preserved.

## Files Changed

- `rust/src/interp.rs` — **new**, `np.interp` with NumPy's full contract,
  a `# Sources and licensing` header, the `mul_add` rationale, and 11
  unit tests.
- `rust/src/boost.rs` — **new**, the four kernels plus `trapezoid` /
  `pairwise_sum`, `BoostError`, the contracted-site rationale, the
  `# Faithfulness notes` on the four preserved defects, and 13 unit
  tests.
- `rust/src/interp_probe.rs`, `rust/src/boost_probe.rs` — **new**,
  registration-only `hazma._core.interp` and `hazma._core.boost`.
- `rust/src/lib.rs` — the four `mod` lines, two `add_submodule` calls, and
  the reconciled paragraphs on the foundation modules and their probes.
- `test/test_core_interp.py` — **new**.
- `test/test_core_boost.py` — **new**, including the `__pyx_capi__` shim.
- `test/parity/cases.py`, `test/parity/test_parity.py`,
  `test/parity/README.md` — `hazma._core.{interp,boost}` added to
  `_CORE_TEST_ONLY_MODULES`, and the three places naming the exempted
  submodules reconciled.
- `hazma/_core.pyi` — the unstubbed-submodule comment now covers all four
  probes (the only change under `hazma/`, non-executable).
- `docs/followups/todo/boost-integral-drops-last-interior-cell.md` +
  `docs/followups/README.md` — the preserved defect.
- **Two canonical patches:**
  `../../phases/phase-03-numerics-foundation.md` (five Task 3.4 criteria
  added during execution) and
  `../../references/numerics-replacements.md` (the measured block).

## Verification

**Commands and their real summary lines**, all on the corpus's capturing
environment (CPython 3.12.12, NumPy 2.5.1, SciPy 1.18.0, macOS/arm64), on
a tree rebuilt with `uv pip install -e . --no-build-isolation` before
anything was run:

| Command | Result |
| --- | --- |
| `pytest -q` (bare, the gate) | `1314 passed, 13 skipped, 5 warnings in 596.25s` |
| `pytest test/test_core_interp.py -q` | `33 passed in 0.46s` (6 classes) |
| `pytest test/test_core_boost.py -q` | `69 passed in 0.91s` (9 classes) |
| `cargo test --manifest-path rust/Cargo.toml --no-default-features` | `67 passed` (24 new: 11 in `interp`, 13 in `boost`) |
| `cargo fmt --check` / `cargo clippy --all-targets -- -D warnings` | clean |
| `markdownlint --dot` over the eight changed `.md` | clean |
| `scripts/agents/preflight.sh --paths "…"` | **RESULT: PASS** |

`+102` on Task 3.3's `1212 passed, 13 skipped`, and all 102 are this
task's tests. **The skip count is unchanged at 13, which is what proves
the parity corpus ran in bit-equality mode**; confirmed directly rather
than inferred —
`tolerances.provenance(manifest)` → `Provenance(exact=True, detail='')`.

**What the tests cover**, by class rather than by count:

*`test/test_core_interp.py`* — `TestAgainstNumpy` (bit-equality with
`np.interp` on all seven live tables over 20,204 abscissae each: 20,000
random interior points, every node, every node nudged ±1e-13, and four
out-of-range; plus 50 random grids with cells of wildly unequal width);
`TestFusedArithmetic` (the fused and unfused forms are shown to differ
somewhere before the Rust is asserted to side with NumPy, so the test
cannot pass vacuously on a non-contracting platform);
`TestClamping`; `TestQuirks` (NaN propagation, the one-point grid's NaN
asymmetry, the duplicate-node tie-break, both infinite-cell rescues, and
the infinite-node short circuit); `TestErrors` (both `ValueError`s, each
asserted against NumPy's own message); `TestDispatch`.

*`test/test_core_boost.py`* — `TestOracle` (the `__pyx_capi__` shim is
the Cython: the export set, every capsule's C signature string, and a
closed form `boost_eng` must reproduce); `TestBoostParameters`;
`TestBoostDeltaFunction` (40,000 random draws at both live product masses
bit-for-bit against the Cython, the window edges located **to the last
bit** by bisection over 400 more draws, unit normalisation, and seven
unphysical or out-of-window argument sets);
`TestBoostIntegrateLinearInterp` (one test per branch — above-table
clamp, below-table tail, straddling the floor, straddling the ceiling,
both partial cells, the interior sum — each pinned by a closed form or by
a sensitivity check on a table entry only that branch reads, and each
bit-equal to the Cython; plus all seven live tables over six boost
regimes × 400 energies); `TestFusedArithmetic` (an independent unfused
reference, used as a discriminator); `TestDroppedInteriorCell`;
`TestTrapezoidSummation` (five table sizes spanning NumPy's pairwise
blocking, plus a check that a sequential sum really would differ);
`TestErrors`; `TestDispatch`.

**Numerical impact:** no public value changes — see the section below.

**Test validity (mutation campaign).** 21 distinct mutations, run
strictly sequentially behind a lock file, with a green baseline asserted
before the campaign and again after it (Task 3.3's
`[poisoned-baseline]` lesson). Round 1: 20 mutations, **17 caught**.
The three survivors were real gaps, and each named the test that was
missing:

| Survivor | Why it survived | Test added |
| --- | --- | --- |
| `interp`: drop the exact-node short circuit | At an ordinary node the interpolation returns `fp[j]` anyway, so the guard is unobservable except on an infinitely wide cell | `TestQuirks::test_an_infinite_node_returns_its_own_value` |
| `boost`: unfuse the delta window bounds | `eminus`/`eplus` never reach the return value — they move the support edge by one double, and no grid sample lands there | `TestBoostDeltaFunction::test_the_window_edges_sit_on_the_same_double_as_the_cython` (bisection on the bit pattern) |
| `boost`: unfuse `k = sqrt(e² − m²)` | Same, and with `m = 0` the two forms are *identical*, so only massive-product draws can see it — three fixed parameter sets saw nothing | the same sweep, widened to 400 random draws over both product masses. An offline search over 4,000 draws found 708 edges that move when `k` is unfused, which is why 400 is enough |

Round 2 re-ran the three survivors plus a new mutation on the line
height (`k0`): 3 of 4 caught, `k` still surviving because the first
version of the edge test used three fixed parameter sets. Round 3, after
widening that test to the random sweep: **1 mutation, 0 survived.**
Final state: all 21 caught, baseline restored green
(`RESTORED BASELINE: GREEN — 102 passed`).

**Deferred:** nothing. Two things are deliberately *not* done rather than
deferred — the boost integral's window-coverage defect is preserved under
rule 1 and filed as a follow-up, and `boost_jac` / `boost_eng` are not
ported because nothing calls them (`rg` at execution time) and Phase 06
Task 6.4 deletes the extension.

## Open Questions

- **Does the fused-arithmetic decision need to be revisited for the
  mediator spectrum tables?** Phase 06's `.pyx` call `np.interp` on their
  own tables and do their own arithmetic around it; whether *those*
  expressions contract is a separate measurement, and this task did not
  make it.

## Plan Impact

**Impact Level:** Phase file patched + reference patched.

No ADR. Nothing revises ADR-0001 or ADR-0002 — the provenance here is
original work plus NumPy's documented behavior (BSD-3-Clause, which
ADR-0002 permits; its rule is that nothing GSL-derived enters the tree),
and no interface, ordering or acceptance criterion outside Task 3.4
moves. The phase file's Task 3.4 block gained five "criteria added during
execution" bullets, because the criterion it shipped with named an oracle
that does not exist and set a bar (per-branch tests) that turned out to be
achievable at bit-equality only by reproducing the compiler's contraction.
`../../references/numerics-replacements.md` gained the measured block for
the same reason Tasks 3.2 and 3.3 patched it: its own prose is what would
lead the next reader to the wrong port.

## Numerical impact

**No public value changes** (verified: `git diff origin/master -- hazma`
is one file, `hazma/_core.pyi`, and every line of the hunk is comment
text — no executable line under `hazma/` is reachable from this diff, on
a tree rebuilt before anything was run). The rest is two PyO3-free Rust
modules that no Python imports and no Rust kernel yet calls, their
registration-only probes, two new test modules, the parity corpus's
served-kernel exemption, and project bookkeeping.

Measured rather than only argued: the bare suite ran the parity corpus in
**bit-equality mode** — `rtol = 0` across all 41 consumed entry points,
179,695 pinned values — and passed, at `1314 passed, 13 skipped`
(`provenance` → `exact=True`).

What the task *did* produce numerically is a foundation that reproduces
the Cython **bit-for-bit** where the Cython is what the corpus records:
zero mismatches on all seven live tables across six boost regimes × 400
energies, zero across 40,000 delta-function draws, and zero on 20,204
`np.interp` abscissae per table. **The Phase 04 drift line for the
kaon/eta/omega/phi family will be measured against these**, so a wrong
choice here would surface as a kernel bug rather than a foundation bug.

**One consequence to declare when Phase 04 lands, not now.** The Rust is
bit-equal to the *contracted* (macOS/arm64) Cython on every platform,
because `f64::mul_add` is fused everywhere. On a target whose C compiler
does not contract — baseline x86-64 is the case that matters, since that
is what the Linux wheels are built for — today's Cython returns the
unfused values, which differ from these by up to **3.6e-12** relative on
the corpus grids. The Phase 04 swap therefore moves those users' numbers
by that much, one-time, and past rule 3's 1e-12 declaration threshold.
Nothing moves in *this* task, because nothing calls the new code.

## Stale-state sweep

Each command run against this branch on 2026-08-10.

**Full change inventory** — `git status --short` plus
`git diff origin/master --stat` after `git add -N .`, re-derived after
the review round: **19 files, +3,539 / −49**. Four new `rust/src/*.rs`,
`rust/src/lib.rs`, two new `test/test_core_*.py`, three `test/parity/*`
sites, `hazma/_core.pyi`, two canonical patches, one follow-up + its
index row, `docs/agents/lessons.md`, and three bookkeeping files. No
untracked file is unaccounted for. (Round 1 recorded `18 files,
+3,265 / −49`, which was measured before the last documentation edits of
that round — the number is now taken after every edit, not during.)

**Numerical-impact statement** — `git diff origin/master -- hazma` is one
file, `hazma/_core.pyi`, comment-only. Corpus in bit-equality mode:

```text
$ python -c "...tolerances.provenance(manifest)..."
Provenance(exact=True, detail='')
$ python -c "...cases.rust_core_kernels()..."
[]
```

**The port is not yet served, and the exemption is honest:**

```text
registered: ['boost', 'interp', 'neutrino', 'photon', 'positron',
             'quad', 'scalar_mediator', 'special', 'vector_mediator']
exempted  : ['boost', 'interp', 'quad', 'special']
$ rg -n 'hazma\._core\.(interp|boost|special|quad)' hazma/
none
```

**The tree under test is this worktree** —
`python -c "import hazma._core; print(hazma._core.__file__)"` →
`…/.claude/worktrees/cython-to-rust-96ef8d/hazma/_core.abi3.so`.

**Stale-sibling sweep** —
`rg -n 'hazma._core.special|_CORE_TEST_ONLY_MODULES'` over the tree finds
four live prose sites naming the exempted set (`test/parity/cases.py`,
`test/parity/README.md`, `test/parity/test_parity.py`,
`hazma/_core.pyi`); **all four now say four submodules**. `rg -n 'Two
further submodules'` → no occurrences. Every other hit is a dated
per-task record (the Task 3.2 and 3.3 notes), which is history rather
than a live claim.

**Citations** —
`scripts/agents/check_doc_citations.py <8 changed .md>` →
`docs scanned: 8; in-repo citations checked: 20; out-of-range or
ambiguous: NONE`.

**Introduced-token scan** — `grep -rnE 'TODO|FIXME|breakpoint\(\)|pdb|^\s*print\('`
over the six new source files → none. Preflight's own
`forbidden tokens` gate agrees.

**Bookkeeping consistency** — this note's `**Status:** Complete`, the
phase README's Tasks-table cell for 3.4, its `**Status:**` header line,
and the project README's Phases row all agree that 3.4 is complete and
3.5 is the only task left in Phase 03. The phase file's frontmatter
stays `status: Not started` because that field is two-state in this
project (every completed phase file flips straight to `Complete`), and
flipping it is Task 3.5's closure step, not this task's.

**Gates** — `scripts/agents/preflight.sh --paths "…" --md "…"` →
`RESULT: PASS`: ten rows PASS (including `markdownlint` over all eight
changed `.md`) and one expected SKIP (`version bump`, not a closing PR).

## Handoff to Next Task

**For the next agent in Phase 03:** only Task 3.5 (dispatch and error
layer) remains. Read `../../PLAN.md`, `../README.md`, `README.md`, then
the phase file — whose Tasks 3.2, 3.3 and 3.4 blocks all now carry
"criteria added during execution".

**Currently safe to assume:**

- **`hazma_core::interp` and `hazma_core::boost` exist and are
  bit-equal to what they replace**, on the corpus's capturing platform.
  `interp::interp(x, xp, fp)` is `np.interp` with NumPy's full contract;
  `boost::{boost_beta, boost_gamma, boost_delta_function,
  boost_integrate_linear_interp}` are the four live routines of
  `hazma/_utils/boost.pyx`. All PyO3-free; the two `hazma._core`
  submodules are test surfaces and must stay importer-free or the parity
  corpus leaves bit-equality mode.
- Task 3.5's dispatch work does not touch either — both already route
  through `dispatch::map_unary` in their probes, so whatever 3.5 decides
  about the four live-behavior questions applies to them for free.

**Currently risky / unknown, for Phases 04–06:**

- **Do not "clean up" a `mul_add` into `a * b + c`, and do not add one
  where this module does not have one.** Both directions move published
  numbers: unfusing costs up to 3.6e-12 against the corpus (past the
  1e-12 `TABULATED` budget), and fusing `boost_beta` — which the Cython
  does *not* contract at any of its ten call sites — moves every boosted
  spectrum. Every Phase 04 kernel needs its own disassembly or bisection;
  there is no house style to apply.
- **`boost_integrate_linear_interp` is wrong near threshold and is
  supposed to stay wrong for now.** All seven tabulated photon spectra
  diverge instead of converging to their rest-frame values as the parent
  slows (6,500×–33,000× one part in 1e12 above rest). The corpus pins
  those values, so a Phase 04 swap that repairs the coverage **fails the
  gate**. The repair is
  [`../../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md),
  blocked until after Phase 06 Task 6.4.
- **The live tables are strided views**, not contiguous buffers
  (`np.loadtxt(...).T` rows), so `PyReadonlyArray1::as_slice` refuses
  them. A Phase 04 kernel should own its table in Rust; anything that
  takes one from Python must copy.
- **A `cdef` with a `.pxd` declaration is callable from Python** through
  `__pyx_capi__` + `ctypes` (`PYFUNCTYPE`, never `CFUNCTYPE`). Reach for
  that rather than adding a temporary shim to a `.pyx` when a
  C-level-only routine needs an oracle. `test/test_core_boost.py`'s
  `cython_boost` is the working example, and it dies with the `.pyx` in
  Phase 06 Task 6.4 — the classes that outlive it are named in that
  module's docstring.
- **An edge that only decides a branch needs a bisection test, not a
  grid.** Three of this task's mutations survived a 40,000-draw sweep
  because they moved a support boundary by one double. Any Phase 04–06
  kernel with a window, a threshold, or a piecewise switch inherits that:
  sample the boundary by bisecting on the bit pattern, and make sure the
  sweep's *parameter* space reaches the branch (with `m = 0` the fused
  and unfused momenta are identical, so only massive draws could see it).
