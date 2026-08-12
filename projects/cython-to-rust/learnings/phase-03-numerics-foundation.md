# Phase 03 Learnings: Numerics foundation

Synthesized at phase close (2026-08-11) from the five task notes
([3.1](../task-notes/phase-03/task-3.1-constants.md),
[3.2](../task-notes/phase-03/task-3.2-specfun.md),
[3.3](../task-notes/phase-03/task-3.3-quadpack.md),
[3.4](../task-notes/phase-03/task-3.4-interp-boost.md),
[3.5](../task-notes/phase-03/task-3.5-dispatch.md)) and
[`../task-notes/phase-03/README.md`](../task-notes/phase-03/README.md).
Read this instead of the notes; the notes are history.

## 1. Implementation Reality Check

The phase delivered the substrate it promised — `constants`, `special`,
`quad`, `interp`, `boost` and `dispatch` in `hazma_core`, every one
unit-tested in Rust and swept from Python against its oracle, with no
kernel swapped and no public value moved. Five of five tasks landed;
no ADR was needed, because ADR-0002 had already fixed the provenance
question that was the only anticipated blocker.

What the plan did not anticipate is that **four of the five tasks had to
add exit criteria during execution**, and all four additions are the same
shape: *the plan's model of an external artifact was a hypothesis, and
only the sweep could refute it.* Phase 02's lesson was "it exists ≠ it is
load-bearing"; this phase's is one level out.

- **Task 3.2**: the plan said "scipy is cephes, so this is
  algorithm-for-algorithm parity". True for `spence` and `k1`, false for
  `kn` — scipy routes integer orders to `kv`, so the *faithful* cephes
  `kn` misses scipy by 5.1e-9, squared into `thermal_cross_section`'s
  prefactor and inside the corpus's own 1e-8 budget for it.
- **Task 3.3**: the break-point contract belongs to `scipy.integrate.quad`
  (Python-level `np.unique` + strictly-interior filtering), not to
  QUADPACK. A documentation-driven port would have made five live call
  sites *error*.
- **Task 3.4**: the compiled Cython's own arithmetic is fused. Clang
  contracts `a*b + c` by default; written unfused the port missed the
  corpus by 3.6e-12 — past the 1e-12 `TABULATED` budget — on the corpus's
  own grids. And the converse holds at `boost_beta`, which is *not*
  contracted at any of its ten call sites.
- **Task 3.5**: the reference's "every public function follows one shape"
  was false. There are four dispatch shapes across the 43 surviving
  top-level `def`s, and two of them disagree with each other about a 0-d
  array.

The one task that did *not* need a criteria patch, 3.1, is the one whose
criterion already said "extract from source and assert bit-equality — no
hand-transcription trust". That is the generalization worth carrying into
Phases 04–07: **write the criterion so it names the oracle and forbids
transcription, and it will survive contact with the artifact.**

## 2. Critical Context for Future Work

- **`hazma_core::{constants, special, quad, interp, boost}` are the
  foundation Phases 04–06 call directly in Rust.** Their Python-visible
  twins (`hazma._core.special`, `.quad`, `.interp`, `.boost`,
  `.dispatch`) are *test surfaces*, exempted wholesale in
  `test/parity/cases.py`'s `_CORE_TEST_ONLY_MODULES` and held honest by
  `test_test_only_core_submodules_have_no_importer`. **A wrapper that
  imports one of them makes it a served kernel and takes the corpus out
  of bit-equality mode.** Do not widen the exemption to quiet a red mode
  check.
- **Name the constants table the `.pyx` you are porting `include`s.**
  `constants::pdg` ← `_utils/constants.pxd` (everything under
  `hazma/spectra/**`); `constants::legacy` ← `_utils/legacy_parameters.pxd`
  (the four mediator spectrum extensions); `constants::derived::<pyx>` ←
  the module-local `DEF`s. The two tables share 19 names and disagree on
  12, asserted as a literal partition — a per-file bit-equality check
  cannot catch a consolidation, because each side would still match
  *some* source.
- **`derived::photon_pion` deliberately mixes both tables.** Its `MPI` /
  `ME` / `MMU` are PDG, its five kinematic literals reproduce bit-exactly
  from the *legacy* masses and from no other table. Recomputing them from
  the header the file includes moves `ENG_MU_PIRF` by 4.7e-5 MeV and every
  charged-pion photon spectrum with it. Phase 04 must not consolidate it.
- **Do not "simplify" `special::bessel_kn` to `spec_math`'s.** The
  upward recurrence on cephes `k0`/`k1` seeds tracks scipy to ≤ 3.4e-15;
  cephes `kn` misses by 5.1e-9, which the corpus's `thermal_cross_section`
  budget would absorb.
- **`quad::quad` is the entry point, not `qagse`/`qagpe`.** It is what
  reproduces scipy's limit ordering and break-point filtering, so twelve
  call sites do not each re-derive them, and `points is None` — not "no
  break point survived" — is what selects `qagse`.
- **Do not add or remove a `mul_add` in `boost.rs` or `interp.rs`.**
  Contraction is a per-*expression* fact, established twice for each site
  (disassembly plus a 16-combination bisection). There is no house style
  to apply; every Phase 04 kernel needs its own measurement.
- **`boost_integrate_linear_interp` is wrong near threshold and stays
  wrong.** It drops one interior cell and double-counts when both bounds
  sit in one cell, so all seven tabulated photon spectra diverge instead
  of converging to their rest-frame values as the parent slows (6,500× to
  33,000× one part in 1e12 above rest). The corpus pins those values, so a
  Phase 04 swap that *repairs* it fails the gate. Blocked until after
  Phase 06 Task 6.4 —
  [`../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md).
- **The dispatch contract is settled and lives in three helpers.**
  `dispatch::map_unary` (33 of the 35 dispatching entry points: 15 with a
  scalar-or-1D energy argument plus the 18 cross sections),
  `dispatch::map_flavors` (the 2 neutrino ones, 3-tuple / `(3, N)`), and
  `dispatch::require_vector` (`partial_widths`). The rule that decided
  every divergence: **each exception the Cython raises explicitly keeps
  its type; only its `assert`s change type** (rules.md rule 9). Kernels
  stay PyO3-free and pass their quantity wording in.
- **`require_vector` checks rank and dtype, never length.** The Cython's
  `pws` handling indexes seven entries and raises `IndexError` from the
  kernel; Phase 06 owns that check.

## 3. Quirk Log & Edge Cases

- **`scipy.special.kn` is not cephes `kn`** — integer orders go to `kv`.
  Only `k0`/`k1` are still cephes in scipy.
- **`spec_math::Polylog::li2` *is* `scipy.special.spence`** (its body is
  `cephes64::spence`, so the convention is Li₂(1−z)). The name is the
  trap, not the function.
- **Negative zero routes to a routine's *zero* branch**, so cephes returns
  `+∞` for `kn(n, -0.0)` while a recurrence seeded on `k0`/`k1` produces
  `∞ + -∞ = NaN`. Any kernel that divides by its argument inherits a
  signed-zero case the underlying cephes routine does not have, and `+0.0`
  passing says nothing about `-0.0`.
- **`np.trapezoid` reduces pairwise**, eight accumulators over 128-element
  blocks, recursing. A sequential sum is a different number (1.8e-15
  relative on the 500-row tables).
- **`np.interp` has two quirks a spec-driven port misses**: a one-point
  grid answers *everything* with `fp[0]`, NaN included (the NaN check
  lives on the multi-point path only), and duplicate abscissae resolve to
  the **last** matching node.
- **The live boost tables are strided views** (rows of a transposed
  `np.loadtxt`), so `PyReadonlyArray1::as_slice` refuses them — use
  `to_vec`.
- **A `cdef` declared in a `.pxd` is callable from Python** through the
  module's `__pyx_capi__` capsules. Use `ctypes.PYFUNCTYPE`, never
  `CFUNCTYPE` — the latter releases the GIL and anything calling back into
  NumPy segfaults with no Python-level error — and assert on the capsule's
  *name*, which is its C signature string.
- **A 0-d array's `__float__` forwards to its element**, and `np.str_`
  subclasses `str`, so `float(np.array("15.0"))` is `15.0`. A dispatch
  layer that accepts a 0-d array by trying the conversion will accept a
  string as a number; ask the dtype's `kind` instead.
- **PyO3's `text_signature` is a claim, not a constraint.** `roundtrip`
  advertised `(x, /)` while `roundtrip(x=1.5)` worked. The Cython entry
  points accept keywords, so a `/` copied into a Phase 04 wrapper is a
  latent public-API narrowing.
- **`clippy::excessive_precision` is on by default** and fires on any
  literal transcribed verbatim with trailing significant zeros.

## 4. Test Infrastructure State

- **Five Python test modules, all against a live oracle**, none of them
  duplicating the parity corpus: `test/test_core_constants.py` (25 tests,
  source-to-source text comparison, no build needed, 0.03s),
  `test_core_special.py` (65, scipy), `test_core_quad.py` (58,
  `scipy.integrate.quad` through a Python-callable probe),
  `test_core_interp.py` (33, `np.interp`), `test_core_boost.py` (69, the
  Cython twin via `__pyx_capi__`), `test_core_dispatch.py` (118, the
  `.pyx` sources' own error strings). `cargo test` carries 69 units.
- **`test_core_dispatch.py` is the template Phases 04–06 copy** for a
  kernel swap: keep every test, swap the probe and the quantity wording,
  add the kernel's numerical tests *beside* rather than merged in.
- **A test whose oracle is something you compiled is scoped to that
  build.** Nineteen bit-equality assertions against the Cython and NumPy
  passed on macOS/arm64 and failed on Linux/x86-64, because
  `-ffp-contract=on` only bites on a target with hardware FMA. Measure the
  property at import (`CYTHON_CONTRACTS`, `NUMPY_CONTRACTS`) and skip
  where it does not hold; loosening to a tolerance is the wrong fix, since
  the worst relative gap sits at a cancellation point where any admitting
  tolerance would hide real defects
  (`docs/agents/lessons.md` `[platform-scoped-oracle-asserted-globally]`).
- **Mutation campaigns are this phase's standard gate** — 13 mutations in
  3.1, 11 in 3.2, 17 in 3.3, 21 in 3.4, 14 in 3.5, each run sequentially
  from a green baseline and re-checked after. Two hard-won rules: **assert
  the baseline before and after and hold a lock** (Task 3.3 poisoned its
  own baseline by running two campaigns concurrently, and rationalised the
  tell for a while), and **`cargo test -- --test-threads=1`**, because the
  default parallelism interleaves `test NAME ... FAILED` lines and a
  scraped failure list then names the wrong tests.
- **A campaign's survivors are results, not failures.** Task 3.4's three
  survivors each moved a *branch boundary* by one double without touching
  a returned value, which no grid sample can see — they are what the
  bisection tests were written for. Task 3.5's single survivor refuted a
  claim in the implementation's own comment (an argument-ordering swap it
  described as load-bearing turned out to be unobservable), and the
  comment was corrected rather than the mutation dropped.
- **A sweep's parameter space is part of its result.** Task 3.3's earlier
  6,000-combination design, capped at two break points, reported *zero*
  subdivision mismatches; they appeared only once 9- and 39-point grids
  entered the draw.
- **Pin `numpy==2.5.1` when building an environment for this project.** A
  fresh `uv pip install -e .` resolves 2.5.2, which puts the parity corpus
  into budget mode (`exact: False`) and turns the bare suite's skip count
  from 13 into 14 — the corpus's own signal working, but it costs a re-run
  if noticed only at the end. The one-second check is in
  [`../task-notes/README.md`](../task-notes/README.md) under Findings.
- **Bare `pytest` at phase close: 1378 passed, 13 skipped** on the
  capturing environment (CPython 3.12.12, macOS/arm64, NumPy 2.5.1, scipy
  1.18.0), parity corpus included and in bit-equality mode. Phase 02
  closed at 1063; this phase added 315 tests and moved no number.

## 5. Follow-on seeds

- **The boost integral's window coverage** —
  [`../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md),
  filed in Task 3.4. A live defect in hazma 2.1.0, reproduced per rule 1
  and pinned in both languages. Blocked until after Phase 06 Task 6.4,
  because the repair needs a declared corpus regeneration.
- **The model-level scalar-energy contract** —
  [`../../../docs/followups/todo/model-spectra-reject-scalar-energies.md`](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md).
  Task 3.5 settled the compiled half (a scalar, a 0-d array and a
  sequence are all accepted below `Theory`); the pure-Python half —
  `Theory.spectra`'s `type(e_gams) == float` branch and the channel
  wrappers' `len(e_gams)` — is still open and ripens with Phases 04–06.
- **The three fine-structure constants and the two mass tables.** Hazma
  holds `1/137.035999084` (`constants.pxd`, pre-CODATA-2022), `1/137`
  (`legacy_parameters.pxd`) and `1/137.04` (`hazma/parameters.py:205`),
  and the two `.pxd` disagree on 12 names. Rule 4 forbids consolidating
  any of it inside this project; a post-port consolidation is a separate
  declared numerical change, and it has to account for the pure-Python
  third α as well.
- **Which PDG edition each `constants.pxd` value came from is recorded
  nowhere** (Task 3.1). The `± uncertainty` annotations are the only
  provenance. Not blocking, and explicitly *not* to be resolved by
  re-sourcing values.
