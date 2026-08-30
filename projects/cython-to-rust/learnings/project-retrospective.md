# Project Retrospective: cython-to-rust

**Project:** cython-to-rust
**Ran:** 2026-08-03 → 2026-08-29 (26 calendar days, 8 phases, 33 tasks)
**Shipped as:** hazma 3.0.0
**Deliverable:** Hazma's compiled layer rebuilt in Rust (PyO3, one abi3
`hazma._core`, maturin-built), zero Cython, and a permanent parity-test
corpus.

This file is the project's durable memory. The seven phase learnings
files beside it hold the per-phase detail and are still the right thing
to read before touching an area they cover; this one holds what only
becomes visible from the whole arc
([ADR-0002](../../../docs/adrs/ADR-0002-read-phase-learnings-not-closed-task-notes.md)).

## 1. Implementation Reality Check

The project delivered its deliverable exactly, and the numbers came out
better than the plan allowed for.

- **All 41 consumed compiled entry points are on `hazma._core`.** The two
  unimported `sigma_xx_to_all` exports were dropped rather than ported,
  as `PLAN.md` §Scope said they would be.
  `find hazma -name "*.pyx" -o -name "*.pxd"` returns nothing; the tree
  went from 32 extension modules to 20 (Phase 00) to one.
- **27 of the 41 entry points reproduce hazma 2.1.0 bit-for-bit**, and
  the other 14 move by at most 5.4e-12 relative. `PLAN.md` opened the
  quad-backed budget at 1e-8 relative and expected to tighten it after
  measurement. It tightened by four orders of magnitude: **fourteen
  budgets tightened across the project and none widened**, and neither
  opening budget (`QUAD_RTOL` 1e-8, `NESTED_RTOL` 1e-6) is claimed by any
  corpus case any more — `QUAD_RTOL` emptied at Task 5.2, `NESTED_RTOL`
  at Task 6.3.
- **The estimate held.** 21–32 focused dev-days across ~33 tasks was the
  plan; 33 tasks is what ran, and no phase was split or merged. The
  ordering constraint `00 → 01 → 02 → 03 → {04, 05} → 06 → 07` was
  respected, though 04 and 05 ran sequentially rather than in the
  parallel the plan permitted.
- **Performance was measured, not chased** (`rules.md` rule 12), and
  arrived anyway: 1.46×–1.93× on a full relic-density calculation,
  1.8×–3.2× on the ⟨σv⟩ kernel, 32×–43× on the mediator positron
  spectra, and 4,180× on a fixed-mass parameter sweep of the mediator
  photon spectra. The large numbers are not Rust-versus-Cython; they are
  a dead memo cache the Cython never populated, which the redesign
  repaired incidentally.

Two things diverged from the plan, both worth recording.

**Phase 00 was worth doing on its own, and the plan said so in advance.**
It deleted roughly 6,500 lines of dead Cython and about 33,000 lines
overall, took the extension count from 32 to 20, and removed the only C++
in the tree — before a single line of Rust existed. `PLAN.md` §Scope
called it "worth doing even if the port stopped there", and that
turned out to be the single best structural decision in the project: it
meant the parity corpus in Phase 01 had to cover 41 entry points instead
of a number nobody had counted, and every later phase worked against a
surface that was known rather than assumed.

**The port surfaced twelve live numerical defects in hazma 2.1.0, and
that was not a goal.** None was found by porting. Every one was found by
writing a statement the original never made — an analytic normalization
check, a sibling-to-sibling diff, a rest-frame limit, a forward-cone
argument, a continuum subtraction — in order to have something to hold
the port to. Section 3 draws the general lesson; the roster is in the
3.0.0 CHANGELOG's `Known issues` and in `docs/followups/todo/`.

**ADRs:** three, all Accepted, none superseded or amended after
acceptance. ADR-0001 (PyO3 + maturin over pybind11) was decided before
the project started and never revisited. ADR-0002 (license-clean
numerics: cephes and netlib-QUADPACK in, the GPL rust-cyphus crates out)
was accepted 2026-08-04 and unblocked Phase 03. ADR-0003 (remove the
broken-on-import `hazma.gamma_ray`) was accepted the same day and fully
discharged across Tasks 0.5 and 0.2. **No phase after 03 needed an ADR at
all**, which is the tell that the architectural questions were genuinely
settled up front.

## 2. Critical Context for Future Work

Contracts this project established that later work must respect.

- **`hazma._core` is the compiled layer, and it has one shape.**
  `rust/src/kernels/` is PyO3-free and `pub`; `rust/src/dispatch.rs` is
  the only PyO3 boundary. A wrapper calls `dispatch::map_unary`,
  `map_flavors` for a `(3, N)` return, `map_unary_try` for a kernel that
  raises, or `require_vector` for an argument that must be a 1-D array.
  Kernels call each other natively, the way the `.pyx` cimported.
- **A `.rs` edit needs `pip install -e .`, not `cargo build`.**
  `cargo build` refreshes `rust/target/`, which nothing Python imports.
  Iterate with `cargo test --no-default-features` — the
  `extension-module` feature must be off or the test harness will not
  link — then reinstall before believing any Python-side result.
- **`crate::quad::quad` is the integrator entry point**, not `qagse` or
  `qagpe` directly — it is what reproduces scipy's limit ordering and
  break-point filtering, and `points is None` (rather than "no break
  point survived") is what selects `qagse`. Both drivers are proven
  against live call sites. Copy a call site's `epsabs` / `epsrel` /
  `points` verbatim into a `const QuadOpts`; because `quad`'s `Err` arm
  depends only on the options and never on the integrand, it is
  unreachable for a `const` value — return `NaN` there rather than
  panicking, and assert the unreachability with a cargo test. `quad`
  short-circuits an empty interval the way `scipy.integrate.quad` does,
  so any kernel whose limits can coincide inherits that.
- **The parity corpus is the numerical gate, and it is scoped to its
  capturing platform.** `test/parity/cases.py` is the single source of
  every entry point's call convention; `python test/parity/generate.py
  --check` verifies the stored arrays. `test/parity/stability.py` declines
  to compare the 494 positions (0.27%) whose values are cancellation
  residue rather than numbers any implementation reproduces.
- **`test/test_theory_aggregation.py` is the model-layer gate the corpus
  cannot be** — identities over `hazma/theory/`'s pure-Python
  aggregation, no golden data, and the only numerical gate in the repo
  that is *not* platform-scoped. Run it either side of any change that
  can reach a spectrum.
- **The two constants tables are still deliberately divergent**, twelve
  names plus `ALPHA_EM`, preserved bit-for-bit under `rules.md` rule 4.
  `rust/src/constants.rs` reproduces the split (`pdg` for
  `hazma/spectra/**`, `legacy` for the mediator kernels,
  `derived::photon_pion` reading both) and a cargo test asserts it stays
  divergent. Consolidating it is §5's first seed and a declared numerical
  change.
- **`pyproject.toml` is the only build entry point**, `[project] version`
  is the version's source of truth, and `hazma.VERSION` reads it back out
  of `importlib.metadata`. See the Phase 07 learnings for the rest of the
  packaging contract.
- **Twelve reproduced defects are load-bearing, not oversights.**
  `rules.md` rule 1 required the port to reproduce them and rule 2 forbade
  regenerating the corpus a repaired kernel would have to pass. Do not
  "fix" one in passing: six are sequenced in
  [`projects/parity-pinned-defect-repair/`](../../parity-pinned-defect-repair/PLAN.md),
  which lands each as a *declared delta* against the committed corpus
  arrays rather than as a regeneration, and that mechanism is the one to
  reuse.

## 3. Quirk Log & Edge Cases

The transferable ones. Per-phase detail is in the seven phase files.

- **Cython's compiled arithmetic is not the arithmetic its source
  reads.** Three separate discoveries, each of which would have failed
  the corpus if missed. (1) Clang contracts `a*b + c` into an FMA by
  default, so a port written unfused missed by 3.6e-12 — past the
  `TABULATED` budget — while `boost_beta` is *not* contracted at any of
  its ten call sites. (2) Cython 3's default `cpow` semantics compile
  `double ** double`, and everything around it in the same expression, in
  `double _Complex`, reaching `cpow` and compiler-rt's `__divdc3`; a
  real-arithmetic transliteration of `sigma_xx_to_s_to_ff` misses
  bit-equality at 355 of 935 points on its electron block alone. (3)
  Reproducing two `cdef float` truncations is what makes
  `dnde_photon_neutral_pion` bit-equal; an all-`f64` spelling lands
  8.5e-9 away. **Read the disassembly; do not infer it from the source.**
  Where clang fuses is one syntactic rule — `EmitFMulAdd` contracts
  `A ± B` when `A` is a syntactic multiply, else when `B` is, decided on
  the C tree Cython emits, where `x ** n` is a `pow` *call* and never a
  multiply — and Phase 06 reproduced 37 sites in two kernels from that
  rule alone.
- **"scipy is cephes" is true per-function, not per-library.** It holds
  for `spence` and `k1` and fails for `kn`, which scipy routes to `kv`:
  the *faithful* cephes `kn` misses scipy by 5.1e-9, squared into
  `thermal_cross_section`'s prefactor. Separately, `spec_math`'s `li2`
  disagrees with `scipy.special.spence` by ≤2e-15, which the muon photon
  kernel's `5/β ≈ 3.5e6` amplifies to 3.15e-11. **The fix was at the
  source, not at the budget:** transcribing cephes `spence` in-tree with
  scipy's FP contraction gives 0 mismatches at 13,000 points.
- **A break-point contract can belong to the Python wrapper rather than
  to the library.** `scipy.integrate.quad`'s `points` handling —
  `np.unique` plus strictly-interior filtering — is Python-level, not
  QUADPACK's. A documentation-driven port would have made five live call
  sites error.
- **Nested quadrature damps; it does not amplify.** Phase 04 billed the ρ
  spectra as the project's numerical stress test and got 1.5e-13, five
  decades inside its class, because the outer integral *averages* the
  inner one over a window. Every task's numerical prediction in that
  phase was wrong, in a different direction each time. **Re-derive; do
  not inherit.**
- **A gate that exists is not a gate that can fail.** The crate's mere
  existence flipped the parity suite out of bit-equality mode, because
  `tolerances.provenance` keyed on "`hazma._core` is importable" rather
  than on whether a kernel was *served*. Two more instances landed in the
  same phase. The question to ask of every criterion is "what turns red
  if this is wrong?"
- **Catastrophic cancellation makes a pinned value a property of the
  platform.** Four scalar elastic cross sections return one libm's
  rounding residue near `e_cm = 2 m_x`; the same Cython gives
  `-1.504e-02` on macOS and `+5.624e-07` on Linux, and no tolerance
  absorbs a sign flip. The fix was to establish, against a 60-digit
  evaluation of the same closed forms, *which positions are residue* and
  decline to pin those — not to guess from which platforms disagree.
- **A `.pyx` whose locals are untyped contracts nothing**, and so does
  one whose locals are all typed but which contains no multiply-add at
  all. Neither shape tells you what the compiler did.

## 4. Test Infrastructure State

- **The parity corpus (`test/parity/`)** is the project's main artifact
  after `hazma._core` itself: 41 entry points, 179,695 pinned values,
  per-case tolerance budgets with a written rationale each, a
  `--check` regenerator, and `assert_no_rust_core` to stop anyone
  regenerating it from a tree where a kernel already runs on Rust. It
  outlives the project and is the gate for
  `parity-pinned-defect-repair`.
- **`test/parity/oracles/`** holds committed corrected-value arrays for
  the defects the corpus pins wrong, and `defects.RESTORED_SOURCES` names
  a git revision per deleted source so a re-capture stays possible: 29
  entries, every case covered. The trick that made it possible is that
  `git show <rev>:<path>` takes a plain SHA as readily as a `^`
  expression, so a task that cannot cite its own commit can cite one that
  already exists.
- **Mutation campaigns on every kernel, and interrogate the survivors.**
  Task 4.4's eleven mutants had two survivors and concluded they were
  unobservable; 4.5's six had one and *fixed* it; 4.6's eleven had two
  and resolved both; 6.2's thirty-seven had sixteen, fourteen of them
  provably identity-equivalent. Ask "can this be lifted out?" and "is
  this the coefficient or the grid?" before writing a limitation into
  the source.
- **One test module per clone-pair, not per entry point**, when the
  independent reference is one function parameterised by pair. And the
  surviving module shape is: an independent Python reference plus the
  against-the-Cython numbers measured *before* deletion, recorded in
  prose where they can no longer be re-run.
- **The gate at close, on the capturing platform:** bare `pytest -q` is
  2231 passed / 15 skipped / 12 subtests (from 1006/13 at the Phase 01
  close), and `cargo test --no-default-features` is 258 passed (from 222
  at the Phase 02 close). Re-derive rather than quoting — these moved at
  nearly every task, and the cargo figure carried in the project's own
  working memory was two tasks stale at close.

## 5. Follow-on seeds

Four filed at close, plus the standing backlog the project sourced.

- **[Consolidate the divergent constants tables](../../../docs/followups/todo/consolidate-the-two-constants-tables.md)**
  — the project's most obvious deferred cleanup, named as out of scope in
  `PLAN.md` from the first day. Three fine-structure constants
  (`1/137.035999084`, `1/137`, `1/137.04`) and two mass tables that
  disagree on twelve names, all preserved bit-for-bit because
  `rules.md` rule 4 required a ported kernel to read the exact constant
  its Cython source read. That constraint lifts at close. It is a
  declared numerical change and should reuse
  `parity-pinned-defect-repair`'s declared-delta mechanism rather than
  regenerating the corpus.
- **[Free-threaded `abi3t` wheels](../../../docs/followups/todo/free-threaded-abi3t-wheels.md)**
  — waiting on PyO3 and on NumPy/SciPy, not on hazma. The crate's only
  mutable module state, the mediator table cache, is already
  `Mutex`-guarded, so the blocker is packaging rather than concurrency.
  Adding per-version free-threaded builds today would reintroduce exactly
  the interpreter matrix the abi3 cutover removed.
- **[The relic-density Boltzmann solve in Rust](../../../docs/followups/todo/relic-density-odes-in-rust.md)**
  — deliberately excluded, and it stays excluded until a profile says
  otherwise. Phase 05 already moved the expensive part (the ⟨σv⟩
  integrand, which was re-entering Python per quadrature node); what
  remains is a compiled SciPy stepper. Note the trap recorded there: at
  `relic_density`'s default `rtol=1e-5` a last-bit input change moves the
  answer by 3.8e-5, so any reimplementation moves the default-tolerance
  result without moving the physics.
- **[Wheels for linux-aarch64 and Windows](../../../docs/followups/todo/wheels-for-aarch64-and-windows.md)**
  — Task 7.2 decided against them deliberately and declined to file, on
  the grounds that `PLAN.md` §Scope recorded the option. Closing the
  project made that record archival, so it is filed now with the decision
  unchanged and the reasons written down.

**The twelve reproduced 2.1.0 defects are the project's most valuable
by-product, and they deserve a conversation the schedule of this project
never gave them.** Six are sequenced in
[`projects/parity-pinned-defect-repair/`](../../parity-pinned-defect-repair/PLAN.md);
the rest are individual `docs/followups/todo/` entries, and the 3.0.0
CHANGELOG's `Known issues` section is the user-facing roster with
magnitudes. Two are worth naming here because they are the ones a user is
most likely to be affected by without noticing: the **boost integral**
diverges instead of converging as a parent approaches rest, which is the
regime every model spectrum passes through near threshold; and
**`thermal_cross_section` never converges**, so ⟨σv⟩ is 0.5%–5% wrong
across the entire freeze-out region and every relic abundance computed
from a mediator model inherits that roughly linearly.

Three smaller items the project surfaced and did not own:
[`hazma.utils`'s redundant public helpers](../../../docs/followups/todo/utils-public-surface-redundant-helpers.md),
[the remaining `sqrt(kallen_lambda(...))` call sites](../../../docs/followups/todo/kallen-under-sqrt-remaining-call-sites.md),
and [four tracked non-source files under `hazma/`](../../../docs/followups/todo/tracked-non-source-files-under-hazma.md).
`hazma/experimental/axial_vector_mediator/__init__.py` is still broken on
import (`from hazma.theory import Theory`, but `hazma.theory` exports
`TheoryAnn` / `TheoryDec`); no follow-up is filed because `experimental/`
is explicitly not a public surface, but it will keep tripping import
sweeps until someone deletes or fixes it.

## 6. Process notes

Three things about *how* the project ran that are worth carrying to the
next one.

- **Write the criterion so it names its oracle and forbids
  transcription.** The one Phase 03 task that needed no criteria patch,
  3.1, is the one whose criterion already said "extract from source and
  assert bit-equality — no hand-transcription trust". Every other task in
  that phase had to add criteria mid-flight, because the plan's model of
  an external artifact was a hypothesis only the sweep could refute. The
  same shape recurs in Phase 00 (three of five tasks patched canonical
  criteria — *the criteria written before a deletion are routinely wrong
  about what the deletion strands*) and in Phase 07 (a criterion scoped
  to a file is a criterion scoped to a guess). **Patching the criterion
  in the same PR is cheaper and more honest than absorbing the difference
  into the diff**. This project did it in three of Phase 00's five
  tasks, all three of Phase 02's, four of Phase 03's five and three of
  Phase 07's four — without once weakening a gate.
- **The running numerical record has to be a file, not a memory.**
  `task-notes/numerical-impact.md` — one entry per task giving the
  function, the grid, and the result, appended in task order — is what
  the 3.0.0 CHANGELOG was assembled from at close, twenty-six days and
  thirty-three tasks after the first entry. It was moved out of the
  working-memory README at Task 5.3 precisely because it had outgrown a
  section. No closing agent could have reconstructed those figures, and
  `PLAN.md` said so in advance.
- **A phase-learnings file that replaces its phase's task notes actually
  works.** ADR-0002 was written mid-project against measured context
  growth (three Phase 04–05 tasks each ended between 513k and 644k
  tokens, with the mandatory documents under 35k of it). Closing this
  project meant reading seven learnings files instead of thirty-three
  task notes, and nothing needed was missing from them.
