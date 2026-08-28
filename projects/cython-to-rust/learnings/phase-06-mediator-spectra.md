# Phase 06 Learnings: Mediator spectra

Synthesized at phase close (2026-08-27) from the four task notes
([6.1](../task-notes/phase-06/task-6.1-table-struct.md),
[6.2](../task-notes/phase-06/task-6.2-decay-spectra.md),
[6.3](../task-notes/phase-06/task-6.3-positron-spectra.md),
[6.4](../task-notes/phase-06/task-6.4-retire-survivors.md)) and
[`../task-notes/phase-06/README.md`](../task-notes/phase-06/README.md).
Read this instead of the notes; the notes are history.

## 1. Implementation Reality Check

The phase delivered what it promised and closed the port. **Seven
consumed entry points** — three mediator decay photon spectra and four
mediator positron spectra — moved to `hazma._core` across Tasks 6.2 and
6.3, each gated on the parity corpus and each deleting its Cython twin in
the same PR. Task 6.4 then deleted the four capi-survivor spectra
extensions and the four `hazma/_utils/` headers, and
`find hazma -name "*.pyx" -o -name "*.pxd"` returns **nothing**. All 41
consumed entry points are on Rust; `setup.py` declares one
`RustExtension` and nothing else.

The redesign the phase file asked for is real rather than nominal. The
four `.pyx` differed in exactly four ways and **all four were data** —
grid start, below-grid policy, which tables are built, and the selector
type — so `crate::kernels::mediator_tables` is one parameterised
implementation with a genuine mass-keyed cache in place of the Cython's
dead predicate, and `mediator_decay_positron.rs` is **one kernel serving
all four positron entry points**, because normalised for the model's name
the two positron `.pyx` were not a clone-pair but the same implementation
twice. Mode strings became an enum parsed once per call instead of once
per quadrature node.

Numerically the phase stayed inside its budgets and tightened them.
Task 6.2's three entry points moved by at most **5.3327e-12** and Task
6.3's four by at most **2.3319e-12**, all of it the integrator's:
`crate::quad` against scipy's QUADPACK, established by setting
`eng_s == ms` to make the boost integrand constant, at which point every
channel agrees with the Cython **to within one ulp**. Seven budgets
tightened across the phase and none widened; Task 6.3 emptied
`NESTED_RTOL`, the last opening budget any corpus case held, so the
project ends with **fourteen budgets tightened and none widened**. One
value moved `NaN → 0.0`, at the legacy `m_e`, which was a repair rather
than drift. Task 6.4 moved nothing: 88 public arrays captured from a
second worktree built at `origin/master` compared **0 of 88 moved**.

Performance came out where the dead-cache bug predicted: **32x–43x** on
the positron pair from release builds of both sides, and 4.2x on the
table build with 4,180x on a fixed-mass parameter sweep.

## 2. Critical Context for Future Work

**The corpus is what pins these kernels now, and it is the only thing
that does.** Every Cython oracle in the tree is gone — the twins, the
`__pyx_capi__` capsules, the `.pxd` constant tables, and the `.pyx`
sources that dispatch messages and mode strings were scanned out of.
Four separate mechanisms that used to *execute* against the Cython are
now frozen rosters or reference implementations with the provenance
recorded in place: `cython_dispatch_messages()` (Task 6.2),
`test_core_mediator_tables.py`'s mode parsers (6.3),
`test_core_boost.py`'s oracle (6.4) and `test_core_constants.py`, which
was deleted outright (6.4). Treat "the port's messages are the Cython's"
as transcription from here on.

**A constant no entry point reads is now pinned by nothing.**
`test_core_constants.py` compared ~220 constants bit-for-bit against
`rust/src/constants.rs` by parsing the two `.pxd` and three `.pyx`; it
was run green (21 passed) against `origin/master` immediately before
deletion, so the transcription is verified as of that moment.
`constants.rs`'s own `cargo` tests keep the structural half — the rule-4
divergences, `photon_pion`'s mix of the two tables, `R_FACTOR`'s
provenance, const-folding — and the corpus keeps the numerical half for
every constant reaching one of the 41 entry points. The gap is stated in
that module's header. Consolidating the two `MASS_E` tables is still the
separate declared change rule 4 requires.

**Re-capturing `test/parity/oracles` is possible but expensive, and it
was nearly made impossible.** `RESTORED_SOURCES` went from 13 entries to
**29** in Task 6.4, covering every case a defect chain reaches. The
recursion that blocked Tasks 6.2 and 6.3 — a task cannot know its own
commit's SHA — dissolves once you notice `capture.py` resolves revisions
with `git show <rev>:<path>`, which accepts a plain SHA as readily as a
`^` expression: name a revision that **already exists** instead of one
computed from a later commit's parent. A complete roster also needs the
**compile closure**, not just the patched files, and a re-capture must
now restore `setup.py` and `pyproject.toml` too.

**The drift these swaps carry is the integrator's, and it is measurable
rather than arguable.** Setting the mediator energy to its mass collapses
the boost integrand to a constant and isolates the transliteration from
the quadrature. A constant integrand is still not reproduced exactly —
`∫₋₁¹ c dcl` lands one ulp off the exact `2c` on **both** sides, at
different `c`.

**`numpy.logspace` and `f64::powf` are not the same code.** The
rest-frame grid is bit-equal to NumPy's on macOS/arm64 and one ulp off it
at ~5% of points on Linux/x86-64 (every measured disagreement exactly one
ulp, worst relative 2.16e-16). "Bit-equal to the Cython" is a
**capturing-platform statement** for anything reading these tables.

## 3. Quirk Log & Edge Cases

- **An unrecognised mode string returns `0.0`, not an error.** Every
  `cdef double` integrand ends in an `if`-chain with no `else`, and a C
  function that falls off its end returns zero. Reproduced under rule 1
  and filed rather than fixed.
- **The two clone-pairs differ in laziness and it is observable.** The
  scalar decay integrand guards each channel with a bitflag `if`; the
  vector one computes all six components then selects, so a
  single-channel call still raises where any component would. Read the
  `.pyx`'s structure, not just its formulae.
- **`pws` is read lazily, so a length check would break a working call.**
  The last partial width is read only inside the boosted line window, so
  a short buffer legitimately succeeds outside it and raises `IndexError`
  inside.
- **The `nan` at the legacy `m_e` was a clang FMA contraction, not a
  constants divergence.** `sqrt(E² − m_e²)` compiles to an FMA that
  subtracts the *rounded* `m_e²` from an exactly-computed square, and
  that rounding is upward by 1.45e-17. Fixed by clamping the radicand.
- **The shipped `e⁺e⁻` line is low by the positron's rest-frame
  velocity** — the box's edges carry `r` and its height does not, so it
  integrates to `pw_ee · r`. Reproduced under rule 1 and filed; it moves
  published numbers above the budgets the four positron cases now hold,
  so it needs a corpus re-capture or a declared exception.
- **`beta == 1` divides by zero in Python and not in Rust.** Both
  implementations pass the shared guard (`beta > 1.0`, not `>=`) and
  compute `1/sqrt(0)`; Rust yields `+inf` under IEEE-754 so the line
  height underflows to `0.0`, while a Python transcription raises
  `ZeroDivisionError`. A reference implementation is an oracle only where
  it is defined.
- **`hazma/_utils/kinematics.pyx.bak` was a *tracked* backup file.** No
  rule in this repo covers one, and no sweep in six phases had named it.

## 4. Test Infrastructure State

`pytest -q` ends the phase at **2231 passed, 15 skipped, 12 subtests**;
`pytest test/parity -q` at **658 passed, 1 skipped**; `cargo test
--no-default-features` at **258 passed** (201 at Phase 05's close).

The suite total *fell* by 158 in Task 6.4 and the accounting matters,
because a shrinking suite is the shape of a disabled gate. Six modules
changed, all measured with `--collect-only`: `test_core_constants`
21 → 0, `test_core_boost` 80 → 50, `test_core_photon_muon` 53 → 29,
`test_core_photon_pion` 73 → 30, `test_core_positron_muon` 47 → 25,
`test_core_positron_pion` 49 → 27, plus a new `test_no_cython_remains`
0 → 4. Every retired test compared against an oracle that no longer
exists; **none of the physics or dispatch coverage was touched**.

Where a retired comparison had a surviving substitute it was repointed
rather than deleted. `test_core_boost.py`'s `TestFusedArithmetic` already
swept all five photon tables × 6 boost regimes × 300 energies and 40,000
delta-function draws bit-for-bit against in-module reference
implementations, on **every** platform — strictly stronger than the
Cython sweeps, which were bit-equal only on macOS/arm64 — so nine further
tests that corroborated a hand-computed value with the Cython were moved
onto the same reference.

New standing gates from this phase: `hazma._core.mediator_tables` as a
sixth `_CORE_TEST_ONLY_MODULES` probe (6.1), one test module per
clone-pair rather than per entry point (6.2), and
`test/test_no_cython_remains.py` (6.4), which asserts the tree-wide
property on sources and build declarations — never on `ImportError`,
because a built `.so` and its generated `.c` outlive a deleted `.pyx`.

## 5. Follow-on seeds

- [`mediator-positron-line-misses-the-electron-velocity`](../../../docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md)
  — the `e⁺e⁻` line's missing `1/r`. Needs a corpus re-capture or a
  declared exception; explicitly not a swap-time fix.
- [`mediator-spectra-accept-unknown-mode-strings`](../../../docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md)
  — silently returning `0.0` for a typo'd mode.
- [`oracle-restore-revisions-for-the-mediator-decay-pyx`](../../../docs/followups/done/oracle-restore-revisions-for-the-mediator-decay-pyx.md)
  — **closed by Task 6.4**; the roster is complete at 29 entries.
- The constants gap in §2: nothing pins a constant that no entry point
  reads. The natural time to close it is the deferred consolidation of
  the two divergent tables (rule 4).

## 6. Process notes

- **A mutation survivor is a statement about the coefficient or about
  the grid; decide which before writing it off.** Fourteen of Task 6.2's
  thirty-seven fused sites survived their mutation *by construction* —
  the coefficient is a power of two, so the product is exact and fusing
  cannot change the sum. Two more survived only because the grid never
  reached `2 m_μ`.
- **The contraction rule predicts; the campaign decides.**
  `head − coef·m_e·m_e` ends in a syntactic multiply, so the rule
  predicts a fusion — and fusing it *loses* bit-equal values.
- **A mutation harness that does not force a rebuild will lag its own
  mutations.** `rm -f hazma/_core.abi3.so` before each install, `test -f`
  after. One of three campaign runs lagged by two iterations.
- **Instrument the extension before trusting a replica.** Four rounds of
  transliterating a `cdef` into Python returned `0.0` where the extension
  returned `NaN`; what settled it was a temporary `def` in the `.pyx`
  returning the intermediates. A Python replica is unfused and the
  shipped C is not.
- **Deleting a `.pyx` does not make its module unimportable.** The built
  `.so` and generated `.c` sit beside the source, are gitignored, and
  survive `git rm` — Task 6.3 used that deliberately to keep both twins
  callable for a drift measurement after `git rm`. Assert on source files
  and build declarations, never with `pytest.raises(ImportError)`, and
  `rm` the orphans before the next install measures extensions nothing
  builds.
- **`python -c` from the repository root imports the cwd's `hazma/`, not
  the installed one.** A cross-tree comparison has to be driven from
  outside both trees, or it silently measures one tree twice.
- **Encode a caught exception by name, not by `hash()`.** Python
  randomises `str` hashing per process, so a hash-based sentinel makes a
  before/after array comparison report spurious movement.
