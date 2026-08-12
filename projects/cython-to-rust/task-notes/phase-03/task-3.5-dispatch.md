# Task 3.5: Dispatch and error layer

**Date:** 2026-08-11
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-03-numerics-foundation.md` (Task 3.5);
`../../references/numerics-replacements.md` ("Entry-point dispatch contract");
`../../rules.md` rules 1 (parity discipline), 8 and 9 (Rust conventions 3–4)
**Related ADRs:** none
**Depends On:** none (Phase 02 Task 2.3 pinned the scaffold's half of the
contract; Tasks 3.2–3.4 already route their probes through it)

## Objective

Settle the argument-dispatch and error contract every Phase 04–06 entry
point will use — including the neutrino 3-tuple / `(3, N)` variant and the
required-1-D-array variant — as one PyO3-side helper module, with each
divergence from the live Cython decided on purpose and its error text
byte-matched against the `.pyx` sources.

## Exit Criteria

From the phase file:

- One generic helper implementing the scalar-or-1D contract
  (`../../references/numerics-replacements.md`, dispatch section),
  including the neutrino tuple/`(3,N)` variant; kernel crates stay
  PyO3-free (rules.md rule 8).
- Error messages byte-match the Cython ones the tests assert on.

Made concrete before implementing:

- Every one of the four divergences the reference records as "Task 3.5
  decides" is decided, implemented, and pinned by a test.
- The message templates are checked against strings **extracted from the
  `.pyx` sources**, not against hand-typed copies.
- `cargo test` covers the PyO3-free half; `test/test_core_dispatch.py`
  covers the Python-visible contract.
- The parity corpus stays in bit-equality mode (`exact=True`).

All met — see Verification.

## Inputs Reviewed

- `../../PLAN.md`, `../README.md`, `../../rules.md`,
  `../../phases/phase-03-numerics-foundation.md`
- `../../references/numerics-replacements.md` — "Entry-point dispatch
  contract" and its "What the Cython actually does today" block
- `rust/src/dispatch.rs`, `rust/src/lib.rs`, `rust/src/kernels.rs`,
  the four probe modules, `test/test_core_dispatch.py`
- `test/parity/cases.py` — `_CORE_SCAFFOLD_NAMES`,
  `_CORE_TEST_ONLY_MODULES`, `rust_core_kernels`
- [`../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md`](../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
- `docs/agents/lessons.md`, `docs/agents/environment.md`

## Findings

### There are four dispatch shapes, not one

The reference's premise — "every public function follows one shape;
implement once as a helper" — is false, and the four measured divergences
it lists are consequences of that rather than the whole story. All 43
surviving top-level `def`s were classified by parsing their source (the
population enumerated, not typed) and then a representative of each shape
was measured on the built tree against a 24-input matrix:

| # | Shape | Entry points | 0-d array | list / tuple | rank guard |
| --- | --- | --- | --- | --- | --- |
| A | scalar-or-1D energies | 15 | `AssertionError` | accepted | yes |
| B | neutrino (3-tuple / `(3, N)`) | 2 | `AssertionError` | accepted | yes |
| C | cross sections | 18 | **float** (`.item()`) | `AttributeError` | **none** |
| D | `partial_widths` | 1 argument | `AssertionError` | accepted | yes |

The other 8 `def`s (`thermal_cross_section` ×2, the four mediator
`dnde_decay_*` pairs) take strictly typed `double` /
`np.ndarray[double]` arguments and dispatch on nothing.

So shapes A and C — 33 of the 35 dispatching entry points — **disagree
with each other** on both the 0-d array and the list, in opposite
directions. A port that transcribed either one would have shipped the
other's users a break.

### The Cython's dtype message is not one string

Measured: the spectra extensions raise
`ValueError: Buffer dtype mismatch, expected 'double' but got 'long'` and
the mediator ones `expected 'float64_t'` — for the same rejection, on the
same input. Both name a C type rather than a dtype, and `'long'` is
platform-width. There is nothing here to byte-match, which is what makes
rewording it the only coherent option.

### A 0-d array's `__float__` forwards to its element

`float(np.array("15.0"))` is `15.0`, because NumPy's 0-d `__float__`
defers to the element and `np.str_` subclasses `str`. PyO3's
`extract::<f64>` goes through `PyNumber_Float`, i.e. exactly that
conversion. A first draft that accepted a 0-d array by *trying* the
extraction therefore returned a number for `roundtrip("15.0")` — a string
accepted as an energy where the Cython raises. Fixed by asking the dtype's
`kind` (`b`/`i`/`u`/`f`) instead of attempting the conversion. **Any
Phase 04–06 check that means "is this numeric?" and answers it with a
float conversion has the same hole.**

### `require_vector` must not check length, and the Cython shows why

`scalar_mediator_decay_spectrum`'s `pws` handling indexes seven entries
and raises `IndexError: Out of bounds on buffer access (axis 0)` from the
*kernel* for anything shorter — measured. That is a kernel precondition,
not a dispatch one, so `require_vector` validates rank and dtype only and
Phase 06 owns the length check. Recorded in the module docs and pinned by
`test_an_empty_array_is_accepted`.

### A mutation campaign can refute the implementation's own comment

Fourteen mutations, thirteen caught. The survivor swapped the sequence
branch and the scalar fallback in `classify` — an ordering the code
comment claimed was load-bearing against the string bug above. It is not:
the only objects with both a `__len__` and a working `__float__` are 0-d
ndarrays, which an earlier arm already took, so all 118 tests stayed green
with the arms reversed. The ordering is fidelity to the Cython's own
predicate order and nothing more; `has_numeric_dtype` is the guard. The
comment was corrected rather than the mutation dropped. **Read a survivor
as a result** — Task 3.4's three named the bisection tests it was missing,
and this one named a false sentence.

## Decisions and Implementation Notes

**The rule that decides every divergence, stated once: each exception the
Cython raises *explicitly* keeps its type, and only its `assert`s change
type** (rules.md rule 9 — today they vanish under `python -O` and leave
the user a downstream failure). Applying it:

| Situation | Cython | Port | Why |
| --- | --- | --- | --- |
| rank > 1 | `AssertionError` (A/B/D), `ValueError` buffer wording (C) | `ValueError`, assert text **verbatim** | rule 9; C's wording is Cython-internal |
| 1-D, dtype ≠ float64 | `ValueError`, two different C-type wordings | `ValueError` naming the dtype | type kept; no single string to match |
| 0-d array | `AssertionError` (A/B/D), float (C) | float | widening; C's own behavior |
| list / tuple | accepted (A/B/D), `AttributeError` (C) | accepted | widening; A/B/D's own behavior |
| not a number, no `__len__` | `TypeError` (CPython's wording) | `TypeError` naming the quantity | type kept |
| `pws` without `__len__` | `ValueError`, explicit `raise` | `ValueError`, **verbatim** | explicit, so unchanged |

Three widenings ride along and none can break a call that works today.

Other decisions:

- **Three helpers over one `classify`, not three implementations.**
  `map_unary` (shapes A and C), `map_flavors` (B), `require_vector` (D).
  Covering C costs no extra code — it is the same function — so what the
  task actually decided is that Phase 05 calls the same helper as Phase
  04, which is what makes the two shapes agree for the first time.
- **`map_flavors` calls its kernel once per energy**, not once per
  (energy, flavor). The Cython computes the three flavors from one shared
  kinematic evaluation; three calls would triple the work and could not be
  assumed to round identically.
- **`hazma/spectra/_neutrino/_muon.pyx:205`'s "Photon energies" is not
  carried over.** The ported `dnde_neutrino_muon` will pass `"Neutrino
  energies"`, matching its `_pion.pyx` sibling. The string is reachable
  only through an exception whose *type* is already changing, so nothing
  that matches on it survives either way, and it names the wrong physical
  quantity. `TestCythonMessageParity` pins the roster from source so the
  defect can neither spread nor vanish unnoticed.
- **The `.pyx` sources are the message oracle.** The test scans them for
  every `assert len(...) == 1, "..."` and `raise ValueError("...")`,
  asserts the roster is exactly the four-plus-one it expects, and renders
  each message through the port with that quantity. A transcribed roster
  would have been the `[hand-written-population-in-a-derived-check]`
  lesson again.
- **A fifth probe module, `hazma._core.dispatch`, and it exists for the
  wording.** Its three probes take `quantity` as an argument, which the
  top-level `roundtrip` (Phase 02's, wording fixed to `"Input values"`)
  cannot — and a test byte-matching `"Photon energies must be 0 or
  1-dimensional."` has to be able to ask for that wording. `roundtrip`
  itself is untouched, so Phase 02's scaffold and its
  `_CORE_SCAFFOLD_NAMES` exemption keep working. The submodule joins
  `cases._CORE_TEST_ONLY_MODULES` under Task 3.2's importer guard —
  fourth instance of `docs/agents/lessons.md` `[gate-disabled-stays-green]`
  in this project, and the mechanism is reused rather than widened.
- **`kernels::roundtrip_flavors` is `[x, -x, 1/x]`, not three copies of
  `x`.** With equal rows, a transposed result, a reversed row order or a
  row written twice would all satisfy a value assertion. Negation and
  reciprocal are both correctly rounded, so the Python-side reference is
  bit-exact and the test argues about no tolerance.
- **Review round 1 (PR #62): the stale-state sweep had two populations
  and only one was swept.** `Task 3.5 decides` — the *pointer* text —
  was swept across the durable docs, but the **statements of the
  pre-decision contract** carry no such token and were left asserting the
  old rules as current fact: `../../learnings/phase-02-rust-scaffold.md`
  §2 still said a 0-d array must be `float64`, that everything else
  raises `ValueError`, and that `map_unary` was the sole helper, and
  `../README.md`'s Task 2.3 finding said "a 0-d array still enforces
  dtype". Fixed by re-sweeping on the *behavior* words rather than the
  task id (`still enforces dtype`, `anything else → ValueError`,
  `single implementation`) — seven live occurrences across five files,
  each now either corrected or carrying a dated supersession note beside
  it. Recorded in `docs/agents/lessons.md` as
  `[settling-a-deferral-has-two-sweeps]`. The `phase-02` working memory
  and Task 2.3's note had *predicted* this ("a Task 3.5 decision that
  changes any of them now turns a named test red"), which made closing
  the loop on them one line each.
- **`dispatch.rs` has no `cargo test` units** and does not need any: every
  line of it is PyO3, so `cargo test --no-default-features` (which links
  libpython but attaches no interpreter) is the wrong harness. The phase
  criterion "`cargo test` covers the foundation GIL-free" is about
  `constants`/`special`/`quad`/`interp`/`boost`; the PyO3 boundary's gate
  is `test/test_core_dispatch.py`. `kernels.rs` gained two units for
  `roundtrip_flavors`.

## Files Changed

- `rust/src/dispatch.rs` — rewritten around a shared `classify` /
  `classify_array`, plus `has_numeric_dtype`, the `numpy.asarray`
  sequence branch, `map_flavors`, `require_vector`, `TypeError` for
  non-numbers, and a module header carrying the measured four-shape
  rationale.
- `rust/src/dispatch_probe.rs` — **new**, registration-only
  `hazma._core.dispatch` (`roundtrip`, `roundtrip_flavors`,
  `roundtrip_vector`, each taking the quantity wording).
- `rust/src/kernels.rs` — `roundtrip_flavors` and two unit tests.
- `rust/src/lib.rs` — `mod dispatch_probe;`, the submodule registration,
  and the reconciled probe paragraph.
- `test/test_core_dispatch.py` — 54 → 118 tests: the sequence path, the
  0-d decision, the flavor shape, `require_vector`, the source-derived
  message parity, and each declared divergence asserted against **both**
  implementations.
- `test/parity/cases.py`, `test/parity/test_parity.py`,
  `test/parity/README.md` — `hazma._core.dispatch` added to
  `_CORE_TEST_ONLY_MODULES`, and the three prose sites reconciled.
- `hazma/_core.pyi` — the unstubbed-submodule comment now covers all five
  probes, and `roundtrip`'s contract paragraph carries the settled rules.
  **The only change under `hazma/`, and every line of it is a comment.**
- `docs/followups/todo/model-spectra-reject-scalar-energies.md` — the
  compiled half recorded as decided; the pure-Python half stays open.
- **Canonical patches:** `../../phases/phase-03-numerics-foundation.md`
  (five Task 3.5 criteria added during execution; frontmatter
  `status: Complete`) and
  `../../references/numerics-replacements.md` (a "settled contract"
  section, plus a pointer from the superseded design sketch).
- **Phase closure:** `../../learnings/phase-03-numerics-foundation.md`
  (**new**), `../../PLAN.md` Phases row, `../README.md` (Phases row,
  Findings, Numerical impact, Files Changed, Verification, Handoff), and
  this phase's `README.md`.

## Verification

Environment: CPython 3.12.12, macOS/arm64, NumPy 2.5.1, scipy 1.18.0,
Cython 3.2.9 — the corpus's capturing environment, confirmed by
`tolerances.provenance(...)` → `Provenance(exact=True, detail='')`. The
tree was cleaned of stale `.c`/`.cpp`/`.so` and rebuilt with
`uv pip install -e . --no-build-isolation` before anything below was run;
`hazma._core.__file__` resolves inside this worktree.

- **`pytest` (bare, the full gate)** → `1378 passed, 13 skipped in
  564.55s`. +64 on Task 3.4's 1314, which is exactly
  `test/test_core_dispatch.py` growing from 54 tests to 118 — no other
  module's count moved. The skip count is unchanged at 13, which is what
  proves the parity corpus ran in **bit-equality mode** (`rtol = 0` across
  all 41 consumed entry points, 179,695 pinned values).
- **`pytest test/test_core_dispatch.py -q`** → `118 passed in 4.19s`;
  population from `--collect-only -q` → `118 tests collected`, in 11
  classes. What they cover, by class: the scalar path (Python float, NumPy
  scalars of five dtypes, `int`/`bool`, all ten special float64 values
  bit-for-bit); the 0-d decision (five numeric dtypes accepted, `<U4` and
  `object` rejected); the 1-D array path (dtype, shape, freshness,
  non-contiguity, read-only, empty); sequences (list, tuple, empty, int
  list, nested); the error paths (four rank cases, five dtypes, ordering,
  three type cases, the string); the parameterised quantity; the flavor
  shape (scalar tuple, `(3, N)` layout, the transpose-invisible length-3
  case, empty, aliasing, strided, read-only, shared error contract);
  `require_vector` (arrays, sequences, strided, empty, seven no-length
  cases, five rank cases, two dtypes); the source-derived message parity
  (roster, every rank message, both `pws` messages); five declared
  divergences asserted against the live Cython; and the signatures.
- **`cargo test --manifest-path rust/Cargo.toml --no-default-features`**
  → `69 passed` (2 new, both on `roundtrip_flavors`).
- **`cargo fmt --check`**, **`cargo clippy --all-targets -- -D warnings`**
  — clean.
- **`scripts/agents/preflight.sh --paths "..."`** → RESULT: PASS.
- **Mutation campaign**: 14 mutations against `rust/src/dispatch.rs` and
  `rust/src/kernels.rs`, applied one at a time from a green baseline,
  reverted after each, with the baseline re-asserted at the end (both
  runs `118 passed`). **13 caught**; the survivor is analysed under
  Findings and produced a source correction.

  | Mutation | Result | First test to fire |
  | --- | --- | --- |
  | rank message wording | caught (9) | `test_multidimensional_arrays_raise_value_error` |
  | dtype message drops the quantity | caught (14) | `test_wrong_dtype_arrays_raise_value_error_naming_the_dtype` |
  | non-numeric raises `ValueError` again | caught (5) | `test_non_numeric_input_raises_type_error` |
  | 0-d array raises like the spectra Cython | caught (20) | `test_zero_dim_float64_array_returns_a_python_float` |
  | 0-d dtype guard removed | caught (2) | `test_zero_dim_non_numeric_arrays_are_rejected` |
  | numeric dtype kinds lose `"b"` | caught (1) | `test_zero_dim_arrays_of_any_numeric_dtype...` |
  | scalar fallback before the sequence branch | **survived** | — (see Findings) |
  | sequences no longer converted | caught (9) | `test_a_sequence_of_floats_is_accepted` |
  | flavor array transposed | caught (5) | `test_array_in_three_by_n_array_out` |
  | flavor rows reversed | caught (3) | `test_rows_are_flavors_not_energies` |
  | `require_vector` loses its no-length guard | caught (8) | `test_anything_without_a_length_is_refused` |
  | `require_vector` accepts a 0-d array | caught (3) | `test_anything_that_is_not_one_dimensional_is_refused` |
  | `require_vector` rank message widened | caught (6) | `test_the_partial_width_messages_are_reproduced_exactly` |
  | flavor probe rows collide | caught (13) | `test_scalar_in_three_tuple_out` |

- **Numerical impact: no public value changes** (verified:
  `git diff origin/master -- hazma` is one file, `hazma/_core.pyi`, and
  every line of the hunk is comment text — no executable line under
  `hazma/` is reachable from this diff). No grid sweep is reported because
  the parity corpus is a stricter grid than any ad-hoc one and it ran in
  bit-equality mode, and because nothing under `hazma/` imports the code
  this task wrote. The task *does* settle five user-visible **exception**
  changes that land with Phases 04–06 — listed in the decision table above
  and logged under `../README.md`'s "Numerical impact so far" so the
  Phase 07 CHANGELOG picks them up.

## Open Questions

- **What `quantity` wording Phase 05 gives the cross sections.** They have
  none today — shape C carries no message at all — so the port invents
  one. `"Center-of-mass energies"` is what this task's tests use as a
  placeholder; the call is Phase 05's and nothing here depends on it.
- The pure-Python half of
  [`../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md`](../../../../docs/followups/todo/model-spectra-reject-scalar-energies.md)
  remains open (`Theory.spectra`'s `type(e_gams) == float` branch and the
  channel wrappers' `len(e_gams)`). The compiled half is now decided and
  the followup records that.

## Plan Impact

**Impact Level:** Phase file patched (and the reference).

- `../../phases/phase-03-numerics-foundation.md` — five Task 3.5
  "criteria added during execution", because the exit criterion "one
  generic helper" describes a tree with one dispatch shape and there are
  four; plus frontmatter `status: Complete` and the phase's own exit
  criteria met.
- `../../references/numerics-replacements.md` — a "settled contract"
  section recording the four shapes, the three helpers, the deciding rule
  and the two measured traps, with a pointer at the top of the superseded
  design sketch. Its own prose is what a next reader would port from, and
  it was wrong in a way that would have narrowed the public API.
- `../../PLAN.md` — the Phases table's row 03 (live status only, no shape
  change).
- No ADR. The decision is confined to this project's port and is fully
  expressed by the phase file plus rules.md rule 9, which already
  committed to the assert tightening.

## Stale-state sweep

Each block below is the command's actual output on this branch.

```text
$ git diff origin/master --stat -- hazma
 hazma/_core.pyi | 22 ++++++++++++++--------
 1 file changed, 14 insertions(+), 8 deletions(-)

$ git status --short
 M docs/followups/todo/model-spectra-reject-scalar-energies.md
 M hazma/_core.pyi
 M projects/cython-to-rust/PLAN.md
 M projects/cython-to-rust/learnings/phase-02-rust-scaffold.md
 M projects/cython-to-rust/phases/phase-03-numerics-foundation.md
 M projects/cython-to-rust/references/numerics-replacements.md
 M projects/cython-to-rust/task-notes/README.md
 M projects/cython-to-rust/task-notes/phase-03/README.md
 M rust/src/dispatch.rs
 M rust/src/kernels.rs
 M rust/src/lib.rs
 M test/parity/README.md
 M test/parity/cases.py
 M test/parity/test_parity.py
 M test/test_core_dispatch.py
?? projects/cython-to-rust/learnings/phase-03-numerics-foundation.md
?? projects/cython-to-rust/task-notes/phase-03/task-3.5-dispatch.md
?? rust/src/dispatch_probe.rs

$ rg -n "Task 3.5 decides|Task 3.5 must make|Task 3.5 must decide" \
      projects/ docs/ test/ rust/ hazma/ | grep -v task-3.5-dispatch.md
projects/cython-to-rust/task-notes/phase-02/task-2.3-plumbing-test.md:330:
  Cython/contract dispatch divergences Task 3.5 must decide.
```

The one survivor is a **closed task note**, which is history and is left
alone. The three *live* forward references were reconciled in this task —
`../README.md` (Findings), `../../learnings/phase-02-rust-scaffold.md`
(§3) and `../../references/numerics-replacements.md` (the Task 2.1
measurement block), each now pointing at the settled contract.

```text
$ rg -c "_CORE_TEST_ONLY_MODULES|_core.dispatch" test/parity/cases.py \
      test/parity/test_parity.py test/parity/README.md hazma/_core.pyi rust/src/lib.rs
test/parity/cases.py:6
test/parity/test_parity.py:5
test/parity/README.md:3
hazma/_core.pyi:1
rust/src/lib.rs:1

$ rg -n "today four|Four further submodules|four test-only|four probes" \
      test/ hazma/ rust/ projects/ --glob '!task-3.5-dispatch.md'
(no occurrences — every "four" reconciled to five)

$ rg -n "TODO|FIXME|breakpoint\(\)|import pdb|print\(" \
      rust/src/dispatch.rs rust/src/dispatch_probe.rs test/test_core_dispatch.py
(no occurrences)

$ python -c "... tolerances.provenance(manifest) ..."
Provenance(exact=True, detail='')

$ scripts/agents/preflight.sh --paths "test/test_core_dispatch.py \
      test/parity/cases.py test/parity/test_parity.py" --md "<10 changed docs>"
RESULT: PASS
```

**Numerical-impact statement:** no public value changes (verified:
`git diff origin/master -- hazma` is `hazma/_core.pyi` only, comment text
throughout; the parity corpus ran in bit-equality mode inside a bare
`pytest` of `1378 passed, 13 skipped`).

## Handoff to Next Task

**Phase 03 is closed.** Read
[`../../learnings/phase-03-numerics-foundation.md`](../../learnings/phase-03-numerics-foundation.md)
first — it supersedes this note and its four siblings.

**Safe to assume:**

- `dispatch::map_unary(obj, quantity, kernel)` serves shapes A and C,
  `map_flavors` shape B (`Fn(f64) -> [f64; 3]`, one call per energy,
  `(3, N)` rows electron/muon/tau), `require_vector` shape D. Kernels stay
  plain `fn(f64, ...) -> f64` and pass their wording in.
- `test/test_core_dispatch.py` is the template a kernel swap copies: keep
  every test, swap the probe and the quantity, add the numerical tests
  beside rather than merged in.
- The messages are pinned against the `.pyx` themselves, so a Phase 04–06
  deletion that removes the last site carrying a wording turns
  `TestCythonMessageParity::test_the_tree_carries_exactly_the_expected_
  message_roster` red. That is the test telling you to update its roster,
  not a defect.

**Risky / unknown:**

- **`require_vector` does not check length.** Phase 06's
  `scalar_mediator_decay_spectrum` must add its own — the Cython indexes
  seven `pws` entries and raises `IndexError` from the kernel today.
- **Phase 05 has to name the cross sections' `quantity`.** They carry no
  message at all today, so the port invents the wording and it is
  user-visible from the first swap.
- **Do not put a non-kernel on `hazma._core` without the submodule
  exemption.** `cases.rust_core_kernels()` counts every public callable
  except the literal name `roundtrip`; anything else flips the corpus out
  of bit-equality mode with nothing turning red.
