# Task 2.3: Cross-language plumbing test

**Date:** 2026-08-09
**Project:** cython-to-rust
**Status:** Complete
**Plan References:** `../../phases/phase-02-rust-scaffold.md` (Task 2.3);
`../../references/numerics-replacements.md` (Entry-point dispatch
contract); `../../rules.md` rule 8 (Rust conventions 3)
**Related ADRs:** ADR-0001 (accepted)
**Depends On:** Task 2.1

## Objective

Pin the `hazma._core` entry-point dispatch contract from the Python side
— every branch of `dispatch::map_unary`, exercised through the scaffold's
`roundtrip` probe — in a test module written to be the template Phases
04–06 copy for each real kernel.

## Exit Criteria

Copied from the phase file's Task 2.3 block:

- pytest exercises the round-trip function for: float in/float out, 1-D
  array in/out (dtype, shape, ownership), 0-d array, wrong-dtype and 2-D
  error paths raising `ValueError` with the contract message (see
  `../../references/numerics-replacements.md`, dispatch contract).
- The same test module is the template later kernel swaps copy.

## Inputs Reviewed

- `../../PLAN.md`, `../README.md` (project working memory),
  `../../phases/phase-02-rust-scaffold.md`, `../../rules.md`.
- `README.md` (this phase's working memory) — Task 2.1's dispatch
  findings, the NumPy-panic ordering constraint, and its handoff line
  naming `map_unary` as the thing Task 2.3 must exercise.
- `../../references/numerics-replacements.md`, "Entry-point dispatch
  contract" **and** its "What the Cython actually does today" subsection.
- `rust/src/{lib,dispatch,kernels}.rs`, `hazma/_core.pyi`.
- `test/test_theory_aggregation.py` (house style for a contract-shaped,
  golden-data-free test module), `test/conftest.py`,
  `pyproject.toml`'s `[tool.pytest.ini_options]`.
- `docs/agents/lessons.md`, `docs/agents/doc-consistency.md`,
  `docs/agents/environment.md`, `docs/agents/preflight.md`.

## Findings

- **Every assertion in the new module was measured against the built
  extension before it was written.** A 26-case probe over `roundtrip`
  (`float`, `int`, `bool`, four NumPy scalar types, 0-d/1-D/2-D/3-D
  arrays in five dtypes, empty, non-contiguous, Fortran-order, read-only,
  `list`, `str`, `None`, `complex`, and the arity/keyword forms) is what
  the tests encode. Three behaviors were *not* predictable from the
  contract prose and are now pinned:
  - **Rank is checked before dtype.** `roundtrip(np.ones((2, 2),
    dtype=np.int64))` reports `must be 0 or 1-dimensional.`, not the
    dtype message. That ordering is the order the checks appear in
    `map_unary`, and reordering them would silently reword a
    user-visible exception.
  - **A 0-d array still enforces dtype**, even though a Python `int` does
    not. The 0-d scalar path lives *inside* the array branch, behind the
    typed view, whereas an `int` reaches `extract::<f64>`. So
    `roundtrip(4)` is `4.0` and `roundtrip(np.array(4))` is a
    `ValueError`.
  - **Non-`float` NumPy scalars are accepted** (`np.float32`, `np.int64`,
    `np.uint8`, `np.bool_`). They are neither `float` subclasses nor
    ndarrays, so they fall past both fast paths to `extract::<f64>`,
    which takes them. The module's docstring called out only `np.float64`
    before this task.
- **`roundtrip` advertised a signature it did not have.** `#[pyo3(
  text_signature = "(x, /)")]` made `inspect.signature` report
  positional-only, while `roundtrip(x=1.5)` worked — `text_signature` is
  a doc claim PyO3 does not enforce (enforcing it needs `#[pyo3(
  signature = (x, /))]`). Measured against the layer being replaced:
  `hazma.spectra._photon._muon.dnde_photon(egam=100.0, emu=200.0)`
  returns `2.0036713127483527e-05` — the Cython entry points are `def`
  functions and take keywords. So the *claim* was the defect, not the
  behavior: copied into a Phase 04 wrapper, a positional-only signature
  would have been a public-API narrowing. Fixed to `"(x)"`, which is also
  what `hazma/_core.pyi` already described.
- **Bit patterns survive the round trip intact, NaN payload included.**
  `struct.pack("<d", roundtrip(float("nan")))` is `000000000000f87f`,
  byte-identical to the input; so are `-0.0`, both infinities, the
  smallest subnormal (`5e-324`) and the largest finite. That is what lets
  the module assert bit-equality everywhere and argue about no
  tolerances — the property Task 2.1 chose the identity kernel for.
- **The freshly-allocated array does not own its data.**
  `roundtrip(a).flags.owndata` is `False` and `.base` is a
  `PySliceContainer` — the `numpy` crate's wrapper around the Rust `Vec`.
  So "fresh" has to be asserted as *non-aliasing* (`result is not
  values`, `result.base is not values`, `np.shares_memory` false, and
  mutating the result leaves the input alone), never as `owndata`. An
  `owndata` assertion would have been red on correct code.
- **A read-only input is accepted and a non-contiguous one is read in
  order.** `map_unary` iterates the ndarray *view*, so `np.arange(6.0)
  [::2]` yields `[0., 2., 4.]` rather than the buffer's first three
  elements. Worth a test of its own: rewriting the array path to read a
  raw pointer would pass every other assertion in the module.

## Decisions and Implementation Notes

- **One module, `test/test_core_dispatch.py`, at the top of `test/`.**
  It sits beside `test_theory_aggregation.py` — the other gate that is a
  contract rather than stored data — and outside `test/parity/`, which is
  generated and platform-scoped. This module is neither: it is
  hand-written, and every assertion in it holds on any platform.
- **No `pytest.importorskip("hazma._core")`.** From Phase 02 on the
  extension is in every build, so a missing `_core` is a build failure
  and must fail loudly. A skip here would be `docs/agents/lessons.md`
  `[gate-disabled-stays-green]` in the one module whose whole job is to
  notice that the cross-language path is broken.
- **The module pins the *target* contract, and says so at each
  divergence.** Two of the reference's four measured Cython/contract
  divergences surface at this layer — a 0-d array (Cython raises,
  `map_unary` returns a float) and a Python list (Cython accepts,
  `map_unary` raises). Both are asserted as the scaffold behaves, with a
  comment at the assertion naming Task 3.5 as the decision point. The
  alternative — leaving them untested because they are unsettled — would
  make the eventual Task 3.5 change invisible.
- **It does not re-pin any value against Cython.** That is
  `test/parity/`'s job at bit-equality across all 41 entry points; a
  second, looser numerical gate here would only be one more thing to keep
  in sync. The module's docstring says this explicitly so a copier does
  not add one.
- **The `text_signature` fix shipped in this task rather than as a
  follow-up.** It is one line in the file whose contract this task
  exists to pin, the test that catches it is one of the module's own, and
  deferring it would have meant a template that advertises a signature
  narrower than the API it replaces. `rust/src/lib.rs` carries the
  reasoning in a doc comment so a future reader does not "restore" the
  `/`.
- **Copy instructions are in the module docstring, not in a project
  doc.** A template whose usage notes live somewhere else stops being
  copied correctly on the first swap; the file names what to change
  (`roundtrip` → the kernel, `QUANTITY` → the kernel's wording) and what
  not to (do not merge plumbing and physics assertions).

## Files Changed

- `test/test_core_dispatch.py` — **new**, 54 tests. Six classes:
  `TestScalarPath` (float / NumPy-scalar / int / bool in, `float` out,
  bit-for-bit over ten special values), `TestZeroDimensionalArray`,
  `TestArrayPath` (dtype, shape, values, empty, non-contiguous,
  read-only, non-aliasing), `TestErrorPaths` (2-D/3-D/Fortran-order/(1,1),
  five wrong dtypes, rank-before-dtype, five non-numeric types, the list
  narrowing, and the quantity-name prefix), `TestSignature`.
- `rust/src/lib.rs` — `roundtrip`'s `text_signature` `"(x, /)"` → `"(x)"`,
  plus the doc comment recording why. No executable change.
- `projects/cython-to-rust/task-notes/phase-02/task-2.3-plumbing-test.md`
  — this note.
- `projects/cython-to-rust/task-notes/phase-02/README.md` — Task 2.3 row
  → Complete, findings, files, verification, handoff; phase closure.
- `projects/cython-to-rust/task-notes/README.md` — Phase 02 row →
  Complete, findings, numerical-impact entry, files, handoff.
- `projects/cython-to-rust/phases/phase-02-rust-scaffold.md` —
  frontmatter `status: Not started` → `Complete`; Task 2.3 exit criteria
  widened to name the `text_signature` fix (see §Plan Impact).
- `projects/cython-to-rust/PLAN.md` — Phases table row 02.
- `projects/cython-to-rust/learnings/phase-02-rust-scaffold.md` — **new**,
  phase closure.

## Verification

**Environment.** The tree was cleaned (`find hazma -name '*.c' -o -name
'*.so' | xargs rm -f`, 40 files) and rebuilt with `uv pip install -e .`
before anything was run, and again after the `lib.rs` edit — `cargo
build` publishes nothing to Python (Task 2.2). Confirmed each time:
`python -c "import hazma._core; print(hazma._core.__file__)"` →
`<worktree>/hazma/_core.abi3.so`, 21 `.so` in the tree.

**The new module.**

```text
.venv/bin/python -m pytest test/test_core_dispatch.py -q
54 passed in 0.27s
```

What the 54 cover, by class:

| Class | Tests | Covers |
| --- | --- | --- |
| `TestScalarPath` | 17 | `float` (10 special values, bit-for-bit), `np.float64` fast path, four other NumPy scalar types, `int`, `bool` |
| `TestZeroDimensionalArray` | 11 | 0-d `float64` → `float` over the same ten values; 0-d wrong dtype rejected |
| `TestArrayPath` | 6 | dtype/shape/values, special values, empty, non-contiguous, read-only, non-aliasing |
| `TestErrorPaths` | 19 | 4 rank errors, 5 dtype errors, rank-before-dtype, 5 non-numeric, the `list` narrowing, 3 quantity-prefix checks |
| `TestSignature` | 1 | `inspect.signature` is `(x)`; the keyword call works |

**Full suite** (bare `pytest`, the gate CI runs):

```text
.venv/bin/python -m pytest -q
1063 passed, 13 skipped, 5 warnings in 552.68s (0:09:12)
```

1076 collected, +54 on Task 2.2's 1022. The skip count is unchanged at
13, which is what shows the parity suite is still in **bit-equality
mode** (forcing budget mode turns
`test_running_on_the_capturing_tree` into a skip — see
`test/parity/README.md`).

**Test validity — mutation campaign.** Six mutations of the production
code, each applied, rebuilt with `uv pip install -e .`, run, and
reverted. Every one is caught, and by the tests whose names claim it:

| Mutation | Result |
| --- | --- |
| M1 `text_signature` back to `"(x, /)"` | 1 failed — `TestSignature` |
| M2 array path returns the input object instead of a fresh array | 2 failed — `test_result_is_a_fresh_array_that_does_not_alias_the_input`, `test_non_contiguous_input_is_read_in_order` |
| M3 dtype checked before rank | 6 failed — all four rank cases, `test_rank_is_checked_before_dtype`, the dimension quantity-prefix case |
| M4 `{quantity}` dropped from the dimension message | 2 failed — `test_rank_is_checked_before_dtype`, quantity-prefix `[dimension]` |
| M5 0-d array rejected instead of taking the scalar path | 10 failed — all of `TestZeroDimensionalArray`'s float cases |
| M6 array path reads the raw buffer instead of the view | 1 failed — `test_non_contiguous_input_is_read_in_order` |

M1 is the stash-proof check for this task's one production change: the
signature test fails with the fix reverted, passes with it.

**Cargo gates**, from the repo root:

```text
cargo fmt --manifest-path rust/Cargo.toml --check                       clean
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings   clean
cargo test --manifest-path rust/Cargo.toml --no-default-features        2 passed
```

**Preflight:** `scripts/agents/preflight.sh --paths
"test/test_core_dispatch.py" --md "<the 6 changed docs>"` → **RESULT:
PASS**, all eleven rows (`version bump` SKIP — not a closing PR).

One invocation trap, hit on the first run and worth carrying: **preflight
resolves every Python tool from `PATH`, and this worktree's `.venv` is
not on it by default** (fish). A bare run reports `FAIL black not
installed`, `WARN isort`, `WARN ruff`, `FAIL pytest` — four rows that say
*missing tool*, not *red gate*, and one of them (`WARN`) is a hole rather
than a pass. Run it as
`env PATH="$PWD/.venv/bin:$PATH" scripts/agents/preflight.sh …`. Note
also that the `import hazma` row passes either way, because Python puts
the repo root on `sys.path` when preflight runs from there — it is not
evidence the venv was used.

**Deferred:** nothing. The exit criteria name array *ownership*, which is
covered as non-aliasing rather than `owndata` — see §Findings for why
`owndata` is `False` on correct code.

## Open Questions

- None opened by this task. The two divergences its assertions pin
  (0-d array, Python list) are Task 3.5's to decide, already tracked in
  `../../references/numerics-replacements.md` and in this phase's
  README; this task adds a failing test to whichever way 3.5 goes, which
  is the outcome that makes the decision visible.

## Plan Impact

**Impact Level:** Update phase file.

Two canonical edits, both in
`../../phases/phase-02-rust-scaffold.md`:

1. **Task 2.3's exit criteria gained a third bullet** naming the
   `text_signature` fix. The task was widened past the plan's wording —
   the plan asked for tests only — so the plan now says so, the same way
   Task 2.1 and Task 2.2 recorded their widenings. The reason it is
   canonical rather than a note: the criterion "the same test module is
   the template later kernel swaps copy" is not satisfiable by a template
   whose probe advertises a narrower signature than the Cython it
   replaces.
2. **The phase's frontmatter `status:` → `Complete`**, with the phase
   learnings written and `PLAN.md`'s Phases row updated. Task 2.3 is the
   last task in Phase 02.

No ADR. Nothing here revises ADR-0001, and the dispatch contract itself
is unchanged — this task pins it, it does not decide it.

## Stale-state sweep

Commands run against this branch
(`claude/cython-to-rust/task-2.3-cross-language-plumbing-test`).

| Check | Command | Result |
| --- | --- | --- |
| Branch / worktree | `git rev-parse --abbrev-ref HEAD`; `--show-toplevel` | `claude/cython-to-rust/task-2.3-cross-language-plumbing-test` (**not** `master`); `…/.claude/worktrees/cython-to-rust-next-99fe4c` |
| Full change inventory | `git status --short`; `git diff origin/master --stat` | 8 files, `+1025 / −37`; 3 added, 5 modified. Nothing untracked left unaccounted (`git add -A` run before the check, so `??` cannot hide a file) |
| Numerical-impact statement | `git diff origin/master -- hazma \| wc -l` | `0` — see below |
| Rust diff is non-executable | `git diff origin/master -- rust` | one hunk in `src/lib.rs`: `text_signature "(x, /)"` → `"(x)"` plus a 6-line doc comment. No statement changed |
| Task 2.3 status siblings | `rg -n "2\.3" projects/cython-to-rust --glob '*.md'` | 20 hits. The four status-bearing ones — phase file §Task 2.3, `phase-02/README.md` table row, its Files/Verification sections, this note's header — all read `Complete (2026-08-09)`. The rest are prose or links |
| Phase-02 status siblings | `rg -n "^status:" phases/phase-02-rust-scaffold.md`; `rg -n "phase-02" PLAN.md task-notes/README.md` | `status: Complete`; `PLAN.md` Phases row and `task-notes/README.md` Phases row both `**Complete (2026-08-09)**` |
| Closure is machine-readable | `scripts/agents/resolve_task.py --project cython-to-rust` | `{"status": "ready", "task_id": "3.1", …, "phase": "03"}` — the resolver has moved off Phase 02, which is the check that the table edits actually parse |
| Stale "2.3 open / not started" | `rg -n "2\.3.*(Not started\|open)" projects/` | 2 hits, both intentional: this sweep row itself, and the struck-through `~~Task 2.3 is the last open task~~` bullet in `phase-02/README.md` |
| Test-count claims | `rg -n "1009 passed\|1022 collected\|1006 passed" projects/cython-to-rust --glob '*.md'` | 20 hits, all inside dated Task 1.4 / 2.1 / 2.2 records. This task's counts are **new rows**, not edits to those — the two live roll-ups (`task-notes/README.md` "suites are merged" bullet, `phase-02/README.md` Verification) were updated to 1063/13 |
| Collected count | `pytest --collect-only -q \| tail -1` | `1076 tests collected` (= 1063 + 13, and +54 on Task 2.2's 1022) |
| Per-class test counts | `pytest test/test_core_dispatch.py --collect-only -q` | 17 / 11 / 6 / 19 / 1 = 54, matching the §Verification table row for row |
| Debug leftovers | `rg -n "TODO\|FIXME\|breakpoint\(\)\|pdb\|print\(" test/test_core_dispatch.py rust/src/lib.rs` | no occurrences |
| Markdown + citations | `scripts/agents/preflight.sh --md "<6 changed docs>"` | `PASS markdownlint` |
| Full gate | `scripts/agents/preflight.sh --paths … --md …` | **RESULT: PASS**, all eleven rows (`version bump` SKIP — not a closing PR) |

**Numerical-impact statement.** No public value changes (verified:
`git diff origin/master -- hazma` is empty — 0 lines). The only
non-documentation change under `rust/` is `roundtrip`'s advertised
signature string; `roundtrip` is the scaffold probe and nothing under
`hazma/` imports it. The stronger statement, from the bare suite above:
the parity corpus ran in **bit-equality mode** — `rtol = 0` across all 41
consumed entry points, 179,695 pinned values — and passed. No ad-hoc grid
sweep is reported because the corpus is a stricter grid than any of them.

## Handoff to Next Task

**Phase 02 is Complete.** The next task is **Phase 03, Task 3.1**.

- Read `../../learnings/phase-02-rust-scaffold.md` rather than this
  phase's three task notes — the learnings are the distillation, the
  notes are history.
- **`test/test_core_dispatch.py` is the template.** Copy it per kernel:
  swap `roundtrip` for the kernel and `QUANTITY` for the wording that
  kernel passes to `map_unary`, keep every test, and add the kernel's
  numerical tests *beside* them rather than merged into them. The
  module's docstring carries these instructions so they travel with the
  file.
- **The contract is now pinned from the Python side, including three
  things the reference prose did not state**: rank is checked before
  dtype, a 0-d array still enforces dtype where a Python `int` does not,
  and non-`float` NumPy scalars are accepted. A Task 3.5 decision that
  changes any of them will turn a named test red rather than passing
  unnoticed.
- **Do not assert `owndata` on a returned array.** It is `False` on
  correct code — the `numpy` crate's `Vec` wrapper owns the buffer.
  Non-aliasing is the assertable property.
- Still risky / unknown, unchanged by this task: `spec_math`'s `li2`
  convention (Task 3.2 pins it), the ill-conditioned-points corpus
  repair, which must land **before** the first Phase 04 swap flips the
  parity gate out of bit-equality mode permanently, and the four
  Cython/contract dispatch divergences Task 3.5 must decide.
