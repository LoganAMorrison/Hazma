# Working Memory: Phase 02 — Rust scaffold

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 02
**Status:** Complete (2026-08-09 — Tasks 2.1 and 2.2 on 2026-08-08,
Task 2.3 on 2026-08-09;
[learnings](../../learnings/phase-02-rust-scaffold.md))
**Plan References:** `../../phases/phase-02-rust-scaffold.md`
**Related ADRs:** ADR-0001 (accepted)
**Depends On:** Phase 01 complete

## Objective

Track live per-task status and phase-scoped findings for the Rust
scaffold.

## Tasks

| # | Task | Depends on | Status | Task Note |
| --- | ------ | ------------ | -------- | ----------- |
| 2.1 | Crate + setuptools-rust integration | — | **Complete (2026-08-08)** | [task-2.1-crate-skeleton.md](task-2.1-crate-skeleton.md) |
| 2.2 | CI, preflight, dev-loop docs | 2.1 | **Complete (2026-08-08)** | [task-2.2-ci-devloop.md](task-2.2-ci-devloop.md) |
| 2.3 | Cross-language plumbing test | 2.1 | **Complete (2026-08-09)** | [task-2.3-plumbing-test.md](task-2.3-plumbing-test.md) |

## Exit Criteria

- ~~All rows Complete; phase file frontmatter `status: Complete`.~~ —
  **met 2026-08-09.**
- ~~Phase learnings at `../../learnings/phase-02-rust-scaffold.md`.~~ —
  **written 2026-08-09.**

## Inputs Reviewed

- `../../phases/phase-02-rust-scaffold.md`; `../README.md`;
  `../../rules.md` rules 6–8 (Rust conventions 1–3; see that file's
  numbering key).

## Findings

- **The scaffold's mere existence switched the parity gate off** (Task
  2.1). `tolerances.provenance` keyed on
  `find_spec("hazma._core") is not None`, which is true in every build
  from this phase on while every value still comes from Cython, so
  Phases 02–03 would have run against 1e-8 budgets instead of
  bit-equality. The predicate now asks whether a kernel is *served*
  (`cases.rust_core_kernels()`), which is what `rules.md` rule 2 says.
  Same change unblocks the corpus repair: `assert_no_rust_core` no
  longer refuses on a bare scaffold. **From the first Phase 04 swap both
  flip, permanently** — so the ill-conditioned-points repair has to land
  before that swap, not after.
- **PyO3 0.29.2 renamed `Bound::downcast` to `Bound::cast`.** Every
  tutorial still says `downcast`.
- **`cargo build` needs three things setuptools-rust supplies for you:**
  an explicit empty `[workspace]` in `rust/Cargo.toml` (a stray
  `Cargo.toml` anywhere above the checkout otherwise captures the crate
  — a home-directory one did), macOS's `-undefined dynamic_lookup`
  (emitted from `rust/build.rs` as a cdylib-only link arg; PyO3 0.29 does
  not add it), and `extension-module` *off* for `cargo test` (it is a
  default-on optional feature; tests run
  `cargo test --no-default-features`).
- **abi3 is verifiable without CI.** The installed file is
  `hazma/_core.abi3.so`, and the exact file built under CPython 3.12.12
  loads and runs under 3.13.7 — in an interpreter with no NumPy
  installed. The wheel stays CPython-tagged, which is correct while
  Cython extensions remain (`lessons.md`
  `[wheel-tag-vs-extension-abi]`) — and the tag names the *building
  interpreter*, so it is `cp312` from the 3.12.12 venv this task used and
  `cp313` from a 3.13.7 one, with `_core.abi3.so` inside either way. The
  invariant is "`cp<XY>`, not `abi3`"; do not quote one build's tag as
  though it were fixed.
- **The `numpy` crate panics, it does not raise, when NumPy is absent.**
  `cast::<PyUntypedArray>` reaches for the array-API capsule; without
  NumPy that is a `PanicException` across the FFI boundary. Found by the
  abi3 probe above. The dispatch helper now takes a `PyFloat` fast path
  first, so a scalar call never touches NumPy. hazma depends on NumPy at
  runtime, so this was latent rather than live — but every Phase 03–06
  kernel inherits the ordering.
- **The live Cython dispatch is not the contract the reference
  described** — 0-d arrays raise instead of taking the scalar path, lists
  are accepted, shape errors are `AssertionError`, and one neutrino
  module's message says "Photon energies". Measured and written into
  `../../references/numerics-replacements.md`; Task 3.5 implements from
  that section and two of the four are public-API narrowings if taken
  silently.
- **The macOS wheels are built on the runner; the Linux wheels are not**
  (Task 2.2). cibuildwheel runs the Linux builds inside a manylinux
  container that cannot see a toolchain installed on the host, so a
  `dtolnay/rust-toolchain` step in `release.yml` serves macOS *only* and
  Linux needs `CIBW_BEFORE_ALL_LINUX` (rustup inside the container) plus
  `CIBW_ENVIRONMENT_LINUX` to put it on `PATH`. Both are in the job now.
  This is the same shape as the `MANIFEST.in`-vs-wheel lesson from
  Task 0.4: two artifacts, two mechanisms, and fixing one has never
  fixed the other.
- **`release.yml` does not run on pull requests**, so nothing about it
  is verifiable from a PR check (Task 2.2). Its `build-wheels` job fires
  on `release: published` and `workflow_dispatch` only. That is why the
  "each wheel contains `hazma/_core.abi3.so`" criterion became a **step
  inside the job** rather than a one-time manual inspection — and why
  the step fails on an empty `wheelhouse/` as well as on a missing
  `.so`. A check that can verify nothing and still report success is
  `docs/agents/lessons.md` `[gate-disabled-stays-green]` waiting to
  happen. **Confirming the job itself needs a manual dispatch, and no
  amount of local work substitutes** — round 1 of PR #56's review made
  exactly that call. `gh workflow run release.yml --ref <branch>` is the
  command; verify the publishing job is gated on
  `github.event_name == 'release'` first, so the dispatch builds without
  publishing. Measured cost: ~17 min for both platforms.
- **`cargo test` and `cargo build` are not rebuilds** (Task 2.2, now
  written into `AGENTS.md` and `docs/agents/environment.md`). They work
  out of `rust/target/`, which nothing Python imports; only
  `pip install -e .` re-links the crate into the tree as
  `hazma/_core.abi3.so`. Measured:
  `python -c "import hazma._core; print(hazma._core.__file__)"` resolves
  to `<worktree>/hazma/_core.abi3.so`. The dev loop that follows is
  cargo for iteration, editable install before any Python-side claim.
- **Only one skill file is markdownlint-red, and it was already**
  (Task 2.2). Of the seven skill files this task edited,
  `.claude/skills/task-pipeline/SKILL.md` reports 7 errors (MD036,
  MD032, MD031) on **both** `origin/master` and this branch; the other
  six report 0. That is
  [`../../../../docs/followups/todo/markdownlint-skips-skill-file-shapes.md`](../../../../docs/followups/todo/markdownlint-skips-skill-file-shapes.md),
  not this diff — so `--md` gets the six clean ones and the count
  comparison above is the evidence, rather than the file being quietly
  dropped.
- **Three dispatch behaviors the reference prose did not state**, all
  measured against the built extension in Task 2.3 and now pinned by
  `test/test_core_dispatch.py`: (a) **rank is checked before dtype**, so
  a 2-D int64 array reports `must be 0 or 1-dimensional.`; (b) **a 0-d
  array still enforces dtype** where a Python `int` does not — the 0-d
  path lives inside the array branch behind the typed view, so
  `roundtrip(4)` is `4.0` and `roundtrip(np.array(4))` is a `ValueError`;
  (c) **non-`float` NumPy scalars are accepted** (`np.float32`,
  `np.int64`, `np.uint8`, `np.bool_`) via the `extract::<f64>` arm. A
  Task 3.5 decision that changes any of them now turns a named test red.
  **It did, and it did (2026-08-11):** Task 3.5 reversed (b) —
  `roundtrip(np.array(4))` is now `4.0`, because a 0-d array *is* a
  scalar and only a non-numeric dtype is rejected there — and the named
  test turned red exactly as this bullet predicted. (a) and (c) stand.
  See [`../../learnings/phase-03-numerics-foundation.md`](../../learnings/phase-03-numerics-foundation.md).
- **`text_signature` is a claim PyO3 does not enforce** (Task 2.3).
  `roundtrip` advertised `(x, /)` while `roundtrip(x=1.5)` worked;
  enforcing positional-only needs `#[pyo3(signature = (x, /))]`. The
  Cython entry points are `def` functions that take keywords — measured,
  `dnde_photon(egam=100.0, emu=200.0)` returns
  `2.0036713127483527e-05` — so keyword-accepting is the target and the
  `/` was a latent public-API narrowing waiting to be copied into every
  Phase 04–06 wrapper. Fixed to `"(x)"`, which is what `hazma/_core.pyi`
  already described.
- **A "fresh" array from the `numpy` crate does not own its data** (Task
  2.3). `roundtrip(a).flags.owndata` is `False` and `.base` is a
  `PySliceContainer` wrapping the Rust `Vec`. Assert non-aliasing
  (`is not`, `.base is not`, `np.shares_memory`, mutate-and-check) rather
  than `owndata`, which is red on correct code.

## Decisions and Implementation Notes

- Crate layout follows `../../rules.md` Rust rule 3: `kernels.rs` is
  PyO3-free and `cargo test`-able, `dispatch.rs` is the single PyO3
  boundary, the five submodule files are registration only.
- Submodules are registered in `sys.modules` under their fully-qualified
  names, so both `from hazma._core import photon` and
  `from hazma._core.photon import <kernel>` work for Phase 04 wrappers.
- `roundtrip` is the identity and allocates a fresh array: the value is
  preserved so plumbing tests need no tolerance, and the fresh
  allocation is what proves the Rust actually ran.
- `Cargo.lock` is committed and shipped in the sdist; `MANIFEST.in`
  carries the crate because `global-include` covers no Rust pattern.
- **Task 2.2 widened its first two exit criteria, deliberately, and
  patched the phase file to say so.** (a) The three cargo gates run in
  CI as a `rust` job, not only in `preflight.sh` — a gate that lives
  only in local discipline enforces nothing, and Phases 03–06 put the
  entire numerics layer behind it. (b) The wheel/abi3 criterion is a
  failing job step rather than an eyeball, because `release.yml` never
  runs on a PR. Both are additions to what the criteria asked for, not
  reinterpretations of them.
- The cargo gates sit **before** pytest in `preflight.sh`: they cost
  seconds and the bare suite costs minutes, so a rustfmt diff should not
  wait behind the parity corpus. They ignore `--paths` (the crate is
  small and always checked whole) and use `--manifest-path` rather than
  `cd rust`, which would leak into the gates after it — the script
  anchors everything to `REPO_ROOT` on purpose.
- Absence rules for the cargo gates: no `rust/` → SKIP (branches cut
  before this phase must still preflight), `rust/` present but no
  `cargo` → WARN, matching how a missing isort is already reported.
  Both branches were executed, not reasoned about — see Verification.
- `cargo_gate()` reads its status with `if capture ...; then` rather
  than the `$?` idiom its four Python-gate siblings use. Not a
  divergence worth undoing: shellcheck flags the older form (SC2181,
  style) and the counts are 5 on `origin/master` and 5 here, so the new
  helper simply did not add a sixth.

## Files Changed

### Task 2.1

New `rust/` crate (`Cargo.toml`, `Cargo.lock`, `build.rs`, and
`src/{lib,dispatch,kernels,photon,positron,neutrino,scalar_mediator,vector_mediator}.rs`);
build wiring in `setup.py`, `pyproject.toml`, `MANIFEST.in`,
`.gitignore`; the `hazma/_core.pyi` stub; the served-kernel predicate in
`test/parity/{cases,tolerances,test_parity,README}.py|md`; and the two
canonical doc patches (`../../phases/phase-02-rust-scaffold.md`,
`../../references/numerics-replacements.md`). Full list in
[task-2.1-crate-skeleton.md](task-2.1-crate-skeleton.md).

### Task 2.2

CI (`.github/workflows/ci.yml`: a `rust` job plus a toolchain step in the
test matrix; `.github/workflows/release.yml`: a host toolchain for macOS,
rustup-in-container for Linux, `hazma._core` added to the wheel test
command, and the abi3 assertion step); the three cargo gates in
`scripts/agents/preflight.sh`; the dev-loop documentation in `AGENTS.md`,
`docs/agents/environment.md` and `docs/agents/preflight.md`; the
rebuild-awareness sweep across `docs/agents/{review-lenses,README}.md`
and seven skill files; two canonical patches
(`../../phases/phase-02-rust-scaffold.md` Task 2.2 exit criteria,
`../../rules.md` Rust rule 1); and one struck-through risk bullet in
[task-2.1-crate-skeleton.md](task-2.1-crate-skeleton.md). 21 files: 20
modified, 1 added. Nothing under `hazma/` or `rust/`. Full list in
[task-2.2-ci-devloop.md](task-2.2-ci-devloop.md).

### Task 2.3

New `test/test_core_dispatch.py` (54 tests in six classes — the template
every Phase 04–06 kernel swap copies); a one-line non-executable change
in `rust/src/lib.rs` (`roundtrip`'s `text_signature` `"(x, /)"` → `"(x)"`
plus the doc comment recording why); the phase-closure bookkeeping
(`../../phases/phase-02-rust-scaffold.md` frontmatter and Task 2.3 exit
criteria, `../../PLAN.md`'s Phases row, `../README.md`, this file) and
`../../learnings/phase-02-rust-scaffold.md`. Nothing under `hazma/`.
Full list in [task-2.3-plumbing-test.md](task-2.3-plumbing-test.md).

## Verification

- Per-task verification lives in each task note. Task 2.1's closing
  state, on the corpus's capturing interpreter (CPython 3.12.12,
  macOS/arm64): bare `pytest -q` green with the parity suite in
  **bit-equality mode** (`test_running_on_the_capturing_tree` passes
  rather than skips), `cargo test --no-default-features` green,
  `cargo fmt --check` and `cargo clippy -- -D warnings` clean, and the
  sdist installs and runs both toolchains in a fresh venv from outside
  the repo. Numbers in the task note.
- **Task 2.2 state (2026-08-08)**, same interpreter, tree cleaned and
  rebuilt first: `scripts/agents/preflight.sh` **RESULT: PASS** across
  all eleven rows, including the three new `PASS … rust/` ones and
  `pytest` at `1009 passed, 13 skipped … 571.34s` — byte-identical to
  Task 2.1's, which is what shows no test outcome moved, with the parity
  suite still in bit-equality mode. All four branches of the new gate
  block (SKIP / WARN / FAIL / PASS) were forced and observed rather than
  reasoned about. The abi3 criterion was measured by running
  `release.yml`'s own assertion script against a real hybrid wheel
  (`cp312`-tagged, `hazma/_core.abi3.so` inside). **Review round 1 then
  ran `release.yml` for real** (dispatch → run 31297673951, `success`;
  `publish` skipped): 10 wheels across both platforms, each reported by
  the assertion step as carrying the extension. PR #56's eight checks
  are green, including the new `rust` job at 30s. Full tables in the
  task note.
- **Task 2.3 state (2026-08-09)**, same interpreter, tree cleaned (40
  stale `.c`/`.so`) and rebuilt first, and rebuilt again after the
  `lib.rs` edit: `test/test_core_dispatch.py` → `54 passed in 0.27s`;
  bare `pytest -q` → `1063 passed, 13 skipped, 5 warnings in 552.68s`
  (1076 collected, +54 on Task 2.2's 1022 — all of them the new module),
  parity suite still in **bit-equality mode** (skip count unchanged at
  13). The three cargo gates green. `scripts/agents/preflight.sh`
  RESULT: PASS. The 54 assertions were validated by a **six-mutation
  campaign** against `rust/src/{dispatch,lib}.rs` — text_signature
  reverted, array path returning the input object, dtype checked before
  rank, `{quantity}` dropped from a message, 0-d array rejected, array
  path reading the raw buffer instead of the view — each applied,
  rebuilt, run, reverted, and each caught by the test whose name claimed
  it. Full tables in the task note.

## Open Questions

- ~~**CI has no Rust toolchain step, and passes anyway on today's runner
  images**~~ — **closed by Task 2.2 (2026-08-08).** The measurement that
  opened it stands (PR #55: all seven checks green first try, hybrid
  build included, ubuntu py3.10–3.14 in 16m29s–19m59s and macos py3.14
  in 16m49s, with no toolchain step anywhere), and it was exactly the
  argument for pinning: the images happened to ship cargo, nothing in
  the repo required them to keep doing so, and an image refresh that
  dropped Rust would have taken every matrix entry down at once. Every
  entry now installs `dtolnay/rust-toolchain@stable` before building,
  `release.yml` gained the container-side toolchain the cibuildwheel job
  needs, and a third job runs the cargo gates.
- ~~**`release.yml` is still unexercised, and cannot be exercised by a
  PR.**~~ — **closed in PR #56's review round 1.** A reviewer refused a
  `Complete` status resting on an unrun exit criterion; the fix was to
  run it, not to soften it. `gh workflow run release.yml --ref <branch>`
  → run 31297673951, conclusion `success`: both `build-wheels` jobs and
  `build-sdist` green, `publish` **skipped** because it is gated on
  `github.event_name == 'release'` (so a dispatch is build-only — check
  that gate before dispatching anything). The assertion step printed
  `5 wheel(s) carry hazma/_core.abi3.so` on each platform: 10 wheels,
  `cp310`–`cp314` × {macOS arm64, manylinux_2_28 x86_64}, the unchanged
  CPython-tagged matrix. Phase 07 Task 7.1 rewrites this job for maturin
  and inherits the same obligation — a workflow without a
  pull-request trigger has to be dispatched deliberately or its criteria
  stay unmeasured (`docs/agents/lessons.md`
  `[unrun-workflow-cannot-close-a-criterion]`).
- ~~setuptools-rust + editable-install rebuild ergonomics under uv~~ —
  **answered by Task 2.1**: `uv pip install -e .` builds Cython and Rust
  in one pass with no extra flags, and re-running it after a `.rs` edit
  is what publishes the change. `cargo build` alone updates
  `rust/target/`, not the installed `hazma/_core.abi3.so`. Task 2.2
  writes this into `docs/agents/` and `AGENTS.md`.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).
Tasks 2.1's and 2.2's own canonical patches are recorded in their task
notes.

## Handoff to Next Task

**Phase 02 is Complete (2026-08-09).** The next task is **Phase 03,
Task 3.1**. Read
[`../../learnings/phase-02-rust-scaffold.md`](../../learnings/phase-02-rust-scaffold.md)
rather than this phase's three task notes — the learnings are the
distillation, the notes are history. The extension's import path was
final from day one: `hazma._core`.

**Currently safe to assume:**

- The crate builds, imports, and ships. `uv pip install -e .` produces
  `hazma/_core.abi3.so` beside the 20 Cython extensions (21 `.so` in the
  tree), `import hazma._core` works, and all five submodules are
  importable both as attributes and as `hazma._core.<name>`.
- **The three cargo gates now run themselves, in two places** (Task
  2.2). `scripts/agents/preflight.sh` runs them ahead of pytest, and
  CI's `rust` job runs the same three; the exact spellings are in the
  phase file's Task 2.2 exit criteria. Note `--no-default-features` on
  the test one, without which the harness fails to link, and
  `--manifest-path rust/Cargo.toml` so you can run them from the repo
  root.
- **`pip install -e .` is the only thing that republishes a `.rs` edit
  to Python.** `cargo build` and `cargo test` work out of
  `rust/target/`. Written into `AGENTS.md`,
  `docs/agents/environment.md`, `docs/agents/preflight.md`, and the
  rebuild-awareness bullet of every review skill — so a future reviewer
  is expected to challenge a cargo-only run cited as a Python result.
- `dispatch::map_unary` is the one implementation of the entry-point
  dispatch contract, and as of Task 2.3 **every branch of it is pinned
  from Python** by `test/test_core_dispatch.py` (54 tests, 0.27s,
  platform-independent), written against `hazma._core.roundtrip`.
  **Task 3.5 (2026-08-11) gave it two siblings** — `map_flavors` (the
  neutrino 3-tuple / `(3, N)` return) and `require_vector`
  (`partial_widths`) — over one shared classification, and grew the
  module to 118 tests.
- **That module is the template Phase 04–06 swaps copy.** Swap
  `roundtrip` for the kernel and `QUANTITY` for the wording that kernel
  passes to `map_unary`, keep every test, and add the kernel's numerical
  tests *beside* them rather than merged in. The copy instructions are in
  the module docstring so they travel with the file. It deliberately does
  not re-pin values against Cython — that is the corpus's job.
- **Do not assert `owndata` on a returned array**; it is `False` on
  correct code. Non-aliasing is the assertable property.
- The parity gate is in bit-equality mode again and stays there until a
  real kernel lands. Do not re-key it on `rust_core_available()`.

**Currently risky / unknown:**

- ~~`release.yml`: wired, locally evidenced, never run.~~ — dispatched
  and green in PR #56's review round 1 (see Open Questions). It stays
  invisible to PR checks, so any *future* change to it needs its own
  dispatch.
- The reference's dispatch contract records four measured divergences
  from the live Cython. Task 2.3 asserted the *target* contract on
  `roundtrip`, with a comment at each of the two that surface at this
  layer (0-d array, Python list) naming Task 3.5 as the decision point.
  **Task 3.5 still has to decide, per divergence**, whether the ported
  entry points keep the Cython behavior or take the declared change —
  what changed is that either answer now moves a named test rather than
  passing unnoticed.
- ~~**Task 2.3 is the last open task in this phase.**~~ — **closed
  2026-08-09.** No task in Phase 02 remains open. The obligation every
  later task inherits is unchanged: a `.rs` edit means an editable
  reinstall before any pytest number is quotable.
