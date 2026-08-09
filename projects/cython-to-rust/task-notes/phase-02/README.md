# Working Memory: Phase 02 — Rust scaffold

**Date:** 2026-08-03 (created)
**Project:** cython-to-rust
**Phase:** 02
**Status:** In Progress (Task 2.1 complete 2026-08-08)
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
| 2.2 | CI, preflight, dev-loop docs | 2.1 | Not started | [task-2.2-ci-devloop.md](task-2.2-ci-devloop.md) |
| 2.3 | Cross-language plumbing test | 2.1 | Not started | [task-2.3-plumbing-test.md](task-2.3-plumbing-test.md) |

## Exit Criteria

- All rows Complete; phase file frontmatter `status: Complete`.
- Phase learnings at `../../learnings/phase-02-rust-scaffold.md`.

## Inputs Reviewed

- `../../phases/phase-02-rust-scaffold.md`; `../README.md`;
  `../../rules.md` rules 6–8.

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
  installed. The wheel stays CPython-tagged
  (`hazma-2.1.0-cp313-cp313-macosx_11_0_arm64.whl`), which is correct
  while Cython extensions remain (`lessons.md`
  `[wheel-tag-vs-extension-abi]`).
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

## Verification

- Per-task verification lives in each task note. Task 2.1's closing
  state, on the corpus's capturing interpreter (CPython 3.12.12,
  macOS/arm64): bare `pytest -q` green with the parity suite in
  **bit-equality mode** (`test_running_on_the_capturing_tree` passes
  rather than skips), `cargo test --no-default-features` green,
  `cargo fmt --check` and `cargo clippy -- -D warnings` clean, and the
  sdist installs and runs both toolchains in a fresh venv from outside
  the repo. Numbers in the task note.

## Open Questions

- **CI has no Rust toolchain step, and passes anyway on today's runner
  images** (Task 2.2 still owns pinning it). Measured on PR #55: all
  seven checks green first try, hybrid build included, on every matrix
  entry — ubuntu py3.10–3.14 in 16m29s–19m59s and macos py3.14 in
  16m49s. So the GitHub-hosted images ship a usable cargo and
  setuptools-rust finds it unconfigured. **Nothing in the repo pins
  that**, and an image refresh that dropped Rust would take the whole
  matrix down at once, which is the argument for Task 2.2's explicit
  step rather than against it. Untested either way: `release.yml`'s
  cibuildwheel job, which does not run on pull requests and *will* need
  a toolchain inside the manylinux container.
- ~~setuptools-rust + editable-install rebuild ergonomics under uv~~ —
  **answered by Task 2.1**: `uv pip install -e .` builds Cython and Rust
  in one pass with no extra flags, and re-running it after a `.rs` edit
  is what publishes the change. `cargo build` alone updates
  `rust/target/`, not the installed `hazma/_core.abi3.so`. Task 2.2
  writes this into `docs/agents/` and `AGENTS.md`.

## Plan Impact

**Impact Level:** None (this file is metadata, not a canonical change).
Task 2.1's own canonical patches are recorded in its task note.

## Handoff to Next Task

**For the next agent working in Phase 02:** read `../../PLAN.md`,
`../README.md`, this file, then the phase file. The extension's import
path is final from day one: `hazma._core`.

**Currently safe to assume:**

- The crate builds, imports, and ships. `uv pip install -e .` produces
  `hazma/_core.abi3.so` beside the 20 Cython extensions (21 `.so` in the
  tree), `import hazma._core` works, and all five submodules are
  importable both as attributes and as `hazma._core.<name>`.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and
  `cargo test --no-default-features` are all green from `rust/`. Task 2.2
  wires exactly those three spellings into `preflight.sh` — note the
  `--no-default-features`, without which the test harness fails to link.
- `dispatch::map_unary` is the one implementation of the entry-point
  dispatch contract; Task 2.3's plumbing tests are written against
  `hazma._core.roundtrip`, which exercises every branch of it.
- The parity gate is in bit-equality mode again and stays there until a
  real kernel lands. Do not re-key it on `rust_core_available()`.

**Currently risky / unknown:**

- CI's Rust toolchain (see Open Questions) and the cibuildwheel job.
- The reference's dispatch contract now records four measured
  divergences from the live Cython. Task 2.3 asserts the *target*
  contract on `roundtrip`; Task 3.5 has to decide, per divergence,
  whether the ported entry points keep the Cython behavior or take the
  declared change.
