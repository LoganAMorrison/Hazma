---
phase: 02
title: Rust scaffold
status: Not started
---

# Phase 02: Rust scaffold

## Goal

A building, testing, CI-green Rust extension skeleton at its final
import path — `hazma._core` — coexisting with the remaining Cython
under the setuptools backend via setuptools-rust (ADR-0001). No physics
yet: the walking skeleton proves the toolchain end to end.

## Prerequisites

- Phase 01 complete (the corpus exists before any Rust lands).
- ADR-0001 (accepted); `../rules.md` rules 6–8.

## Tasks

### Task 2.1: Crate + setuptools-rust integration

**Task note:** [`../task-notes/phase-02/task-2.1-crate-skeleton.md`](../task-notes/phase-02/task-2.1-crate-skeleton.md)
**Depends on:** —

**Exit criteria:**

- `rust/Cargo.toml` (edition 2024) with `pyo3` (`abi3-py310`,
  `extension-module`) and `numpy` crates; `rust/src/lib.rs` defines
  `#[pymodule] _core` with empty submodules (`photon`, `positron`,
  `neutrino`, `scalar_mediator`, `vector_mediator`) plus one trivial
  round-trip function (array in → array out) for plumbing tests.
- `setup.py` gains `RustExtension("hazma._core", ...)`;
  `pip install -e .` (uv env) builds Cython + Rust in one pass;
  `python -c "import hazma._core"` works.
- `hazma/_core.pyi` stub started; py.typed unaffected.

### Task 2.2: CI, preflight, and dev-loop documentation

**Task note:** [`../task-notes/phase-02/task-2.2-ci-devloop.md`](../task-notes/phase-02/task-2.2-ci-devloop.md)
**Depends on:** Task 2.1

**Exit criteria:**

- CI installs the Rust toolchain on both OS matrices; full matrix
  green; wheel-build job (release.yml) still succeeds with the hybrid
  build. **Hybrid wheels stay CPython-version-tagged** (the Cython
  extensions force that; the 10-wheel matrix is unchanged until
  Phase 07) — what is verified here is extension-level only: each
  wheel contains `hazma/_core.abi3.so`, i.e. the Rust extension is
  built against the limited API (`abi3-py310`). Distribution-level
  abi3 wheel tags and the 2-wheel matrix are asserted in Phase 07,
  never earlier.
- `scripts/agents/preflight.sh` grows `cargo fmt --check`,
  `cargo clippy -- -D warnings`, `cargo test` (skipped gracefully when
  `rust/` absent, so pre-Phase-02 branches still preflight).
- `docs/agents/` env notes + `AGENTS.md` Commands section document the
  rebuild loop (when a `.rs` edit requires re-running the editable
  install vs. plain `cargo test`).

### Task 2.3: Cross-language plumbing test

**Task note:** [`../task-notes/phase-02/task-2.3-plumbing-test.md`](../task-notes/phase-02/task-2.3-plumbing-test.md)
**Depends on:** Task 2.1

**Exit criteria:**

- pytest exercises the round-trip function for: float in/float out,
  1-D array in/out (dtype, shape, ownership), 0-d array, wrong-dtype
  and 2-D error paths raising `ValueError` with the contract message
  (see `../references/numerics-replacements.md`, dispatch contract).
- The same test module is the template later kernel swaps copy.

## Exit Criteria

- All tasks complete; CI + preflight green; hybrid (CPython-tagged)
  wheels build on both platforms, each containing an
  `hazma/_core.abi3.so` built against the limited API.
- No public API change; `hazma._core` exists but nothing imports it yet.
- Phase learnings written to `../learnings/phase-02-rust-scaffold.md`.
