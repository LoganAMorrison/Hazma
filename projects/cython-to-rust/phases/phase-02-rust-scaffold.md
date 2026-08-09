---
phase: 02
title: Rust scaffold
status: Complete
---

# Phase 02: Rust scaffold

## Goal

A building, testing, CI-green Rust extension skeleton at its final
import path — `hazma._core` — coexisting with the remaining Cython
under the setuptools backend via setuptools-rust (ADR-0001). No physics
yet: the walking skeleton proves the toolchain end to end.

## Prerequisites

- Phase 01 complete (the corpus exists before any Rust lands).
- ADR-0001 (accepted); `../rules.md` rules 6–8 (Rust conventions 1–3:
  edition 2024 and the cargo gates; one cdylib named `hazma._core` with
  per-domain submodules; kernels are PyO3-free). Rule 9 (edge guards)
  binds the porting phases, not the scaffold. See that file's numbering
  key — the flat and per-section schemes are both in use.

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
- **The parity gate still runs in bit-equality mode.** Added by Task 2.1
  on 2026-08-08, because the task's own deliverable would otherwise
  silently switch it off: `test/parity/tolerances.provenance` counted
  "`hazma._core` is importable" as a divergence, which is true from the
  moment the scaffold exists and false as a statement about the values —
  every kernel still runs on Cython through Phase 03. The predicate now
  asks whether `hazma._core` *serves* a kernel
  (`cases.rust_core_kernels`), which is what `rules.md` rule 2 says, and
  `assert_no_rust_core` keys on the same thing so the corpus-repair
  follow-up is not blocked by a scaffold.

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
  `rust/` absent, so pre-Phase-02 branches still preflight). The
  spellings that actually run — added by Task 2.2 on 2026-08-08, because
  two of the three carry a load-bearing flag — are
  `cargo fmt --manifest-path rust/Cargo.toml --check`,
  `cargo clippy --manifest-path rust/Cargo.toml --all-targets --
  -D warnings`, and
  `cargo test --manifest-path rust/Cargo.toml --no-default-features`.
- `docs/agents/` env notes + `AGENTS.md` Commands section document the
  rebuild loop (when a `.rs` edit requires re-running the editable
  install vs. plain `cargo test`).
- **The cargo gates also run in CI, and the wheel assertion is a job
  step rather than an eyeball.** Both added by Task 2.2 on 2026-08-08,
  widening the first two bullets rather than reinterpreting them.
  Reasons, in order: (1) `preflight.sh` is local discipline that nothing
  enforces, and Phases 03–06 land the whole numerics layer in Rust, so
  the gates belong somewhere that fails a PR — `ci.yml` gains a `rust`
  job running the same three commands; (2) `release.yml` does not run on
  pull requests, so "each wheel contains `hazma/_core.abi3.so`" could
  otherwise only ever be checked by hand at release time — it is now a
  step in `build-wheels` that fails the job, and it also fails when
  `wheelhouse/` is empty, since a check that verifies nothing must not
  look green.

### Task 2.3: Cross-language plumbing test

**Task note:** [`../task-notes/phase-02/task-2.3-plumbing-test.md`](../task-notes/phase-02/task-2.3-plumbing-test.md)
**Depends on:** Task 2.1

**Exit criteria:**

- pytest exercises the round-trip function for: float in/float out,
  1-D array in/out (dtype, shape, ownership), 0-d array, wrong-dtype
  and 2-D error paths raising `ValueError` with the contract message
  (see `../references/numerics-replacements.md`, dispatch contract).
- The same test module is the template later kernel swaps copy.
- **`roundtrip`'s advertised signature matches the one that works, and
  matches the Cython convention it replaces.** Added by Task 2.3 on
  2026-08-09, widening the task past "tests only" because the criterion
  above is not satisfiable otherwise: `#[pyo3(text_signature = "(x, /)")]`
  made `inspect.signature` report positional-only while `roundtrip(x=1.5)`
  worked, and the Cython entry points are `def` functions that accept
  keywords (measured: `dnde_photon(egam=…, emu=…)` returns a value). A
  template that advertises a narrower signature than the API it replaces
  propagates a public-API narrowing into every Phase 04–06 wrapper that
  copies it. Fixed to `"(x)"`, which is also what `hazma/_core.pyi`
  already described.

## Exit Criteria

- All tasks complete; CI + preflight green; hybrid (CPython-tagged)
  wheels build on both platforms, each containing an
  `hazma/_core.abi3.so` built against the limited API.
- No public API change; `hazma._core` exists but nothing imports it yet.
- Phase learnings written to `../learnings/phase-02-rust-scaffold.md`.
