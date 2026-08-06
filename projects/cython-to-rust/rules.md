# cython-to-rust — Cross-Cutting Rules

Rules every task in this project must follow. Repo-wide invariants
(preflight gate, PR conventions, versioning) are not restated here —
see `AGENTS.md` and `docs/agents/`.

## Parity discipline

1. **The corpus gates every swap.** A kernel's Python wrapper is
   repointed to `hazma._core` only when the Rust implementation passes
   the Phase 01 parity corpus within that function's budget in
   `test/parity/tolerances.*`. The Cython twin is deleted in the same PR
   as the swap — no dual-implementation drift window.
2. **Corpus data is generated only from pre-port Cython.** The corpus
   manifest records the git SHA and environment that produced it.
   Never regenerate reference arrays from a tree in which any kernel
   already runs on Rust. Widening a tolerance budget requires a one-line
   justification in the tolerance file and a note in the task note.
3. **Every numerical shift is declared.** If a swapped kernel moves any
   value beyond 1e-12 relative, the PR body and the working-memory
   "Numerical impact so far" section record the function, grid, and
   max shift — even when the corpus tolerance absorbs it. The Phase 07
   closing CHANGELOG aggregates these.

## Constants

1. **Bit-parity first, cleanup second.** Ported code uses the exact
   constant values its Cython source used — including the known
   divergences between `_utils/constants.pxd` and the legacy
   `parameters.pxd` tables (see `references/cython-inventory.md`,
   "Bugs" §3). Consolidating to one table is a *separate, declared*
   numerical change after the port, not a silent side effect of it.
   The relocated `legacy_parameters` header keeps its values verbatim,
   with one settled exception: its malformed `WIDTH_K` / `WIDTH_PI`
   entries were deleted on 2026-08-05 because no module referenced
   them, so there was no parity to preserve. Deleting a name nothing
   reads is not a numerical change; changing a value something reads
   still is, and this rule governs the latter.

## Licensing

1. **No GSL-derived code in the repo or its dependency graph** (ADR-0002).
   Numerics provenance is restricted to: public-domain netlib QUADPACK,
   cephes-lineage code (`spec_math` or in-tree cephes translations),
   and original work. Every vendored or translated routine cites its
   upstream source and license in a header comment.

## Rust conventions

1. Edition 2024; `cargo fmt --check` and `cargo clippy -- -D warnings`
   are part of the preflight gate from Phase 02 on; `cargo test` runs
   the kernel unit tests (analytic limits, edge cases) — the Python-side
   corpus tests remain the cross-language gate.
2. **The extension is `hazma._core`** — one cdylib, PyO3 submodules per
   domain (`photon`, `positron`, `neutrino`, `scalar_mediator`,
   `vector_mediator`). Public Python import paths never change; wrapper
   modules re-export from `hazma._core` behind the existing names.
3. Kernel functions are plain `fn(f64, ...) -> f64` in modules free of
   PyO3 types; the PyO3 layer (dispatch, error mapping, array glue)
   lives separately. This keeps `cargo test` GIL-free and the math
   readable next to the Cython it replaces.
4. Every `.pyx` numeric edge guard survives the port explicitly:
   threshold short-circuits (`E − M < f64::EPSILON` → rest frame),
   below-threshold zeros, and the β→0 singularity guard. Cython
   `assert`s become unconditional error returns (declared once in the
   CHANGELOG as a tightening — today they compile out under `-O`).

## Process

1. **Verify-before-delete.** Phase 00 deletions re-run the importer
    check (`rg` for the module path) at execution time rather than
    trusting the inventory snapshot, and each deletion PR states the
    check in its body.
2. Phases land in order; within Phases 04–06 tasks may proceed in any
    order consistent with the cimport DAG in
    `references/cython-inventory.md`.
3. Performance claims require a benchmark in the task note (hyperfine
    or pytest-benchmark), compared against the pre-swap Cython on the
    same machine. Expected wins (mediator spectra, thermal ⟨σv⟩) are
    side effects to measure, not goals to chase at parity's expense.
