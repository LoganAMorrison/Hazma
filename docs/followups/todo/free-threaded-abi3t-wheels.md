# Ship free-threaded wheels once abi3t settles

- **Added:** 2026-08-29
- **Source:** `projects/cython-to-rust/learnings/project-retrospective.md` §5
- **Scope:** cross-cutting
- **Status:** open
- **Triggers / blockers:** waiting on the ecosystem, not on hazma. Ripens
  when PyO3 supports a stable free-threaded ABI (`abi3t`) rather than
  requiring a per-version build against `Py_GIL_DISABLED`, and when
  hazma's runtime dependencies (NumPy, SciPy, matplotlib, scikit-image)
  publish free-threaded wheels for the same interpreters. Re-check both
  before scoping the work.

## Why

The cython-to-rust port replaced a per-CPython wheel matrix with a single
`cp310-abi3` wheel per platform, which is the whole packaging argument
for abi3 and the reason `release.yml` builds two artifacts instead of
ten. Free-threaded CPython does not currently participate in that: a
free-threaded build needs its own extension compiled against
`Py_GIL_DISABLED`, so supporting it today would reintroduce exactly the
per-interpreter matrix the port removed — one extra wheel per platform
per free-threaded CPython.

Waiting is cheap, and the code side already looks ready. `hazma._core`'s
kernels are pure functions over `f64`, and its only mutable module state
— the mass-keyed mediator table cache
(`rust/src/kernels/mediator_tables.rs:320`) — is a
`LazyLock<TableCache<_>>` whose slot is already behind a `Mutex`, so it
is `Sync` by construction rather than by the GIL. What blocks
free-threading here is packaging, not concurrency.

## What

Mostly one piece, once the ABI allows it: enable the free-threaded build
as a `[tool.maturin]` / `Cargo.toml` feature and add the matrix rows in
`.github/workflows/release.yml`. Two things to settle at that point:

1. **Confirm the concurrency claim rather than inheriting it.** The
   `Mutex`-guarded cache is sound under concurrent access today, but
   `rust/src/` is the only place that was checked; re-run the sweep
   (`rg 'static |Mutex|RwLock|OnceLock|LazyLock|thread_local|RefCell'
   rust/src/`) against whatever the crate looks like then, and add a
   concurrent-access test that would actually fail if the cache raced.
2. **Weigh the wheel count if `abi3t` still is not stable.** If PyO3
   requires per-version builds against `Py_GIL_DISABLED` when this
   ripens, adding them reintroduces the per-interpreter matrix the abi3
   cutover removed. That may still be worth it, but it is a trade rather
   than a free addition, and it should be made deliberately.

## Entry points

- `rust/Cargo.toml` — the `abi3-py310` feature on the `pyo3` dependency
- `pyproject.toml` `[tool.maturin]`
- `.github/workflows/release.yml` — the `maturin-action` matrix
- `rust/src/kernels/mediator_tables.rs:320` — the `Mutex`-guarded cache
- `projects/cython-to-rust/learnings/phase-06-mediator-spectra.md` — where
  the cache was introduced and why

## Risks / open questions

- **Dependency readiness gates the user-visible benefit.** A
  free-threaded `hazma._core` is not usable if SciPy has no free-threaded
  wheel for the same interpreter, so the trigger is the ecosystem's, not
  PyO3's alone.
