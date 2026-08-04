# ADR 0001: Replace Cython with Rust + PyO3, packaged by maturin

**Date:** 2026-08-03
**Status:** Accepted
**Scope:** Project-scoped (applies only within `projects/cython-to-rust/`).

## Context

Hazma's compiled layer is 32 Cython extension modules whose maintenance
cost is concentrated in the toolchain, not the physics: 3 of the last 45
commits touched `.pyx` files and two of those were forced migrations
(NumPy 2.0 / SciPy API removals; build-system rework). Standing pains:
a `scipy>=1.13` *build-time* ABI pin caused by
`scipy.special.cython_special` cimports; 10 version-specific wheels per
release (cp310–cp314 × 2 platforms) that must be rebuilt for every new
CPython; the eager-extension-declaration hack required to avoid
mistagged wheels; Cython 3 deprecations (`DEF`, implicit relative
cimports) already breaking unbuilt files; and no unit-test story for the
compiled math outside Python.

The August 2026 analysis (see `../references/cython-inventory.md`)
established the live surface is small (20 surviving extensions — 19
kernel modules plus one C-level helper — exposing 43 public defs of
which 41 are consumed, ~2,500–3,000 lines of distinct logic), scalar
float64 math with no OpenMP, no C++ classes, and no complex numbers —
i.e. cheap to port.
Candidates evaluated: Rust + PyO3 + maturin, and pybind11 (+
scikit-build-core). Estimated effort is comparable (21–32 focused days
either way).

## Decision

Port the live compiled surface to **Rust**, exposed through **PyO3** as
a **single `hazma._core` extension** (submodules per domain), built with
**abi3-py310** so one wheel per platform covers all supported CPythons.
During the migration (Phases 02–06) the Rust extension is built
alongside the remaining Cython via **setuptools-rust** under the
existing setuptools backend, so `pip install -e .` keeps producing one
importable package and the extension already lives at its final import
path `hazma._core`. At cutover (Phase 07) the build backend switches to
**maturin** and setuptools/Cython are removed.

pybind11 is rejected because it retains the cost centers this project
exists to remove: a C-family toolchain with per-platform compiler
variance, CMake in place of setup.py, no stable-ABI wheels (the 10-wheel
matrix survives), and the same undefined-behavior class in which this
audit found real bugs (out-of-bounds vector write, uninitialized read).
Its genuine advantages — C++ familiarity for physicist contributors and
marginally smoother hybrid builds — are outweighed given kernel churn is
~1 commit/year and the Python-facing API layer stays pure Python.

## Consequences

- **Positive:** wheel matrix drops 10 → 2 (linux x86_64, macOS arm64)
  and stops tracking CPython releases; the scipy build-ABI pin and
  `numpy.get_include()` coupling disappear; kernel math becomes
  GIL-free `cargo test`-able; memory safety by construction; cargo
  replaces vendored/ad-hoc dependency handling; adding
  aarch64/Windows wheels later is a CI-matrix line, not a port.
- **Negative:** contributors touching kernels need Rust; source builds
  need a Rust toolchain; two build systems coexist during Phases 02–06;
  quadrature moves off QUADPACK-via-scipy so numerical drift must be
  measured and declared (versioning.md numerical-change carve-out).
- **Mitigation:** kernel churn is near zero and transliterated math
  reads like the Cython it replaces; the public API and all wrapper
  modules stay pure Python; the Phase 01 parity corpus gates every
  swap; setuptools-rust coexistence is bounded to the migration window;
  drift budgets are set per function in the corpus tolerance file
  (`test/parity/`), and ADR-0002 keeps the integrator QUADPACK-faithful
  to minimize drift at the source.
