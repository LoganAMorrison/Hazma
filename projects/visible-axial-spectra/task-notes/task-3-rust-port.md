# Task 3: Rust port of the Task 1-2 numerics

**Date:** 2026-10-16
**Project:** visible-axial-spectra
**Status:** Complete
**Plan References:** Task 3 and "Implementation layering" in projects/visible-axial-spectra/PLAN.md
**Related ADRs:** ADR-0001, ADR-0002 (project-scoped)
**Depends On:** Task 2

## Objective

Move the axial tensor algebra and the LO three-pion currents and phase-space
integration from NumPy into Rust kernels behind `hazma._core.axial`, leaving
the public Python API unchanged and keeping the NumPy implementation as a
test oracle.

## Exit Criteria

- [x] The existing axial test suite passes unchanged against the Rust-backed
      functions.
- [x] Every ported function agrees with the NumPy oracle to a few units in the
      last place on its existing test inputs, with any larger difference
      measured and justified.
- [x] Cargo unit tests cover the Ward identities, threshold zeros and the
      conditioning guard.
- [x] `references/check_public_impact.py` stays bit-identical.
- [x] Performance against the oracle is recorded.

## Inputs Reviewed

- `PLAN.md` Task 3, "Implementation layering" and "Numerical impact", via a
  subagent brief.
- ADR-0001 (project) for units, metric and projection conventions.
- `hazma/form_factors/axial/_tensors.py`, `_three_pi.py` and their tests at
  `e39bf50f`.
- `rust/src/lib.rs`, `dispatch.rs`, `kernels.rs`, `boost.rs::pairwise_sum`,
  `vector_mediator.rs`.
- `test/parity/cases.py::rust_core_kernels` and `test_parity.py`'s served-kernel
  roster check.

## Findings

- NumPy 2.5.4 on macOS/arm64 fuses complex multiplies into FMAs, so the
  oracle's `H H*` has a nonzero imaginary diagonal of order 1e-16 while the
  kernel's is exactly real. Two more sources of rounding difference are
  summation order and NumPy's evaluation of complex divided by real as a
  multiply by the reciprocal; the real-valued three-pion currents are
  bit-identical. The spectral functions add the quadrature rules, which
  genuinely differ (see Decisions).
- The parity suite asserts that every served `hazma._core` callable is a
  ported Cython entry point, so a new-physics submodule needs its own
  exemption (`_CORE_NEW_PHYSICS_MODULES`) rather than a corpus case.
- A first sample-major layout made `hadronic_tensor` 5x slower than NumPy
  (two transposing copies and an unsized collect); the component-major
  layout of ADR-0002 removed both.

## Decisions and Implementation Notes

- ADR-0002 records the split: Python validates and reshapes to
  `(components, N)`; Rust maps a per-sample kernel; the tensor guard and its
  messages live in Rust with the NumPy precedence (overflow, then negative
  density, then conditioning) as the batch error.
- Gauss-Legendre nodes come from Newton's method in Rust
  (`axial_three_pi::gauss_legendre`), not a port of NumPy's eigensolver.
  Against mpmath at order 511 the kernel's nodes are good to about an ulp
  and its weights to 3e-12 relative; `leggauss` has the same node accuracy
  but weights good to 2e-10. The integral inherits the weight differences,
  so the oracle comparison is a relative tolerance of 1e-13 rather than a
  count of ulps. The Newton stop is 4 ulps with one further step, because
  the rounding of `value / derivative` can never meet a 1-ulp stop.
- Sums over many terms use `boost::pairwise_sum`, matching NumPy's
  reduction of a contiguous axis; the 16-term tensor sums are sequential,
  matching NumPy's reduction of a batched `(4, 4, N)` array.
- Review fixes: the oracle comparison for spectral functions sweeps seven
  orders and three masses; a cargo test pins the `128 ε` factor and the
  diagonal term of the conditioning bound; and the Gauss-Legendre
  documentation states the measured accuracy of both rules.
- Review fix: `_momentum` rescales each `q` by an exact power of two to
  `0.5 <= q[0] < 1` before the timelike check and both projection kernels,
  which are invariant under `q → λq`. At `q` scaled by `2e-162`, `q·q`
  underflowed: `tensor_projections` passed its guard with `(0.0, 1.0)`
  against the true `(0.0543, 1.163)`, `current_projections` was 30% low,
  and a spacelike `q` passed the timelike check. The fix also accepts large
  timelike `q` such as `(1e308, 0, 0, 0)`, which was refused. In-range
  values are unchanged, since the rescaling is exact.
- `num-complex` is a direct dependency (already in the lock tree through
  `numpy`) so PyO3-free kernels can use complex arithmetic.
- Module names follow the plan's table: `axial_tensors`, `axial_three_pi`,
  `hazma._core.axial`. The oracle is `axial_oracle.py`, not `oracle.py`,
  because the test tree imports helpers by basename.

## Files Changed

- `rust/src/kernels/axial_tensors.rs`, `axial_three_pi.rs` — kernels and cargo
  tests.
- `rust/src/axial.rs`, `rust/src/lib.rs` — the `hazma._core.axial` submodule.
- `rust/src/dispatch.rs` — `components` / `component_major` batch layout.
- `rust/src/kernels.rs`, `rust/Cargo.toml`, `rust/Cargo.lock` — registration,
  `num-complex`.
- `hazma/form_factors/axial/_tensors.py`, `_three_pi.py` — wrappers call the
  kernels.
- `hazma/_core.pyi`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`
  — submodule roster comment and the built-extension roster assertions.
- `test/form_factors/axial/axial_oracle.py`, `test_oracle.py` — NumPy oracle and
  comparison.
- `test/form_factors/axial/test_tensors.py` — momentum-scale regression tests.
- `test/parity/cases.py`, `test/parity/README.md` — new-physics exemption.
- `projects/visible-axial-spectra/adrs/ADR-0002-rust-kernels-and-numpy-oracle.md`.

## Verification

- `cargo test --no-default-features --features test-probes`: 314 passed,
  15 of them axial (Ward identities, threshold zeros, conditioning guard
  and its `128 ε` bound, chiral limit, Gauss-Legendre exactness).
- `cargo clippy --all-features --all-targets -D warnings` and
  `cargo fmt --check`: clean.
- `pytest -n 0 test/form_factors/axial -q`: 273 passed (192 existing,
  unchanged, plus 81 oracle comparisons).
- `pytest test/parity -q`: 723 passed, 1 skipped.
- `references/check_public_impact.py`: maximum absolute difference 0.0,
  "unchanged on the recorded grid".
- `scripts/agents/preflight.sh` (full suite, cargo gates): pytest 2799
  passed, 16 skipped, 127 subtests; cargo fmt, clippy and test PASS. Its
  ruff, black and markdownlint findings in this diff's files were fixed
  afterward and re-checked file by file.

## Numerical impact

Existing public values are bit-identical (`check_public_impact.py`). The
axial functions move by rounding only; `test_oracle.py` measures it against
an oracle verified bit-identical to `origin/master`'s implementation on the
same inputs. Worst cases on macOS/arm64: `hadronic_tensor` 1.4 ulps of
`|H_mu||H_nu|`; `current_projections` 2 ulps of `gamma**2 (L + 3T)`;
`tensor_projections` 3.7e-3 of its own roundoff budget; integration 2 ulps
(last axis) and 5 ulps (other axes); three-pion currents and weights 0;
spectral functions a relative difference of 3.2e-14 at order 505 (orders 8
to 512, both channel classes, pion masses 134.9768, 139.57 and 0), from the
two quadrature rules differing; the tolerance is 1e-13.

Outside rounding, `current_projections` and `tensor_projections` change
only where `q·q` underflowed or overflowed: they returned wrong values for
`q` near `1e-162` and refused finite timelike `q` near `1e308`, and now
return the unit-scale values to 5.8 eps.

## Performance

Minimum of five runs, `N = 100000` samples, macOS/arm64; the two
`spectral_functions` rows are the minimum of 21 runs.

| function | NumPy oracle (ms) | Rust (ms) |
|---|---|---|
| `hadronic_tensor` | 1.48 | 1.37 |
| `current_projections` | 2.40 | 1.29 |
| `tensor_projections` | 16.36 | 11.80 |
| `integrate_spectral_functions` | 2.45 | 1.41 |
| three-pion `current` (Rust includes 1.33 ms of shared validation) | 0.87 | 1.60 |
| `spectral_functions`, 1000 energies, order 64 | 0.19 | 0.20 |
| `spectral_functions`, 1000 energies, order 512 | 1.12 | 3.29 |

At order 512 the Rust path spends 2.32 ms building the rule, which NumPy
cached with `lru_cache`; see ADR-0002's mitigation. The extra Newton step
that makes the nodes converge costs about 0.4 ms over the earlier 1.88 ms.

## Open Questions

None.

## Plan Impact

**Impact Level:** ADR required

ADR-0002 (project-scoped) records the Rust/Python split and the oracle, the
anticipated Task 3 ADR in `PLAN.md`. No phase or task ordering changed.

## Stale-state sweep

Run against the branch with untracked files marked by `git add -N .`.

- **Identifier sweep.** `rg -n --hidden 'axial_oracle|
  _CORE_NEW_PHYSICS_MODULES|component_major|hazma\._core\.axial'
  projects/visible-axial-spectra/ docs/ README.md hazma/ test/ .claude/
  .codex/ .github/` hit only this diff's files, `PLAN.md`'s layering table
  (KEPT: Task 3 row already names `hazma._core.axial`), ADR-0002 and the
  project README (KEPT), and `test/parity/` (EDITED in this task). No hit in
  `docs/`, `.claude/` or `.codex/`.
- **Submodule roster sweep.** `git grep -n -E 'five|"vector_mediator"'`
  found three stale rosters, all EDITED: `hazma/_core.pyi` (now lists
  `axial`), and the `served` sets in `.github/workflows/ci.yml` and
  `.github/workflows/release.yml`, whose "exactly the served submodules"
  assertion would have failed on the built wheel. `docs/agents/
  lessons-examples.md` ("exactly the five per-domain submodules") records
  PR #88 as history (KEPT). `rust/src/lib.rs` already says six. AGENTS.md
  and `docs/agents/environment.md` count the six `test-probes` probe
  modules, which this task did not change (KEPT).
- **Line-number citation sweep.** `check_doc_citations.py --changed-vs
  origin/master` reports no docs to check (no changed Markdown cites
  `file:line`).
- **Forward-looking phrase sweep.** `rg -n '(Task [0-9]+ will|will be
  added|still pending|today: ?stub|currently|In Progress)'` over the
  project and `hazma/form_factors/axial` hits only status headers (KEPT).
- **Count sweep.**

| Claim location | Command | Actual | Status |
|---|---|---|---|
| lib.rs "six per-domain submodules" | `dir(hazma._core)` modules | six | KEPT |
| README "314 Rust tests" | `cargo test` | 314 | EDITED |
| README "273 tests" | `pytest -n 0 test/form_factors/axial` | 273 | EDITED |
| parity README "today six" test-only | `_CORE_TEST_ONLY_MODULES` | six | KEPT |

- **Numerical-impact statement.** No public value changes (verified:
  `references/check_public_impact.py`, 164 values, maximum absolute
  difference 0.0). The axial primitives move by rounding only, bounded in
  `test_oracle.py`.
- **Exit Criteria to test mapping.**

| Exit criterion | Satisfied by |
|---|---|
| Existing suite unchanged | `test/form_factors/axial/` (192 tests) |
| Agreement with oracle | `test_oracle.py` (81 comparisons) |
| Cargo Ward, threshold, guard | `kernels/axial_*.rs` unit tests (15) |
| Public values bit-identical | `check_public_impact.py` |
| Performance recorded | `## Performance` above |

- **Task-note self-consistency.** `**Status:** Complete` matches the
  README Tasks row; every file in Files Changed appears in the diff, which
  also changes the CI workflows listed under Files Changed.

## Handoff to Next Task

- Task 4 adds `axial_poles` to the same `hazma._core.axial` submodule and
  the same oracle pattern; read ADR-0002 first.
- `_CORE_NEW_PHYSICS_MODULES` in `test/parity/cases.py` must list
  `hazma._core.axial_vector_mediator` when Task 5 creates it.
