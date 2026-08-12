---
phase: 03
title: Numerics foundation
status: Complete
---

# Phase 03: Numerics foundation

## Goal

The shared numeric substrate every kernel port stands on: constants,
special functions, the QUADPACK port, `np.interp`-exact interpolation,
the boost integrals, and the scalar/array dispatch layer. Everything
here is unit-tested against scipy/analytic references before any kernel
uses it.

## Prerequisites

- Phase 02 complete. **ADR-0002 is Accepted (2026-08-04)** — Tasks
  3.2/3.3 are ungated, and their source provenance is fixed by it:
  cephes lineage and netlib QUADPACK only, nothing GSL-derived.
- Read `../references/numerics-replacements.md` in full.

## Tasks

### Task 3.1: Constants module

**Task note:** [`../task-notes/phase-03/task-3.1-constants.md`](../task-notes/phase-03/task-3.1-constants.md)
**Depends on:** —

**Exit criteria:**

- `rust/src/constants.rs` carries every `DEF` from
  `_utils/constants.pxd` and every value from
  `_utils/legacy_parameters.pxd`, in **two distinct namespaces**
  preserving the known divergences verbatim (rules.md rule 4).
- A test extracts values from the `.pxd` sources (script or generated
  fixture) and asserts bit-equality — no hand-transcription trust.
- Derived module-local `DEF`s (e.g. `eng_mu_pi_rf`) become `const fn`
  or literal consts with the same float semantics.

### Task 3.2: Special functions

**Task note:** [`../task-notes/phase-03/task-3.2-specfun.md`](../task-notes/phase-03/task-3.2-specfun.md)
**Depends on:** —

**Exit criteria:**

- `spence`, `bessel_k1`, `bessel_kn` exposed via a thin
  `rust/src/special.rs` over `spec_math` (or in-tree cephes translation
  on any gap — ADR-0002 fallback).
- Convention pinned: Rust `spence`-wrapper matches
  `scipy.special.spence` (Li₂(1−z) convention) on a grid covering
  (0,1), [1,∞), z→0⁺, z=1, z=2 — rtol ≤ 1e-13.
- `k1`/`kn` swept vs scipy over the thermal domain incl. large-argument
  underflow region — rtol ≤ 1e-13.

**Criteria added during execution** (2026-08-09; the gap the first
bullet's parenthetical anticipated turned out to be in *scipy*, not in
`spec_math`, so the shape of the answer had to be recorded here):

- **`kn` is not `spec_math`'s `bessel_kn`, and the deviation is
  declared.** `scipy.special.kn` dispatches integer orders to `kv`;
  only `k0`/`k1` are still cephes there. Measured over
  `x ∈ [1e-8, 300]`, cephes `kn(2, ·)` misses scipy by up to **5.1e-9**
  relative — past this task's gate by four orders and inside the parity
  corpus's 1e-8 budget for `thermal_cross_section`, whose prefactor
  squares it. `Kₙ` is therefore built from the upward recurrence
  `K_{m+1} = K_{m-1} + (2m/x)·K_m` seeded on cephes `k0`/`k1`, measured
  at ≤ 3.4e-15 against scipy for every order n = 0..5.
- **The `kn` underflow criterion is bounded at scipy's flush point.**
  The 1e-13 gate holds for `k1` through the whole tail (both sides
  reach zero together near `x = 742`), and for `kn` up to `x ≈ 698`,
  where scipy flushes to zero while `K₂` is still `3.9e-305`. Above
  that the two disagree wholesale by construction; hazma cannot reach
  it (`thermal_cross_section` short-circuits above `x = 300`). The
  boundary is pinned in `test/test_core_special.py`, not merely
  documented.
- **The corpus's served-kernel predicate stays sound.** Exposing the
  shim to Python as `hazma._core.special` — needed because the oracle
  is scipy, which lives in Python — otherwise reads as three ported
  kernels and drops the parity corpus out of bit-equality mode for the
  rest of the project. `cases._CORE_TEST_ONLY_MODULES` exempts the
  submodule, and `test_test_only_core_submodules_have_no_importer`
  makes the exemption conditional on nothing under `hazma/` importing
  it. Task 2.1 fixed this same class once already
  (`docs/agents/lessons.md`, `[gate-disabled-stays-green]`).

### Task 3.3: QUADPACK port (qk15, qk21, qelg, qags, qagp)

**Task note:** [`../task-notes/phase-03/task-3.3-quadpack.md`](../task-notes/phase-03/task-3.3-quadpack.md)
**Depends on:** —

**Exit criteria:**

- Finite-interval `qags` and `qagp` in `rust/src/quad.rs`, translated
  from netlib QUADPACK Fortran (provenance header per rules.md rule 5),
  closure-based API carrying `epsabs`/`epsrel`/`limit`/breakpoints.
- **Breakpoint preprocessing contract, pinned empirically against
  scipy** (do not design it from the QUADPACK docs alone): determine
  by experiment what `scipy.integrate.quad(points=...)` does with
  unsorted lists, duplicates, points coinciding with the endpoints,
  and points outside `[a, b]` — then replicate that behavior
  (including any raised errors) exactly. Both degenerate cases occur
  live: the spectra calls pass `points=[-1, 1]` on the interval
  `[-1, 1]` (all breakpoints endpoint-coincident), and the thermal
  ⟨σv⟩ calls pass `[2, m_med/mx, 2·m_med/mx]` where the mediator
  points can exceed the upper bound `max(50/x, 100|150)` for heavy
  mediators (out-of-interval). Tests cover three parameter regimes per
  thermal call: breakpoints interior (resonance active), breakpoints
  at/near threshold, and breakpoints outside the interval (inactive).
- Unit tests: QUADPACK's own reference problems, plus every live
  integrand *shape* from the call-site table in
  `../references/numerics-replacements.md`, compared against
  `scipy.integrate.quad` at matching settings — agreement within 10×
  the requested tolerance, and within 1e-12 rel on smooth cases.
- Error/abnormal-termination behavior mapped (roundoff, max
  subdivisions, invalid breakpoints) — returns a Result, never panics
  across FFI.

**Criteria added during execution** (2026-08-10; the second bullet's
"pin it empirically" instruction was right, and what it turned up
reshapes the other three):

- **The breakpoint contract belongs to scipy, not to QUADPACK, and both
  live degeneracies are *discards*.** `scipy.integrate.quad` filters in
  Python before `qagpe` ever sees the list — `np.unique`, then strictly
  interior — so `points=[-1, 1]` on `[-1, 1]` leaves **zero** breakpoints
  and the heavy-mediator thermal entries are dropped. Consequence: five of
  the twelve live call sites run `qagpe` with an *empty* list, because
  `points is None` — not "no breakpoint survived" — is what selects
  `qagse`. The port therefore has to keep that dispatch distinction even
  though it is almost never observable (measured: the two routines agree
  on value, `neval` and `last` in every one of 3,776 random combinations
  that converged, and differ only once `limit` is exhausted).
- **Only `qk21` is on the live path.** Both `qagse` and `qagpe` evaluate
  with the 21-point rule and nothing else, so `qk15` — which this task's
  first criterion names — is reachable from no hazma call site. It is
  ported anyway, and earns its place as an independent second rule for
  the cross-checks rather than as production code.
- **The agreement criterion is met with two orders of headroom, and its
  boundary is `limit`.** Over 11,274 random (integrand, tolerance, limit,
  points) combinations against scipy 1.18.0: the 4,461 runs that
  converged reproduced scipy's `neval` and `last` on all but **5**
  (0.11%), with the value within 3.6e-2 of the requested tolerance (the
  criterion allows 10x) and 8.2e-11 relative at worst. The 6,813 that
  exhausted `limit` can separate without bound — 4.5e-5 in that sweep,
  11% on a hand-picked case — because Wynn's ε-algorithm is chaotic on a
  sequence that is not converging; identical subdivision plus a few ulp
  in the table is enough. Termination flags agreed on all 11,274. No live
  integrand shape reaches the second regime — each returns `ier = 0`, and
  `test/test_core_quad.py` asserts it rather than assuming it.
- **Two heuristics inside the adaptive loop need purpose-built inputs.**
  `qagpe`'s `ndin` flag and `qagse`'s roundoff counters change only
  *which* subinterval is bisected next, so they survive every test built
  from the reference problems and the live shapes — a mutation campaign
  found both. The inputs that expose them were found by mutating and
  searching (`sin(293.25/x)` over a 39-point grid moves by a factor of
  48 without `ndin`; a near-delta spike at 0.16309 with `points=[0.5]`
  moves by 2,800 when the 0.99 roundoff threshold is relaxed) and are
  pinned in `TestAdaptiveHeuristics` at a deliberately coarse
  `rtol = 1e-6`, both sitting in the limit-exhausted regime.
- **The corpus's served-kernel predicate stays sound.** The scipy oracle
  lives in Python, so the port is exposed as `hazma._core.quad` and joins
  `hazma._core.special` in `cases._CORE_TEST_ONLY_MODULES`, under the
  same importer guard Task 3.2 built. Same mechanism, not a widened
  exemption.

### Task 3.4: Interpolation + boost kernels

**Task note:** [`../task-notes/phase-03/task-3.4-interp-boost.md`](../task-notes/phase-03/task-3.4-interp-boost.md)
**Depends on:** Task 3.1

**Exit criteria:**

- `interp` replicating `np.interp` exactly (ascending grid, edge
  clamping, node hits) — property-tested against numpy over random
  grids.
- `boost_beta`/`boost_gamma`/`boost_delta_function` +
  `boost_integrate_linear_interp` ported with per-branch unit tests
  (interior, both partial edge cells, below-table `1/E` tail,
  above-table clamp, β→0 guard) pinned against the Cython originals
  via dedicated micro-fixtures captured in Phase 01.

**Criteria added during execution** (2026-08-10; the second bullet's
"micro-fixtures captured in Phase 01" turned out not to exist, and what
replaced them changed what the other criteria could be held to):

- **The oracle is the live Cython, not a fixture.** Phase 01's corpus
  enumerates top-level `def`s in the surviving `.pyx`, and every routine
  in this task is `cdef` — private to the C level and invisible to the
  corpus by construction, so no micro-fixture was ever captured and none
  could have been. `boost.pxd` declares the `cdef`s, which makes Cython
  export them through `hazma._utils.boost.__pyx_capi__` as capsules, so
  `test/test_core_boost.py` calls the *live* kernel through `ctypes` at
  whatever arguments a test picks. Two mechanical constraints, both
  found the hard way: the shim must use `ctypes.PYFUNCTYPE` rather than
  `CFUNCTYPE`, because the latter releases the GIL and
  `boost_integrate_linear_interp` calls `np.trapezoid` (a `CFUNCTYPE`
  call segfaults); and the capsule's *name* is its C signature, so a
  changed argument list is checkable rather than a stack corruption
  waiting to happen.
- **The port must reproduce the C compiler's fused multiply-adds, and
  the criterion is bit-equality rather than a tolerance.** Clang
  contracts `a*b + c` by default and the corpus's capturing platform
  (macOS/arm64) does so at five sites in `boost.pyx` and one inside
  NumPy's `arr_interp`. Written the obvious unfused way the port misses
  the corpus by up to **3.6e-12** relative *on the corpus's own grids*
  for the seven tabulated photon spectra — past the 1e-12 `TABULATED`
  budget, so the Phase 04 swap would have failed its own gate. With the
  contraction reproduced via `f64::mul_add` the port is bit-equal at
  every one of those points. Each site was established twice, by
  disassembling the shipped `.so` and by bisecting the 16 combinations
  against the live kernel. **Where the Cython does not contract, the
  port must not either:** `boost_beta` spells its square as
  `(mass/energy) ** 2`, whose rounded product completes before the
  subtraction, and none of its ten inlining call sites contract it.
- **The interior sum drops a whole cell, and that is reproduced.**
  `np.trapezoid(yy[ilow:ihigh], x=x[ilow:ihigh])` is exclusive at the
  top while the upper partial-cell term starts at `x[ihigh]`, so
  `[x[ihigh-1], x[ihigh]]` is covered by nothing — and when the window
  reaches past the table the table's **final row contributes to no term
  at all**. Systematic and one-signed (the boosted spectrum is always
  slightly low). Preserved per rules.md rule 1 and pinned in both
  languages; the repair is
  [`../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md),
  blocked until after Phase 06 Task 6.4 because it needs a declared
  corpus regeneration.
- **`interp`'s contract is NumPy's, quirks included.** The exit
  criterion names three behaviors; there are two more that a
  spec-driven port would miss. A one-point grid answers *everything*
  with `fp[0]`, NaN included, because NumPy's NaN check lives on the
  multi-point path only; and duplicate abscissae resolve to the **last**
  matching node, not the first. Both are pinned against `np.interp` in
  the same assertion that pins them, so the pin cannot drift from the
  thing it pins.
- **The corpus's served-kernel predicate stays sound.** `interp` and
  `boost` are exposed as `hazma._core.interp` and `hazma._core.boost`
  because both oracles live in Python, and both join
  `cases._CORE_TEST_ONLY_MODULES` under the importer guard Task 3.2
  built — the same mechanism, not a widened exemption. Third instance of
  `docs/agents/lessons.md` `[gate-disabled-stays-green]` in this project.

### Task 3.5: Dispatch and error layer

**Task note:** [`../task-notes/phase-03/task-3.5-dispatch.md`](../task-notes/phase-03/task-3.5-dispatch.md)
**Depends on:** —

**Exit criteria:**

- One generic helper implementing the scalar-or-1D contract
  (`../references/numerics-replacements.md`, dispatch section),
  including the neutrino tuple/`(3,N)` variant; kernel crates stay
  PyO3-free (rules.md rule 8).
- Error messages byte-match the Cython ones the tests assert on.

**Criteria added during execution** (2026-08-11; the reference's premise
that "every public function follows one shape" is false, and what the
measurement found reshapes both criteria above):

- **There are four dispatch shapes across the 43 surviving top-level
  `def`s, not one, and two of them disagree with each other.** Classified
  from source and then measured on the built tree: 15 entry points
  dispatch on `hasattr(x, '__len__')` and reject a 0-d array (12 photon,
  2 positron, and `scalar_mediator_decay_spectrum`); 18 cross-section
  entry points dispatch on `hasattr(...) and x.ndim > 0` and *accept* a
  0-d array via `.item()` while rejecting a list with `AttributeError`;
  the 2 neutrino entry points are the first shape with the 3-tuple /
  `(3, N)` return; and `partial_widths` is a required-1-D argument with
  its own two messages. The port therefore ships **three**
  helpers — `map_unary`, `map_flavors`, `require_vector` — over one
  classification, and Phase 05's cross sections use the same one as
  Phase 04's spectra.
- **The rule that decides every divergence, stated once: each exception
  the Cython raises *explicitly* keeps its type, and only its `assert`s
  change type** (rules.md rule 9). So a rank error becomes `ValueError`
  carrying the assert's message verbatim, a dtype error stays
  `ValueError`, a non-number stays `TypeError`, and `partial_widths`'
  explicit `raise ValueError` keeps type and wording. Three widenings ride
  along, none of which can break a working call: a 0-d array takes the
  scalar path (what the 18 cross sections already do), a list or tuple is
  accepted (what those same 17 already do), and the dtype message names
  the dtype rather than a C type — the Cython has no one string to match
  here, saying `expected 'double'` in the spectra and `expected
  'float64_t'` in the mediator modules for the same rejection.
- **`hazma/spectra/_neutrino/_muon.pyx:205`'s "Photon energies" is not
  carried over.** The ported `dnde_neutrino_muon` passes `"Neutrino
  energies"`, matching its `_pion.pyx` sibling: the string is reachable
  only through an exception whose *type* is already changing, so nothing
  that matches on it survives either way, and it names the wrong physical
  quantity. `TestCythonMessageParity` pins the roster from source so the
  anomaly cannot quietly reappear.
- **Byte-matching is done against the `.pyx` sources, not against typed
  copies.** `test/test_core_dispatch.py` scans the surviving `.pyx` for
  every `assert len(...) == 1, "..."` and `raise ValueError("...")`,
  asserts the roster is exactly the four-plus-one it expects, and renders
  each message through the port with that quantity.
- **The corpus's served-kernel predicate stays sound.**
  `hazma._core.dispatch` — three probes taking the quantity wording as an
  argument, which the fixed-wording `roundtrip` cannot — joins
  `cases._CORE_TEST_ONLY_MODULES` under the importer guard Task 3.2
  built. Fourth instance of `docs/agents/lessons.md`
  `[gate-disabled-stays-green]` in this project; the mechanism is reused,
  not widened.

## Exit Criteria

- All tasks complete; `cargo test` covers the foundation GIL-free; the
  scipy-comparison suite passes in CI.
- No kernel swapped yet — `hazma._core` still unreferenced by wrappers.
- Phase learnings written to `../learnings/phase-03-numerics-foundation.md`.
