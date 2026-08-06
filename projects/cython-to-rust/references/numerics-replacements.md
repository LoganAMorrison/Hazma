# Reference: Numerics replacements — quadrature, special functions, interpolation

**Audience:** Phase 01 (corpus tolerances), Phase 03 (foundation), and
any Phase 04–06 task that touches an integrand.
**Nature:** Grounded facts + spec.

## SciPy C-level dependencies being replaced

### `scipy.special.cython_special` (the build-time ABI pin)

Only three functions are cimported anywhere live:

| Function | Live call sites | Math |
| --- | --- | --- |
| `spence` | `spectra/_photon/_muon.pyx:13,113` (`spence(xm) - spence(xp)`) | Dilogarithm. **Convention trap:** scipy's `spence(z)` = Li₂(1−z), not Li₂(z). |
| `k1` | `_c_scalar_mediator_cross_sections.pyx:1361`, `_c_vector_mediator_cross_sections.pyx:606` (`k1(x*z)` in `__thermal_cross_section_integrand`) | Modified Bessel K₁ (Maxwell–Boltzmann weight). |
| `kn` | scalar `:1404`, vector `:650` (`kn(2, x)` in the ⟨σv⟩ prefactor `x/(2*kn(2,x))**2`) | Modified Bessel K_n, integer order. |

This cimport is why `pyproject.toml` pins `scipy>=1.13` in
`[build-system].requires` (exported C symbols must ABI-match between
build and runtime). The pin disappears when these move to Rust.

**Replacement:** the `spec_math` crate (v0.1.6, June 2026,
**MIT OR Apache-2.0**, pure Rust) is a cephes re-implementation:

- `Bessel` trait: `bessel_k0`, `bessel_k0e`, `bessel_k1`, `bessel_k1e`,
  `bessel_kn(n)`.
- `Polylog` trait: `li2` — which per its docs delegates to
  `cephes64::spence`.

scipy's `spence`, `k1`, `kn` are themselves cephes wrappers, so this is
algorithm-for-algorithm parity, not merely value parity. Two
implementation-time checks are mandatory (Task 3.2):

1. Pin the `li2`/`spence` argument convention against
   `scipy.special.spence` on a grid spanning (0, 1), (1, ∞), and the
   branch point — do not assume which convention `li2` exposes.
2. Sweep `bessel_k1`/`bessel_kn` vs scipy over the thermal-integral
   domain (arguments roughly `x·z ∈ [2, 3000]`, watch underflow at large
   argument — cephes K1 underflows near x≈705) at rtol ≤ 1e-13.

Fallback if `spec_math` has a gap or a parity miss: vendor a direct Rust
translation of the specific cephes routine (`spence.c` ≈ 100 lines,
`k1.c`/`kn.c` similar; cephes licensing is permissive — scipy ships it
under its BSD stack).

### `scipy.integrate.quad` (QUADPACK) call sites

All live sites call `quad` from Cython with a `cdef`-function callback
(Cython auto-wraps it as a Python callable → QUADPACK re-enters Python
per node). Sites and their settings:

| Call site | Interval | Settings |
| --- | --- | --- |
| `spectra/_photon/_pion.pyx:123` | cosθ ∈ [−1, 1] | `points=[-1,1]` (QAGP), `epsabs=1e-10`, `epsrel=1e-5` |
| `spectra/_photon/_rho.pyx:52,123` | cosθ | `epsabs=1e-10`, `epsrel=1e-5`; integrand itself calls `_pion`'s quad → **nested adaptive quadrature** |
| `spectra/_positron/_pion.pyx:58` | cosθ | `epsabs=1e-10`, `epsrel=1e-4` |
| `spectra/_neutrino/_pion.pyx:124,127` | energy-space | two quads, scipy default tolerances (`epsabs=1.49e-8`, `epsrel=1.49e-8`), integer selector via `args` |
| scalar `thermal_cross_section` (`:1370` region) | z ∈ [2, max(50/x, 100)] | `points=[2, ms/mx, 2ms/mx]` (QAGP) |
| vector `thermal_cross_section` (`:615` region) | z ∈ [2, max(50/x, 150)] | `points=[2, mv/mx, 2mv/mx]` (QAGP) |
| 4 × mediator spectrum modules | cosθ ∈ [−1, 1] | `points=[-1,1]`, `epsabs=1e-10`, `epsrel=1e-5` |

**Replacement decision (ADR-0002):** port finite-interval QUADPACK —
`qk15`/`qk21` rules, the `qelg` ε-algorithm extrapolation, `qags`, and
`qagp` — to Rust **from the public-domain netlib Fortran sources**
(QUADPACK by Piessens et al. is public domain; scipy vendors that exact
Fortran, GSL's GPL reimplementation is *not* the source we translate
from). This gives scipy-matching subdivision behavior, which is what
keeps corpus drift near zero, at ~1,500–2,500 lines of Rust. Infinite
intervals (`qagi`) are not needed — every live integral is finite.

**Breakpoint degeneracies present in the live calls** (they shape the
Task 3.3 preprocessing contract): the four spectra/mediator-spectrum
`points=[-1, 1]` calls pass breakpoints that coincide with *both*
integration endpoints; the thermal ⟨σv⟩ calls pass
`[2, m_med/mx, 2·m_med/mx]`, whose lower entry equals the lower bound
and whose mediator entries can exceed the upper bound
`max(50/x, 100|150)` for heavy mediators. What scipy does with
sorted/duplicate/endpoint-coincident/out-of-interval points must be
pinned *empirically* and replicated exactly, errors included — do not
derive the contract from QUADPACK documentation alone (Task 3.3).

Oracle strategy: primary oracle is scipy itself, via (a) direct
Python-side comparisons on each live integrand shape and (b) the Phase
01 corpus. See ADR-0002 for what the cyphus crates may and may not be
used for.

Expected drift: with a faithful QUADPACK port, quad-backed spectra
should reproduce to ~1e-12 relative or better; budget 1e-8 in the
corpus tolerance file and tighten after measurement. The nested ρ
integral is the stress test and gets its own budget line.

### `np.interp` semantics

`np.interp(x, xp, fp)` is called inside `cdef` functions in the kaon/eta
photon family and both cross-section spectrum-table paths. The Rust
`interp` routine must replicate exactly:

- linear interpolation on an ascending grid;
- **clamping** outside the grid: returns `fp[0]` below, `fp[-1]` above
  (no extrapolation, no error);
- exact-node hits return the node value.

Note the kaon/eta family separately implements a `1/E`-weighted
power-law tail *below* the table minimum inside
`boost_integrate_linear_interp` — that is part of the boost integral
(next section), not of `np.interp`.

### `boost_integrate_linear_interp` (`hazma/_utils/boost.pyx`)

The one genuinely subtle numeric in the foundation (~90 live lines).
Integrates `y/x` between `E·γ(1∓β)` over a tabulated `(x, y)` spectrum:
`np.trapezoid` over interior whole cells + closed-form linear-interpolant
partial-cell corrections at both edges + analytic `1/E` tail when the
lower bound is below `x[0]` + hard clamp above `x[-1]`; edge detection
uses a 1e-6 absolute tolerance; `assert 0.0 < beta < 1.0` guards the
β→0 singularity (callers short-circuit to the rest-frame value when
`E − M < DBL_EPSILON`). Port with dedicated unit tests per branch
(interior, both partial-cell edges, below-table tail, above-table clamp,
β→0 short-circuit) before any consumer kernel ports.

`boost_delta_function` (boosted Dirac δ for two-body lines) is closed
form: `1/(2γβk₀)` inside the boosted support window, else 0.

## The cyphus crates (assessed 2026-08-03)

Logan's 2020–2022 GSL ports at github.com/rust-cyphus, evaluated for
reuse. Test runs on rustc 1.96.0:

| Crate | Verdict on quality | License | Reusable? |
| --- | --- | --- | --- |
| `cyphus-integration` (3.8k lines) | GSL `qag/qags/qagp/qagi` + `qk` + ε-table. After deleting a stale `#![feature(const_option)]` gate (stabilized since), **43/44 tests pass**; the one failure is doubly-infinite `qagi`, which Hazma never uses. Builder API (`epsabs/epsrel/limit/order/singular_points`) mirrors the scipy surface. | **GPL-3** (explicit GSL copyright headers — "Adapted from the GNU Scientific Library", Brian Gough copyright) | Not in-repo (license). See ADR-0002: manual dev-time cross-check oracle only. |
| `cyphus-specfun` (20.8k lines) | Compiles clean; **99/102 tests pass** (failures: `cyl_bessel_yn_e`, `exprel_n_e`, `choose_e` — none needed here). Has `cyl_bessel_k*`; **no dilog/spence**. | **GPL-3** (GSL port) | Not in-repo. `spec_math` (cephes) is the better parity match for scipy anyway — scipy's k1/kn/spence are cephes, not GSL. |
| `cyphus-interpolation` (2.8k lines) | FITPACK/dierckx curfit/splev port + GSL-style `interp1d` with accel. | No license file; mixed GSL-idiom provenance | Out of scope — Hazma's spline use is pure-Python scipy and stays. |
| `cyphus-diffeq` (3.9k lines) | Hairer dopri5/dop853/radau/rodas ports. | No license file | Out of scope — relic-density ODEs are Python-level scipy and stay. Possible future interest if relic density ever moves to Rust. |

Key conclusions: (1) GPL-3 provenance bars vendoring or linking any of
it into MIT-licensed Hazma — Logan authored the ports but cannot
relicense GSL-derived work; (2) their value is as *independent oracles*
and as proof the QUADPACK-class port is a known, bounded job (done once
already); (3) for the specfun surface Hazma needs, the cephes lineage
(`spec_math`) is both license-clean *and* numerically closer to scipy
than the GSL lineage. Formalized in ADR-0002.

## Entry-point dispatch contract (Phase 03, Task 3.5)

Every public function follows one shape; implement once as a helper:

- Accept `Bound<'_, PyAny>`: `float`/0-d array → scalar path → Python
  `float`; 1-D array (via `PyReadonlyArray1<f64>`) → array path →
  `PyArray1<f64>`. Anything else → `ValueError` matching the current
  message ("Photon energies must be 0 or 1-dimensional.").
- Neutrino pair returns a 3-tuple (scalar in) or `(3, N)` `PyArray2`
  (array in) — the one non-uniform return shape.
- Current Cython `assert`s become explicit `PyValueError`/`PyAssertionError`
  raises; note today's asserts vanish under `-O` — replicating them as
  unconditional checks is a (desirable, tiny) behavior tightening to
  note in the CHANGELOG.
