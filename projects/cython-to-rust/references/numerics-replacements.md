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

scipy's `spence` and `k1` are themselves cephes wrappers, so for those
two this is algorithm-for-algorithm parity, not merely value parity.
Two implementation-time checks were mandatory (Task 3.2):

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

**Measured, Task 3.2 (2026-08-09, scipy 1.18.0) — the third sentence
above was wrong about `kn`.** Running check 2 is what found it:

- `spence` and `k1` are the same cephes routine on both sides and agree
  to a few ulp: max relative deviation **2.4e-15** over the `spence`
  grid, **1.2e-15** over `k1`'s. `li2` does expose scipy's convention
  (`Li₂(1−z)`), because it delegates to `cephes64::spence` — but the
  *name* says the other one, which is why check 1 exists.
- **`scipy.special.kn` is not cephes `kn`.** It dispatches integer
  orders to `kv`, and only `k0`/`k1` remain cephes. `spec_math`'s
  faithful cephes `kn` therefore misses scipy by up to **5.1e-9**
  relative over `x ∈ [1e-8, 300]`, at `x = 9.531` — just below that
  routine's own `x = 9.55` branch switch. The fix is not
  the vendoring fallback above — a hand-translated cephes `kn` would
  reproduce the same miss. `rust/src/special.rs` builds `Kₙ` from the
  upward recurrence `K_{m+1} = K_{m-1} + (2m/x)·K_m` seeded on cephes
  `k0`/`k1`, which tracks scipy to **≤ 3.4e-15** for every order
  n = 0..5.
- **Underflow is where the two `kn` part company.** scipy flushes to
  zero from `x ≈ 698`, while `K₂(697.88)` is `3.9e-305` — a normal
  double, not a lost one — and the recurrence keeps returning values
  until its `exp(-x)` seeds die near `x = 742`. `k1` has no such split:
  both sides decay into the subnormals and reach zero together.
  Unreachable from hazma (`thermal_cross_section` short-circuits above
  `x = 300`, where `K₂ ≈ 3.7e-132`) and pinned in
  `test/test_core_special.py`.
- **The `cython_special` C symbols and the `scipy.special` ufuncs return
  bit-identical values** for all three (checked through `__pyx_capi__`
  on the same grids), so a test may use the ufunc as the oracle for
  what the `.pyx` actually calls. That equivalence is itself a test
  (`TestOracleIdentity`) rather than an assumption.

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
Task 3.3 preprocessing contract): the five spectra/mediator-spectrum
`points=[-1, 1]` calls pass breakpoints that coincide with *both*
integration endpoints; the thermal ⟨σv⟩ calls pass
`[2, m_med/mx, 2·m_med/mx]`, whose lower entry equals the lower bound
and whose mediator entries can exceed the upper bound
`max(50/x, 100|150)` for heavy mediators. What scipy does with
sorted/duplicate/endpoint-coincident/out-of-interval points must be
pinned *empirically* and replicated exactly, errors included — do not
derive the contract from QUADPACK documentation alone (Task 3.3).

**Measured, 2026-08-10 (Task 3.3), against scipy 1.18.0.** The contract
is not QUADPACK's at all — it is three lines of Python in
`scipy/integrate/_quadpack_py.py`'s `_quad`: `np.unique(points)`, then
`[a < p]`, then `[p < b]`, after `quad` has already ordered the limits
(`flip, a, b = b < a, min(a, b), max(a, b)`) and with the result negated
afterwards if it flipped. So: **sort ascending, drop duplicates, keep
only strictly interior points.** Endpoint-coincident points, points
outside `[a, b]` and `NaN` all vanish silently; `-0.0` and `0.0` collapse
to one entry. The only errors are `ValueError`s raised from
`quad`: an unattainable tolerance, and `limit <= npts` counted **after**
filtering (scipy's message quotes the caller's unfiltered length, which
is a message defect, not a contract).

Two consequences the paragraph above did not anticipate:

- **Both live degeneracies are discards.** `points=[-1, 1]` on `[-1, 1]`
  leaves *zero* break points, and a heavy mediator drops both of the
  thermal call's mediator entries. Five of the twelve live call sites
  therefore run `qagpe` with an empty break-point list.
- **`points is None` selects `qagse`, not "no break point survived".**
  scipy dispatches before it filters, so those five sites run `qagpe`.
  That matters rarely but not never: `qagpe` measures the "smallest
  interval" by subdivision level and `qagse` by interval length, so
  `qagpe` extrapolates one bisection earlier. Over 3,776 random
  (integrand, tolerance, limit) combinations the two agreed on value,
  `neval` and `last` in every run that converged, and differed in 45 —
  all of them runs that exhausted `limit`.

Also measured there, and worth carrying into Phases 04–06: only `qk21`
is on the live path. `qagse` and `qagpe` both evaluate with the 21-point
rule and nothing else, so `qk15` is reachable from no hazma call site.

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

### Measured, Task 3.4 (2026-08-10): fused arithmetic, and a dropped cell

Three facts the two sections above are silent about, each of which
changes what the port has to be. All were measured against the live
Cython through `hazma._utils.boost.__pyx_capi__` (the `cdef`s are
declared in `boost.pxd`, so Cython exports them as capsules and `ctypes`
can call them — with `PYFUNCTYPE`, since `CFUNCTYPE` drops the GIL and
the integral calls back into NumPy).

**1. The compiler contracts, and the port has to as well.** Clang's
default is `-ffp-contract=on`, so `a*b + c` becomes a fused
multiply-add. The corpus's capturing platform (macOS/arm64) contracts
eight distinct expressions across these routines — `1 - β²` in both the
integral and the line, `e² - m²` and `e0² - m²` and `e ∓ βk` in the line,
and `y1 - m·x1`, `0.5·m·(x2 + lb) + b` and the accumulation itself in
each partial cell (twelve `mul_add` call sites in the Rust) — plus
`slope·(x - xp[j]) + fp[j]` inside NumPy's own `arr_interp`. Written
unfused, the Rust port misses the corpus by up to
**3.6e-12** relative on the corpus's own grids for the seven tabulated
photon spectra, against the 1e-12 `TABULATED` budget; written with
`f64::mul_add` at those sites it is **bit-equal at every point**. The
sites were established twice over — by disassembling the shipped
`hazma/_utils/boost.cpython-312-darwin.so` for `fmsub`/`fmadd`, and by
bisecting all 16 on/off combinations against the live kernel (only the
all-on combination reaches zero mismatches).

The converse matters as much: `boost_beta` spells its square as
`(mass/energy) ** 2`, whose rounded product completes before the
subtraction, and **none** of its ten inlining call sites contract
`1 - t` (checked in `_eta`, `_kaon`, `_positron/_pion`). Fusing it would
move every boosted spectrum. "The compiler contracts" is a per-expression
fact, not a per-file one.

**2. `np.trapezoid` reduces pairwise, not sequentially.** The interior
sum goes through `ndarray.sum`, which runs eight accumulators over
128-element blocks and recurses above that. A sequential sum in Rust is a
different number — up to 1.8e-15 relative on the 500-row tables. The port
mirrors the blocking; the pin is a comparison against the live
`np.trapezoid` rather than a comment.

**3. The interior sum never covers its last cell.**
`np.trapezoid(yy[ilow:ihigh], x=x[ilow:ihigh])` is exclusive at the top
while the upper partial-cell term starts at `x[ihigh]`, so
`[x[ihigh-1], x[ihigh]]` belongs to no term. When the window reaches past
the table, `ihigh` is the last index and the upper term is skipped
entirely, so the table's **final row contributes to nothing** — checked
by replacing it with a value six orders larger and getting a bit-identical
answer from both implementations. The error is systematic and one-signed:
the boosted spectrum is always slightly low. Preserved per `rules.md`
rule 1; repair tracked in
[`../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md`](../../../docs/followups/todo/boost-integral-drops-last-interior-cell.md).

Two smaller facts for Phase 04. The live tables are rows of a transposed
`np.loadtxt` result, so they are **strided views, not contiguous
buffers** — anything taking a `PyReadonlyArray1` must copy rather than
`as_slice`. And `np.interp` has two behaviors the section above does not
list: a one-point grid answers everything with `fp[0]`, NaN included
(NumPy's NaN check lives on the multi-point path only), and duplicate
abscissae resolve to the *last* matching node.

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

### What the Cython actually does today (measured, Task 2.1)

The target shape above is a *design*, not a transcription. The live
dispatch is `if hasattr(x, '__len__')` — 54 occurrences across 15 `.pyx`,
with 17 `assert len(energies.shape) == 1` guards behind it, e.g.
`hazma/spectra/_photon/_muon.pyx:148-153`. Measured on the built tree
(`hazma.spectra._photon._muon.dnde_photon`,
`hazma.spectra._positron._muon.dnde_positron_muon`,
`hazma.spectra._neutrino._muon.dnde_neutrino_muon`), it diverges from the
contract in four ways. Each is a call Task 3.5 must make on purpose,
because three of them are user-visible:

1. **A 0-d array raises**, it does not take the scalar path. `ndarray`
   defines `__len__` on the type, so `hasattr` is true for every array;
   the guard then sees `shape == ()` and fails with
   `AssertionError: … must be 0 or 1-dimensional.` — a message that
   names the shape it just rejected.
2. **A Python list is accepted.** `np.array(egam)` converts it before
   the memoryview cast, so `dnde_photon([10.0, 20.0], 200.0)` works
   today. `PyReadonlyArray1<f64>` will not accept one, so a faithful
   port must either call `np.asarray` at the boundary or declare the
   narrowing.
3. **Shape errors are `AssertionError`, not `ValueError`** (and vanish
   under `python -O`). Dtype errors *are* `ValueError`, but with
   Cython's own wording: `Buffer dtype mismatch, expected 'double' but
   got 'long'`.
4. **`hazma/spectra/_neutrino/_muon.pyx:205` says "Photon energies"**
   where its sibling `hazma/spectra/_neutrino/_pion.pyx:261` says
   "Neutrino energies" — a
   copy-paste defect in a string the port would otherwise carry over
   verbatim under `rules.md` rule 1.

Items 1 and 2 are the same layer as
[`docs/followups/todo/model-spectra-reject-scalar-energies.md`](../../../docs/followups/todo/model-spectra-reject-scalar-energies.md),
which records the model-level half (`Theory.spectra` and
`Theory.positron_spectra` reject the scalar energies they document).
Resolving that follow-up by normalizing at the public boundary also
settles item 1 here; deciding them separately risks two different
answers.
