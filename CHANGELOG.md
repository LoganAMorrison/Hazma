# Changelog

All notable user-facing changes to Hazma are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/) over the
public Python API as defined in [`docs/versioning.md`](docs/versioning.md).

Sections: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
`Security`, and `Known issues`. The last one is outside Keep a Changelog's
set and exists because a release can *measure* a defect without repairing
it: an entry there names a wrong number the release ships knowingly, with
its magnitude and a pointer to the tracked repair. **Numerical changes go
under `Changed` with the magnitude stated** — a spectrum that moves is a
user-facing change even when no signature did.

## [3.0.0] — 2026-08-29

**Hazma's compiled layer is now Rust.** The twenty Cython extension
modules are gone, replaced by a single abi3 extension, `hazma._core`,
built by [maturin](https://www.maturin.rs/); no `.pyx`, no `.pxd`, and no
transpiler remain in the tree. Wheels are tagged `cp310-abi3` — one per
platform, valid on every CPython from 3.10 onward, instead of one per
(CPython, platform) pair — and a source build now needs a Rust toolchain
on `PATH` rather than Cython, NumPy and SciPy build-time headers. The
public Python API is unchanged: same module paths, same names, same
keyword arguments, same return shapes and units. Delivered as the
`cython-to-rust` project
([`projects/cython-to-rust/PLAN.md`](projects/cython-to-rust/PLAN.md)).

**The major bump is an API removal, not a number.** `hazma.gamma_ray` and
`hazma.deprecated.rambo` are gone; see `Removed`. Of the 41 compiled
entry points the port moved, 27 reproduce 2.1.0 **bit-for-bit** and the
other 14 move by at most **5.4e-12** relative — see the table under
`Changed`. The one substantive numerical change in this release is the
two-body threshold repair, also under `Changed`, and it predates the
Rust work.

**Read `Known issues` before upgrading a published analysis.** Porting
each kernel meant stating what it computes precisely enough to check, and
that surfaced twelve live numerical defects that hazma 2.1.0 already had.
This release **reproduces all twelve deliberately** — a port that quietly
changed numbers would have had no way to prove it changed only the ones
it meant to — so no result moves because of them here. Several of them do
affect published numbers, one by percent-level amounts across the whole
freeze-out region, and two change the *shape* of a spectrum rather than
its normalization.

### Added

- **`hazma.utils.two_body_momentum(cme, m1, m2)` — the common
  center-of-mass three-momentum of a two-body state, in MeV.** Evaluates
  the factored form of the Källén polynomial with the heavier mass
  subtracted first, which is algebraically identical to
  `sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)` but does not
  cancel at threshold. Broadcasts over `cme` and the masses. This is now
  the single definition of that momentum inside hazma; see `Changed`
  below for the values it moves.

- **`hazma.spectra.dnde_photon_fsr` — exact FSR photon spectra from a
  user-supplied squared matrix element.** The maintained replacement for
  the removed `hazma.gamma_ray.gamma_ray_fsr` (cython-to-rust ADR-0003),
  designed per [`docs/adrs/ADR-0001`](docs/adrs/ADR-0001-fsr-generator-takes-both-matrix-elements.md)
  and resolving
  [`docs/followups/done/msqrd-driven-fsr-generator.md`](docs/followups/done/msqrd-driven-fsr-generator.md).
  The caller supplies the squared matrix elements of both the radiative
  and the non-radiative process; every initial-state factor cancels in
  the ratio, so there is no rate float, no `isp_masses`, and no
  decay/annihilation branch. A two-body non-photon final state is
  integrated by deterministic Dalitz-strip quadrature; higher
  multiplicities run seeded RAMBO Monte Carlo at the reduced invariant
  mass, built on `hazma.phase_space`. Returns the new
  `hazma.spectra.FSRSpectrum` NamedTuple `(dnde, error)` in MeV⁻¹, with
  the one-sigma integration error a first-class output. Validated in
  `test/spectra/test_dnde_photon_fsr.py` against exact tree-level matrix
  elements (numerical Dirac traces / scalar QED) pinned to the analytic
  mediator-model spectra `dnde_xx_to_v_to_ffg`, `dnde_xx_to_s_to_ffg`,
  and `dnde_xx_to_v_to_pipig` at `rtol=1e-5`, plus Ward-identity,
  soft-photon, Altarelli-Parisi, and flat-matrix-element phase-space
  checks.

### Removed

- **`hazma.gamma_ray` — the whole module.** It could not be imported in
  any released version (it transitively imported the long-deleted
  `hazma.rambo`), so no working user code depended on it. Decided in
  [`projects/cython-to-rust/adrs/ADR-0003`](projects/cython-to-rust/adrs/ADR-0003-remove-gamma-ray-module.md).
  Both public functions have a named replacement, **neither a drop-in**:
  - `gamma_ray_decay` → `hazma.spectra.dnde_photon`, the live n-body
    path over `hazma.phase_space`. Argument order and keywords differ,
    and it includes FSR by default (`include_fsr=True`).
  - `gamma_ray_fsr` → `hazma.spectra.dnde_photon_fsr` (see `Added`
    above). It takes the non-radiative squared *matrix element* rather
    than a rate float, and has no `isp_masses`.

- **`hazma.deprecated.rambo`.** The last module under
  `hazma/deprecated/`, superseded by the pure-NumPy `hazma.phase_space`
  it already warned users toward on import. Removing anything from
  `hazma/deprecated/` is `major` per
  [`docs/versioning.md`](docs/versioning.md); the package is now empty.

- **The `hazma._gamma_ray` and `hazma._phase_space` Cython extensions**
  (private, never re-exported, and the only C++ in the tree) and the
  never-built `hazma/rh_neutrino/_rh_neutrino_fsr_four_body.pyx`. This cut
  the extension count from 25 to 20 before the Rust port took it the rest
  of the way to one; see `Changed`. No public value changes: every
  compiled-backed public entry point is bit-for-bit identical across the
  deletion (159 arrays over `np.logspace(-2, 3, 200)` MeV).

- **The Cython build machinery, and with it the build-time dependency on
  Cython, NumPy and SciPy.** `setup.py`, `MANIFEST.in`, `setup.cfg`'s
  `[aliases]` section, `requirements.txt` and the `Dockerfile` are
  deleted; `pyproject.toml` is the only build entry point and
  `[build-system] requires` is `["maturin>=1.5,<2.0"]`. The now-empty
  private `hazma/_utils/` package is deleted too — it held only Cython
  headers. Nothing public imported any of it.

### Changed

- **Two-body kinematics near threshold move; everything else moves by
  roundoff at most.** `hazma.utils.cross_section_prefactor` and the
  two-body branches of `hazma.phase_space` (`Rambo.cross_section`,
  `Rambo.integrate`, `TwoBody.integrate`) and `hazma.deprecated.rambo`
  built the incoming momentum as
  `sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)`. The Källén
  polynomial sums four terms of size `cme**4` that cancel to zero at the
  threshold `cme = m1 + m2`, so close to threshold the result was
  dominated by roundoff. All of these now call the new
  `hazma.utils.two_body_momentum`, whose factored form has no such
  cancellation. Measured over 21 mass pairs from {e, μ, π⁰, π±, K±, p},
  `cross_section_prefactor` moves by ≤2e-15 relative at
  `cme ≥ 1.1 ×` threshold (≤5e-16 at `≥ 2 ×`), 2.0e-13 at `1.01 ×`, 1.3e-8 at
  `1 + 1e-6`, and 1.3e-4 within 1e-10 of threshold; `TwoBody.integrate`
  moves by ≤4e-15 at `1.01 ×` threshold and 2.7e-9 at `1 + 1e-10`. The
  new values are the accurate ones — relative error against an
  exact-rational reference is ≤4.4e-16 everywhere, where the old form
  reached 4.0e-2 within 1e-12 of threshold. **Three behavior changes at
  the edges of the physical domain:** (1) exactly at threshold `cross_section_prefactor`
  now returns `+inf` (the physical divergence of the flux factor as the
  relative velocity vanishes) rather than a large finite number; (2) for
  unequal masses, where `m1 + m2` rounds, the threshold is now resolved
  to the last bit, so a `cme` that rounds just below it returns NaN
  instead of a plausible-looking number; and (3) **the whole
  below-threshold region is now NaN.** `kallen_lambda` is negative only
  *between* its two roots `|m1 - m2|` and `m1 + m2`, and turns positive
  again below the lower one, so for `cme < |m1 - m2|` the old form
  returned a finite, physically meaningless momentum — for example
  `two_body_momentum(1.0, 10.0, 1.0)` would have given `48.98979…` and
  `cross_section_prefactor(10.0, 1.0, 1.0)` `0.005103…`, despite a
  threshold of 11. Both are NaN now. Every current caller integrates by
  Monte Carlo with a percent-level statistical error and none evaluates
  below threshold, so no published result is affected. Resolves
  [`docs/followups/done/cross-section-prefactor-threshold-cancellation.md`](docs/followups/done/cross-section-prefactor-threshold-cancellation.md).

- **Every compiled spectrum and cross section is now computed in Rust,
  and 27 of the 41 entry points return the identical double.** The
  fourteen that move are listed below with the worst relative shift
  measured against the values 2.1.0 produced, over the reference grids in
  `test/parity/cases.py` on the platform they were captured on
  (macOS/arm64). Each figure is the largest single-value disagreement
  anywhere in that entry point's grid, not a typical one; the median
  value in every case is unchanged bit-for-bit.

  | Entry point | Worst relative shift | Cause |
  | --- | --- | --- |
  | `scalar_mediator_decay_spectrum` | 5.3e-12 | boost quadrature |
  | `dnde_decay_s`, `dnde_decay_s_pt` (positron) | 2.3e-12 | boost quadrature |
  | `dnde_decay_v`, `dnde_decay_v_pt` (positron) | 1.5e-12 | boost quadrature |
  | `dnde_decay_v`, `dnde_decay_v_pt` (photon) | 1.2e-12 | boost quadrature |
  | `hazma.spectra.dnde_photon_charged_rho` | 1.5e-13 | boost quadrature |
  | `VectorMediator.thermal_cross_section` | 2.1e-14 | Bessel-`K` prefactor |
  | `hazma.spectra.dnde_positron_charged_pion` | 5.5e-15 | quadrature |
  | `hazma.spectra.dnde_photon_neutral_rho` | 3.2e-15 | boost quadrature |
  | `ScalarMediator.thermal_cross_section` | 3.1e-15 | Bessel-`K` prefactor |
  | `hazma.spectra.dnde_photon_charged_pion` | 2.6e-15 | quadrature |
  | `hazma.spectra.dnde_neutrino_charged_pion` | 9.7e-16 | quadrature |

  The four mediator decay entry points are what
  `ScalarMediator`/`VectorMediator` (and their `HiggsPortal`,
  `HeavyQuark`, `KineticMixing` and `QuarksOnly` subclasses) call from
  `total_spectrum`, `spectra`, `total_positron_spectrum` and
  `positron_spectra`; each `_pt` twin is bit-for-bit identical to its
  array form. **"Quadrature" means one specific thing:** hazma no longer
  calls `scipy.integrate.quad`, which binds Fortran QUADPACK, but an
  in-tree Rust translation of the same netlib QUADPACK routines. It is
  the same algorithm reaching a different-by-one-bit interval-bisection
  decision, not a different method — established by driving the boost
  integrand to a constant, at which point every mediator channel agrees
  with 2.1.0 **to within one ulp**. The remaining 27 entry points — the
  seven tabulated meson photon spectra, both muon spectra, the neutral
  pion, the neutrino muon, and all 16 mediator cross sections other than
  the two ⟨σv⟩ above — are bit-for-bit identical.

  This is a *tightening*, not a loosening: the parity suite's tolerance
  budgets were narrowed for all fourteen of these entry points on the
  measurements above and widened for none, and neither of the two
  as-ported opening budgets (`QUAD` at 1e-8, `NESTED` at 1e-6) is claimed
  by any entry point any more.

- **Four behavior changes at invalid or degenerate inputs.** No valid
  call is affected by any of them.
  1. The seven tabulated meson photon spectra return `NaN` for a `NaN`
     photon energy where they raised `IndexError`, and raise `ValueError`
     for a `NaN` parent energy where they raised `AssertionError`. Cython
     `assert`s became unconditional raises across the port; every
     exception the old code raised *explicitly* keeps its type.
  2. The array-argument mediator decay spectra (`dnde_decay_v`,
     `dnde_decay_s` and their positron twins) raise `ValueError` rather
     than `TypeError` for a scalar energy, and now accept a `list` where
     they refused one. The Python wrappers dispatch a scalar to the `_pt`
     form, so no working call reaches either path.
  3. Two `TypeError` messages lost their Cython wording, which advised
     `use 'cython.cpow(True)'` — a compiler directive that no longer
     exists. Raised at the degenerate masses `e_cm = 2 m_x`, `m_s = 2 m_l`
     and `m_v = 2 m_π`; the type is unchanged.
  4. The mediator decay spectra no longer emit `scipy`'s
     `IntegrationWarning`. The Cython discarded the error estimate the
     warning refers to, so no value ever depended on it.

- **Packaging: one abi3 wheel per platform, built by maturin.** A release
  publishes two wheels (`cp310-abi3` for macOS arm64 and manylinux
  x86_64) plus an sdist, replacing a ten-wheel matrix of cp310–cp314 ×
  two platforms. The sdist carries `hazma/` and `rust/` only — 264 files
  against 2.1.0's 415 — and building from it requires `cargo` on `PATH`,
  because `pip` cannot install a Rust toolchain for you. Windows and
  linux-aarch64 wheels are still not built; see
  [`docs/followups/todo/wheels-for-aarch64-and-windows.md`](docs/followups/todo/wheels-for-aarch64-and-windows.md).

### Fixed

- **The mediator positron spectra no longer return `NaN` at exactly
  0.510998928 MeV.** At that one positron energy — the legacy electron
  mass, and the first abscissa of the shipped positron tables — every
  continuum mode of `dnde_decay_s`, `dnde_decay_v` and their `_pt` twins
  returned `NaN`, because the C compiler contracted
  `sqrt(eng_p*eng_p - me*me)` into a fused multiply-add whose radicand
  rounds negative by 1.45e-17. The port keeps the fused spelling and
  clamps the radicand at zero, which moves that single double from `NaN`
  to `0.0` — the value both of its neighbors already returned — and no
  other energy's arithmetic. Resolves
  [`docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md`](docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md).

### Known issues

Twelve defects that hazma 2.1.0 already had, found while porting and
**reproduced here rather than repaired**. None is new in
3.0.0 and none changes a number relative to 2.1.0; each is tracked in
[`docs/followups/todo/`](docs/followups/todo/) and most are sequenced for
repair in
[`projects/parity-pinned-defect-repair/`](projects/parity-pinned-defect-repair/PLAN.md).
Repairing them here would have destroyed the port's only evidence that it
changed nothing else. Ordered by how much they can move a published
result:

- **[The boost integral mis-covers its window at both ends.](docs/followups/todo/boost-integral-drops-last-interior-cell.md)**
  The worst defect in the list. When the boosted window falls inside a
  single table cell — the regime every model spectrum passes through near
  threshold — two partial-cell terms overlap and cover about two whole
  cells instead of the sliver between the bounds, and the over-count
  diverges as the parent slows. All seven tabulated photon spectra
  therefore **diverge** instead of converging to their own rest-frame
  value as the parent approaches rest: at `E_γ = m/10` and a parent one
  part in 1e12 above rest, `dnde_photon_eta` returns 767.2 against the
  0.02313 it returns exactly at rest. Separately, when the window reaches
  past the table the final row contributes to nothing at all.
- **[`thermal_cross_section` returns its integrator's initial estimate.](docs/followups/todo/thermal-cross-section-quadrature-never-converges.md)**
  The quadrature never converges, so ⟨σv⟩ is 0.5%–5% off the true
  integral for every `x = m_χ/T` above about 5 — that is, across the
  entire freeze-out region. Relic abundance goes as 1/⟨σv⟩, so any
  `relic_density` computed for a `ScalarMediator`- or
  `VectorMediator`-family model inherits that error roughly linearly.
- **[Four scalar elastic cross sections cancel away every significant bit.](docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md)**
  `sigma_xl_to_xl`, `sigma_xpi_to_xpi`, `sigma_xpi0_to_xpi0` and
  `sigma_xs_to_xs` form a difference of two `atan`s that cancels
  completely near `e_cm = 2 m_x` at small width, returning the wrong sign
  and a fabricated pole. The returned value there is the platform's
  rounding residue, not a number any implementation reproduces.
- **[The charged-pion photon spectrum returns zero in the forward cone.](docs/followups/todo/charged-pion-photon-spectrum-misses-the-forward-cone.md)**
  A hard zero over the top ~25% of the support at `γ_π = 10`, worth 0.041%
  of the yield there and 2.96% at `γ_π = 36`. It **compounds** through the
  ρ rather than merely propagating — the charged ρ keeps only 0.5366 of
  its endpoint at `γ_ρ = 10` where a pure boost predicts 0.945 — and
  reaches the mediator photon spectra, where repairing it would move 2,013
  of 29,295 reference values by up to a factor of 8.8 (at an absolute
  7.3e-10). This one changes the *shape* of a low-energy tail, which is
  the kind of error a limit calculation notices.
- **[The muon positron spectrum divides by its normalization.](docs/followups/todo/positron-muon-spectrum-normalization-inverted.md)**
  `dnde_positron_muon` integrates to 0.0374% below one positron per muon.
  Its neutrino sibling applies the same Michel normalization the right way
  round, so the two files genuinely disagree and only one is wrong.
- **[The mediator positron line misses the electron velocity.](docs/followups/todo/mediator-positron-line-misses-the-electron-velocity.md)**
  The `e⁺e⁻` line integrates to `pw_ee · r` rather than `pw_ee`, `r` being
  the positron's rest-frame velocity: the box's edges carry the factor and
  its height does not. Worth 3.3e-5 at `m = 125` MeV, 1.4e-6 at 600 MeV,
  and divergent as `m → 2 m_e`.
- **[The η′ two-photon line carries one photon instead of two.](docs/followups/todo/eta-prime-two-photon-line-missing-factor-two.md)**
  A missing factor of 2 on the `η′ → γγ` line of `dnde_photon_eta_prime`.
- **[The φ photon lines sit at the daughter meson's energy.](docs/followups/todo/phi-photon-lines-use-the-daughter-meson-energy.md)**
  `dnde_photon_phi`'s lines are placed using the daughter meson's energy
  where the photon's is meant, so they land at the wrong energies.
- **[The charged pion's `π → e ν` neutrino line is added twice.](docs/followups/todo/neutrino-pion-electron-line-counted-twice.md)**
  `dnde_neutrino_charged_pion` sums two contributions and both carry the
  line, measured by continuum subtraction at exactly 2.0000 copies. The
  electron-neutrino yield is 0.0123% high integrated, 0.062% high locally
  on the plateau at `E_π = 200` MeV.
- **[The muon photon spectrum's rest frame stops short of the endpoint.](docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md)**
  `dnde_photon_muon` cuts the rest-frame spectrum at `y = 1 − √r` where the
  kinematic endpoint is `y = 1 − r`, leaving a hard zero over the top
  0.2543 MeV of the support (where the spectrum is 5.34e-7 MeV⁻¹) and a
  discontinuity in `E_μ` at rest.
- **[Both rho photon spectra return the boost integrand at rest.](docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md)**
  At exactly `E_ρ = m_ρ` the rest-frame branch returns the integrand
  rather than the integral, so the result is short by a factor of `E_γ`
  and carries MeV⁻² where the spectrum is MeV⁻¹. The guard is absolute, so
  it fires at that one double and no other.
- **[Two vector cross sections raise `TypeError` at `e_cm = 2 m_x`.](docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md)**
  Two of the six vector channels raise at the annihilation threshold where
  the other four return `inf` or `nan`. Inconsistent rather than wrong,
  but it makes a threshold scan fail on channel choice.

Two further items are contract rather than arithmetic problems and are
tracked the same way:
[the model spectrum dicts reject the scalar energies their docstrings advertise](docs/followups/todo/model-spectra-reject-scalar-energies.md),
and
[the mediator spectra return `0.0` for an unrecognised mode string](docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md)
instead of raising.

## [2.1.0] — 2026-08-03

**This release changes numbers.** Gamma-ray spectra from the GeV
vector-mediator models and every detector-convolved spectrum move. Plots
and limits produced with 2.0.2 will not reproduce exactly. Read the
`Changed` section before upgrading a published analysis.

### Changed

- **Detector-convolved spectra move everywhere.**
  `hazma.parameters.convolved_spectrum_fn`, and with it
  `Theory.total_conv_spectrum_fn` and
  `Theory.total_conv_positron_spectrum_fn`, built the Gaussian detector
  response with a width set by the *reconstructed* energy rather than the
  true energy, then normalized over the true energy. Where a detector's
  resolution degrades with energy this leaked sharp spectral features to
  arbitrarily high reconstructed energies. For a 10 MeV scalar decaying to
  e⁺e⁻ observed by e-ASTROGAM — resolution 0.5% at 10 MeV rising to 20% at
  30 MeV — the FSR spectrum cuts off at ~4.95 MeV, but the convolved
  spectrum carried a spurious ~1e-7 tail from 25 MeV out to 1 GeV. That
  tail is now identically zero and total photon number is conserved to
  3e-5. The response is now R(e | e′) with width set by the true energy
  e′, normalized over the reconstructed energy and not renormalized
  afterwards. The old normalization was incidentally cancelling grid
  undersampling error; removing it exposed an 8% pointwise error at the
  default `n_pts=1000`, so the convolution integral now runs on its own
  grid sized from the finest resolution in range (capped at 200k points).
  This converges to better than 0.05% at a cost of roughly 6× runtime per
  call.

- **GeV vector-mediator photon spectra were a factor of 2 low in FSR.**
  The Altarelli-Parisi functions in `hazma.spectra`
  (`dnde_photon_ap_fermion`, `dnde_photon_ap_scalar`) give the spectrum
  radiated by a *single* particle (α/2π), whereas Eq. 4.6 of
  [arXiv:1907.11846](https://arxiv.org/abs/1907.11846) and the deprecated
  `hazma.utils` implementations give the pair-summed result (α/π). This
  convention is now documented, and five call sites in `VectorMediatorGeV`
  (and `KineticMixingGeV`, `BLGeV`) that applied the per-leg spectrum once
  for a charged pair were corrected: the FSR component of
  `dnde_photon_e_e`, `dnde_photon_mu_mu`, `dnde_photon_pi_pi`,
  `dnde_photon_k_k`, and `dnde_photon_pi_pi_pi0_pi0` **doubles**. The
  corrected e⁺e⁻ result now agrees exactly with the pair-summed
  `hazma.utils.dnde_altarelli_parisi_fermion`.

- **Minimum dependency versions raised** to `numpy>=2.0` and
  `scipy>=1.13`, and `scikit-image` is now a declared runtime dependency.
  The `requires-python` upper bound (`<3.13`) is removed; Python 3.10
  through 3.14 are supported and tested.

- `hazma.spectra.boost` no longer offers SciPy's removed
  `scipy.integrate.quadrature`. The `"quadrature"` method name is retained
  as a backwards-compatible alias for `"quad"`, and the default method is
  now `"quad"`. Results change where `"quadrature"` was previously
  selected.

### Fixed

- `VectorMediatorGeV.dnde_photon_k_k` counted the charged-kaon decay
  spectrum once for a K⁺K⁻ final state; it is now counted twice. The
  decay component of this channel **doubles**, independently of the FSR
  fix above.

- `VectorMediatorGeV.dnde_photon_pi_pi_pi0_pi0` used charged-*kaon* decay
  spectra for a π⁺π⁻π⁰π⁰ final state, copy-pasted from a neighboring
  function. It now uses charged-pion spectra. This channel's decay
  contribution changes shape entirely, not by a constant factor.

- `energy_res` callbacks that accept only scalars (for example
  `lambda e: 0.1 if e < 10 else 0.2`) raised when passed an array. Both
  call sites now route through a helper that uses the vectorized call when
  the callback supports it and falls back to element-wise evaluation
  otherwise. A callback returning a single value for an array input is
  treated as a constant resolution. The convolution grid is also now built
  only when there is a continuum spectrum to convolve, so line-only
  convolutions no longer fail.

- `hazma.limits` now works from a wheel: `hazma/limits/data` was missing
  from the packaged distribution.

- Migrated off APIs removed in NumPy 2.0 and SciPy 1.14/1.15
  (`scipy.integrate.trapz`/`simps`, `np.trapz`, `np.float_`,
  `np.complex_`, `pkg_resources`). User-facing `"trapz"` and `"simps"`
  method names are unchanged.

Thanks to Chris Cappiello for reporting the energy-resolution and
Altarelli-Parisi issues.

## [2.0.2] — 2024-07-30

Baseline. This changelog was introduced after 2.0.2 shipped; earlier
releases are not itemized here. See the
[release history](https://github.com/LoganAMorrison/Hazma/releases) and
the git log for changes prior to this entry.
