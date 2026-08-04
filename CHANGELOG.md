# Changelog

All notable user-facing changes to Hazma are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/) over the
public Python API as defined in [`docs/versioning.md`](docs/versioning.md).

Sections: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
`Security`. **Numerical changes go under `Changed` with the magnitude
stated** — a spectrum that moves is a user-facing change even when no
signature did.

## [Unreleased]

Nothing yet.

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
