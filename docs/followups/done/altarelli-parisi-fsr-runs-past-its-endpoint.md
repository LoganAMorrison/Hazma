# The Altarelli-Parisi FSR spectra run past the photon endpoint

- **Added:** 2026-10-24
- **Source:** the class-fix sweep of visible-axial-spectra Task 8
  ([PR #133](https://github.com/LoganAMorrison/Hazma/pull/133)), whose
  `rust/src/kernels/axial_radiation.rs` clips the same spectrum at
  `(Q² − 4m²)/2Q`
- **Scope:** cross-cutting
- **Status:** done. `hazma/spectra/altarelli_parisi.py::_dndx_photon_fsr`
  keeps the spectrum where `s(1 − x)/m² ≥ max(e, (1 + m_Y/m)²)`, and the
  four public functions take `partner_mass` (`m_Y`, by default the
  radiator's own mass). `hazma/spectra/_nbody.py` passes each radiator's
  partner. Both endpoints are pinned in
  `test/spectra/test_altarelli_parisi.py`. No parity-corpus value moves,
  because the corpus evaluates only compiled entry points and none of
  them calls this module, so the repair takes no `C<n>` label.
- **Triggers / blockers:** none.

## Why

`_dndx_photon_fsr` kept the spectrum wherever the bracket
`ln(s(1 − x)/m²) − 1` was positive, which is `x < 1 − e m²/s`. A radiator
of mass `m` recoiling against a partner of mass `m_Y` in a system of
invariant mass `sqrt(s)` can emit a photon only up to
`x = 1 − (m + m_Y)²/s`. For an equal pair that is `1 − 4m²/s`, inside
the bracket's zero, so `dnde_photon_ap_fermion` and
`dnde_photon_ap_scalar` were nonzero past the endpoint. For a muon pair at
`sqrt(s) = 300` MeV the endpoint is 75.58 MeV, but the spectrum ran to
99.42 MeV and was 3.3e-6 MeV⁻¹ at the midpoint of the excess.

A fixed `4m²` clip would have been right for every equal pair but wrong
for `hazma.spectra.dnde_photon`, which applies the AP spectrum to every
pair of a final state. For a massless partner the endpoint is
`1 − m²/s`, past the bracket's zero, so the old cut was already the
binding one. A `4m²` clip would have deleted real radiation there,
including every `(ν, μ)` pair with invariant mass between `1.65 m_μ` and
`2 m_μ`. The partner-aware cut leaves a partner lighter than
`(sqrt(e) − 1) m ≈ 0.649 m` exactly as it was.

## Measurement

Measured against `origin/master` at `ac5484cc`, with "before" evaluated
from a detached checkout of that commit sharing the same `hazma._core`.
Monte-Carlo phase space was seeded identically in both runs. Spectra
were sampled on 160 log-spaced energies from 0.01 MeV to half the
center-of-mass energy, and "photon energy" is the trapezoidal
`∫ E dN/dE` over that grid.

One leg, equal pair, share of the photon energy removed:

```text
          sqrt(s)=2.2m  300 MeV   500 MeV   1 GeV     2 GeV
e         3.5e-1        -         1.7e-8    3.9e-9    8.8e-10
mu        3.5e-1        5.8e-2    6.6e-3    8.2e-4    1.3e-4
pi        3.4e-1        -         1.1e-2    2.9e-4    1.1e-5
k         3.4e-1        -         -         -         5.3e-3
```

Public values, relative change of the photon energy:

```text
spectra.dnde_photon mu mu, cme=300            -1.0e-2
spectra.dnde_photon pi pi, cme=500            -2.1e-3
spectra.dnde_photon mu pi, cme=400            -9.8e-3
spectra.dnde_photon e mu pi, cme=600          -6.3e-2
spectra.dnde_photon ve mu mu, cme=400         -3.0e-3
spectra.dnde_photon pi pi pi0, cme=600        -4.0e-5
VectorMediatorGeV total, e_cm=250             -3.1e-3   (mu mu -1.9e-2)
VectorMediatorGeV total, e_cm=500             -1.2e-3
VectorMediatorGeV total, e_cm=1000            -5.8e-5
VectorMediatorGeV total, e_cm=2000            -6.3e-6
RHNeutrino(300, 1e-3, "e") total              -1.9e-2   (e pi -1.15e-1)
RHNeutrino(300, 1e-3, "mu") total             -2.6e-3
RHNeutrino(600, 1e-3, "mu") total             -2.3e-3
```

The model is `VectorMediatorGeV(mx=100, mv=1000, gvxx=1, gvuu=1,
gvdd=-1, gvss=0, gvee=1, gvmumu=1)` with no neutrino couplings. Every
moved value falls, except in `RHNeutrino`'s `ν e μ` channel, where 42 of
160 energies rise by up to 0.2%. Its invariant-mass distributions carry
negative probabilities, so removing FSR from those bins raises the sum;
that defect is
[its own follow-up](../todo/rhn-three-body-invariant-mass-probabilities-go-negative.md).

## Entry points

- `hazma/spectra/altarelli_parisi.py` — `_dndx_photon_fsr`.
- `hazma/spectra/_nbody.py` — `_make_fsr`, `_dnde_two_body`,
  `_dnde_photon_fsr`.
- `hazma/vector_mediator/_gev/spectra.py` — equal-pair callers, which take
  the default.
- `hazma/utils.py` — the deprecated `dnde_altarelli_parisi_*` keep their
  own copy of the old cut, which is
  [its own follow-up](../todo/deprecated-altarelli-parisi-keeps-the-old-endpoint.md).
