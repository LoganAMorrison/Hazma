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

The two tables come from different methods, because a coarse grid cannot
resolve the band the repair removes. Near `E = sqrt(s)/2` a 160-point
log grid is spaced about as widely as that band. On such a grid one muon
leg at 300 MeV loses 6.10% rather than 5.75%, and one pion leg at
500 MeV loses 0.718% rather than 1.06%.

**One leg, equal pair.** The share of the photon energy removed is
`∫ E dN/dE` from the endpoint `E_end = (s − 4m²)/(2 sqrt(s))` to the old
cut `E_old = (s − e m²)/(2 sqrt(s))`, divided by the same integral from 0
to `E_old`, both by adaptive quadrature. A massless partner keeps exactly
the pre-repair cut, so it stands in for the "before" spectrum, and the
command runs on this tree alone:

```python
import numpy as np
from scipy import integrate
from hazma import spectra
from hazma.parameters import muon_mass as m

sqrts = 300.0
s = sqrts**2
e_end = (s - 4 * m**2) / (2 * sqrts)
e_old = (s - np.e * m**2) / (2 * sqrts)

def f(e):
    return e * spectra.dnde_photon_ap_fermion(np.array([e]), s, m, partner_mass=0.0)[0]

total = integrate.quad(f, 0.0, e_old, points=[e_end], limit=500)[0]
removed = integrate.quad(f, e_end, e_old, limit=200)[0]
print(removed / total)  # 5.75e-2
```

```text
          sqrt(s)=2.2m  300 MeV   500 MeV   1 GeV     2 GeV
e         3.5e-1        5.3e-8    1.7e-8    3.9e-9    8.8e-10
mu        3.5e-1        5.8e-2    6.6e-3    8.2e-4    1.4e-4
pi        3.4e-1        4.3e-1    1.1e-2    2.9e-4    1.2e-5
k         3.4e-1        -         -         8.5e-1    5.3e-3
```

**Public values.** These are measured against `origin/master` at
`ac5484cc`, with "before" evaluated from a detached checkout of that
commit sharing the same `hazma._core`. Monte-Carlo phase space was seeded
identically in both runs. Each spectrum was sampled from 0.01 MeV to half
the center-of-mass energy on the union of 160 log-spaced and 4000
evenly spaced energies, which resolves the removed band. "Photon energy"
is the trapezoidal `∫ E dN/dE` over that grid. The table gives its
relative change:

```text
spectra.dnde_photon mu mu, cme=300            -9.6e-3
spectra.dnde_photon pi pi, cme=500            -3.0e-3
spectra.dnde_photon mu pi, cme=400            -1.0e-2
spectra.dnde_photon e mu pi, cme=600          -6.3e-2
spectra.dnde_photon ve mu mu, cme=400         -2.9e-3
spectra.dnde_photon pi pi pi0, cme=600        -4.1e-5
VectorMediatorGeV total, e_cm=250             -2.9e-3   (mu mu -1.8e-2)
VectorMediatorGeV total, e_cm=500             -1.2e-3
VectorMediatorGeV total, e_cm=1000            -5.8e-5
VectorMediatorGeV total, e_cm=2000            -7.2e-6
RHNeutrino(300, 1e-3, "e") total              -1.9e-2   (e pi -1.14e-1)
RHNeutrino(300, 1e-3, "mu") total             -2.5e-3
RHNeutrino(600, 1e-3, "mu") total             -2.3e-3
```

The model is `VectorMediatorGeV(mx=100, mv=1000, gvxx=1, gvuu=1,
gvdd=-1, gvss=0, gvee=1, gvmumu=1)` with no neutrino couplings. Every
moved value falls, except in `RHNeutrino(300, 1e-3, "e")`'s `ν e μ`
channel. There, 1723 of the 4158 energies between 3.3 and 148 MeV rise,
by up to 1.1%, and the model's total rises at 661 energies by up to
4.9e-4. The channel's invariant-mass distributions carry negative
probabilities, so removing FSR from those bins raises the sum. That
defect is
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
