# The axial mediator's muon photons ignore the muon polarization

- **Added:** 2026-10-23
- **Source:** `projects/visible-axial-spectra/` Task 6, deferred by user
  decision, and carried into Task 7's cascades
- **Scope:** project
- **Status:** open
- **Triggers / blockers:** a gamma-ray limit on `AxialVectorMediator`
  that is set by the `mu mu` or `pi pi pi0` photon continuum near its
  endpoint, where the angular correlation moves the boosted spectrum
  most, or a cross-check of the model's photon yields against a
  polarization-aware reference.

## Why

Task 6 decays the muons of `mu mu` and of `π → μν` with their spin: the
e⁺ and neutrino yields come from `rust/src/kernels/polarized_muon.rs`,
whose densities carry the `b(x) cos θ` term of the V−A decay. The
radiative photons of `μ → e ν ν̄ γ` do not. They come from Hazma's
`photon_muon` kernel, which ports the unpolarized spectrum of
arXiv:hep-ph/9909265 (Kuno and Okada) and so has no angular term to
boost. For a polarized muon in flight the photon's lab spectrum differs
from the unpolarized one by the boosted `P b_γ(x_γ) cos θ` term, in the
same way the e⁺ spectrum does; the e⁺ mean energy per π⁺ moves by 4% from the
spin, and the photon shift has not been measured.

The lab-frame cascades of `AxialVectorMediator.mediator_decay_yields`
carry the same omission. `Cascade::muon_photons` and
`Cascade::charged_pion_photons` in `rust/src/kernels/polarized_cascade.rs`
fold the unpolarized `dnde_photon_muon` over each muon's lab energy,
although the e⁺ and neutrino yields of that cascade use the muon's lab
helicity.

## What

Derive or transcribe the polarized radiative muon decay photon density
`a_γ(x_γ) + b_γ(x_γ) cos θ`, integrated over the electron and
neutrinos, with the finite electron mass as the infrared regulator that
`photon_muon` already uses; Kuno and Okada, Rev. Mod. Phys. 73, 151
(2001), give the polarized distribution. Add it to
`rust/src/kernels/polarized_muon.rs` as a fourth `Daughter`, or as its
own type sharing the module's boost helpers, so `MuonDecay::dnde` and
`counts` serve photons too. Then route the photon arms of
`axial_yields::Channel::photons` and
`ThreePionParents::charged_pion_photons` through it, with the pion
chain's muon fully polarized, and the photon arms of the cascade,
`Cascade::muon_photons` and `Cascade::charged_pion_photons`, through it
with the muon's lab helicity. Pin the unpolarized limit `P = 0` against
`photon_muon` to its tolerance, and measure the moved `mu mu` and
`pi pi pi0` photon yields, in the rest frame and in the cascade, for
`CHANGELOG.md`.

## Entry points

- `rust/src/kernels/axial_yields.rs`: `Channel::photons`,
  `ThreePionParents::charged_pion_photons`, `photon_response` and the
  `# Decays` section that records the omission.
- `rust/src/kernels/polarized_muon.rs`: `MuonDecay`, `Daughter` and
  the boost helpers.
- `rust/src/kernels/polarized_cascade.rs`: `Cascade::muon_photons`,
  `Cascade::charged_pion_photons` and the module docs that record the
  omission.
- `rust/src/kernels/photon_muon.rs`: the unpolarized spectrum and its
  source.
- `projects/visible-axial-spectra/task-notes/task-6-rest-frame-yields.md`:
  the decision and its reasoning. `task-7-polarized-cascades.md` extends
  it to the cascades.

## Risks / open questions

The polarized radiative spectrum is usually quoted differential in the
photon and electron energies with the electron angle; integrating out
the electron at finite mass needs the same care as `photon_muon`'s
port, and the result should reproduce that kernel at `P = 0` before the
`cos θ` term is trusted.
