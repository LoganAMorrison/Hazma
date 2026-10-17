# Scope sources and acceptance references

**Audience:** Tasks 3-16.
**Nature:** Grounded facts: reference values with their source labels, the
channel validity table, and the map from tasks to the integration plan.

All labels refer to the mediator extension note, version 11. Values are the
current, corrected ones; superseded values from earlier note versions are not
used as gates. Agreement targets follow the integration plan: analytic
identities at 1e-10 to 1e-8 relative where conditioning permits, deterministic
quadrature at 1e-6, binned cascades at 1e-3, Monte Carlo within its reported
error.

## Acceptance reference values

| Quantity | Value | Note label | Task |
|---|---|---|---|
| Master formula vs spin-1 f fbar rate at zero width | 5e-16 relative | `sec:ax-match` | 5 |
| Master formula vs explicit spinors, all V and A couplings nonzero | 2e-14 relative | `sec:ax-match` | 5 |
| Pure-vector limit vs Hazma sigma v(pi+ pi-), Gamma(V -> pi pi), Gamma(V -> pi0 gamma) | <= 3e-14 relative | `sec:hazma-pi0g` | 11 |
| a1 fit (m_A = 1230 MeV, Gamma_A = 420 MeV): kappa_A, beta | 0.1539, 0.9408 | `sec:ax-match` | 11 |
| BR(tau -> pi- pi- pi+ nu), BR(tau -> pi- 2pi0 nu) from the fit | 9.03%, 9.23% (PDG 8.99, 9.26) | `sec:ax-match` | 11 |
| Mean e+ energy per pi+ at rest, polarized / unpolarized muon | 36.94069288635647 / 38.430221 MeV | `sec:k22` | 6 |
| Cascade f_eff, A -> pi+ pi- pi0, 600/450 MeV (spin-corrected; unpolarized 0.4022) | 0.3967 | `tab:spd-cascade` | 9 |
| Cascade f_eff, A -> pi+ pi- pi0, 900/500 MeV (spin-corrected; unpolarized 0.3750) | 0.3705 | `tab:spd-cascade` | 9 |
| gamma gamma line f_eff at m_chi = 67.488 MeV (pi0) and 273.93 MeV (eta) | 0.724, 0.601 | `sec:spd-feff` | 9 |
| Secluded L_mu-L_tau: sigma v(chi chibar -> Z'Z') | 3.73093e-9 GeV^-2 = 4.35523e-26 cm^3/s | `sec:axl6-cmb` | 14 |
| Secluded L_mu-L_tau: conditional f_eff, p_ann / Planck (3.2e-28) | 0.952771, 1.286% | `sec:axl6-cmb` | 9 |
| NA62 invisible-pi0 bound on kappa = g_A^chi g_q/M^2 at m_chi = 60 MeV | <= 5.34e-6 GeV^-2 | `sec:ax-inv` | 15 |
| Illustrative hypercharge card excluded by NA62 | 1.355 MeV < m_chi < m_pi0/2 - 5.5 eV | `sec:ax` | 15 |
| Illustrative hypercharge card embeddability in H1H2S | M_Z' >= 15.55 GeV or abs(g_A^chi) <= 0.0268 | `sec:ax` | 12 |
| Resonance card: Gamma_Z', tau_Z' (QED-corrected widths, matched pi0 gamma) | 1.912926e-15 GeV, 3.440865e-10 s | `sec:c8` | 16 |
| Resonance card: Gamma(Z' -> pi0 e+ e-), its BR, total width, BR_inv | 1.444933e-24 GeV, 7.553524e-10, 1.9129256950e-15 GeV, 0.9749965435 | `sec:c10` | 16 |
| Resonance card relic abundance (external reference, not a Hazma output) | 0.119307 +- 0.000030 | `sec:n9` | 16 |
| Restricted LO resolved-photon fractions at 450/500 MeV | about 9.12e-4 / 3.14e-3 | integration plan | 8 |

The resonance card is `m_chi = 99.987725 MeV`, `M_Z' = 200 MeV`, `abs(g_A^chi) =
3e-4`, `g_au = 3e-8`, `G_A = g_au diag(1, -1, -1)`. The illustrative hypercharge
card is `g' z_H1 = 2e-3`, `M_Z' = 2.5 GeV`, `g_A^chi = 1`; it is a
simplified-model reference that is NA62-excluded below the pi0 pole, unscreened,
and not embeddable in the minimal completion.

## Channel validity (note `sec:ax-match`)

| Final state | Mechanism | Partial wave | Declared validity |
|---|---|---|---|
| pi+ pi- | isovector vector: contact + rho | s (g_V^chi), p (g_A^chi) | up to about 1 GeV |
| K Kbar | vector: contact + rho, omega, phi | s/p | indicative |
| pi0 gamma, eta gamma, eta' gamma | vector: WZW + VMD | s/p | up to about 1.1 GeV; eta' order of magnitude |
| pi+ pi- pi0 | vector isoscalar (omega, phi); axial isovector (contact, pion pole, Z' rho pi, a1 -> rho pi) | V: s; A: p (A_T) + s (A_L) | up to about 1.1 GeV (V), 1.6 GeV (A) |
| 3pi0 | axial, pion pole only | s (A_L) | as above |
| eta pi pi | vector box anomaly; axial eta and eta' poles | s/p; s | qualitative |
| gamma gamma | axial longitudinal only | s | LO poles + triangles |
| l+ l-, nu nubar | tree level | s/p | exact |
| Z'Z' | t/u-channel chi exchange only (no dark Higgs) | s | simplified model |

The LO three-pion current of Task 2 is declared to 600 MeV. K Kbar, eta' gamma,
isoscalar-axial transverse channels and anything above 1.6 GeV are outside the
resonance model's range.

## Map to the integration plan and the gap list

| Task | Integration-plan PR | Gap-list label |
|---|---|---|
| 1 | PR0 | - |
| 2 | PR1 | - |
| 3 | PR0-PR1 Rust port | - |
| 4 | PR2 | - |
| 5 | PR3 | - |
| 6 | PR4 | - |
| 7 | PR5 | - |
| 8 | PR6 | - |
| 9 | PR7 | - |
| 10 | PR8 | - |
| 11 | PR3 follow-on | AX-3b |
| 12 | - | UV-1, UV-2 |
| 13 | - | VAL-1 |
| 14 | - | PAIR-1 |
| 15 | - | LAB-1 |
| 16 | - | DOC-1, PKG-1 |

## Outside this project

Standalone PRs against existing models, each with its own CHANGELOG entry:

- FIX-1: the vector-mediator pi0 gamma cross section prefactor (1/3456 against
  the 1/384 implied by its own width, `rust/src/kernels/vector_xs.rs`) together
  with the test that pins the wrong value; the VMD pi0 gamma scalar-branch unit
  conversion; the inverted alpha(s) mask in
  `hazma/form_factors/vector/_alpha.py`.
- FIX-2: `_f_eff_helper` positron doubling and electron-energy argument, and the
  Dirac population factor, for existing models.
- PKG-2: removal of `hazma.experimental.pseudo_scalar_mediator` and
  `hazma.experimental.axial_vector_mediator`, which fail to import in every
  released version; rebuild of the stale Rust core.

Follow-on project: `pseudo-scalar-mediator` (gap-list PS-0, PS-1, PS-2),
requiring this project through Task 9 and Task 13.
