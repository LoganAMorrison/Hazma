# ADR 0003 Embed the longitudinal meson poles through the Proca contact

**Date:** 2026-10-17
**Status:** Accepted
**Scope:** Project-scoped within visible-axial-spectra.

## Context

An axial source `Z'_mu qbar gamma^mu gamma5 G_A q` mixes with the neutral
pseudoscalars through `-c_P Z'.dP`, `c_P = f Tr(G_A lambda_P)`. The mixing is
proportional to the source momentum `k`, so it never reaches a physical
on-shell `Z'`, but it does reach the dark matter through the Proca term
`k^mu k^nu/(s M^2)`. Two shortcuts are tempting and wrong: treating the
pole as a spin-zero mediator with its own Breit-Wigner, and inserting the
mediator width in a denominator shared by the transverse and longitudinal
parts. Version 1 of the mediator note did the second, which put a spurious
zero in the s-wave `A_L` term at `s = M^2` (note `sec:ax`, eq:ax-master).
Task 4 also needs the `eta`-`eta'` mixing, the light-quark anomaly, and the
invisible `chi chibar` cut in one consistent block, so that Task 5's rates and
Task 15's invisible branching ratios read the same residues.

## Decision

- **The Proca contact carries the poles.** Contracting the source current
  with the propagator leaves `-J.k/M^2` for any `k`-parallel hadronic current,
  so the longitudinal response is a non-resonant contact with no pole at
  `s = M^2`. The width dresses only the transverse term. The hadronic object
  Task 4 returns is therefore independent of `M` and of `Gamma_Z'`, except
  through an invisible cut, whose strength carries `1/M^2`.
- **One mixing insertion, guarded.** The full quadratic kernel over
  `(a, pi3, eta8, eta0)`, with `a` the source-longitudinal mode
  (`K_aa = M^2`, `K_ai = i sqrt(s) c_i`), defines the exact amplitude
  `h_eff/D_a` with `h_eff = h_a - K_am D^-1 h_m` and
  `D_a = M^2 - s c^T D^-1 c`. Task 4 keeps `h_eff`, which is first order in
  the mixing, and drops the Schur correction to `D_a`. Its size per pole,
  `epsilon_P = |c_P^2 s/(M^2 (s - m_P^2 + i m_P Gamma_P))|`, is exposed as
  `AxialMesonPoles.mixing_expansion` with the guard `MIXING_GUARD = 0.1`
  (note `sec:ax-match`). The exact total, `|s c^T D^-1 c|/M^2`, is at most
  the sum over the three poles, so the per-pole guard can understate it up
  to threefold. A rate must reject points above the guard; the form factor
  does not know `M` and cannot.
- **The meson block is inverted whole.** `D(s)` is the full complex
  `3 x 3` inverse propagator in the physical basis `(pi0, eta, eta')`, with
  the single LO angle `theta_P = -15.4` degrees and a common `f = 92.2` MeV
  in the isospin limit. Any absorptive insertion is added to the whole
  matrix, not to one diagonal width, because a rotation does not commute
  with adding a fresh diagonal width. The invisible cut is a rank-one update
  along the source direction `a_P = Tr(G_A lambda_P)` itself, so
  Sherman-Morrison gives the exact inverse in closed form,
  `a^T D^-1 b = a^T Delta^-1 b/(1 + i gamma a^T Delta^-1 a)`, with no
  linear solve. A meson with `a_P = 0` drops out exactly.
- **The invisible cut is inserted once.** Moving the `a`-line pair cut onto
  the mesons gives `y_P = kappa c_P`, `kappa = 2 m_chi g_A^chi/M^2`, and the
  rank-one absorptive matrix `i s beta_chi kappa^2 c c^T/(8 pi)`. Its
  dispersive part renormalizes the pole inputs and is omitted. At each pole
  it adds exactly `Gamma(P -> chi chibar) = m_P beta_chi y_P^2/(8 pi)`
  (note `eq:ax-inv`) to the visible width. It is not a leading effect: at
  each pole `Gamma_inv/Gamma_P = epsilon_P Pi_aa/M^2`, with `Pi_aa` the
  `a`-line pair cut, so wherever the guard passes it is smaller than the
  dropped real Schur term by `Pi_aa/M^2`. It is kept because it is the
  residue's physical consequence, which Task 15 bounds, and because a
  line shape near an open pair threshold should carry the width it
  implies.
- **Light quarks reach `gamma gamma` only through the poles.** The
  Bardeen-form WZW has no direct source-photon-photon vertex, so the light
  anomaly is the pole sum, which tends to `N_c sum_q g_A^q Q_q^2` for
  `s >> m_P^2` by completeness of the U(3) generators. Leptons and charm,
  which have no mesons here, enter through their perturbative triangles
  `N_c Q_f^2 g_A^f [1 - tau_f f(tau_f)]` (note `eq:ax-gg`). Then
  `A_L(gamma gamma) = s alpha^2 |A(s)|^2/(8 pi^3)` and `A_T = 0`.

## Consequences

- **Positive:** The returned densities are hadronic and mediator-free, as
  `AxialSpectralResult` declares of every source density, so Task 5's
  master formula applies them unchanged. Task 15's invisible widths come
  from the same `c_P` the rates use, so a limit and a line cannot disagree
  about a residue.
- **Negative:** The first-order treatment fails inside narrow peaks even
  for small couplings (the note's example has `epsilon_pi = 2.4` at the
  `pi0` peak for `g_A^u = -g_A^d = 1e-3`, `M = 500` MeV), and the guard is
  a rejection, not a resummation. A UV completion with a physical
  pseudoscalar `A` interferes with the contact and multiplies the residue
  by `m_A^2/(m_A^2 - m_P^2)`; this ADR does not cover it.
- **Mitigation:** The guard value and its formula are exported, so Task 13's
  validity table can declare the region rather than each caller rederiving
  it. Task 12 owns the UV completions, and a physical `A` enters there as a
  separate coherent contribution.
