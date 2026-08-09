//! The two constant tables hazma's Cython layer compiles against.
//!
//! Hazma carries **two** tables that disagree, and the disagreement is
//! load-bearing: `hazma/_utils/constants.pxd` is `include`d by the twelve
//! spectra extensions and `hazma/_utils/legacy_parameters.pxd` by the four
//! mediator ones, and the two give different masses, branching ratios and
//! fine-structure constants for the same particles. Merging them would move
//! published spectra, so this module reproduces both, verbatim, in two
//! namespaces — [`pdg`] and [`legacy`] — exactly as
//! `projects/cython-to-rust/rules.md` rule 4 (Constants 1) requires.
//! Consolidation is a separate, declared numerical change; do not "fix" a
//! value here.
//!
//! # Sources
//!
//! Every value is transcribed from the `.pxd` byte-for-byte, including the
//! `± uncertainty` annotations, which are the Cython's own. Those
//! annotations are the only provenance the Cython recorded — it names no
//! edition — so the citations below are for the *tables*, not per value,
//! and this port deliberately re-sources nothing:
//!
//! - Particle masses, widths and branching ratios: Particle Data Group,
//!   *Review of Particle Physics* — <https://pdg.lbl.gov/>, whose constants
//!   reviews are indexed at
//!   <https://pdg.lbl.gov/2025/reviews/constants_atomic_and_related.html>.
//!   Every mass in [`pdg`] is bit-equal to its counterpart in the
//!   pure-Python `hazma/parameters.py` (checked, all fourteen), which cites
//!   "PDG March 2022"; several entries (m(tau) = 1776.86 +- 0.12 MeV, for
//!   one) are older than the current edition. A handful of branching ratios
//!   are marked in the source as taken from Pythia 8.306 rather than the
//!   PDG, and say so inline.
//! - Fine-structure constant: [`pdg::ALPHA_EM`] is `1/137.035999084(21)`, a
//!   pre-2022 CODATA adjustment — CODATA 2022 revised α⁻¹ to
//!   137.035999177(21) (Mohr, Newell, Taylor & Tiesinga,
//!   [arXiv:2409.03787](https://arxiv.org/abs/2409.03787);
//!   <https://physics.nist.gov/cgi-bin/cuu/Value?alphinv>). [`legacy`] uses
//!   the cruder `1/137`, and `hazma/parameters.py:205` a third value again,
//!   `1/137.04` — the masses are the only part of the tree that agrees with
//!   itself. All three are kept as they are; reconciling them is the
//!   consolidation this port is forbidden to perform.
//!
//! # Layout
//!
//! | Module | Ported from | Consumed by |
//! | --- | --- | --- |
//! | [`pdg`] | `hazma/_utils/constants.pxd` | the `hazma/spectra/**` kernels |
//! | [`legacy`] | `hazma/_utils/legacy_parameters.pxd` | the four mediator spectrum kernels |
//! | [`derived`] | module-local `DEF`s in individual `.pyx` | that one kernel module |
//!
//! `test/test_core_constants.py` re-parses all three sources and this file
//! and asserts every value is bit-equal, so nothing here rests on careful
//! typing. It also reconstructs [`derived`]'s seven hard-coded literals from
//! the formulas their comments imply and pins which table each came from.
#![allow(clippy::excessive_precision)]
// The literals are transcribed from the `.pxd` verbatim so that a diff
// against the Cython source is meaningful. Several carry trailing zeros
// (`0.9998770`, `0.0023900`) that clippy would rather see dropped; dropping
// them would make the two files stop matching character-for-character
// without changing a single f64.

/// `hazma/_utils/constants.pxd` — the PDG-era table.
///
/// `include`d by all twelve `hazma/spectra/**` extensions -- a different
/// twelve from the fourteen masses below, which is a collision worth
/// reading twice. All fourteen masses are bit-equal to
/// `hazma/parameters.py`'s; its `ALPHA_EM` is not.
pub mod pdg {
    // =========================================================
    // ---- Masses in MeV --------------------------------------
    // =========================================================

    pub const MASS_E: f64 = 0.5109989461; // m[e-] = 0.5109989461 ± 3.1e-09
    pub const MASS_MU: f64 = 105.6583745; // m[mu-] = 105.6583745 ± 2.4e-06
    pub const MASS_TAU: f64 = 1776.86; // m[tau-] = 1776.86 ± 0.12
    pub const MASS_PI0: f64 = 134.9768; // m[pi0] = 134.9768 ± 0.0005
    pub const MASS_PI: f64 = 139.57039; // m[pi+] = 139.57039 ± 0.00018
    pub const MASS_ETA: f64 = 547.862; // m[eta] = 547.862 ± 0.017
    pub const MASS_ETAP: f64 = 957.78; // m[eta'(958)] = 957.78 ± 0.06
    pub const MASS_K: f64 = 493.677; // m[K+] = 493.677 ± 0.016
    pub const MASS_K0: f64 = 497.611; // m[K0] = 497.611 ± 0.013
    pub const MASS_KL: f64 = 497.611; // m[K(L)0] = 497.611 ± 0.013
    pub const MASS_KS: f64 = 497.611; // m[K(S)0] = 497.611 ± 0.013
    pub const MASS_RHO: f64 = 775.26; // m[rho(770)0] = 775.26 ± 0.23
    pub const MASS_OMEGA: f64 = 782.66; // m[omega(782)] = 782.66 ± 0.13
    pub const MASS_PHI: f64 = 1019.461; // m[phi(1020)] = 1019.461 ± 0.016

    // =========================================================
    // ---- π⁺ Branching Ratios --------------------------------
    // =========================================================

    // BR(μ+, νμ) = (99.98770±0.00004) %
    pub const BR_PI_TO_MU_NUMU: f64 = 0.9998770;
    // BR(e+, νe) = ( 1.230±0.004  )×10−4
    pub const BR_PI_TO_E_NUE: f64 = 1.230e-4;

    // =========================================================
    // ---- π⁰ Branching Ratios --------------------------------
    // =========================================================

    // BR(γ, γ) = (98.823±0.034) %
    pub const BR_PI0_TO_A_A: f64 = 98.823e-2;
    // BR(e+, e−, γ) = (1.174±0.035) %
    pub const BR_PI0_TO_E_E_A: f64 = 1.174e-2;
    // BR(e+, e+, e−, e−) = (3.34±0.16 )×10−5
    pub const BR_PI0_TO_E_E_E_E: f64 = 3.34e-5;

    // =========================================================
    // ---- η Branching Ratios ---------------------------------
    // =========================================================

    // BR(γ, γ) = (39.41±0.20) %
    pub const BR_ETA_TO_A_A: f64 = 39.41e-2;
    // BR(π0, π0, π0) = (32.68±0.23) %
    pub const BR_ETA_TO_PI0_PI0_PI0: f64 = 32.68e-2;
    // BR(π0, γ, γ) = ( 2.56±0.22)×10−4
    pub const BR_ETA_TO_PI0_A_A: f64 = 2.56e-4;
    // BR(π+, π−, π0) = (22.92±0.28) %
    pub const BR_ETA_TO_PI_PI_PI0: f64 = 22.92e-2;
    // BR(π+, π−, γ) = ( 4.22±0.08) %
    pub const BR_ETA_TO_PI_PI_A: f64 = 4.22e-2;
    // BR(e+, e−, γ) = ( 6.9±0.4 )×10−3
    pub const BR_ETA_TO_E_E_A: f64 = 6.9e-3;
    // BR(μ+, μ−, γ) = ( 3.1±0.4 )×10−4
    pub const BR_ETA_TO_MU_MU_A: f64 = 3.1e-4;
    // BR(μ+, μ−) = ( 5.8±0.8 )×10−6
    pub const BR_ETA_TO_MU_MU: f64 = 5.8e-6;
    // BR(π+, π−, e+, e−)  = ( 2.68±0.11)×10−4
    pub const BR_ETA_TO_PI_PI_E_E: f64 = 2.68e-4;
    // BR(e+, e−, e+, e−) = ( 2.40±0.22)×10−5
    pub const BR_ETA_TO_E_E_E_E: f64 = 2.40e-5;

    // =========================================================
    // ---- ρ⁰ Branching Ratios --------------------------------
    // =========================================================

    pub const BR_RHO_TO_PI_PI: f64 = 0.9988447;
    // BR(π⁰, γ) = 4.7e-4
    pub const BR_RHO_TO_PI0_A: f64 = 4.7e-4;
    // BR(η, γ) = 3.00e-4
    pub const BR_RHO_TO_ETA_A: f64 = 3.00e-4;
    // BR(π⁺, π⁻, π⁰) = 1.01e-4
    pub const BR_RHO_TO_PI_PI_PI0: f64 = 1.01e-4;
    // BR(e⁺, e⁻) = 4.72e-5
    pub const BR_RHO_TO_E_E: f64 = 4.72e-5;
    // BR(μ⁺, μ⁻) = 4.55e-5
    pub const BR_RHO_TO_MU_MU: f64 = 4.55e-5;
    // BR(π⁰, π⁰, γ) = 4.5e-5
    pub const BR_RHO_TO_PI0_PI0_A: f64 = 4.5e-5;
    // BR(π⁺, π⁻, π⁺, π⁻) = 1.8e-5
    pub const BR_RHO_TO_PI_PI_PI_PI: f64 = 1.8e-5;
    // BR(π⁺, π⁻, π⁰, π⁰) = 1.6e-5
    pub const BR_RHO_TO_PI_PI_PI0_PI0: f64 = 1.6e-5;
    // BR(π⁺, π⁻, γ) = 9.9e-3
    pub const BR_RHO_TO_PI_PI_A: f64 = 9.9e-3;

    // =========================================================
    // ---- ρ⁺ Branching Ratios --------------------------------
    // =========================================================

    pub const RHOP_TO_PI_PI0: f64 = 0.9995502;
    // BR(π±, γ) = 4.5e-4
    pub const RHOP_TO_PI_A: f64 = 4.5e-4;

    // =========================================================
    // ---- K-Long Branching Ratios ----------------------------
    // =========================================================

    // BR(π0, π0, π0) = (19.52±0.12 ) %
    pub const BR_KL_TO_PI0_PI0_PI0: f64 = 19.52e-2;
    // BR(π+, π−, π0) = (12.54±0.05 ) %
    pub const BR_KL_TO_PI_PI_PI0: f64 = 12.54e-2;
    // BR(π±, e∓, νe) = (40.55±0.11 ) %
    pub const BR_KL_TO_PI_E_NUE: f64 = 40.55e-2;
    // BR(π±, μ∓, νμ) =  (27.04±0.07 ) %
    pub const BR_KL_TO_PI_MU_NUMU: f64 = 27.04e-2;
    // BR(π+, π−) = ( 1.967±0.010)×10−3
    pub const BR_KL_TO_PI_PI: f64 = 1.967e-3;
    // BR(π0, π0) = ( 8.64±0.06 )×10−4
    pub const BR_KL_TO_PI0_PI0: f64 = 8.64e-4;
    // BR(γ, γ) = ( 5.47±0.04 )×10−4
    pub const BR_KL_TO_A_A: f64 = 5.47e-4;
    // BR(π0, π±, e∓, ν) = ( 5.20±0.11 )×10−5
    pub const BR_KL_TO_PI0_PI_E_NU: f64 = 5.20e-5;
    // BR(π±, e∓, ν, e+, e−) = ( 1.26±0.04 )×10−5
    pub const BR_KL_TO_PI_E_E_E_NU: f64 = 1.26e-5;

    // =========================================================
    // ---- K-Short Branching Ratios ---------------------------
    // =========================================================

    // BR(π+, π−) = (69.20±0.05) %
    pub const BR_KS_TO_PI_PI: f64 = 69.20e-2;
    // BR(π0, π0) = (30.69±0.05) %
    pub const BR_KS_TO_PI0_PI0: f64 = 30.69e-2;
    // BR(π+, π−, e+, e−) = ( 4.79±0.15)×10−5
    pub const BR_KS_TO_PI_PI_E_E: f64 = 4.79e-5;
    // BR(π±, e∓, νe) =  ( 7.04±0.08)×10−4
    pub const BR_KS_TO_PI_E_NUE: f64 = 7.04e-4;
    // BR(γ, γ) = ( 2.63±0.17)×10−6
    pub const BR_KS_TO_A_A: f64 = 2.63e-6;
    // BR(π+, π−, π0) = ( 3.5+1.1−0.9)×10−7
    pub const BR_KS_TO_PI_PI_PI0: f64 = 3.5e-7;
    // BR(π+, π−, γ) = ( 1.79±0.05)×10−3
    pub const BR_KS_TO_PI_PI_A: f64 = 1.79e-3;
    // BR(π0, γ, γ) =  ( 4.9±1.8 )×10−8
    pub const BR_KS_TO_PI0_A_A: f64 = 4.9e-8;
    // BR(π0, e+, e−) = ( 3.0+1.5−1.2)×10−9
    pub const BR_KS_TO_PI0_E_E: f64 = 3e-9;
    // BR(π0, μ+, μ−) = ( 2.9+1.5−1.2)×10−9
    pub const BR_KS_TO_PI0_MU_MU: f64 = 2.9e-9;

    // =========================================================
    // ---- K⁰ Branching Ratios --------------------------------
    // =========================================================

    pub const BR_K0_TO_KL: f64 = 0.5;
    pub const BR_K0_TO_KS: f64 = 0.5;

    // =========================================================
    // ---- K⁰' (K⁰-star) Branching Ratios ---------------------
    // =========================================================

    // Taken from Pythia8306
    pub const BR_K0STAR_TO_K_PI: f64 = 0.6649467;
    pub const BR_K0STAR_TO_K0_PI0: f64 = 0.3326633;
    pub const BR_K0STAR_TO_K0_A: f64 = 0.0023900;

    // =========================================================
    // ---- K⁺, K⁻ Branching Ratios ----------------------------
    // =========================================================

    // BR(μ+, νμ) = (63.56 ± 0.11) %
    pub const BR_K_TO_MU_NUMU: f64 = 63.56e-2;
    // BR(e+, νe) = (1.582 ± 0.007)×10−5
    pub const BR_K_TO_E_NUE: f64 = 1.582e-5;
    // BR(π+, π0) = (20.67 ± 0.08 ) %
    pub const BR_K_TO_PI_PI0: f64 = 20.67e-2;
    // BR(π+, π+, π−) = (5.583 ± 0.024) %
    pub const BR_K_TO_PI_PI_PI: f64 = 5.583e-2;
    // BR(π+, π0, π0) = (1.760 ± 0.023) %
    pub const BR_K_TO_PI_PI0_PI0: f64 = 1.760e-2;
    // BR(π0, e+, νe) = (5.07 ± 0.04) %
    pub const BR_K_TO_E_NUE_PI0: f64 = 5.07e-2;
    // BR(π0, μ+, νμ)   (3.352 ± 0.033) %
    pub const BR_K_TO_MU_NUMU_PI0: f64 = 3.352e-2;
    // BR(π0, π0, e+, νe) = (2.55 ± 0.04)×10−5
    pub const BR_K_TO_E_NUE_PI0_PI0: f64 = 2.55e-5;
    // BR(π+, π−, e+, νe) =  (4.247 ± 0.024)×10−5
    pub const BR_K_TO_E_NUE_PI_PI: f64 = 4.247e-5;
    // Taken from Pythia8306 (can't find in PDG)
    pub const BR_K_TO_MU_NUMU_PI0_PI0: f64 = 0.0000140;
    // BR(π+, π−, μ+, νμ) =  (1.4 ± 0.9)×10−5
    pub const BR_K_TO_MU_NUMU_PI_PI: f64 = 1.4e-5;
    // BR(e+, νe, e+, e−) =  (2.48 ± 0.20 )×10−8
    pub const BR_K_TO_E_E_E_NUE: f64 = 2.48e-8;
    // BR(μ+, νμ, e+, e−) =  (7.06 ± 0.31 )×10−8
    pub const BR_K_TO_MU_E_E_NUMU: f64 = 7.06e-8;
    // BR(e+, νe, μ+, μ−) =  (1.7 ± 0.5  )×10−8
    pub const BR_K_TO_MU_MU_E_NUE: f64 = 1.7e-8;
    // BR(π+, e+, e−) = (3.00 ± 0.09 )×10−7
    pub const BR_K_TO_PI_E_E: f64 = 3.00e-7;
    // BR(π+, μ+, μ−) = (9.4 ± 0.6  )×10−8
    pub const BR_K_TO_PI_MU_MU: f64 = 9.4e-8;

    // =========================================================
    // ---- K⁺' (K-star) Branching Ratios ----------------------
    // =========================================================

    pub const BR_KSTAR_TO_K0_PI: f64 = 0.6660067;
    pub const BR_KSTAR_TO_K_PI0: f64 = 0.3330033;
    pub const BR_KSTAR_TO_K_A: f64 = 0.0009900;

    // =========================================================
    // ---- Eta' Branching Ratios ------------------------------
    // =========================================================

    // BR(π⁺, π⁻, η) = (42.5 ± 0.5) %
    pub const BR_ETAP_TO_PI_PI_ETA: f64 = 42.5e-2;
    // BR(ρ⁰, γ) = (29.5 ± 0.4) % (including non-resonant π+ + π− + γ)
    pub const BR_ETAP_TO_RHO_A: f64 = 29.5e-2;
    // BR(π⁰, π⁰, η) = (22.4 ± 0.5) %
    pub const BR_ETAP_TO_PI0_PI0_ETA: f64 = 22.4e-2;
    // BR(ω, γ) = ( 2.52 ± 0.07) %
    pub const BR_ETAP_TO_OMEGA_A: f64 = 2.52e-2;
    // BR(γ, γ) = ( 2.307 ± 0.033) %
    pub const BR_ETAP_TO_A_A: f64 = 2.307e-2;
    // BR(π⁰, π⁰, π⁰) = ( 2.50 ± 0.17 )×10−3
    pub const BR_ETAP_TO_PI0_PI0_PI0: f64 = 2.50e-3;
    // BR(μ⁺, μ⁻, γ) = (1.13 ± 0.28)×10−4
    pub const BR_ETAP_TO_MU_MU_A: f64 = 1.13e-4;
    // BR(ω, e⁺, e⁻) = ( 2.0 ± 0.4  )×10−4
    pub const BR_ETAP_TO_OMEGA_E_E: f64 = 2e-4;
    // BR(π⁺, π⁻, π⁰) = (3.61 ± 0.17)×10−3
    // BR(π⁺, π⁻, π⁰) = (3.8 ± 0.5)×10−3 (S-wave)
    pub const BR_ETAP_TO_PI_PI_PI0: f64 = 3.61e-3;
    // BR(π∓, ρ±) = (7.4 ± 2.3)×10−4
    pub const BR_ETAP_TO_PI_RHOP: f64 = 7.4e-4;
    // BR(π⁺, π⁻, π⁺, π⁻) = (8.4 ± 0.9)×10−5
    pub const BR_ETAP_TO_PI_PI_PI_PI: f64 = 8.4e-5;
    // BR(π⁺, π⁻, π⁰, π⁰) = (1.8 ± 0.4)×10−4
    pub const BR_ETAP_TO_PI_PI_PI0_PI0: f64 = 1.8e-4;
    // BR(π⁺, π⁻, e⁺, e⁻) = (2.4 +1.3 −1.0)×10−3
    pub const BR_ETAP_TO_PI_PI_E_E: f64 = 2.4e-3;
    // BR(γ, e⁺, e⁻) = (4.91 ± 0.27)×10−4
    pub const BR_ETAP_TO_E_E_A: f64 = 4.91e-4;
    // BR(π⁰, γ, γ) = (3.20 ± 0.24)×10−3
    // BR(π⁰, γ, γ) = (6.2 ± 0.9)×10−4 (non resonant)
    pub const BR_ETAP_TO_PI0_A_A: f64 = 3.20e-3;

    // =========================================================
    // ---- ω Branching Ratios ---------------------------------
    // =========================================================

    // BR(π⁺, π⁻, π⁰) = 89.2 ± 0.7 %
    pub const BR_OMEGA_TO_PI_PI_PI0: f64 = 89.2e-2;
    // BR(π⁰, γ) = 8.34 ± 0.26 %
    pub const BR_OMEGA_TO_PI0_A: f64 = 8.34e-2;
    // BR(π⁺, π⁻) = 1.53 +0.11 −0.13 %
    pub const BR_OMEGA_TO_PI_PI: f64 = 1.53e-2;
    // BR(η, γ) = 4.5e-4 ± 0.4e-4
    pub const BR_OMEGA_TO_ETA_A: f64 = 4.5e-4;
    // BR(π⁰, e⁺, e⁻) = 7.7e-4 ± 0.6e-4
    pub const BR_OMEGA_TO_PI0_E_E: f64 = 7.7e-4;
    // BR(π⁰, μ⁺, μ⁻) = 1.34e-4 ± 0.18e-4
    pub const BR_OMEGA_TO_PI0_MU_MU: f64 = 1.34e-4;
    // BR(e⁺, e⁻) = 7.39e-5 ± 0.19e-5
    pub const BR_OMEGA_TO_E_E: f64 = 7.39e-5;
    // BR(μ⁺, μ⁻) = 7.4e-5 ± 1.8e-5
    pub const BR_OMEGA_TO_MU_MU: f64 = 7.4e-5;
    // BR(π⁰, π⁰, γ) = 6.7e-5 ± 1.1e-5
    pub const BR_OMEGA_TO_PI0_PI0_A: f64 = 6.7e-5;

    // =========================================================
    // ---- φ Branching Ratios ---------------------------------
    // =========================================================

    // BR(K⁺, K⁻) = (49.2 ± 0.5) %
    pub const BR_PHI_TO_K_K: f64 = 49.2e-2;
    // BR(KL, KS) = (34.0 ± 0.4) %
    pub const BR_PHI_TO_KL_KS: f64 = 34.0e-2;
    // PDG: BR(ρ, π⁰) + BR(ρ⁺, π⁻) + BR(ρ⁻, π⁺) +  BR(π⁺, π⁻, π⁰) = (15.24 ± 0.33) %
    // The below is taken from Pythia8306
    pub const BR_PHI_TO_RHOP_PI: f64 = 0.0420984;
    pub const BR_PHI_TO_RHO_PI0: f64 = 0.0420984;
    pub const BR_PHI_TO_PI_PI_PI0: f64 = 0.0270000;
    // BR(η, γ) = (1.303 ± 0.025) %
    pub const BR_PHI_TO_ETA_A: f64 = 1.303e-2;
    // BR(π⁰, γ) = (1.32 ± 0.06)×10−3
    pub const BR_PHI_TO_PI0_A: f64 = 1.32e-3;
    // BR(e⁺, e⁻) = (2.974 ± 0.034)×10−4
    pub const BR_PHI_TO_E_E: f64 = 2.974e-4;
    // BR(μ⁺, μ⁻) = (2.86 ± 0.19)×10−4
    pub const BR_PHI_TO_MU_MU: f64 = 2.86e-4;
    // BR(η, e⁺, e⁻) = (1.08 ± 0.04)×10−4
    pub const BR_PHI_TO_ETA_E_E: f64 = 1.08e-4;
    // BR(π⁺, π⁻) = (7.3 ± 1.3)×10−5
    pub const BR_PHI_TO_PI_PI: f64 = 7.3e-5;
    // BR(ω, π⁰) = (4.7 ± 0.5)×10−5
    pub const BR_PHI_TO_OMEGA_PI0: f64 = 4.7e-5;
    // BR(π⁺, π⁻, γ) = (4.1 ± 1.3)×10−5
    pub const BR_PHI_TO_PI_PI_A: f64 = 4.1e-5;
    // BR(f₀(980), γ) = (3.22 ± 0.19)×10−4
    pub const BR_PHI_TO_F0980_A: f64 = 3.22e-4;
    // BR(π⁰, π⁰, γ) = (1.12 ± 0.06)×10−4
    pub const BR_PHI_TO_PI0_PI0_A: f64 = 1.12e-4;
    // BR(π⁺, π⁻, π⁺, π⁻) = (3.9 +2.8 −2.2)×10−6
    pub const BR_PHI_TO_PI_PI_PI_PI: f64 = 3.9e-6;
    // BR(π⁰, e⁺, e⁻) = (1.33 +0.07 −0.10)×10−5
    pub const BR_PHI_TO_PI0_E_E: f64 = 1.33e-5;
    // BR(π⁰, η, γ) = (7.27 ± 0.30)×10−5
    pub const BR_PHI_TO_PI0_ETA_A: f64 = 7.27e-5;
    // BR(a₀(980), γ) = (7.6 ± 0.6)×10−5
    pub const BR_PHI_TO_A0980_A: f64 = 7.6e-5;
    // BR(η'(958), γ) = (6.22 ± 0.21)×10−5
    pub const BR_PHI_TO_ETAP_A: f64 = 6.22e-5;
    // BR(μ⁺, μ⁻, γ) = (1.4 ± 0.5)×10−5
    pub const BR_PHI_TO_MU_MU_A: f64 = 1.4e-5;

    // =========================================================
    // ---- Decay Widths ---------------------------------------
    // =========================================================

    pub const WIDTH_E: f64 = 0.0; // Γ[e-] = 0.0 ± 0.0
    pub const WIDTH_MU: f64 = 2.9959837e-16; // Γ[mu-] = 2.9959837e-16 ± 3e-22
    pub const WIDTH_TAU: f64 = 2.267e-09; // Γ[tau-] = 2.267e-09 ± 4e-12
    pub const WIDTH_PI0: f64 = 7.81e-06; // Γ[pi0] = 7.81e-06 ± 1.2e-07
    pub const WIDTH_PI: f64 = 2.5284e-14; // Γ[pi+] = 2.5284e-14 ± 5e-18
    pub const WIDTH_ETA: f64 = 0.00131; // Γ[eta] = 0.00131 ± 5e-05
    pub const WIDTH_ETAP: f64 = 0.188; // Γ[eta'(958)] = 0.188 ± 0.006
    pub const WIDTH_K: f64 = 5.317e-14; // Γ[K+] = 5.317e-14 ± 9e-17
    // DEF WIDTH_K0 = None # Γ[K0] = None ± None
    pub const WIDTH_KL: f64 = 1.287e-14; // Γ[K(L)0] = 1.287e-14 ± 5e-17
    pub const WIDTH_KS: f64 = 7.3508e-12; // Γ[K(S)0] = 7.3508e-12 ± 2.9e-15
    pub const WIDTH_RHO: f64 = 149.1; // Γ[rho(770)0] = 149.1 ± 0.8
    pub const WIDTH_OMEGA: f64 = 8.68; // Γ[omega(782)] = 8.68 ± 0.13
    pub const WIDTH_PHI: f64 = 4.249; // Γ[phi(1020)] = 4.249 ± 0.013

    // =========================================================
    // ---- Other Constants ------------------------------------
    // =========================================================

    // 1/137.035999084(21)
    pub const ALPHA_EM: f64 = 1.0 / 137.035999084; // Fine structure constant.
    pub const RATIO_E_MU_MASS_SQ: f64 = (MASS_E / MASS_MU) * (MASS_E / MASS_MU);

    // FA = 0.0119 ± 0.0001
    pub const F_A_PI: f64 = 0.0119;
    // FV = 0.0254 ± 0.0017
    pub const F_V_PI: f64 = 0.0254;
    // FV slope parameter a= 0.10 ± 0.06
    pub const F_V_PI_SLOPE: f64 = 0.10;

    pub const F_A_K: f64 = 0.042;
    pub const F_V_K: f64 = 0.096;

    pub const DECAY_CONST_PI: f64 = 130.41; // PDG convention
    pub const DECAY_CONST_K: f64 = 156.1; // PDG convention
}

/// `hazma/_utils/legacy_parameters.pxd` — the older table.
///
/// `include`d by the four mediator spectrum extensions
/// (`{scalar,vector}_mediator/*_{decay_spectrum,positron_spec}.pyx`) and by
/// nothing else. Its masses, branching ratios and α differ from [`pdg`];
/// see [`self`] for why they stay that way.
///
/// One departure from verbatim predates this task: the `WIDTHS` section is
/// empty because two malformed, never-referenced entries were deleted on
/// 2026-08-05 (see the comment in the `.pxd`).
pub mod legacy {
    // MASSES (MeV)
    pub const MASS_E: f64 = 0.510998928; // electron
    pub const MASS_MU: f64 = 105.6583715; // muon
    pub const MASS_PI0: f64 = 134.9766; // neutral pion
    pub const MASS_PI: f64 = 139.57018; // Charged pion
    pub const MASS_K0: f64 = 497.61; // neutral kaon
    pub const MASS_K: f64 = 493.68; // charged Kaon
    pub const MASS_ETA: f64 = 547.86; // eta
    pub const MASS_ETAP: f64 = 957.8; // eta prime
    pub const MASS_RHO: f64 = 775.3; // rho
    pub const MASS_OMEGA: f64 = 782.7; // omega

    // BRANCHING RATIOS
    pub const BR_PI0_TO_GG: f64 = 0.9882; // Pi0   -> g   + g
    pub const BR_PI_TO_MUNU: f64 = 0.9998; // pi    -> mu  + nu
    pub const BR_PI_TO_ENU: f64 = 0.000123; // pi    -> e  + nu

    pub const BR_KS_TO_PIPI: f64 = 0.6920; // ks    -> pi  + pi
    pub const BR_KS_TO_PI0PI0: f64 = 0.3069; // ks    -> pi0 + pi0

    pub const BR_KL_TO_PIENU: f64 = 0.4055; // kl    -> pi  + e   + nu
    pub const BR_KL_TO_PIMUNU: f64 = 0.2704; // kl    -> pi  + mu  + nu
    pub const BR_KL_TO_3PI0: f64 = 0.1952; // kl    -> pi0 + pi0  + pi0
    pub const BR_KL_TO_2PIPI0: f64 = 0.1254; // kl    -> pi  + pi  + pi0

    pub const BR_K_TO_MUNU: f64 = 0.6356; // k     -> mu  + nu
    pub const BR_K_TO_PIPI0: f64 = 0.2067; // k     -> pi  + pi0
    pub const BR_K_TO_3PI: f64 = 0.05583; // k     -> pi  + pi  + pi
    pub const BR_K_TO_PI0ENU: f64 = 0.0507; // k     -> pi0 + e   + nu
    pub const BR_K_TO_PI0MUNU: f64 = 0.03352; // k     -> pi0 + mu  + nu
    pub const BR_K_TO_PI2PI0: f64 = 0.01760; // k     -> pi  + pi0 + pi0

    pub const BR_ETA_TO_GG: f64 = 0.3941; // eta   -> g   + g
    pub const BR_ETA_TO_3PI0: f64 = 0.3268; // eta   -> pi0 + pi0 + pi0
    pub const BR_ETA_TO_2PIPI0: f64 = 0.2292; // eta   -> pi  + pi  + pi0
    pub const BR_ETA_TO_2PIG: f64 = 0.0422; // eta   -> pi  + pi  + g
    pub const BR_ETAP_TO_2PIETA: f64 = 0.429; // eta'  -> pi  + pi  + eta
    pub const BR_ETAP_TO_RHOG: f64 = 0.291; // eta'  -> rho + g
    pub const ETAP_BR_PI0_PI0_ETA: f64 = 0.222; // eta'  -> pi0 + pi0 + eta
    pub const BR_ETAP_TO_OMEGAG: f64 = 0.0275; // eta'  -> omega + g
    pub const BR_ETAP_TO_GG: f64 = 0.0220; // eta'  -> g   + g
    pub const BR_ETAP_TO_3PI0: f64 = 0.0214; // eta'  -> pi0 + pi0 + pi-
    pub const BR_ETAP_TO_MUMUG: f64 = 0.0108; // eta'  -> mu  + mu  + g

    pub const BR_OMEGA_TO_2PIPI0: f64 = 0.892; // omega -> pi + pi   + pi0
    pub const BR_OMEGA_TO_PI0G: f64 = 0.0828; // omega -> pi0 + g
    pub const BR_OMEGA_TO_2PI: f64 = 0.0153; // omega -> pi + pi

    // WIDTHS
    //
    // Deliberately empty. This table used to define WIDTH_K and WIDTH_PI with
    // `**` where a decimal exponent was meant -- `3.3406**-13.` is
    // exponentiation, evaluating to 1.5498e-7 rather than a width of order
    // 1e-13 MeV -- and no including module referenced either name. Deleted
    // rather than repaired: constants.pxd is the canonical source for decay
    // widths and carries both PDG-cited (Gamma[K+] = 5.317e-14 MeV,
    // Gamma[pi+] = 2.5284e-14 MeV). See
    // docs/followups/done/legacy-parameters-width-exponent-bug.md.

    // MISC.
    pub const ALPHA_EM: f64 = 1.0 / 137.0; // Fine structure constant.
    pub const RATIO_E_MU_MASS_SQ: f64 = (MASS_E / MASS_MU) * (MASS_E / MASS_MU);
    pub const F_A_PI: f64 = 0.0119;
    pub const F_V_PI: f64 = 0.0254;
    pub const F_V_PI_SLOPE: f64 = 0.1;
    pub const F_A_K: f64 = 0.042;
    pub const F_V_K: f64 = 0.096;
    pub const DECAY_CONST_PI: f64 = 130.41; // PDG convention
    pub const DECAY_CONST_K: f64 = 156.1; // PDG convention
}

/// Module-local `DEF`s — constants a single `.pyx` defines for itself.
///
/// Each submodule is named for the `.pyx` it comes from and holds exactly
/// that file's `DEF`s, so a Phase 04 kernel port has one place to look and
/// the coverage check in `test/test_core_constants.py` can be total.
///
/// Two kinds live here and they behave differently:
///
/// - *Computed* `DEF`s (`R`, `ENG_MU_PI_RF`, …). Cython evaluates these at
///   compile time from whichever table the file `include`s, so they track
///   that table. They are `const` expressions here for the same reason,
///   written in the same association order so the rounding matches.
/// - *Hard-coded* `DEF`s (`R_FACTOR`, `BETA_MU_PIRF`, …). Someone evaluated
///   a formula once and pasted the digits, so these are frozen against
///   whatever the table said that day. [`derived::photon_pion`]'s five are frozen
///   against [`legacy`] even though that file `include`s [`pdg`] — see
///   there. They stay literals, which is both what the Cython has and what
///   Rust permits (`sqrt` and `ln` are not `const`).
pub mod derived {
    /// `hazma/spectra/_photon/_pion.pyx`.
    ///
    /// **Mixed provenance, preserved deliberately.** The file `include`s
    /// [`pdg`](super::pdg), so `MPI` / `ME` / `MMU` below are PDG values —
    /// but its five hard-coded pion/muon kinematic literals reproduce
    /// bit-exactly from [`legacy`](super::legacy)'s masses and from no other
    /// table. Recomputing them from `pdg` moves `ENG_MU_PIRF` by 4.7e-5 MeV
    /// and every photon spectrum from charged-pion decay with it, so they
    /// are left exactly as the Cython has them.
    /// `test/test_core_constants.py::test_photon_pion_literals_come_from_the_legacy_table`
    /// is what turns that claim into a check.
    pub mod photon_pion {
        use super::super::pdg;

        /// Maximum photon energy in the muon rest frame, `(m_mu^2 - m_e^2) /
        /// (2 m_mu)` over the [`legacy`](super::super::legacy) masses, in MeV.
        pub const ENG_GAM_MAX_MURF: f64 = 52.82795006985128;
        /// Maximum photon energy in the pion rest frame: [`ENG_GAM_MAX_MURF`]
        /// boosted by [`GAMMA_MU_PIRF`] / [`BETA_MU_PIRF`], in MeV.
        pub const ENG_GAM_MAX_PIRG: f64 = 69.78345771948752;
        /// Muon energy in the pion rest frame, `(m_pi^2 + m_mu^2) / (2 m_pi)`
        /// over the [`legacy`](super::super::legacy) masses, in MeV.
        pub const ENG_MU_PIRF: f64 = 109.77820123634007;
        /// Charged pion mass in MeV. PDG value — unlike the literals above.
        pub const MPI: f64 = pdg::MASS_PI;
        /// Electron mass in MeV. PDG value — unlike the literals above.
        pub const ME: f64 = pdg::MASS_E;
        /// Muon mass in MeV. PDG value — unlike the literals above.
        pub const MMU: f64 = pdg::MASS_MU;
        /// Muon velocity in the pion rest frame, dimensionless.
        pub const BETA_MU_PIRF: f64 = 0.27138337509758564;
        /// Muon Lorentz factor in the pion rest frame, dimensionless.
        pub const GAMMA_MU_PIRF: f64 = 1.0389919859434902;
    }

    /// `hazma/spectra/_photon/_rho.pyx`. Three aliases, no arithmetic.
    pub mod photon_rho {
        use super::super::pdg;

        /// Charged pion mass in MeV.
        pub const MPI: f64 = pdg::MASS_PI;
        /// Neutral pion mass in MeV.
        pub const MPI0: f64 = pdg::MASS_PI0;
        /// Rho(770) mass in MeV.
        pub const MRHO: f64 = pdg::MASS_RHO;
    }

    /// `hazma/spectra/_positron/_muon.pyx`.
    pub mod positron_muon {
        use super::super::pdg;

        /// Electron-to-muon mass ratio, dimensionless.
        pub const R: f64 = pdg::MASS_E / pdg::MASS_MU;
        /// [`R`] squared.
        pub const R2: f64 = R * R;
        /// Michel-spectrum normalization,
        /// `1 / (1 - 8r^2 + 8r^6 - r^8 - 12 r^4 ln(r^2))` at `r =` [`R`].
        ///
        /// Hard-coded, and unlike [`super::photon_pion`]'s literals it is
        /// frozen against the *PDG* table, so it agrees with the [`R`] beside
        /// it. The `.pyx` comment above it writes the log term as
        /// `12 r^2 ln(r^2)`; the exponent is a typo — only `r^4` reproduces
        /// the digits, and `test/test_core_constants.py` pins that.
        pub const R_FACTOR: f64 = 1.0001870858234163;
    }

    /// `hazma/spectra/_positron/_pion.pyx`.
    ///
    /// Every value computed, so this module tracks [`pdg`](super::pdg)
    /// with nothing frozen. Note `ENG_MU_PI_RF` is the same physical
    /// quantity as [`photon_pion::ENG_MU_PIRF`] and *not* the same
    /// number — different table, and the two spellings differ by an
    /// underscore.
    pub mod positron_pion {
        use super::super::pdg;

        /// Muon mass in MeV.
        pub const MMU: f64 = pdg::MASS_MU;
        /// Charged pion mass in MeV.
        pub const MPI: f64 = pdg::MASS_PI;
        /// Electron mass in MeV.
        pub const ME: f64 = pdg::MASS_E;
        /// Muon energy in the pion rest frame, in MeV.
        pub const ENG_MU_PI_RF: f64 = 0.5 * (MPI * MPI + MMU * MMU) / MPI;
        /// Electron energy in the pion rest frame, in MeV.
        pub const ENG_E_PI_RF: f64 = 0.5 * (MPI * MPI + ME * ME) / MPI;
        /// Muon Lorentz factor in the pion rest frame, dimensionless.
        pub const GAMMA_MU: f64 = ENG_MU_PI_RF / MMU;
    }

    /// `hazma/spectra/_neutrino/_muon.pyx`.
    ///
    /// `R`, `R2` and `R_FACTOR` are the same three the positron muon
    /// kernel defines; `R4` and `R6` are extra.
    pub mod neutrino_muon {
        use super::super::pdg;

        /// Electron-to-muon mass ratio, dimensionless.
        pub const R: f64 = pdg::MASS_E / pdg::MASS_MU;
        /// [`R`] squared.
        pub const R2: f64 = R * R;
        /// [`R`] to the fourth.
        pub const R4: f64 = R2 * R2;
        /// [`R`] to the sixth.
        pub const R6: f64 = R4 * R2;
        /// Michel-spectrum normalization; see
        /// [`super::positron_muon::R_FACTOR`].
        pub const R_FACTOR: f64 = 1.0001870858234163;
    }
}

#[cfg(test)]
mod tests {
    use super::{derived, legacy, pdg};

    /// The whole point of two namespaces: they disagree, and rule 4 says the
    /// disagreement survives the port. If this ever passes trivially,
    /// someone has consolidated the tables without declaring it.
    #[test]
    fn the_two_tables_disagree_where_the_cython_says_they_do() {
        for (name, a, b) in [
            ("MASS_E", pdg::MASS_E, legacy::MASS_E),
            ("MASS_MU", pdg::MASS_MU, legacy::MASS_MU),
            ("MASS_PI0", pdg::MASS_PI0, legacy::MASS_PI0),
            ("MASS_PI", pdg::MASS_PI, legacy::MASS_PI),
            ("MASS_K0", pdg::MASS_K0, legacy::MASS_K0),
            ("MASS_K", pdg::MASS_K, legacy::MASS_K),
            ("MASS_ETA", pdg::MASS_ETA, legacy::MASS_ETA),
            ("MASS_ETAP", pdg::MASS_ETAP, legacy::MASS_ETAP),
            ("MASS_RHO", pdg::MASS_RHO, legacy::MASS_RHO),
            ("MASS_OMEGA", pdg::MASS_OMEGA, legacy::MASS_OMEGA),
            ("ALPHA_EM", pdg::ALPHA_EM, legacy::ALPHA_EM),
            (
                "RATIO_E_MU_MASS_SQ",
                pdg::RATIO_E_MU_MASS_SQ,
                legacy::RATIO_E_MU_MASS_SQ,
            ),
        ] {
            assert_ne!(a.to_bits(), b.to_bits(), "{name} must stay divergent");
        }
    }

    /// …and agree on the handful the Cython copied across unchanged, so the
    /// test above is measuring a real split rather than two unrelated tables.
    #[test]
    fn the_two_tables_agree_where_the_cython_says_they_do() {
        for (name, a, b) in [
            ("F_A_PI", pdg::F_A_PI, legacy::F_A_PI),
            ("F_V_PI", pdg::F_V_PI, legacy::F_V_PI),
            ("F_V_PI_SLOPE", pdg::F_V_PI_SLOPE, legacy::F_V_PI_SLOPE),
            ("F_A_K", pdg::F_A_K, legacy::F_A_K),
            ("F_V_K", pdg::F_V_K, legacy::F_V_K),
            (
                "DECAY_CONST_PI",
                pdg::DECAY_CONST_PI,
                legacy::DECAY_CONST_PI,
            ),
            ("DECAY_CONST_K", pdg::DECAY_CONST_K, legacy::DECAY_CONST_K),
        ] {
            assert_eq!(a.to_bits(), b.to_bits(), "{name} must stay identical");
        }
    }

    /// `photon_pion`'s frozen literals were computed from `legacy`, but the
    /// same file's aliases come from `pdg`. Both halves are asserted here so
    /// a future "cleanup" that unifies them fails in Rust as well as Python.
    #[test]
    fn photon_pion_mixes_the_two_tables() {
        assert_eq!(
            derived::photon_pion::MMU.to_bits(),
            pdg::MASS_MU.to_bits(),
            "the aliases follow the included pdg table"
        );
        let from_legacy = 0.5
            * (legacy::MASS_PI * legacy::MASS_PI + legacy::MASS_MU * legacy::MASS_MU)
            / legacy::MASS_PI;
        assert_eq!(
            derived::photon_pion::ENG_MU_PIRF.to_bits(),
            from_legacy.to_bits(),
            "the frozen literals do not"
        );
        assert_ne!(
            derived::photon_pion::ENG_MU_PIRF.to_bits(),
            derived::positron_pion::ENG_MU_PI_RF.to_bits(),
            "the two spellings are two different numbers"
        );
    }

    /// The muon-decay normalization is frozen against `pdg`, so it must
    /// reproduce from the `R` sitting next to it. `ln` is not `const`, which
    /// is why the constant is a literal and this check is a runtime test.
    #[test]
    fn r_factor_reproduces_from_the_pdg_mass_ratio() {
        let r = derived::positron_muon::R;
        let (r2, r4) = (r * r, r * r * r * r);
        let r6 = r4 * r2;
        let r8 = r4 * r4;
        let expected = 1.0 / (1.0 - 8.0 * r2 + 8.0 * r6 - r8 - 12.0 * r4 * r2.ln());
        assert_eq!(
            derived::positron_muon::R_FACTOR.to_bits(),
            expected.to_bits()
        );
        assert_eq!(
            derived::neutrino_muon::R_FACTOR.to_bits(),
            derived::positron_muon::R_FACTOR.to_bits()
        );
    }

    /// Cython folds `DEF`s at compile time in Python; these are `const`
    /// expressions folded by rustc. Same operations, same order, so the same
    /// bits — asserted rather than assumed, against a runtime recomputation.
    #[test]
    fn const_folding_matches_a_runtime_evaluation() {
        let (mpi, mmu, me) = (pdg::MASS_PI, pdg::MASS_MU, pdg::MASS_E);
        assert_eq!(
            derived::positron_pion::ENG_MU_PI_RF.to_bits(),
            (0.5 * (mpi * mpi + mmu * mmu) / mpi).to_bits()
        );
        assert_eq!(
            derived::positron_pion::ENG_E_PI_RF.to_bits(),
            (0.5 * (mpi * mpi + me * me) / mpi).to_bits()
        );
        assert_eq!(
            derived::neutrino_muon::R6.to_bits(),
            {
                let r2 = (me / mmu) * (me / mmu);
                r2 * r2 * r2
            }
            .to_bits()
        );
    }
}
