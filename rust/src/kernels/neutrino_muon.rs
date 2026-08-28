//! The neutrino spectra from muon decay, ported from
//! `hazma/spectra/_neutrino/_muon.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::neutrino`] is the Python-visible half.
//!
//! # The physics
//!
//! `μ⁻ → e⁻ ν̄_e ν_μ` puts one neutrino of each light flavor in the final
//! state, so this kernel returns two non-zero rows and never a tau one.
//! In the scaled variable `x = 2E/m_μ` the rest-frame spectra are the
//! standard Michel forms
//!
//! ```text
//! dN/dx |_e  = 12 C ,        dN/dx |_μ = 2 C (3 + r²(3 − x) − 5x + 2x²)/(1−x)² ,
//! C = N x² (1 − r² − x)² / (1 − x) ,     0 < x < 1 − r² ,
//! ```
//!
//! with `r = m_e/m_μ` and `N =` [`R_FACTOR`], and in flight the same
//! polynomials are integrated over the boost cone in closed form between
//! the kinematic limits `x∓`. Like [`super::positron_muon`], and unlike
//! the pion and rho spectra, this kernel reaches no special function and
//! no integrator — which is what puts `spectra.neutrino.muon` in the
//! parity corpus's `EXACT` class (`test/parity/tolerances.py`): the budget
//! is `rtol = 0`, so this module must be *bit-equal* to the Cython on the
//! capturing platform, not merely close.
//!
//! # Why `mul_add`, and where it is *not* used
//!
//! `objdump -d hazma/spectra/_neutrino/_muon.cpython-312-darwin.so | grep
//! -c 'fmadd\|fmsub\|fnmadd\|fnmsub'` prints **14**, all of them inside
//! `c_muon_decay_spectrum_point` because clang inlines the rest-frame
//! helper into it: three in the rest-frame branch, four in the boosted
//! electron polynomial and seven in the boosted muon one. Each is named in
//! a comment at its site with the instruction it reproduces.
//!
//! Four expressions the same disassembly leaves *un*fused, and which this
//! module therefore also leaves unfused:
//!
//! * `1.0 - R**2 - x` — the `1 − r²` half is a folded constant
//!   ([`XMAX_RF`]) and the subtraction of `x` is a plain `fsub`;
//! * `1.0 - (MASS_MU/emu)**2` inside `beta`, which [`crate::boost`] already
//!   documents as unfused everywhere — the `.pyx` spells the boost inline
//!   rather than calling `boost_beta`, and clang treats it the same way;
//! * `xm ** 2 + 3 * R4` and `xp ** 2 + …` in the electron polynomial, both
//!   `fmul` then `fadd`;
//! * `gam ** 2 * x * (1.0 ∓ beta)`, a plain `fmul` chain.
//!
//! # Compile-time constants
//!
//! The `.pyx` declares `R`, `R2`, `R4`, `R6` and `R_FACTOR` as `DEF`s and
//! the generated C folds every combination of them it writes inline —
//! `2/m_μ`, `1 − r²`, `3r²`, `3r⁴`, `6r⁴`, `2r⁴(r² − 3)`. They are `const`
//! here and pinned against the immediates the shipped object loads, in
//! [`tests::the_folded_constants_match_the_shipped_object_code`].
//!
//! They live in this module rather than in
//! [`crate::constants::derived`] because that namespace mirrored the
//! `DEF`s of `.pyx` files that were still on disk, and this kernel's was
//! deleted by the same task that added this module — the same call Task
//! 4.5 made for `derived::photon_rho`. Task 6.4 has since deleted the
//! rest, so the distinction is now historical; the constants stay here
//! because this is the one kernel that reads them.

use super::neutrino_flavors::NeutrinoSpectrumPoint;
use crate::constants::pdg::{MASS_E, MASS_MU};

/// Electron-to-muon mass ratio, dimensionless — the `.pyx`'s `R`.
const R: f64 = MASS_E / MASS_MU;
/// [`R`] squared.
const R2: f64 = R * R;
/// [`R`] to the fourth.
const R4: f64 = R2 * R2;
/// [`R`] to the sixth.
const R6: f64 = R4 * R2;
/// The Michel normalization `1 / (1 − 8r² + 8r⁶ − r⁸ − 12r⁴ln(r²))`.
///
/// Hard-coded in the `.pyx` exactly as [`super::positron_muon`]'s twin is,
/// and to the same digits. The comment above it writes the log term's
/// exponent as `r²`; that is a typo — only `r⁴` reproduces the digits, and
/// [`tests::r_factor_is_the_quartic_normalization`] pins it.
const R_FACTOR: f64 = 1.000_187_085_823_416_3;

/// `2/m_μ`, MeV⁻¹ — the rest-frame `dN/dE ← dN/dx` Jacobian, folded
/// because `MASS_MU` is a compile-time constant on that branch. The
/// in-flight branch divides by the muon's *energy* and cannot be folded.
const TWO_OVER_MASS_MU: f64 = 2.0 / MASS_MU;
/// The upper edge of the rest-frame support in `x`: `1 − r²`.
const XMAX_RF: f64 = 1.0 - R2;
/// `3r²`, the boosted electron polynomial's linear coefficient and the
/// rest-frame muon polynomial's.
const THREE_R2: f64 = 3.0 * R2;
/// `3r⁴`, the boosted electron polynomial's intercept.
const THREE_R4: f64 = 3.0 * R4;
/// `6r⁴`, the coefficient of the log in the electron row and of the
/// simple-pole difference in the muon row.
const SIX_R4: f64 = 6.0 * R4;
/// `2r⁴(r² − 3)`, the coefficient of the log in the muon row.
///
/// The `.pyx` spells it `2 * R4 * (-3 + R2)`, which is **not** `−6r⁴`: the
/// `r²` correction is 7.8e-6 relative and the two constants are distinct
/// doubles. [`tests::the_folded_constants_match_the_shipped_object_code`]
/// pins both so a transcription that collapses them fails.
const TWO_R4_TIMES_R2_MINUS_THREE: f64 = 2.0 * R4 * (-3.0 + R2);

/// The rest-frame neutrino spectra of a muon at rest, MeV⁻¹.
///
/// # Parameters
///
/// * `enu` — the neutrino energy, MeV.
///
/// # Returns
///
/// The three flavors' `dN/dE` in MeV⁻¹, all exactly zero outside
/// `0 < x < 1 − r²` and the tau row always zero. `NaN` propagates: both
/// edge comparisons are false for a `NaN`, so it falls through to the
/// arithmetic exactly as the Cython does.
#[must_use]
pub fn dnde_neutrino_muon_rest_frame(enu: f64) -> NeutrinoSpectrumPoint {
    let pre = TWO_OVER_MASS_MU;
    let x = pre * enu;

    if x <= 0.0 || x >= XMAX_RF {
        return NeutrinoSpectrumPoint::ZERO;
    }

    let xm = 1.0 - x;
    let x2 = x * x;
    // `fsub d4, d4, d0`: the `1 − r²` half is the folded `XMAX_RF`, and
    // subtracting `x` is not a contraction candidate.
    let gap = XMAX_RF - x;
    let common = R_FACTOR * x2 * (gap * gap) / xm;

    let dndxe = 12.0 * common;
    // `fadd d3, d3, d3`: the Cython's `2.0 * common`, which clang emits as
    // `c + c`. Exact either way, and written to match the instruction.
    let two_common = common + common;
    // `fmadd d5, d6, d7, d5`: 3 + r²(3 − x).
    let poly = (3.0 - x).mul_add(R2, 3.0);
    // `fmadd d0, d0, d6, d5`: the previous term − 5x.
    let poly = x.mul_add(-5.0, poly);
    // `fmadd d0, d2, d5, d0`: the previous term + 2x².
    let poly = x2.mul_add(2.0, poly);
    let dndxm = two_common * poly / (xm * xm);

    NeutrinoSpectrumPoint {
        electron: dndxe * pre,
        muon: dndxm * pre,
        tau: 0.0,
    }
}

/// The neutrino spectra `dN/dE` in MeV⁻¹ from the decay of a muon of
/// energy `emu`.
///
/// # Parameters
///
/// * `enu` — the neutrino energy, MeV.
/// * `emu` — the muon's total energy, MeV.
///
/// # Returns
///
/// The three flavors' `dN/dE` in MeV⁻¹, the tau row always zero. Exactly
/// zero everywhere for a muon below its own rest mass, and outside the
/// boosted support `0 < x < (1 + β)(1 − r²)`.
///
/// A muon within one `f64::EPSILON` *MeV* of rest takes the rest-frame
/// branch rather than computing `β`, which would otherwise be `0` and
/// divide out the `1/(2β)` prefactor. That is the Cython's guard, in its
/// own units: the comparison is on `E − m` in MeV, not on a dimensionless
/// ratio.
#[must_use]
// Not a disguised equality test, which is what `float_equality_without_abs`
// is looking for: `emu >= MASS_MU` is already established above, so this is
// the one-sided "within one epsilon MeV of rest" threshold the Cython
// writes, and `.abs()` would change nothing.
#[allow(clippy::float_equality_without_abs)]
pub fn dnde_neutrino_muon(enu: f64, emu: f64) -> NeutrinoSpectrumPoint {
    if emu < MASS_MU {
        return NeutrinoSpectrumPoint::ZERO;
    }

    if emu - MASS_MU < f64::EPSILON {
        return dnde_neutrino_muon_rest_frame(enu);
    }

    let e_to_x = 2.0 / emu;
    let x = e_to_x * enu;
    let gam = emu / MASS_MU;
    // `fmul` then `fsub`: the squared ratio is rounded before the
    // subtraction, same as `boost::boost_beta`'s `(m/E)²`.
    let ratio = MASS_MU / emu;
    let beta = (1.0 - ratio * ratio).sqrt();
    let one_plus_beta = beta + 1.0;
    // `fadd d4, d3, d3`: the Cython's `2.0 * beta`, emitted as `β + β`.
    let pre = R_FACTOR * e_to_x / (beta + beta);

    if x <= 0.0 || one_plus_beta * XMAX_RF <= x {
        return NeutrinoSpectrumPoint::ZERO;
    }

    let gam2x = gam * gam * x;
    let xm = gam2x * (1.0 - beta);
    // `fminnm d13, d0, d1`: the Cython's `fmin(xmax_rf, ...)`, whose
    // arguments are the other way round. `fmin` is symmetric for finite
    // operands and returns the non-`NaN` one otherwise, which is exactly
    // what `f64::min` does.
    let xp = (gam2x * one_plus_beta).min(XMAX_RF);

    let xmm = 1.0 - xm;
    let xpm = 1.0 - xp;
    let diff = xm - xp;
    let sum = xm + xp;
    let xm2 = xm * xm;
    let xp2 = xp * xp;
    // Written once; the `.pyx` spells `log(xpm / xmm)` in both rows and
    // clang emits one call.
    let log_ratio = (xpm / xmm).ln();

    // -- electron row -------------------------------------------------
    // `fadd d0, d1, d0`: xm² + 3r⁴, unfused.
    let poly = xm2 + THREE_R4;
    // `fmadd d0, d12, d13, d0`: + xm·xp.
    let poly = xm.mul_add(xp, poly);
    // `fadd d0, d14, d0`: + xp², unfused.
    let poly = xp2 + poly;
    // `fmadd d0, d11, d15, d0`: + 3r²(xm + xp).
    let poly = sum.mul_add(THREE_R2, poly);
    // `fadd d0, d0, d0`: the Cython's leading `2 *`.
    let poly = poly + poly;
    // `fmadd d9, d11, d1, d0`: −3(xm + xp) + the doubled polynomial.
    let bracket = sum.mul_add(-3.0, poly);
    // `fmul d1, d0, d1` then `fmadd d1, d10, d9, d1`.
    let electron = (pre + pre) * diff.mul_add(bracket, log_ratio * -SIX_R4);

    // -- muon row -----------------------------------------------------
    // `fmadd d2, d12, d3, d2`: −9 + 4xm.
    let low = xm.mul_add(4.0, -9.0);
    // `fmadd d3, d13, d4, d3` then `fmul d3, d3, d14`: (9 − 4xp)·xp².
    let high = xp.mul_add(-4.0, 9.0) * xp2;
    // `fmadd d2, d4, d2, d3`: xm²(−9 + 4xm) + the term above.
    let cubic = xm2.mul_add(low, high) / 3.0;
    // `fmadd d1, d1, d11, d2`: 3r²(xm − xp)(xm + xp) + the cubic.
    let acc = (THREE_R2 * diff).mul_add(sum, cubic);
    // `fsub d2, d3, d2`: 2xp/xpm² − 2xm/xmm², i.e. the Cython's
    // `(−2xm)/xmm² + 2xp/xpm²` with the terms exchanged. Both are one
    // rounded subtraction of the same two quotients.
    let poles = (xp + xp) / (xpm * xpm) - (xm + xm) / (xmm * xmm);
    // `fmadd d1, d2, d3, d1`: + r⁶ · the pole difference.
    let acc = poles.mul_add(R6, acc);
    // `fmadd d1, d2, d3, d1`: + 6r⁴(1/xmm − 1/xpm).
    let acc = (1.0 / xmm - 1.0 / xpm).mul_add(SIX_R4, acc);
    // `fmadd d0, d0, d2, d1`: + 2r⁴(r² − 3)·log(xpm/xmm).
    let muon = pre * log_ratio.mul_add(TWO_R4_TIMES_R2_MINUS_THREE, acc);

    NeutrinoSpectrumPoint {
        electron,
        muon,
        tau: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        R, R_FACTOR, R2, R4, R6, SIX_R4, THREE_R2, THREE_R4, TWO_OVER_MASS_MU,
        TWO_R4_TIMES_R2_MINUS_THREE, XMAX_RF, dnde_neutrino_muon, dnde_neutrino_muon_rest_frame,
    };
    use crate::constants::pdg::MASS_MU;

    /// Every folded constant, against the literal the shipped
    /// `_muon.cpython-312-darwin.so` loads at that site.
    ///
    /// Read out of `objdump -d` as the `mov`/`movk` sequences that build
    /// each immediate. This is the check that Rust's const evaluator
    /// folded these the way clang did — a difference of one ulp in any of
    /// them would move the spectrum, and the parity budget for this entry
    /// point is zero.
    #[test]
    fn the_folded_constants_match_the_shipped_object_code() {
        assert_eq!(R2.to_bits(), 0x3ef8_86bb_bae1_538a);
        assert_eq!(THREE_R4.to_bits(), 0x3e1c_3279_514a_a944);
        assert_eq!(SIX_R4.to_bits(), 0x3e2c_3279_514a_a944);
        assert_eq!(THREE_R2.to_bits(), 0x3f12_650c_cc28_fea8);
        assert_eq!(R6.to_bits(), 0x3d0c_d0c5_0627_0b13);
        assert_eq!(TWO_R4_TIMES_R2_MINUS_THREE.to_bits(), 0xbe2c_326a_e8e8_2630);
        assert_eq!(XMAX_RF.to_bits(), 0x3fef_ffce_f288_8a3d);
        assert_eq!(TWO_OVER_MASS_MU.to_bits(), 0x3f93_621b_0149_5ffd);
        assert_eq!(R_FACTOR.to_bits(), 0x3ff0_00c4_2c77_e3d0);

        // The muon row's log coefficient is *not* the electron row's with
        // a sign flip, and the difference is 7.8e-6 relative — small
        // enough that a tolerance-based test would miss the collapse.
        assert_ne!(TWO_R4_TIMES_R2_MINUS_THREE, -SIX_R4);
        assert!((TWO_R4_TIMES_R2_MINUS_THREE / -SIX_R4 - 1.0).abs() < 1e-5);
    }

    /// `R_FACTOR` is the quartic normalization its `.pyx` comment
    /// misstates as quadratic.
    ///
    /// The same literal `_positron/_muon.pyx` carries, with the same
    /// wrong exponent in the comment above it. Re-derived here because a
    /// port that recomputed it from the comment would land 0.3% away and
    /// every value below would move.
    #[test]
    fn r_factor_is_the_quartic_normalization() {
        let r2 = R * R;
        let from_code = 1.0
            / (1.0 - 8.0 * r2 + 8.0 * r2 * r2 * r2 - r2 * r2 * r2 * r2 - 12.0 * r2 * r2 * r2.ln());
        assert!(
            (from_code - R_FACTOR).abs() <= 1e-15 * R_FACTOR,
            "quartic form gives {from_code}, R_FACTOR is {R_FACTOR}"
        );
        assert_eq!(R2.to_bits(), r2.to_bits());
        assert_eq!(R4.to_bits(), (r2 * r2).to_bits());
    }

    /// The rest-frame support is exactly `(0, 1 − r²)` in `x`, and the
    /// upper edge is a **removable** boundary rather than a step.
    ///
    /// The energy that scales to exactly `x = 1 − r²` is searched for
    /// rather than computed, because `TWO_OVER_MASS_MU * (XMAX_RF /
    /// TWO_OVER_MASS_MU)` need not round back to `XMAX_RF`. Having found
    /// it, the test states two things:
    ///
    /// 1. the kernel returns `+0.0` there — compared on the **bit
    ///    pattern**, since a signed zero from the arithmetic path would
    ///    pass an `==` against the guard's `+0.0`;
    /// 2. it would return `+0.0` there **with the guard relaxed to
    ///    `>`**, because the `(1 − r² − x)²` factor has a double root on
    ///    that edge. A mutation campaign duly survived swapping `>=` for
    ///    `>`; this records that the survivor is unobservable *by
    ///    construction* rather than for want of a test, so a future
    ///    reader does not go hunting for the gap. The lower edge is not
    ///    like that — `x <= 0.0` guards a division and a mutation there
    ///    is caught.
    #[test]
    fn the_rest_frame_support_is_open_at_both_edges() {
        let mut at_max = XMAX_RF / TWO_OVER_MASS_MU;
        for _ in 0..4 {
            if TWO_OVER_MASS_MU * at_max == XMAX_RF {
                break;
            }
            at_max = f64::from_bits(at_max.to_bits() + 1);
        }
        assert_eq!(
            TWO_OVER_MASS_MU * at_max,
            XMAX_RF,
            "no energy found that scales exactly to the upper edge"
        );
        for value in dnde_neutrino_muon_rest_frame(at_max).to_array() {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }
        // Point 2: the polynomial vanishes on the edge, so the guarded
        // and unguarded spellings agree there.
        let x = TWO_OVER_MASS_MU * at_max;
        let gap = XMAX_RF - x;
        assert_eq!(gap.to_bits(), 0.0_f64.to_bits());
        let unguarded = R_FACTOR * x * x * (gap * gap) / (1.0 - x);
        assert_eq!(unguarded.to_bits(), 0.0_f64.to_bits());

        for value in dnde_neutrino_muon_rest_frame(0.0).to_array() {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }
        assert!(dnde_neutrino_muon_rest_frame(-1.0) == super::NeutrinoSpectrumPoint::ZERO);
        let inside = dnde_neutrino_muon_rest_frame(20.0);
        assert!(inside.electron > 0.0 && inside.muon > 0.0);
    }

    /// No kernel here ever writes a tau neutrino.
    ///
    /// The row exists because the public return shape is `(3, N)`, not
    /// because a muon makes tau neutrinos. Asserted on the bit pattern
    /// across every branch so that a future edit which starts filling it
    /// has to say so.
    #[test]
    fn the_tau_row_is_always_a_positive_zero() {
        for (enu, emu) in [
            (20.0, MASS_MU),
            (20.0, 150.0),
            (1e-3, 1e5),
            (1e9, 150.0),
            (20.0, 1.0),
        ] {
            assert_eq!(
                dnde_neutrino_muon(enu, emu).tau.to_bits(),
                0.0_f64.to_bits(),
                "tau row non-zero at enu = {enu}, emu = {emu}"
            );
        }
    }

    /// A muon below its own rest mass has no spectrum at all.
    #[test]
    fn a_muon_below_threshold_gives_three_zeros() {
        let point = dnde_neutrino_muon(20.0, MASS_MU * 0.999_999);
        assert_eq!(point, super::NeutrinoSpectrumPoint::ZERO);
    }

    /// A muon at rest takes the rest-frame branch verbatim.
    #[test]
    fn a_muon_at_rest_is_the_rest_frame_branch() {
        for enu in [1.0, 10.0, 30.0, 52.0] {
            assert_eq!(
                dnde_neutrino_muon(enu, MASS_MU).to_array(),
                dnde_neutrino_muon_rest_frame(enu).to_array()
            );
        }
    }

    /// The boosted form approaches the rest-frame form as the parent
    /// stops.
    ///
    /// The point is that the `E − m < DBL_EPSILON` short circuit guards a
    /// removable singularity rather than a discontinuity: the boosted
    /// closed form has a `1/(2β)` prefactor that is finite in the limit.
    /// 1e-5 relative because that prefactor amplifies cancellation as
    /// `β → 0`, and the smallest `β` the routing admits is
    /// `sqrt(2 f64::EPSILON / m_μ) ≈ 6.5e-9`.
    #[test]
    fn the_boosted_form_approaches_the_rest_frame_form() {
        let emu = MASS_MU + 1e-9;
        for enu in [5.0, 20.0, 40.0] {
            let boosted = dnde_neutrino_muon(enu, emu);
            let rest = dnde_neutrino_muon_rest_frame(enu);
            for (got, want) in boosted.to_array().iter().zip(rest.to_array()) {
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs(),
                    "at enu = {enu}: boosted {got} vs rest frame {want}"
                );
            }
        }
    }

    /// Both flavors integrate to exactly one neutrino, and **this kernel
    /// applies the Michel normalization the right way round**.
    ///
    /// The statement about this kernel that owes nothing to the Cython:
    /// `μ → e ν̄_e ν_μ` emits exactly one neutrino of each light flavor, so
    /// both rows must integrate to 1.
    ///
    /// That they do is worth pinning, because the sibling kernel gets it
    /// wrong. `_positron/_muon.pyx:28` **divides** by [`R_FACTOR`] where
    /// the normalization has to multiply, so every positron value is low
    /// by `1/N²` — 0.0374% — and
    /// `docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`
    /// tracks it. `_neutrino/_muon.pyx` writes the same literal into
    /// `common = R_FACTOR * x² …` as a **factor**, which is correct, and
    /// the port keeps each file as it is. So a reader who has met the
    /// positron defect must not "fix" this one to match: the two really do
    /// disagree, and only one of them is wrong.
    ///
    /// Simpson on 100_001 panels over the rest-frame support. 1e-6
    /// relative: the integrand is smooth and vanishes quadratically at
    /// both edges, so the composite rule converges at its full order here
    /// — unlike the positron muon spectrum, whose square-root endpoint
    /// costs it three decades. That still separates 1 from `1/N` (1.9e-4
    /// away) and from `1/N²` (3.7e-4 away) by two decades.
    #[test]
    fn both_flavors_integrate_to_one_neutrino_each() {
        let (lo, hi) = (0.0, XMAX_RF / TWO_OVER_MASS_MU);
        let n = 100_001_usize;
        let h = (hi - lo) / (n - 1) as f64;
        let mut totals = [0.0_f64; 2];
        for index in 0..n {
            let weight = if index == 0 || index == n - 1 {
                1.0
            } else if index % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let point = dnde_neutrino_muon_rest_frame(lo + h * index as f64);
            totals[0] += weight * point.electron;
            totals[1] += weight * point.muon;
        }
        for (flavor, total) in ["electron", "muon"].iter().zip(totals) {
            let integral = total * h / 3.0;
            assert!(
                (integral - 1.0).abs() < 1e-6,
                "the {flavor} row integrates to {integral}, not to one neutrino"
            );
        }
        // And the defect the positron sibling carries is two decades
        // outside that budget, so this test discriminates rather than
        // merely passing.
        const { assert!(1.0 - 1.0 / (R_FACTOR * R_FACTOR) > 1e-4) };
    }

    /// The boost conserves neutrino number: the in-flight rows integrate
    /// to the same one neutrino each the rest-frame ones do.
    ///
    /// This is what makes the closed-form boost integral testable without
    /// the Cython. Trapezoid on 200_001 points from 0 to the boosted
    /// endpoint. 1e-5 relative: the in-flight spectra have a kink where
    /// `xp` meets its `1 − r²` clip, and a composite rule of this order
    /// resolves it no better.
    #[test]
    fn the_boost_conserves_neutrino_number() {
        let emu = 500.0;
        let beta = (1.0 - (MASS_MU / emu) * (MASS_MU / emu)).sqrt();
        let hi = (1.0 + beta) * XMAX_RF * emu / 2.0;
        let n = 200_001_usize;
        let h = hi / (n - 1) as f64;
        let mut totals = [0.0_f64; 2];
        for index in 0..n {
            let weight = if index == 0 || index == n - 1 {
                0.5
            } else {
                1.0
            };
            let point = dnde_neutrino_muon(h * index as f64, emu);
            totals[0] += weight * point.electron;
            totals[1] += weight * point.muon;
        }
        for (flavor, total) in ["electron", "muon"].iter().zip(totals) {
            let integral = total * h;
            assert!(
                (integral - 1.0).abs() < 1e-5,
                "the boosted {flavor} row integrates to {integral}, not to one"
            );
        }
    }

    /// Above the boosted endpoint the spectrum is exactly zero, and the
    /// guard is strict at the endpoint itself.
    #[test]
    fn the_boosted_support_closes_at_its_endpoint() {
        let emu = 500.0;
        let beta = (1.0 - (MASS_MU / emu) * (MASS_MU / emu)).sqrt();
        // `x = (1 + beta) * XMAX_RF` exactly: the guard is `<=`, so this
        // returns zero while anything below it does not.
        let endpoint = (1.0 + beta) * XMAX_RF * emu / 2.0;
        assert_eq!(
            dnde_neutrino_muon(endpoint, emu),
            super::NeutrinoSpectrumPoint::ZERO
        );
        assert!(dnde_neutrino_muon(endpoint * 0.999, emu).electron > 0.0);
        assert_eq!(
            dnde_neutrino_muon(endpoint * 10.0, emu),
            super::NeutrinoSpectrumPoint::ZERO
        );
    }
}
