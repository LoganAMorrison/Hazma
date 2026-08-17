//! The photon spectrum from radiative muon decay, ported from
//! `hazma/spectra/_photon/_muon.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::photon`] is the Python-visible half. Phase 04 Task
//! 4.4 (`_photon/_pion`) and Phase 06 (the mediator spectra) call
//! [`dnde_photon_muon`] and [`dnde_photon_muon_rest_frame`] natively, the
//! way their `.pyx` twins `cimport` `dnde_photon_muon_point` today, which
//! is why both are `pub`.
//!
//! # The physics
//!
//! The radiative decay `μ → e ν ν̄ γ`, from arXiv:hep-ph/9909265 ("Muon
//! Decay and Physics Beyond the Standard Model"), which the `.pyx`
//! docstring cites. In the muon rest frame the spectrum is written in
//! `y = 2E_γ/m_μ` and `r = (m_e/m_μ)²` as
//!
//! ```text
//! dN/dE = 2 · α/(3π y m_μ) · [ P₁(y)(1 − y)/12 + P₂(y) ln((1 − y)/r) ]
//! ```
//!
//! with `P₁ = −102 + 46y − 101y² + 55y³` and
//! `P₂ = 3 − 5y + 6y² − 6y³ + 2y⁴`. In flight the same distribution is
//! integrated over the boost cone in closed form between the kinematic
//! limits `x∓ = x·w∓`, which produces the dilogarithm term
//! `spence(x₋) − spence(x₊)` this kernel is named for — the only spectra
//! kernel in Phase 04 that reaches a special function.
//!
//! That is what puts `spectra.photon.muon` in the parity corpus's
//! `SPECFUN` class (`test/parity/tolerances.py`, `rtol = 1e-13`): the
//! budget exists for [`crate::special::spence`] standing in for scipy's
//! cephes `spence`, and for nothing else here.
//!
//! # Two libm calls the Cython makes and a port would not
//!
//! `y**3` and `y**4` in the rest-frame branch are **`pow` calls**, not
//! repeated multiplication. Cython emits `pow(y, 3.0)` for `y**3`, and
//! clang folds `pow(x, 2.0)` to `x*x` (exact) but leaves the cubic and
//! quartic alone — the shipped
//! `hazma/spectra/_photon/_muon.cpython-312-darwin.so` calls `_pow` twice
//! inside the inlined rest frame, with `3.0` and `4.0` in `d1`. So this
//! module writes [`f64::powf`], which lowers to the same libm `pow`;
//! `powi` or `y*y*y` would be a different number. `y**2` is a plain
//! `fmul` and is spelled `y * y` here to match.
//!
//! # Why `mul_add`, and where it is *not* used
//!
//! Twenty-two FMA instructions, for the reason [`crate::boost`] documents
//! at length: clang contracts `a * b + c` by default
//! (`-ffp-contract=on`) and the corpus was captured from a macOS/arm64
//! build that does. The sites are read out of the disassembly, not
//! guessed. `objdump -d` the shipped object shows 17 `fmadd` plus 5
//! `fmla` — one scalar and four **`fmla.2d`**, because clang vectorizes
//! the two log-coefficient polynomials in `x₋` and `x₊` into one 2-wide
//! Horner chain. A vector FMA is the same operation per lane, so each
//! lane is transcribed here as an ordinary [`f64::mul_add`] chain. All 22
//! live in `dnde_photon_muon_point`; the rest frame is inlined into it
//! once (the `.pyx` calls it from one branch, unlike its positron
//! sibling).
//!
//! Expressions the same disassembly leaves *un*fused, and which this
//! module therefore also leaves unfused:
//!
//! * `1 − β²` inside [`crate::boost::boost_beta`] — `fmul` then `fsub`,
//!   as that function's own docs record;
//! * `(x₋ − x₊) · P` and `P₁·(1 − y)/12` — the divisions that follow
//!   break the contraction;
//! * `log((1−x₊)/r)·log(x₊)`, which clang emits as an `fnmul` so that the
//!   *other* half of that difference can be the fused one.
//!
//! # Constant folding
//!
//! Four compile-time constants the generated C folds are `const` here
//! too, each pinned against the immediate the disassembly builds:
//! `r = (m_e/m_μ)²`, the rest-frame endpoint `1 − m_e/m_μ`, `1 − r`, and
//! `3π`.

use crate::boost;
use crate::constants::pdg::{ALPHA_EM, MASS_E, MASS_MU};
use crate::special::spence;

/// Electron-to-muon mass ratio, dimensionless. Not folded on its own —
/// it exists so [`R`] and [`Y_MAX`] are written the way the `.pyx` writes
/// them.
const MASS_RATIO: f64 = MASS_E / MASS_MU;
/// `r = (m_e/m_μ)²`, dimensionless — the `.pyx`'s function-local `r`,
/// which is the argument scale inside every logarithm below.
const R: f64 = MASS_RATIO * MASS_RATIO;
/// `1 − m_e/m_μ`, the rest-frame endpoint in `y`. Note it is `1 − r^(1/2)`
/// and not `1 − r`: the rest-frame guard and the boosted guard use
/// *different* edges, which is the `.pyx`'s own asymmetry.
const Y_MAX: f64 = 1.0 - MASS_RATIO;
/// `1 − r`, the boosted branch's kinematic edge in `x·w`.
const ONE_MINUS_R: f64 = 1.0 - R;
/// `3π`, folded because both its factors are compile-time constants. It
/// divides the fine-structure constant in the rest frame and multiplies
/// the muon energy in flight.
const THREE_PI: f64 = 3.0 * std::f64::consts::PI;

/// The photon spectrum `dN/dE` from a muon **at rest**.
///
/// # Parameters
///
/// * `egam` — the photon energy, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` outside `0 < y < 1 − m_e/m_μ` in
/// the scaled variable `y = 2E_γ/m_μ`. A `NaN` propagates: both edge
/// comparisons are false for a `NaN`, so it falls through to the
/// arithmetic exactly as the Cython does — there is no `fmax`/`fmin`
/// clipping on this branch.
#[must_use]
pub fn dnde_photon_muon_rest_frame(egam: f64) -> f64 {
    // `2 * egam` is exact, and clang emits it as `egam + egam`.
    let y = (2.0 * egam) / MASS_MU;

    if y <= 0.0 || y >= Y_MAX {
        return 0.0;
    }

    let pre = ALPHA_EM / ((THREE_PI * y) * MASS_MU);
    let ym = 1.0 - y;

    // `y**2` is an `fmul`; `y**3` and `y**4` are libm `pow` calls. See
    // the module docs — this asymmetry is clang's, not Cython's.
    let y2 = y * y;
    let y3 = y.powf(3.0);
    let y4 = y.powf(4.0);

    // `fmadd d0, d8, d1, d0` / `fmadd d12, d11, d1, d0` /
    // `fmadd d12, d0, d1, d12`: -102 + 46y - 101y^2 + 55y^3, folded
    // left to right.
    let poly1 = y3.mul_add(55.0, y2.mul_add(-101.0, y.mul_add(46.0, -102.0)));
    // `fmadd d1, d8, d1, d13` / `fmadd d1, d11, d2, d1` /
    // `fmadd d11, d0, d2, d1` / `fmadd d8, d0, d1, d11`:
    // 3 - 5y + 6y^2 - 6y^3 + 2y^4.
    let poly2 = y4.mul_add(2.0, y3.mul_add(-6.0, y2.mul_add(6.0, y.mul_add(-5.0, 3.0))));

    // `fadd d9, d9, d9`: the `.pyx`'s `2.0 * pre`, which clang emits as
    // `pre + pre`. Exact either way, and written to match the
    // instruction.
    // `fmadd d0, d8, d0, d11`: poly2 * log((1-y)/r) + poly1*(1-y)/12 —
    // the division by 12 leaves the first product unfused.
    (pre + pre) * poly2.mul_add((ym / R).ln(), (poly1 * ym) / 12.0)
}

/// The photon spectrum `dN/dE` from a muon of total energy `emu`.
///
/// # Parameters
///
/// * `egam` — the photon energy, MeV.
/// * `emu` — the muon's total energy, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹. Exactly `0.0` for a muon below its own rest mass and
/// outside the boosted support `0 ≤ x < (1 − r)/(1 − β)`.
///
/// A muon within one `f64::EPSILON` *MeV* of rest takes
/// [`dnde_photon_muon_rest_frame`] rather than computing `β`, which would
/// otherwise divide out of the `1/β` prefactors. That is the Cython's
/// guard, in its own units: the comparison is on `E − m` in MeV, not on a
/// dimensionless ratio.
///
/// A `NaN` `egam` propagates here, unlike in the positron muon kernel:
/// this branch clips nothing with `fmax`/`fmin`, so `x` stays `NaN`, both
/// support comparisons are false, and the `NaN` reaches the arithmetic.
#[must_use]
// Not a disguised equality test, which is what `float_equality_without_abs`
// looks for: `emu >= MASS_MU` is already established above, so this is the
// one-sided "within one epsilon MeV of rest" threshold the Cython writes,
// and `.abs()` would change nothing.
#[allow(clippy::float_equality_without_abs)]
pub fn dnde_photon_muon(egam: f64, emu: f64) -> f64 {
    if emu < MASS_MU {
        return 0.0;
    }

    if emu - MASS_MU < f64::EPSILON {
        return dnde_photon_muon_rest_frame(egam);
    }

    let gamma = boost::boost_gamma(emu, MASS_MU);
    let beta = boost::boost_beta(emu, MASS_MU);
    let y = (2.0 * egam) / MASS_MU;
    let x = y * gamma;

    // `w = 1 - beta*cos(theta)` at the two ends of the boost cone. `wm`
    // is computed before the support test because the test's denominator
    // is the same double.
    let wm = 1.0 - beta;
    if x < 0.0 || x >= ONE_MINUS_R / wm {
        return 0.0;
    }
    let wp = if x < ONE_MINUS_R / (1.0 + beta) {
        1.0 + beta
    } else {
        ONE_MINUS_R / x
    };

    let xp = x * wp;
    let xm = x * wm;

    // -- Polynomial contribution -------------------------------------
    // `-92 + 21*xp` appears twice in the source and is computed once:
    // `fmadd d4, d6, d5, d4`.
    let edge = xp.mul_add(21.0, -92.0);
    // `fmadd d3, d3, d5, d4`, `fmla.d d3, d4, v6[1]`, `fmadd d3, d6, d4, d3`:
    // 191 + 21*xm^2 + xm*edge + xp*edge, folded left to right.
    let inner = xp.mul_add(edge, edge.mul_add(xm, (xm * xm).mul_add(21.0, 191.0)));
    // `fmadd d2, d2, d3, d4`: 102 + (xm*xp)*inner. The product is spelled
    // `xp * xm` in the object code and `xm * xp` in the `.pyx`; IEEE
    // multiplication is commutative, so the double is the same.
    let poly = (xp * xm).mul_add(inner, 102.0);
    let mut result = ((xm - xp) * poly) / (12.0 * xm * xp * beta);

    // -- Logarithmic contributions -----------------------------------
    let log_m = ((1.0 - xm) / R).ln();
    let log_p = ((1.0 - xp) / R).ln();

    // The two Horner chains clang runs as one `fmla.2d` pair — lane 1 in
    // `xm`, lane 0 in `xp`. A vector FMA is the same operation per lane.
    let poly_m = xm
        .mul_add(-2.0, 9.0)
        .mul_add(xm, -18.0)
        .mul_add(xm, 18.0)
        .mul_add(xm, 9.0);
    let poly_p = xp
        .mul_add(2.0, -9.0)
        .mul_add(xp, 18.0)
        .mul_add(xp, -18.0)
        .mul_add(xp, -9.0);

    result += (log_m * poly_m) / (3.0 * xm * beta);
    result += (log_p * poly_p) / (3.0 * xp * beta);

    // -- Products of logarithms --------------------------------------
    let five_over_beta = 5.0 / beta;
    // `fnmul d0, d0, d1` then `fmadd d0, d3, d9, d0`: the second product
    // is rounded and negated, and only the first one is fused into the
    // difference.
    let log_xm = xm.ln();
    let log_xp = xp.ln();
    result = five_over_beta.mul_add(log_m.mul_add(log_xm, -(log_p * log_xp)), result);

    // `fmadd d0, d9, d15, d0`: 4*log((1-xp)/(1-xm)) fused onto the
    // already-rounded 7*log(xp/xm).
    let ratio_term = ((1.0 - xp) / (1.0 - xm))
        .ln()
        .mul_add(4.0, (xp / xm).ln() * 7.0);
    result = (4.0 / (3.0 * beta)).mul_add(ratio_term, result);

    // -- PolyLog terms -----------------------------------------------
    result = five_over_beta.mul_add(spence(xm) - spence(xp), result);

    (result * ALPHA_EM) / (emu * THREE_PI)
}

#[cfg(test)]
mod tests {
    use super::{
        ALPHA_EM, MASS_MU, ONE_MINUS_R, R, THREE_PI, Y_MAX, dnde_photon_muon,
        dnde_photon_muon_rest_frame,
    };
    use crate::constants::pdg::MASS_E;

    /// The rest-frame endpoint in energy: `y = 1 − m_e/m_μ` scaled back
    /// by `m_μ/2`, in MeV.
    const REST_FRAME_ENDPOINT: f64 = 0.5 * MASS_MU * Y_MAX;

    /// Every folded constant, against the literal the shipped
    /// `_muon.cpython-312-darwin.so` loads at that site.
    ///
    /// Read out of `objdump -d` as the `movk` sequences that build each
    /// immediate (little-endian halfwords, high halfword last). This is
    /// the check that Rust's const evaluator folded these the way clang
    /// did — a difference of one ulp in any of them would move the
    /// spectrum, and this entry point's parity budget is 1e-13.
    #[test]
    fn folded_constants_match_the_shipped_object_code() {
        assert_eq!(R.to_bits(), 0x3ef8_86bb_bae1_538a);
        assert_eq!(Y_MAX.to_bits(), 0x3fef_d861_7a30_552c);
        assert_eq!(ONE_MINUS_R.to_bits(), 0x3fef_ffce_f288_8a3d);
        assert_eq!(THREE_PI.to_bits(), 0x4022_d97c_7f33_21d2);
        assert_eq!(ALPHA_EM.to_bits(), 0x3f7d_e3d4_2a1e_89a9);
        assert_eq!(MASS_MU.to_bits(), 0x405a_6a22_cecc_814d);
    }

    /// `Y_MAX` is `1 − r^(1/2)` and `ONE_MINUS_R` is `1 − r`, and their
    /// distance from each other is two hundred times their distance
    /// from 1.
    ///
    /// The one transcription error this port could make that no swept
    /// grid would catch loudly: both constants are within 5e-3 of 1, so
    /// swapping them moves the rest-frame endpoint by 0.25 MeV and leaves
    /// a spectrum that still looks like a spectrum. Pinned as a
    /// separation, not as two literals — and asserted in a `const` block,
    /// because both sides are compile-time constants and clippy refuses a
    /// runtime `assert!` on one.
    #[test]
    fn the_two_kinematic_edges_are_different_constants() {
        const { assert!(Y_MAX < ONE_MINUS_R) };
        const { assert!(ONE_MINUS_R - Y_MAX > 4.0e-3) };
        assert_eq!(Y_MAX.to_bits(), (1.0 - MASS_E / MASS_MU).to_bits());
        assert_eq!(ONE_MINUS_R.to_bits(), (1.0 - R).to_bits());
    }

    /// The rest-frame support is exactly `(0, 1 − m_e/m_μ)` in `y`,
    /// closed nowhere.
    ///
    /// Compared on the bit pattern rather than with `==`, for the reason
    /// `positron_muon` gives: relaxing an edge comparison leaves the
    /// argument falling through to arithmetic whose sign of zero `==`
    /// cannot distinguish from the guard's `+0.0`.
    #[test]
    fn the_rest_frame_vanishes_outside_its_support_including_both_edges() {
        assert_eq!(
            dnde_photon_muon_rest_frame(0.0).to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            dnde_photon_muon_rest_frame(REST_FRAME_ENDPOINT).to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(dnde_photon_muon_rest_frame(-1.0), 0.0);
        assert_eq!(dnde_photon_muon_rest_frame(MASS_MU), 0.0);
        assert!(dnde_photon_muon_rest_frame(REST_FRAME_ENDPOINT * 0.999) > 0.0);
    }

    /// `y**3` is `pow(y, 3.0)`, and this test fails if it is written as
    /// `y*y*y`.
    ///
    /// The two disagree by an ulp at plenty of arguments, and the
    /// difference reaches the spectrum through `poly1`. Rather than
    /// asserting on the spectrum — where a one-ulp `poly1` is diluted —
    /// the check is made directly against the libm call the disassembly
    /// shows, at an argument searched for so the test is not hostage to
    /// one lucky value.
    #[test]
    fn the_cubic_and_quartic_go_through_libm_pow() {
        let mut cubic_differs = false;
        let mut quartic_differs = false;
        let mut y: f64 = 0.1;
        for _ in 0..4096 {
            cubic_differs |= y.powf(3.0).to_bits() != (y * y * y).to_bits();
            quartic_differs |= y.powf(4.0).to_bits() != ((y * y) * (y * y)).to_bits();
            y = f64::from_bits(y.to_bits() + 1);
        }
        assert!(
            cubic_differs && quartic_differs,
            "powf and repeated multiplication agreed everywhere sampled, so \
             this test can no longer tell the two spellings apart"
        );
    }

    /// The rest-frame branch fires within one `f64::EPSILON` MeV of rest,
    /// and the boosted form is continuous across that guard.
    ///
    /// The guard is a removable singularity — the boosted expression
    /// carries `1/β` prefactors — so the two forms must agree just off
    /// it, or a published spectrum would step. 2e-5 relative at
    /// `E − m = 1e-9` MeV (`β ≈ 1.4e-6`): the boosted form differences
    /// nearly-equal logarithms there, and that is how far the
    /// cancellation has eaten. Measured, not chosen.
    #[test]
    fn the_boosted_form_approaches_the_rest_frame_form_as_beta_vanishes() {
        for fraction in [0.05, 0.2, 0.5, 0.9] {
            let egam = REST_FRAME_ENDPOINT * fraction;
            let rest = dnde_photon_muon(egam, MASS_MU);
            let nearly_rest = dnde_photon_muon(egam, MASS_MU + 1e-9);
            assert!(rest > 0.0);
            assert!(
                (nearly_rest - rest).abs() <= 2e-5 * rest,
                "at egam = {egam}: boosted {nearly_rest} vs rest frame {rest}"
            );
        }
    }

    /// Below the muon threshold, and at a negative photon energy, the
    /// spectrum is exactly zero — but at `E_γ = 0` it is `NaN`.
    ///
    /// The zero is not a guard the `.pyx` has. `E_γ = 0` gives `x = 0`,
    /// which passes both support comparisons (`0 < 0` is false), and the
    /// closed form then divides by `x₋ = x₊ = 0` and takes `ln 0`. The
    /// Cython returns `NaN` there and so does this port; pinned because
    /// the corpus samples no exact zero and a "helpful" guard would be a
    /// silent divergence.
    #[test]
    fn the_spectrum_vanishes_below_threshold_but_is_nan_at_zero_energy() {
        assert_eq!(dnde_photon_muon(10.0, MASS_MU * 0.999_999), 0.0);
        assert_eq!(dnde_photon_muon(-1.0, 500.0), 0.0);
        assert!(dnde_photon_muon(0.0, 500.0).is_nan());
        assert!(dnde_photon_muon(10.0, 500.0) > 0.0);
    }

    /// The boosted endpoint is `(1 − r)·E_μ(1 + β)/2` and the spectrum
    /// dies exactly there.
    ///
    /// Derived from the support test `x < (1 − r)/(1 − β)` with
    /// `x = 2E_γ γ/m_μ` and `1/(γ(1 − β)) = γ(1 + β)`, which is a
    /// statement about the kinematics rather than about the Cython — a
    /// boosted photon cannot carry more than the forward-cone endpoint.
    ///
    /// Probed a part in `10³` below the edge rather than a part in `10⁹`:
    /// inside the last `0.1%` the closed form's terms cancel to a
    /// residual that is sometimes slightly *negative*, which
    /// [`the_spectrum_is_finite_and_signed_only_by_cancellation_at_the_edge`]
    /// bounds. Both implementations do it; it is the formula, not the
    /// port.
    #[test]
    fn the_boosted_support_ends_at_the_forward_cone_endpoint() {
        for emu in [110.0, 150.0, 500.0, 1500.0] {
            let beta = crate::boost::boost_beta(emu, MASS_MU);
            let endpoint = 0.5 * ONE_MINUS_R * emu * (1.0 + beta);
            assert!(dnde_photon_muon(endpoint * (1.0 - 1e-3), emu) > 0.0);
            assert_eq!(dnde_photon_muon(endpoint * (1.0 + 1e-9), emu), 0.0);
        }
    }

    /// The spectrum is finite everywhere and non-negative everywhere
    /// except a sliver at the endpoint, where it is bounded.
    ///
    /// Cheap, and it is what a dropped sign or a mis-transcribed
    /// polynomial coefficient breaks first: every term of the boosted
    /// expression is individually signed and they cancel to something
    /// positive only if all of them are right.
    ///
    /// The exception is real and belongs to the closed form: inside the
    /// last `0.1%` of the support the cancellation overshoots and the
    /// result dips negative. Measured against the Cython twin on a
    /// 4001-point grid, that dip reaches `2.78e-4` of the value at
    /// `0.99` of the endpoint — the *same* fraction at every parent
    /// energy from `110 MeV` to `10⁵ MeV`, because it depends only on the
    /// scaled variable. The bound below is `1e-3`, so it has 3.6x
    /// headroom and still rejects anything structural.
    #[test]
    fn the_spectrum_is_finite_and_signed_only_by_cancellation_at_the_edge() {
        for emu in [MASS_MU, 110.0, 150.0, 500.0, 1500.0, 1e5] {
            let beta = crate::boost::boost_beta(emu, MASS_MU).max(0.0);
            let endpoint = 0.5 * ONE_MINUS_R * emu * (1.0 + beta);
            let reference = dnde_photon_muon(0.99 * endpoint, emu);
            assert!(reference > 0.0);
            let n = 4001_usize;
            for index in 0..n {
                let egam = endpoint * (index as f64 + 0.5) / n as f64;
                let value = dnde_photon_muon(egam, emu);
                assert!(value.is_finite(), "dN/dE = {value} at {egam}, {emu}");
                if egam < 0.999 * endpoint {
                    assert!(value >= 0.0, "dN/dE = {value} at {egam}, {emu}");
                } else {
                    assert!(
                        value >= -1e-3 * reference,
                        "the endpoint cancellation overshot: dN/dE = {value} \
                         at {egam}, {emu}"
                    );
                }
            }
        }
    }

    /// The in-flight closed form **is** the boost integral of the
    /// rest-frame spectrum.
    ///
    /// This is the statement about the kernel that owes nothing to the
    /// Cython, and it is the one the `.pyx` never made: the boosted
    /// expression is not an independent formula but the rest-frame
    /// distribution smeared over the decay cone, so it must reproduce
    /// that smearing numerically. For an isotropic massless daughter of
    /// rest-frame energy `E'`, the lab energy `E = γE'(1 + β cosθ*)` is
    /// uniform in `cosθ*`, giving
    ///
    /// ```text
    /// dN/dE(E) = ∫ dE' f(E') / (2 β γ E'),
    ///            E/(γ(1+β)) ≤ E' ≤ min(E'_max, E/(γ(1−β)))
    /// ```
    ///
    /// with `f` the rest-frame spectrum. Every coefficient of the
    /// in-flight polynomial and every logarithm enters this comparison; a
    /// dropped term, a swapped `x₊`/`x₋`, or a wrong `1/β` power lands at
    /// `O(1)` against the bound. It also pins the prefactor's `1/E_μ`
    /// against the rest frame's `1/m_μ`, which no fixed-`E_μ` sweep can
    /// isolate.
    ///
    /// The `f` here is [`rest_frame_to_the_true_endpoint`], **not**
    /// [`dnde_photon_muon_rest_frame`] — the shipped rest-frame branch
    /// stops `0.25 MeV` short of the kinematic endpoint, which
    /// [`the_two_branches_disagree_about_the_rest_frame_endpoint`]
    /// records as a live defect. Using it here would confuse that defect
    /// with a failure of this identity; with the correct endpoint the
    /// identity holds to **machine precision** wherever the boost window
    /// is not truncated, which is itself the evidence that `1 − r` is the
    /// endpoint the closed form was derived with.
    ///
    /// Simpson on 40_001 panels in `ln E'`, which is the substitution the
    /// integrand asks for — `f(E') ~ 1/E'` at small argument, so
    /// `f(E') dE'/E'` becomes a decaying exponential in `ln E'`. The
    /// bound is 5e-9 relative: measured against the Cython twin at these
    /// exact grids (worst case 2.0e-9, at the largest fraction; the
    /// untruncated cases come in at 1e-15 or exactly zero), and set by
    /// the composite rule's residual against the `ln(1 − y)` endpoint
    /// curvature rather than by anything the kernel does. Energies are
    /// sampled on both sides of the `w₊` branch switch — `emu = 500` at
    /// the small fractions takes `w₊ = 1 + β`, everything past `0.3`
    /// takes `w₊ = (1 − r)/x` — so both halves of the closed form are
    /// exercised.
    #[test]
    fn the_in_flight_form_is_the_boost_integral_of_the_rest_frame_form() {
        for emu in [110.0, 150.0, 500.0, 1500.0] {
            let gamma = crate::boost::boost_gamma(emu, MASS_MU);
            let beta = crate::boost::boost_beta(emu, MASS_MU);
            let endpoint = 0.5 * ONE_MINUS_R * emu * (1.0 + beta);

            for fraction in [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.6, 0.9] {
                let egam = endpoint * fraction;
                let lo = egam / (gamma * (1.0 + beta));
                let hi = (egam / (gamma * (1.0 - beta))).min(0.5 * MASS_MU * ONE_MINUS_R);
                assert!(lo < hi, "empty boost window at egam = {egam}, emu = {emu}");

                let (u_lo, u_hi) = (lo.ln(), hi.ln());
                let n = 40_001_usize;
                let h = (u_hi - u_lo) / (n - 1) as f64;
                let mut total = 0.0;
                for index in 0..n {
                    let weight = if index == 0 || index == n - 1 {
                        1.0
                    } else if index % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    // dE'/E' = du, so the 1/E' of the boost kernel is
                    // absorbed by the substitution and the integrand is
                    // just f(E').
                    total +=
                        weight * rest_frame_to_the_true_endpoint((u_lo + h * index as f64).exp());
                }
                let want = total * h / (3.0 * 2.0 * beta * gamma);
                let got = dnde_photon_muon(egam, emu);

                assert!(want > 0.0);
                assert!(
                    (got - want).abs() <= 5e-9 * want,
                    "closed form {got} vs boost integral {want} at \
                     egam = {egam}, emu = {emu}"
                );
            }
        }
    }

    /// [`dnde_photon_muon_rest_frame`] with the endpoint the kinematics
    /// give: `y < 1 − r`, not `y < 1 − √r`.
    ///
    /// Written out rather than parameterised into the kernel because the
    /// kernel must keep shipping the `.pyx`'s guard (rule 1); this is the
    /// reference the two tests below measure it against.
    fn rest_frame_to_the_true_endpoint(egam: f64) -> f64 {
        let y = (2.0 * egam) / MASS_MU;
        if y <= 0.0 || y >= ONE_MINUS_R {
            return 0.0;
        }
        let pre = ALPHA_EM / ((THREE_PI * y) * MASS_MU);
        let ym = 1.0 - y;
        let (y2, y3, y4) = (y * y, y.powf(3.0), y.powf(4.0));
        let poly1 = y3.mul_add(55.0, y2.mul_add(-101.0, y.mul_add(46.0, -102.0)));
        let poly2 = y4.mul_add(2.0, y3.mul_add(-6.0, y2.mul_add(6.0, y.mul_add(-5.0, 3.0))));
        (pre + pre) * poly2.mul_add((ym / R).ln(), (poly1 * ym) / 12.0)
    }

    /// The rest-frame branch stops short of the endpoint the boosted
    /// branch uses, and the port reproduces the gap.
    ///
    /// `hazma/spectra/_photon/_muon.pyx:41` guards the rest frame with
    /// `y >= 1.0 - MASS_E / MASS_MU`, i.e. `y < 1 − √r`, while
    /// `dnde_photon_muon_point` two functions down uses `1 − r` — and
    /// `1 − r` is the kinematic endpoint `(m_μ² − m_e²)/(2m_μ)`, which
    /// `hazma/spectra/_photon/_pion.pyx:16` also hard-codes as
    /// `ENG_GAM_MAX_MURF = 52.82795006985128`. So the rest-frame branch
    /// returns exactly `0` over the top `0.2543 MeV` (0.48%) of the
    /// spectrum's support, where the spectrum is still
    /// `5.34e-7 MeV⁻¹` — a step, not a taper.
    ///
    /// This is a live defect in hazma 2.1.0, not something the port
    /// introduced, and `projects/cython-to-rust/rules.md` rule 1 says to
    /// reproduce it rather than repair it: the parity corpus pins the
    /// truncated values. Filed as
    /// `docs/followups/todo/photon-muon-rest-frame-endpoint-uses-the-wrong-power-of-r.md`.
    ///
    /// The bound on the step is `1e-6` relative — the two forms are the
    /// same arithmetic below the cut, so they agree to rounding there,
    /// and above it the reference is positive while the shipped branch is
    /// exactly zero.
    #[test]
    fn the_two_branches_disagree_about_the_rest_frame_endpoint() {
        let cut = 0.5 * MASS_MU * Y_MAX;
        let true_endpoint = 0.5 * MASS_MU * ONE_MINUS_R;
        assert!(cut < true_endpoint);
        assert!((true_endpoint - cut - 0.254_263_792_848_824_7).abs() < 1e-12);

        // Below the cut the shipped branch and the reference agree.
        for fraction in [0.1, 0.5, 0.9, 0.999] {
            let egam = cut * fraction;
            let shipped = dnde_photon_muon_rest_frame(egam);
            let reference = rest_frame_to_the_true_endpoint(egam);
            assert!(reference > 0.0);
            assert!((shipped - reference).abs() <= 1e-6 * reference);
        }

        // Above it the shipped branch is a hard zero while the spectrum
        // is not, and the value at the cut is the size of the step.
        let at_cut = rest_frame_to_the_true_endpoint(cut * (1.0 + 1e-12));
        assert!(
            (at_cut - 5.335_612e-7).abs() < 1e-13,
            "step size moved: {at_cut}"
        );
        for fraction in [1.000_001, 1.001, 1.002, 1.004] {
            let egam = cut * fraction;
            assert!(egam < true_endpoint);
            assert_eq!(dnde_photon_muon_rest_frame(egam), 0.0);
            assert!(rest_frame_to_the_true_endpoint(egam) > 0.0);
        }
    }

    /// A `NaN` photon energy propagates on **both** branches.
    ///
    /// Worth pinning because the sibling positron muon kernel does the
    /// opposite: there `fmax`/`fmin` swallow the `NaN`. Nothing clips
    /// here, so a port that "helpfully" guarded would differ from the
    /// Cython at an input the corpus never samples.
    #[test]
    fn a_nan_photon_energy_propagates_on_both_branches() {
        assert!(dnde_photon_muon(f64::NAN, MASS_MU).is_nan());
        assert!(dnde_photon_muon(f64::NAN, 500.0).is_nan());
    }
}
