//! The positron spectrum from muon decay, ported from
//! `hazma/spectra/_positron/_muon.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::positron`] is the Python-visible half.
//!
//! # The physics
//!
//! Michel's spectrum for `μ⁻ → e⁻ ν̄_e ν_μ`, written in the scaled
//! variable `x = 2E/E_μ` and boosted analytically rather than by
//! quadrature. The rest-frame shape is
//!
//! ```text
//! dN/dx = -2 √(x² − 4r²) · (4r² + x(−3 − 3r² + 2x)) / N,  2r < x < 1 + r²
//! ```
//!
//! with `r = m_e/m_μ` and `N` the normalization
//! [`constants::derived::positron_muon::R_FACTOR`]. In flight the same
//! polynomial is integrated over the boost cone in closed form between
//! the kinematic limits `x∓`, which is why this kernel — unlike the
//! photon muon spectrum — reaches no special function and no
//! integrator. That is what puts `spectra.positron.muon` in the parity
//! corpus's `EXACT` class (`test/parity/tolerances.py`): the budget is
//! `rtol = 0`, so this module must be *bit-equal* to the Cython on the
//! capturing platform, not merely close.
//!
//! # Why `mul_add`, and where it is *not* used
//!
//! Seven multiply-adds here are written `a.mul_add(b, c)` rather than
//! `a * b + c`, for the reason [`crate::boost`] documents at length:
//! clang contracts `a * b + c` into a fused multiply-add by default
//! (`-ffp-contract=on`) and the corpus was captured from a macOS/arm64
//! build that does. The sites are not guessed. Disassembling the shipped
//! `hazma/spectra/_positron/_muon.cpython-312-darwin.so` shows exactly
//! nine FMA instructions, all of them inside
//! `dnde_positron_muon_point` because clang inlines both `dndx` helpers
//! into it: two in the rest-frame branch (which is inlined twice, hence
//! four instructions) and five in the boosted branch. Each is named in a
//! comment at its site, with the instruction it reproduces.
//!
//! Three expressions the same disassembly leaves *un*fused, and which
//! this module therefore also leaves unfused — pattern-matching would
//! get all three wrong:
//!
//! * `x² − 4r²` and `x² − r₂₂` are an `fmul` followed by an `fadd`
//!   against a negated operand — a folded constant in the first case,
//!   `(1 − β²)·(−4r²)` in the second — and not an `fmsub`;
//! * `1 − β²` inside `γ²` is likewise `fmul` then `fsub`;
//! * `(4x)/3 + (−3 − 3r²)` is a plain `fadd` — the division breaks the
//!   contraction.
//!
//! [`crate::boost::boost_beta`] is unfused for its own measured reason,
//! stated there.
//!
//! # Constant folding
//!
//! Every combination of `r` the Cython writes inline — `2r`, `1 + r²`,
//! `4r²`, `8r²`, `−3 − 3r²`, `3 + 3r²`, `2/m_μ` — is a compile-time
//! constant in the generated C, so it is a `const` here too. Rust's
//! const evaluator rounds each operation to the nearest `f64` exactly as
//! C's constant folder does, and the values were checked against the
//! literals the disassembly loads.

use crate::boost;
use crate::constants::derived::positron_muon::{R, R_FACTOR, R2};
use crate::constants::pdg::{MASS_E, MASS_MU};

/// Lower edge of the rest-frame spectrum in `x`: `2r`.
const TWO_R: f64 = 2.0 * R;
/// Upper edge of the rest-frame spectrum in `x`: `1 + r²`.
const ONE_PLUS_R2: f64 = 1.0 + R2;
/// `4r²`, the constant under the square root and the polynomial's
/// intercept.
const FOUR_R2: f64 = 4.0 * R2;
/// `8r²`, the boosted polynomial's intercept.
const EIGHT_R2: f64 = 8.0 * R2;
/// `−3 − 3r²`, folded by the C compiler in both branches.
const NEG_THREE_MINUS_THREE_R2: f64 = -3.0 - 3.0 * R2;
/// `3 + 3r²`, the same constant with the opposite sign as the Cython
/// spells it in the `x₊` half.
const THREE_PLUS_THREE_R2: f64 = 3.0 + 3.0 * R2;
/// `2/m_μ`, MeV⁻¹ — the rest-frame scaling `pre`, folded because
/// `MASS_MU` is a compile-time constant on that branch. The in-flight
/// branch divides by the muon's *energy* and cannot be folded.
const TWO_OVER_MASS_MU: f64 = 2.0 / MASS_MU;

/// The rest-frame spectrum `dN/dx` at scaled positron energy `x`.
///
/// # Parameters
///
/// * `x` — the scaled positron energy `2E/m_μ`, dimensionless.
///
/// # Returns
///
/// `dN/dx`, dimensionless, and exactly `0.0` outside `(2r, 1 + r²)`.
/// `NaN` propagates: both edge comparisons are false for a `NaN`, so it
/// falls through to the arithmetic exactly as the Cython does.
#[must_use]
pub fn dndx_rest_frame(x: f64) -> f64 {
    if x <= TWO_R || x >= ONE_PLUS_R2 {
        return 0.0;
    }

    // `fmul` then `fadd` against the negated constant — not an `fmsub`.
    let root = (x * x - FOUR_R2).sqrt();
    // `fmadd d2, d0, d3, d2`: −3 − 3r² + 2x.
    let inner = x.mul_add(2.0, NEG_THREE_MINUS_THREE_R2);
    // `fmadd d0, d0, d2, d3`: 4r² + x·inner.
    let poly = x.mul_add(inner, FOUR_R2);

    (-2.0 * root) * poly / R_FACTOR
}

/// The in-flight spectrum `dN/dx` at scaled energy `x` and parent speed
/// `beta`.
///
/// The rest-frame polynomial integrated over the boost cone in closed
/// form between the kinematic limits `x∓`, which the Cython clips to the
/// rest-frame support with `fmax`/`fmin` before testing whether the
/// window survived.
///
/// # Parameters
///
/// * `x` — the scaled positron energy `2E/E_μ`, dimensionless.
/// * `beta` — the muon's speed in units of `c`, dimensionless.
///
/// # Returns
///
/// `dN/dx`, dimensionless. Exactly `0.0` for an unphysical `beta`
/// (negative, or above 1) and for an empty window; the rest-frame form
/// below `beta = f64::EPSILON`, where the `1/(2β)` prefactor would
/// otherwise divide by zero.
///
/// A `NaN` `x` does **not** propagate here, unlike in
/// [`dndx_rest_frame`]: `max`/`min` return their non-`NaN` operand (as
/// `fmaxnm`/`fminnm` do for the Cython's `fmax`/`fmin`), so both limits
/// collapse onto the rest-frame support and a finite number comes back.
/// Reproduced rather than chosen, and pinned in
/// `test/test_core_positron_muon.py`.
#[must_use]
// `!(0.0..=1.0).contains(&beta)` is *not* the same function: `contains`
// is false for a `NaN`, so the negation would return 0.0 where the
// Cython's two comparisons are both false and fall through to the
// arithmetic. The port reproduces the fall-through.
#[allow(clippy::manual_range_contains)]
pub fn dndx(x: f64, beta: f64) -> f64 {
    if beta < 0.0 || beta > 1.0 {
        return 0.0;
    }

    if beta < f64::EPSILON {
        return dndx_rest_frame(x);
    }

    // `fmul` then `fsub`: the squared beta is rounded before the
    // subtraction, same as `boost::boost_beta`'s `(m/E)²`.
    let one_minus_beta2 = 1.0 - beta * beta;
    let gamma2 = 1.0 / one_minus_beta2;
    let r22 = FOUR_R2 * one_minus_beta2;
    let root = (x * x - r22).sqrt();

    // `fmsub d3, d2, d5, d0` and `fmadd d0, d2, d5, d0`: x ∓ β·root,
    // each fused. `(-beta).mul_add(root, x)` is the fused `fmsub` —
    // Rust has no separate spelling, and the product is exact either way.
    let xm = (gamma2 * (-beta).mul_add(root, x)).max(TWO_R);
    let xp = (gamma2 * beta.mul_add(root, x)).min(ONE_PLUS_R2);

    // The Cython writes this `if xm > xp or xp < xm`. The second
    // disjunct is the first one spelled backwards — same predicate, same
    // NaN behavior (neither fires) — so one comparison is the whole
    // guard, and clang emits exactly one `fcmp`/`b.gt` pair for it.
    if xm > xp {
        return 0.0;
    }

    // `fadd` — the division by 3 breaks contraction here.
    let inner_m = (4.0 * xm) / 3.0 + NEG_THREE_MINUS_THREE_R2;
    // `fmadd d4, d3, d4, d6`: 8r² + x₋·inner.
    let term_m = xm.mul_add(inner_m, EIGHT_R2);

    let inner_p = THREE_PLUS_THREE_R2 - (4.0 * xp) / 3.0;
    // `fmadd d1, d1, d0, d5`: −8r² + inner·x₊.
    let term_p = xp * inner_p.mul_add(xp, -EIGHT_R2);

    // `fmadd d0, d3, d4, d0`: x₋·term₋ + x₊·term₊.
    let numerator = xm.mul_add(term_m, term_p);

    // `fadd d1, d2, d2`: the Cython's `2 * beta`, which clang emits as
    // `β + β`. Exact either way, and written to match the instruction.
    numerator / ((beta + beta) * R_FACTOR)
}

/// The positron spectrum `dN/dE` from the decay of a muon of energy
/// `emu`.
///
/// # Parameters
///
/// * `e` — the positron (or electron) energy, MeV.
/// * `emu` — the muon's total energy, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹. Exactly `0.0` below either threshold — a muon with
/// `emu < m_μ`, or a positron at or below its own rest mass.
///
/// A muon within one `f64::EPSILON` *MeV* of rest takes the rest-frame
/// branch rather than computing `β`, which would otherwise be `0` and
/// divide out the `1/(2β)` prefactor. That is the Cython's guard, in its
/// own units: the comparison is on `E − m` in MeV, not on a
/// dimensionless ratio.
#[must_use]
// Not a disguised equality test, which is what `float_equality_without_abs`
// is looking for: `emu >= MASS_MU` is already established above, so this
// is the one-sided "within one epsilon MeV of rest" threshold the Cython
// writes, and `.abs()` would change nothing.
#[allow(clippy::float_equality_without_abs)]
pub fn dnde_positron_muon(e: f64, emu: f64) -> f64 {
    if emu < MASS_MU || e <= MASS_E {
        return 0.0;
    }

    if emu - MASS_MU < f64::EPSILON {
        let pre = TWO_OVER_MASS_MU;
        pre * dndx_rest_frame(pre * e)
    } else {
        let beta = boost::boost_beta(emu, MASS_MU);
        let pre = 2.0 / emu;
        pre * dndx(pre * e, beta)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EIGHT_R2, FOUR_R2, NEG_THREE_MINUS_THREE_R2, ONE_PLUS_R2, THREE_PLUS_THREE_R2,
        TWO_OVER_MASS_MU, TWO_R, dnde_positron_muon, dndx, dndx_rest_frame,
    };
    use crate::constants::derived::positron_muon::{R, R_FACTOR, R2};
    use crate::constants::pdg::{MASS_E, MASS_MU};

    /// Every folded constant, against the literal the shipped
    /// `_muon.cpython-312-darwin.so` loads at that site.
    ///
    /// Read out of `objdump -d` as the `movk` sequences that build each
    /// immediate (little-endian halfwords, high halfword last). This is
    /// the check that Rust's const evaluator folded these the way clang
    /// did — a difference of one ulp in any of them would move the
    /// spectrum, and the parity budget for this entry point is zero.
    #[test]
    fn folded_constants_match_the_shipped_object_code() {
        assert_eq!(TWO_R.to_bits(), 0x3f83_cf42_e7d5_69fb);
        assert_eq!(ONE_PLUS_R2.to_bits(), 0x3ff0_0018_86bb_bae1);
        assert_eq!(FOUR_R2.to_bits(), 0x3f18_86bb_bae1_538a);
        assert_eq!(EIGHT_R2.to_bits(), 0x3f28_86bb_bae1_538a);
        assert_eq!(NEG_THREE_MINUS_THREE_R2.to_bits(), 0xc008_0024_ca19_9852);
        assert_eq!(THREE_PLUS_THREE_R2.to_bits(), 0x4008_0024_ca19_9852);
        assert_eq!(TWO_OVER_MASS_MU.to_bits(), 0x3f93_621b_0149_5ffd);
    }

    /// The rest-frame support is exactly `(2r, 1 + r²)`, closed nowhere.
    ///
    /// Both edges are compared **on the bit pattern**, not with `==`.
    /// The lower one needs it: relaxing `x <= TWO_R` to `x < TWO_R`
    /// leaves `x = 2r` falling through to arithmetic whose square root
    /// is zero, so the result is a *signed* zero that `==` cannot tell
    /// from the `+0.0` the guard returns. A mutation campaign found that
    /// exact survivor; this is the assertion that kills it.
    #[test]
    fn rest_frame_vanishes_outside_its_support_including_both_edges() {
        assert_eq!(dndx_rest_frame(TWO_R).to_bits(), 0.0_f64.to_bits());
        assert_eq!(dndx_rest_frame(ONE_PLUS_R2).to_bits(), 0.0_f64.to_bits());
        assert_eq!(dndx_rest_frame(0.0), 0.0);
        assert_eq!(dndx_rest_frame(-1.0), 0.0);
        assert_eq!(dndx_rest_frame(2.0), 0.0);
        assert!(dndx_rest_frame(0.5) > 0.0);
    }

    /// `∫ dN/dx dx = 1/N²`, **not** 1 — the shipped normalization is
    /// inverted, and the port reproduces it.
    ///
    /// The un-normalized polynomial integrates to exactly `1/N` over
    /// `(2r, 1 + r²)`, where `N =` [`R_FACTOR`]: that is the closed form
    /// `1 − 8r² + 8r⁶ − r⁸ − 12r⁴ln(r²)` the `.pyx` comment names, and
    /// `scipy.integrate.quad` reproduces it to 1e-16 relative
    /// (0.999812949171142 against 0.9998129491711419). Normalizing
    /// therefore means dividing by `1/N`, i.e. **multiplying** by `N`.
    /// `hazma/spectra/_positron/_muon.pyx:28` divides instead, so every
    /// value is low by `1/N²` — 0.0374% — and so is everything built on
    /// it.
    ///
    /// This is a live defect in hazma 2.1.0, not something the port
    /// introduced, and `projects/cython-to-rust/rules.md` rule 1 says to
    /// reproduce it rather than repair it: the parity corpus pins the
    /// low values, so a "fix" here fails the gate that governs the swap.
    /// Filed as
    /// `docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`.
    ///
    /// Simpson on 200_001 panels. The square-root endpoint at `x = 2r`
    /// leaves the derivative unbounded, so the composite rule converges
    /// at `O(h^1.5)` there rather than `O(h⁴)` and lands 3.3e-6 short —
    /// hence 1e-5, measured rather than chosen. That still separates
    /// `1/N²` from `1/N` (1.9e-4 away) and from `1` (3.7e-4 away) by
    /// more than a decade, which is the discrimination this test needs.
    #[test]
    fn rest_frame_spectrum_carries_the_inverted_normalization() {
        let (lo, hi) = (TWO_R, ONE_PLUS_R2);
        let n = 200_001_usize;
        let h = (hi - lo) / (n - 1) as f64;
        let mut total = 0.0;
        for index in 0..n {
            let weight = if index == 0 || index == n - 1 {
                1.0
            } else if index % 2 == 1 {
                4.0
            } else {
                2.0
            };
            total += weight * dndx_rest_frame(lo + h * index as f64);
        }
        let integral = total * h / 3.0;
        let shipped = 1.0 / (R_FACTOR * R_FACTOR);
        assert!(
            (integral - shipped).abs() < 1e-5,
            "rest-frame dN/dx integrates to {integral}, not the shipped {shipped}"
        );
        // The correct answer is two factors of N away, and far outside the
        // quadrature error above — so this pins the defect, not the rule.
        assert!((integral * R_FACTOR * R_FACTOR - 1.0).abs() < 1e-5);
    }

    /// `dndx` reduces to the rest-frame form as the parent stops.
    ///
    /// Below `f64::EPSILON` that is the Cython's explicit branch; the
    /// point of the test is that the branch agrees with the closed-form
    /// boosted expression it guards, so the guard is a removable
    /// singularity rather than a discontinuity. 1e-7 relative because the
    /// boosted form's `1/(2β)` prefactor amplifies cancellation as
    /// `β → 0`; at `β = 1e-9` the two forms already agree that far.
    #[test]
    fn boosted_form_approaches_the_rest_frame_form_as_beta_vanishes() {
        for x in [0.05, 0.2, 0.5, 0.9] {
            let rest = dndx_rest_frame(x);
            let nearly_rest = dndx(x, 1e-9);
            assert!(
                (nearly_rest - rest).abs() <= 1e-7 * rest.abs(),
                "at x = {x}: boosted {nearly_rest} vs rest frame {rest}"
            );
        }
    }

    /// The window guard is strict: `x₋ == x₊` still evaluates.
    ///
    /// The Cython writes `if xm > xp`, so the degenerate window — the one
    /// point where the square root vanishes and both kinematic limits
    /// coincide — is *inside* the support and returns a number, not zero.
    /// Relaxing the comparison to `>=` is invisible on any swept grid,
    /// because reaching it needs an `x` for which `x·x − r₂₂` is exactly
    /// `0.0`; a mutation campaign duly survived it. The argument is
    /// searched for rather than computed, since `r₂₂.sqrt()` need not
    /// square back to `r₂₂`.
    #[test]
    fn the_degenerate_window_is_inside_the_support() {
        let beta = 0.6;
        let r22 = FOUR_R2 * (1.0 - beta * beta);

        let mut x = r22.sqrt();
        for _ in 0..4 {
            if x * x - r22 == 0.0 {
                break;
            }
            x = f64::from_bits(x.to_bits() + 1);
        }
        assert_eq!(x * x - r22, 0.0, "no x found with a vanishing square root");

        // x₋ and x₊ are the same double here, so `>` passes and `>=`
        // would not — and the value is well away from zero.
        assert!(dndx(x, beta) > 0.0);
    }

    /// The rest-frame short circuit fires at exactly `f64::EPSILON`.
    ///
    /// Worth pinning separately because **this branch is unreachable from
    /// [`dnde_positron_muon`]**: that function already routes anything
    /// with `E − m_μ < f64::EPSILON` MeV to the rest frame, and the
    /// smallest `beta` that survives the routing is
    /// `sqrt(2·f64::EPSILON/m_μ) ≈ 6.5e-9`, thirty million times the
    /// threshold here. So no end-to-end test can see this boundary, and a
    /// mutation that moved it to `2·f64::EPSILON` survived the whole
    /// suite. It is kept because the Cython has it, and pinned here.
    #[test]
    fn the_rest_frame_short_circuit_fires_at_exactly_one_epsilon() {
        let x = 0.5;
        let below = f64::from_bits(f64::EPSILON.to_bits() - 1);
        assert_eq!(dndx(x, below).to_bits(), dndx_rest_frame(x).to_bits());
        // At the threshold itself the boosted expression is used, and it
        // is not the same double — which is what makes the boundary
        // observable at all.
        assert_ne!(
            dndx(x, f64::EPSILON).to_bits(),
            dndx_rest_frame(x).to_bits()
        );
    }

    /// Unphysical parent speeds and empty windows return exactly zero.
    ///
    /// `1.0 + 1e-16` is deliberately *not* used as the "just above 1"
    /// probe: it rounds to `1.0`, which takes the `β = 1` path rather
    /// than the guard. `f64::EPSILON` is the smallest step that lands
    /// above.
    #[test]
    fn boosted_form_is_zero_where_the_kinematics_are_empty() {
        assert_eq!(dndx(0.5, -1e-30), 0.0);
        assert_eq!(dndx(0.5, 1.0 + f64::EPSILON), 0.0);
        // Above the boosted endpoint the window closes: x₋ clips up to
        // 2r and x₊ down to 1 + r², and the pair crosses.
        assert_eq!(dndx(1e3, 0.9), 0.0);
    }

    /// Both thresholds of the public kernel, at and either side of each.
    #[test]
    fn dnde_vanishes_below_either_threshold() {
        assert_eq!(dnde_positron_muon(10.0, MASS_MU * 0.999_999), 0.0);
        assert_eq!(dnde_positron_muon(MASS_E, 500.0), 0.0);
        assert_eq!(dnde_positron_muon(MASS_E * 0.5, 500.0), 0.0);
        assert!(dnde_positron_muon(10.0, 500.0) > 0.0);
    }

    /// A muon at rest takes the rest-frame branch, and the two spectra
    /// are related by the `2/m_μ` Jacobian rather than by the boosted
    /// formula.
    #[test]
    fn a_muon_at_rest_scales_the_rest_frame_spectrum_by_the_jacobian() {
        for e in [1.0, 10.0, 30.0, 52.0] {
            let expected = TWO_OVER_MASS_MU * dndx_rest_frame(TWO_OVER_MASS_MU * e);
            assert_eq!(dnde_positron_muon(e, MASS_MU).to_bits(), expected.to_bits());
        }
    }

    /// The boost conserves positron number: the in-flight spectrum
    /// integrates to the same `1/N²` the rest-frame one does.
    ///
    /// This is the statement about the kernel that owes nothing to the
    /// Cython — the closed-form boost integral is only correct if it
    /// preserves the norm — and it is asserted against the rest frame's
    /// own value rather than against 1, for the reason
    /// [`rest_frame_spectrum_carries_the_inverted_normalization`] gives.
    ///
    /// Trapezoid on 400_001 points from `m_e` to the endpoint. 1e-6
    /// relative: the in-flight spectrum has a kink where the two
    /// kinematic branches meet, and a composite rule of this order
    /// resolves it no better.
    #[test]
    fn in_flight_spectrum_conserves_positron_number() {
        let emu = 500.0;
        let (lo, hi) = (MASS_E, emu);
        let n = 400_001_usize;
        let h = (hi - lo) / (n - 1) as f64;
        let mut total = 0.0;
        for index in 0..n {
            let weight = if index == 0 || index == n - 1 {
                0.5
            } else {
                1.0
            };
            total += weight * dnde_positron_muon(lo + h * index as f64, emu);
        }
        let integral = total * h;
        let shipped = 1.0 / (R_FACTOR * R_FACTOR);
        assert!(
            (integral - shipped).abs() < 1e-6,
            "in-flight dN/dE integrates to {integral}, not the shipped {shipped}"
        );
    }

    /// `R_FACTOR` is the normalization its `.pyx` comment misstates.
    ///
    /// Task 3.1 pinned the literal against the `1 − 8r² + 8r⁶ − r⁸ −
    /// 12r⁴ln(r²)` form and recorded that the Cython comment's `12r²`
    /// exponent is a typo. Re-asserted here because this is the first
    /// module to *use* the constant: a port that recomputed it from the
    /// comment would land 0.3% away and every value below would move.
    #[test]
    fn r_factor_is_the_quartic_normalization_not_the_comment_s_quadratic() {
        let r2 = R * R;
        let from_code = 1.0
            / (1.0 - 8.0 * r2 + 8.0 * r2 * r2 * r2 - r2 * r2 * r2 * r2 - 12.0 * r2 * r2 * r2.ln());
        assert!(
            (from_code - R_FACTOR).abs() <= 1e-15 * R_FACTOR,
            "quartic form gives {from_code}, R_FACTOR is {R_FACTOR}"
        );
        assert_eq!(R2.to_bits(), r2.to_bits());
    }
}
