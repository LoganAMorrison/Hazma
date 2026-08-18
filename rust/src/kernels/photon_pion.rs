//! The photon spectra from charged- and neutral-pion decay, ported from
//! `hazma/spectra/_photon/_pion.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::photon`] is the Python-visible half. Phase 04
//! Task 4.5 (`_photon/_rho`) integrates [`dnde_photon_charged_pion`] and
//! [`dnde_photon_neutral_pion`] over the ρ's two-body kinematics, the way
//! its `.pyx` twin `cimport`s `dnde_photon_charged_pion_point` today,
//! which is why both are `pub`.
//!
//! # The physics
//!
//! **Charged pion.** `π± → μ± ν` (BR 0.9998770) dominates, so the photon
//! spectrum is the *muon's* radiative spectrum boosted out of the pion
//! rest frame, plus the pion's own radiative decays `π → ℓ ν γ` for
//! `ℓ ∈ {μ, e}`. In the pion rest frame the muon is monochromatic at
//! `ENG_MU_PIRF`, so the lab-frame spectrum is a single integral over the
//! photon's angle `cos θ` relative to the pion:
//!
//! ```text
//! dN/dE = ∫_{-1}^{1} dcosθ  J · [ BR_μν · (dN/dE)_μ(E', E_μ^πRF)
//!                               + BR_μν · (dN/dE)_{π→μνγ}(E')
//!                               + BR_eν · (dN/dE)_{π→eνγ}(E') ]
//! ```
//!
//! with `E' = E γ_π (1 − β_π cosθ)` the pion-rest-frame photon energy and
//! `J = 1/(2 γ_π |1 − β_π cosθ|)` the Jacobian. The radiative matrix
//! element carries the pion's vector and axial form factors
//! [`F_V_PI`](crate::constants::pdg::F_V_PI) (with its linear
//! energy-dependence slope) and [`F_A_PI`](crate::constants::pdg::F_A_PI),
//! and the decay constant `f_π = F_π/√2 ≈ 92.2 MeV`.
//!
//! **Neutral pion.** `π⁰ → γγ` (BR 0.98823) is a box of height
//! `2·BR/(E_π β)` between `E_π(1 ∓ β)/2` — the flat boost of a two-photon
//! line, exact and closed-form.
//!
//! # Mixed-provenance constants
//!
//! This `.pyx` is the one Phase 03 Task 3.1 singled out: it `include`s
//! `hazma/_utils/constants.pxd`, so `MPI`/`ME`/`MMU` are PDG values, but
//! its hard-coded `ENG_MU_PIRF` reproduces bit-exactly from the *legacy*
//! mass table and from no other. [`crate::constants::derived::photon_pion`]
//! carries both halves with the reasoning; nothing here recomputes any of
//! them. `ENG_GAM_MAX_MURF`, `ENG_GAM_MAX_PIRG`, `BETA_MU_PIRF` and
//! `GAMMA_MU_PIRF` live there too but are unreachable from this module:
//! the `.pyx`'s `eng_gam_max` is the only reader and nothing calls it or
//! declares it in `_pion.pxd`, so it is not ported.
//!
//! # `float` is not `double`, and the neutral pion depends on it
//!
//! `dnde_photon_neutral_pion_point` declares `cdef float beta` and
//! `cdef float ret_val` — **single** precision, in a file where every
//! other local is a `double`. The shipped object confirms it: two
//! `fcvt s, d` / `fcvt d, s` round trips, one after the `fsqrt` and one
//! on the returned quotient. Both survive here as `as f32 as f64`, which
//! is not a rounding nicety — it moves the neutral-pion spectrum in the
//! eighth significant figure, and the parity corpus pins it at
//! `EXACT` (`rtol = 0`).
//!
//! # Why `mul_add`, and where it is *not* used
//!
//! Nineteen FMA instructions, for the reason [`crate::boost`] documents at
//! length: clang contracts `a * b + c` by default (`-ffp-contract=on`) and
//! the corpus was captured from a macOS/arm64 build that does. The sites
//! are read out of the disassembly, not guessed — `objdump -d` the shipped
//! `hazma/spectra/_photon/_pion.cpython-312-darwin.so` shows 15 in
//! `dnde_pi_to_lnug` and 4 in `charged_pion_integrand`, and **none** in
//! either point function. Two neighbours look fusable and are not:
//!
//! * `1 − β²` inside [`crate::boost::boost_beta`] — `fmul` then `fsub`, as
//!   that function's own docs record, and both point functions inline it;
//! * `2 r² − 2 r x` and `r² − r x`, whose *first* operand is an `fnmul`
//!   (`−(x·2r)`, `−(x·r)`) so that the second can be the fused one. The
//!   negation is exact, so this is a spelling of the same double — but it
//!   fixes which product gets the extra precision.
//!
//! **Nothing tests the 15 sites in [`dnde_pi_to_lnug`], and nothing can.**
//! They sit inside a quadrature: unfusing one moves the integrand in its
//! last bit, and the integral does not carry that bit out. Measured, not
//! assumed — unfusing `F_A² + F_V²` leaves the worst corpus difference at
//! `2.618e-15`, the *same* figure the correct port produces, plus 120
//! `cargo` tests and 73 per-kernel tests green. Task 4.3's muon kernel had
//! the same class of mutation caught by its bit-equality sweep; this
//! kernel has no bit-equality mode to catch it with, because it replaces
//! scipy's QUADPACK. So the map here is defended by the disassembly
//! reading above and by review, not by a gate — which is worth knowing
//! before "simplifying" one of them. The four sites in
//! [`charged_pion_integrand`] *are* covered: they are outside the
//! integrand's own arithmetic, and unfusing the Doppler factor turns a
//! corpus block red.
//!
//! # Constant folding
//!
//! Five compile-time constants the generated C folds are `const` here too,
//! each pinned against the immediate the disassembly builds: `m_π²`,
//! `F_A²`, `12√2`, `24 π m_π`, and `f_π = F_π·(1/√2)`. `f_π` is a module
//! global in the `.pyx` rather than a `DEF`, so clang loads it from
//! memory; it is still a compile-time constant here because
//! `FRAC_1_SQRT_2` is. The spelling matters: `F_π/√2` written as
//! `130.41 * (1.0/2.0_f64.sqrt())` is **one ulp** below
//! `130.41 * FRAC_1_SQRT_2`, which is what C's `M_SQRT1_2` gives.

use crate::boost;
use crate::constants::derived::photon_pion::{ENG_MU_PIRF, ME, MMU, MPI};
use crate::constants::pdg::{
    ALPHA_EM, BR_PI_TO_E_NUE, BR_PI_TO_MU_NUMU, BR_PI0_TO_A_A, DECAY_CONST_PI, F_A_PI, F_V_PI,
    F_V_PI_SLOPE, MASS_PI, MASS_PI0,
};
use crate::kernels::photon_muon;
use crate::quad::{DEFAULT_LIMIT, QuadOpts, quad};

/// The pion decay constant `f_π = F_π/√2 ≈ 92.2 MeV`.
///
/// The `.pyx` spells it `DECAY_CONST_PI * M_SQRT1_2` as a module-level
/// `cdef double`, so it is a runtime global there and a `const` here.
/// [`std::f64::consts::FRAC_1_SQRT_2`] is bit-equal to C's `M_SQRT1_2`;
/// see the module docs for the one-ulp trap in the other spelling.
const FPI: f64 = DECAY_CONST_PI * std::f64::consts::FRAC_1_SQRT_2;

/// `m_π²` in MeV², folded — the `.pyx` writes `MPI*MPI` and clang builds
/// the product as a single immediate.
const MPI_SQ: f64 = MPI * MPI;

/// `F_A²`, dimensionless. Folded for the same reason as [`MPI_SQ`]: the
/// `.pyx`'s `F_A_PI*F_A_PI` is two `DEF`s and never reaches a register.
const F_A_PI_SQ: f64 = F_A_PI * F_A_PI;

/// `12√2`, dimensionless, folded out of the two `12 * sqrt(2) * ...`
/// products. `sqrt` is not `const` in Rust, so this is written from
/// [`std::f64::consts::SQRT_2`], which is what clang's compile-time
/// `sqrt(2)` rounds to.
const TWELVE_SQRT_2: f64 = 12.0 * std::f64::consts::SQRT_2;

/// `24 π m_π` in MeV, folded — the leading three factors of
/// [`dnde_pi_to_lnug`]'s denominator, all compile-time constants.
const TWENTY_FOUR_PI_MPI: f64 = 24.0 * std::f64::consts::PI * MPI;

/// The photon spectrum `dN/dE` from radiative pion decay `π → ℓ ν γ`,
/// in the **pion rest frame**.
///
/// # Parameters
///
/// * `egam` — the photon energy in the pion rest frame, MeV.
/// * `ml` — the charged lepton's mass, MeV: [`MMU`] or [`ME`].
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` outside `0 ≤ x ≤ 1 − r` in the
/// scaled variables `x = 2E_γ/m_π`, `r = (m_ℓ/m_π)²`. A `NaN` `egam`
/// propagates: the guard is written as a rejection (`x < 0 || 1−r < x`),
/// both of whose comparisons are false for a `NaN`, so the `NaN` falls
/// through to the arithmetic — the same way it does in
/// [`crate::kernels::photon_muon`], and the opposite of what an
/// acceptance-shaped guard would do.
#[must_use]
// `x < 0.0 || (1.0 - r) < x` is not `!(0.0..=(1.0 - r)).contains(&x)`:
// `contains` is false for a `NaN`, so `!contains` is *true* and the
// "simplification" would return zero where the Cython falls through to
// the arithmetic and propagates the `NaN`. Same class as
// `crate::kernels::positron_muon`'s β guard, opposite direction —
// which is why each one has to be read rather than pattern-matched.
#[allow(clippy::manual_range_contains)]
pub fn dnde_pi_to_lnug(egam: f64, ml: f64) -> f64 {
    // `2 * egam` is exact, and clang emits it as `egam + egam`.
    let x = (2.0 * egam) / MPI;
    let mass_ratio = ml / MPI;
    let r = mass_ratio * mass_ratio;

    if x < 0.0 || (1.0 - r) < x {
        return 0.0;
    }

    let one_minus_x = 1.0 - x;
    let x_minus_1 = x - 1.0;
    let x_minus_2 = x - 2.0;

    // `fmadd d2, d0, d2, d1`: the vector form factor's energy dependence,
    // 1 + slope*(1-x), fused before the F_V_PI scale.
    let f_v = one_minus_x.mul_add(F_V_PI_SLOPE, 1.0) * F_V_PI;

    // `fadd d3, d8, d9` / `fadd d3, d3, #-1.0`: (r + x) - 1.
    let r_plus_x_minus_1 = (x + r) - 1.0;

    // -- f: the log-free numerator ------------------------------------
    // m_pi^2 x^4 (F_A^2 + F_V^2), built as four separate `fmul`s off the
    // folded m_pi^2 rather than a `powi`.
    let mut term_x4 = MPI_SQ;
    term_x4 *= x;
    term_x4 *= x;
    term_x4 *= x;
    term_x4 *= x;
    // `fmadd d5, d2, d2, d5`: F_A^2 is the folded addend, F_V^2 the
    // fused product.
    term_x4 *= f_v.mul_add(f_v, F_A_PI_SQ);

    // r*r - r*x + r - 2 (x-1)^2. `fnmul d5, d8, d9` then
    // `fmadd d5, d9, d9, d5`: the -r*x is a plain negated product and the
    // r*r is the fused one. Then `fmadd d5, d6, d11, d5` for the last
    // term, with the 2 folded into the first factor's sign.
    let neg_r_x = -(x * r);
    let mut poly_r = r.mul_add(r, neg_r_x);
    // `fadd d5, d9, d5` — the operand order is the Cython's; addition is
    // commutative and exactly rounded, so `+=` is the same double.
    poly_r += r;
    poly_r = (x_minus_1 * -2.0).mul_add(x_minus_1, poly_r);

    // 12 sqrt(2) f_pi m_pi r (x-1) x^2, left to right.
    let twelve_sqrt2_fpi = FPI * TWELVE_SQRT_2;
    let mut term_fv = twelve_sqrt2_fpi * MPI;
    term_fv *= r;
    term_fv *= x_minus_1;
    term_fv *= x;
    term_fv *= x;

    // `fmadd d16, d8, d17, d9`: r - 2x, fused, then + 1 unfused.
    // `fmadd d1, d1, d10, d15`: F_A*(r - 2x + 1) + F_V*x.
    let f_v_x = x * f_v;
    let form_factors = (x.mul_add(-2.0, r) + 1.0).mul_add(F_A_PI, f_v_x);

    // `fnmul d1, d1, d7` then `fmadd d1, d4, d5, d1`: the second term is
    // negated as a product and the first is fused onto it.
    let mut bracket = -(form_factors * term_fv);
    bracket = term_x4.mul_add(poly_r, bracket);

    // -24 f_pi^2 r (x-1) (4 r (x-1) + (x-2)^2), with the sign folded into
    // the literal and `fmadd d4, d4, d11, d5` for the inner sum.
    let mut term_fpi2 = FPI * -24.0;
    term_fpi2 *= FPI;
    term_fpi2 *= r;
    term_fpi2 *= x_minus_1;
    let inner = (r * 4.0).mul_add(x_minus_1, x_minus_2 * x_minus_2);
    bracket = term_fpi2.mul_add(inner, bracket);

    let f = r_plus_x_minus_1 * bracket;

    // -- g: the numerator's logarithmic half --------------------------
    // 12 sqrt(2) f_pi r (x-1)^2, then times log(r/(1-x)).
    let mut log_prefactor = r * twelve_sqrt2_fpi;
    log_prefactor *= x_minus_1;
    log_prefactor *= x_minus_1;
    let log_term = (r / one_minus_x).ln() * log_prefactor;

    // m_pi x^2 (F_A (x - 2r) - F_V x): `fmadd d2, d9, d6, d8` for x - 2r,
    // then `fnmsub d2, d2, d10, d15` for the difference.
    let mut m_pi_x2 = x * MPI;
    m_pi_x2 *= x;
    let axial = r.mul_add(-2.0, x).mul_add(F_A_PI, -f_v_x);

    // sqrt(2) f_pi (2r^2 - 2rx - x^2 + 2x - 2), folded left to right with
    // the -2rx again supplied as an `fnmul`.
    let two_r = r + r;
    let neg_two_r_x = -(x * two_r);
    let mut poly_g = two_r.mul_add(r, neg_two_r_x);
    poly_g = (-x).mul_add(x, poly_g);
    poly_g = x.mul_add(2.0, poly_g);
    poly_g -= 2.0;

    let g = m_pi_x2.mul_add(axial, poly_g * (FPI * std::f64::consts::SQRT_2)) * log_term;

    // -- normalization -------------------------------------------------
    // 24 pi m_pi f_pi^2 (r-1)^2 (x-1)^2 r x, with the first three folded.
    let r_minus_1 = r - 1.0;
    let mut denominator = FPI * TWENTY_FOUR_PI_MPI;
    denominator *= FPI;
    denominator *= r_minus_1;
    denominator *= r_minus_1;
    denominator *= x_minus_1;
    denominator *= x_minus_1;
    denominator *= r;
    denominator *= x;

    ((g + f) * ALPHA_EM) / denominator
}

/// The `cos θ` integrand of the charged-pion photon spectrum.
///
/// # Parameters
///
/// * `cl` — the cosine of the photon's angle to the pion, lab frame.
/// * `egam` — the photon energy in the lab frame, MeV.
/// * `epi` — the pion's total energy in the lab frame, MeV.
///
/// # Returns
///
/// The integrand in MeV⁻¹ (the `cos θ` measure is dimensionless), summed
/// over `π → μνγ`, `π → eνγ` and the boosted muon's own radiative
/// spectrum.
///
/// The Jacobian carries `|1 − β cosθ|`, so this is finite at the
/// forward/backward edges for every `β < 1`; the `.pyx` nevertheless
/// passes `points=[-1, 1]` to `quad`, which — see
/// [`dnde_photon_charged_pion`] — survives scipy's filtering as *no*
/// break point at all.
#[must_use]
pub fn charged_pion_integrand(cl: f64, egam: f64, epi: f64) -> f64 {
    let beta_pi = boost::boost_beta(epi, MASS_PI);
    let gamma_pi = boost::boost_gamma(epi, MASS_PI);

    // `fmsub d13, d0, d8, d12`: 1 - beta*cos(theta), fused, and reused by
    // both the Doppler factor and the Jacobian.
    let doppler = (-beta_pi).mul_add(cl, 1.0);
    let eng_gam_pi_rf = (egam * gamma_pi) * doppler;
    // `fadd d1, d11, d11`: the .pyx's `2.0 * gamma_pi`, exact either way.
    let jac = 1.0 / ((gamma_pi + gamma_pi) * doppler.abs());

    // The three contributions accumulate onto a `0.0` seed through three
    // `fmadd`s. `jac * BR_mu` is computed once and used twice, matching
    // the object code.
    let weight_mu = jac * BR_PI_TO_MU_NUMU;
    let mut result = 0.0;
    result = weight_mu.mul_add(
        photon_muon::dnde_photon_muon(eng_gam_pi_rf, ENG_MU_PIRF),
        result,
    );
    result = weight_mu.mul_add(dnde_pi_to_lnug(eng_gam_pi_rf, MMU), result);
    (jac * BR_PI_TO_E_NUE).mul_add(dnde_pi_to_lnug(eng_gam_pi_rf, ME), result)
}

/// `scipy.integrate.quad`'s arguments at the `.pyx`'s one call site.
///
/// `epsabs`/`epsrel` are copied verbatim from
/// `hazma/spectra/_photon/_pion.pyx`; `limit` is scipy's default, which
/// the call site reaches by passing no keyword. `points=[-1, 1]` selects
/// `qagpe` even though **both** entries are discarded — scipy filters
/// break points to the strictly interior ones in Python before QUADPACK
/// sees them, so this is one of the five live call sites Phase 03
/// Task 3.3 found running `qagpe` over an empty list.
const CHARGED_PION_QUAD: QuadOpts<'static> = QuadOpts {
    epsabs: 1e-10,
    epsrel: 1e-5,
    limit: DEFAULT_LIMIT,
    points: Some(&[-1.0, 1.0]),
};

/// The photon spectrum `dN/dE` in MeV⁻¹ from charged-pion decay.
///
/// # Parameters
///
/// * `egam` — the photon energy, MeV.
/// * `epi` — the pion's total energy, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` for a pion below its own rest
/// mass. A `NaN` `epi` fails that comparison and reaches the quadrature,
/// where the integrand is `NaN` throughout and the result is `NaN` — the
/// Cython does the same, since scipy returns the QUADPACK value and hazma
/// reads `[0]` without looking at the warning.
///
/// The quadrature's termination flag is discarded for the same reason:
/// the `.pyx` subscripts `quad(...)[0]`, so a `ier != 0` that scipy would
/// have raised an `IntegrationWarning` for is invisible to hazma today
/// and stays invisible here.
#[must_use]
pub fn dnde_photon_charged_pion(egam: f64, epi: f64) -> f64 {
    if epi < MASS_PI {
        return 0.0;
    }

    let mut integrand = |cl: f64| charged_pion_integrand(cl, egam, epi);
    match quad(&mut integrand, -1.0, 1.0, &CHARGED_PION_QUAD) {
        Ok(outcome) => outcome.value,
        // Unreachable, and asserted so by
        // `charged_pion_quad_options_are_always_accepted` below:
        // `QuadError` is a statement about the *options*
        // (`epsabs > 0`, `limit` above the surviving break-point count),
        // never about the integrand, and these options are `const`.
        // `NaN` rather than a panic, for the reason `crate::boost` gives
        // — `dispatch::map_unary` evaluates element by element and has no
        // per-element error channel, so a panic would take down a whole
        // array where the Cython would have raised once.
        Err(_) => f64::NAN,
    }
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from neutral-pion decay.
///
/// # Parameters
///
/// * `egam` — the photon energy, MeV.
/// * `epi` — the pion's total energy, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹: a box of height `2·BR(π⁰→γγ)/(E_π β)` over
/// `E_π(1 − β)/2 ≤ E_γ ≤ E_π(1 + β)/2`, and exactly `0.0` outside it or
/// for a pion below its own rest mass. A `NaN` `egam` returns `0.0` —
/// both range comparisons are false, which is also what the Cython's
/// chained comparison does.
///
/// **Both `β` and the returned height are rounded to `f32`**, because the
/// `.pyx` declares them `cdef float`. See the module docs; this is the
/// difference between passing and failing the corpus's `EXACT` budget for
/// `spectra.photon.neutral_pion`.
#[must_use]
pub fn dnde_photon_neutral_pion(egam: f64, epi: f64) -> f64 {
    if epi < MASS_PI0 {
        return 0.0;
    }

    let ratio = MASS_PI0 / epi;
    // `fcvt s3, d3` / `fcvt d3, s3`: `cdef float beta`.
    let beta = f64::from((1.0 - ratio * ratio).sqrt() as f32);

    // `/ 2.0` is emitted as `* 0.5`, which is the same double.
    let lower = (epi * (1.0 - beta)) * 0.5;
    let upper = (epi * (1.0 + beta)) * 0.5;

    if lower <= egam && egam <= upper {
        // `fcvt s0, d0` / `fcvt d0, s0`: `cdef float ret_val`. The
        // numerator `2·BR` is folded.
        f64::from(((BR_PI0_TO_A_A * 2.0) / (epi * beta)) as f32)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CHARGED_PION_QUAD, F_A_PI_SQ, FPI, MPI_SQ, TWELVE_SQRT_2, TWENTY_FOUR_PI_MPI,
        charged_pion_integrand, dnde_photon_charged_pion, dnde_photon_neutral_pion,
        dnde_pi_to_lnug,
    };
    use crate::constants::derived::photon_pion::{ME, MMU};
    use crate::constants::pdg::{BR_PI0_TO_A_A, MASS_PI, MASS_PI0};
    use crate::quad::{Ier, quad};

    /// Every folded constant, against the immediate `objdump -d` builds
    /// out of `movk` halfwords in the shipped
    /// `_pion.cpython-312-darwin.so`. A transposed halfword is exactly
    /// what this catches; so is the one-ulp `f_π` spelling trap.
    #[test]
    fn the_folded_constants_match_the_shipped_immediates() {
        assert_eq!(MPI_SQ.to_bits(), 0x40d3_05f9_3371_1326);
        assert_eq!(F_A_PI_SQ.to_bits(), 0x3f22_8fa4_a337_fdf4);
        assert_eq!(TWELVE_SQRT_2.to_bits(), 0x4030_f876_ccdf_6cda);
        assert_eq!(TWENTY_FOUR_PI_MPI.to_bits(), 0x40c4_8dae_039c_4a1d);
        // Loaded from memory rather than an immediate — it is a module
        // global in the `.pyx`. `130.41 * (1.0 / sqrt(2))` is
        // `0x40570daed2a0781a`, one ulp low.
        assert_eq!(FPI.to_bits(), 0x4057_0dae_d2a0_781b);
    }

    /// The options are `const`, so `quad` can only reject them for
    /// reasons that do not depend on the integrand. This is the assertion
    /// the `Err(_) => NaN` arm in [`dnde_photon_charged_pion`] leans on.
    #[test]
    fn charged_pion_quad_options_are_always_accepted() {
        let mut f = |_: f64| 1.0;
        let outcome = quad(&mut f, -1.0, 1.0, &CHARGED_PION_QUAD)
            .expect("the .pyx's quad options are valid for any integrand");
        assert_eq!(outcome.ier, Ier::Ok);
        // ∫_{-1}^{1} 1 dx = 2, and a constant is exact under any
        // Gauss–Kronrod rule.
        assert_eq!(outcome.value, 2.0);
    }

    /// The radiative spectrum vanishes outside `0 ≤ x ≤ 1 − r`, at both
    /// ends and for both leptons. `x = 1 − r` is `E_γ = (m_π² − m_ℓ²) /
    /// (2 m_π)`.
    #[test]
    fn the_radiative_pion_spectrum_vanishes_outside_its_support() {
        for ml in [MMU, ME] {
            let endpoint = (MASS_PI * MASS_PI - ml * ml) / (2.0 * MASS_PI);
            assert_eq!(dnde_pi_to_lnug(-1.0, ml), 0.0);
            assert_eq!(dnde_pi_to_lnug(endpoint * 1.000_001, ml), 0.0);
            assert!(dnde_pi_to_lnug(endpoint * 0.999, ml) != 0.0);
        }
    }

    /// The `.pyx`'s guard is a *rejection* — `x < 0 or (1-r) < x` — and
    /// both comparisons are false for a `NaN`, so a `NaN` photon energy
    /// reaches the arithmetic and propagates rather than being clamped to
    /// zero. Written down because the opposite reading is the natural one
    /// and is wrong; `test/test_core_photon_pion.py` pins the same
    /// behaviour against the Cython twin.
    #[test]
    fn a_nan_photon_energy_propagates_through_the_radiative_pion_spectrum() {
        assert!(dnde_pi_to_lnug(f64::NAN, MMU).is_nan());
        assert!(dnde_pi_to_lnug(f64::NAN, ME).is_nan());
        // The charged-pion integrand inherits it, and so does the
        // spectrum: every quadrature node is `NaN`.
        assert!(charged_pion_integrand(0.0, f64::NAN, 500.0).is_nan());
        assert!(dnde_photon_charged_pion(f64::NAN, 500.0).is_nan());
    }

    /// `π → e ν γ` is *not* helicity-suppressed the way `π → e ν` is —
    /// the photon carries the angular momentum — so the electron channel's
    /// radiative spectrum is comparable to the muon's rather than
    /// `(m_e/m_μ)²` below it. This pins the sign and the rough scale of
    /// the whole `f + g` construction, which a dropped term would break.
    #[test]
    fn both_radiative_channels_are_positive_and_comparable() {
        for egam in [1.0, 5.0, 20.0, 50.0, 65.0] {
            let electron = dnde_pi_to_lnug(egam, ME);
            assert!(electron > 0.0, "pi -> e nu gamma at {egam} MeV");
        }
        // The muon channel closes at (m_pi^2 - m_mu^2)/(2 m_pi) = 29.8
        // MeV, well below the electron channel's 69.8 MeV.
        for egam in [1.0, 5.0, 20.0] {
            assert!(dnde_pi_to_lnug(egam, MMU) > 0.0, "pi -> mu nu gamma");
        }
    }

    /// The neutral-pion spectrum is a box: flat between the two edges,
    /// zero outside, and its area is `2·BR(π⁰→γγ)` — two photons per
    /// decay, weighted by the branching fraction. Checked at a boost
    /// where the `f32` rounding is visible, so the area holds only to
    /// single precision.
    #[test]
    fn the_neutral_pion_spectrum_is_a_box_of_area_two_br() {
        let epi = 500.0;
        let beta = (1.0 - (MASS_PI0 / epi) * (MASS_PI0 / epi)).sqrt();
        let lower = epi * (1.0 - beta) / 2.0;
        let upper = epi * (1.0 + beta) / 2.0;

        let height = dnde_photon_neutral_pion(0.5 * (lower + upper), epi);
        // Flat: the same double at three interior points.
        for frac in [0.25, 0.5, 0.75] {
            let egam = lower + frac * (upper - lower);
            assert_eq!(
                dnde_photon_neutral_pion(egam, epi).to_bits(),
                height.to_bits()
            );
        }
        assert_eq!(dnde_photon_neutral_pion(lower * 0.999, epi), 0.0);
        assert_eq!(dnde_photon_neutral_pion(upper * 1.001, epi), 0.0);

        // width * height = 2*BR, to f32 precision because both `beta` and
        // the height are rounded there.
        let area = (upper - lower) * height;
        assert!(
            (area - 2.0 * BR_PI0_TO_A_A).abs() < 1e-6 * 2.0 * BR_PI0_TO_A_A,
            "area {area} is not 2*BR = {}",
            2.0 * BR_PI0_TO_A_A
        );
    }

    /// The `f32` truncations are load-bearing: an all-`f64` spelling of
    /// the same box differs in the eighth significant figure, which is
    /// four orders of magnitude past the corpus's `EXACT` budget for this
    /// entry point.
    #[test]
    fn the_neutral_pion_float_truncations_move_the_value() {
        let (egam, epi) = (250.0, 500.0);
        let ratio = MASS_PI0 / epi;
        let all_f64 = (BR_PI0_TO_A_A * 2.0) / (epi * (1.0 - ratio * ratio).sqrt());
        let shipped = dnde_photon_neutral_pion(egam, epi);
        assert_ne!(shipped.to_bits(), all_f64.to_bits());
        let relative = ((shipped - all_f64) / all_f64).abs();
        assert!(
            (1e-9..1e-6).contains(&relative),
            "f32 rounding should show up around 1e-8 relative, got {relative:e}"
        );
    }

    /// Below its own rest mass each pion returns exactly zero, at the
    /// threshold each is already nonzero somewhere in its support.
    #[test]
    fn both_pions_are_zero_below_threshold() {
        assert_eq!(dnde_photon_charged_pion(10.0, MASS_PI * 0.999), 0.0);
        assert_eq!(dnde_photon_neutral_pion(60.0, MASS_PI0 * 0.999), 0.0);
        assert!(dnde_photon_charged_pion(10.0, MASS_PI) > 0.0);
        assert!(dnde_photon_neutral_pion(MASS_PI0 / 2.0, MASS_PI0) > 0.0);
    }

    /// The charged-pion spectrum is positive over the bulk of its support
    /// and falls to zero above the boosted endpoint
    /// `E_γ^max = γ_π(1 + β_π) · ENG_GAM_MAX_PIRG`.
    #[test]
    fn the_charged_pion_spectrum_is_positive_then_vanishes() {
        let epi = 500.0;
        let gamma = epi / MASS_PI;
        let beta = (1.0 - (MASS_PI / epi) * (MASS_PI / epi)).sqrt();
        // ENG_GAM_MAX_PIRG is the pion-rest-frame maximum; boosting it
        // forward gives the lab-frame edge.
        let edge = 69.783_457_719_487_52 * gamma * (1.0 + beta);
        for egam in [0.5, 5.0, 50.0, 200.0] {
            assert!(
                dnde_photon_charged_pion(egam, epi) > 0.0,
                "charged pion spectrum at {egam} MeV"
            );
        }
        assert!(dnde_photon_charged_pion(edge * 1.05, epi) < 1e-12);
    }

    /// The quadrature converges everywhere hazma is used. Phase 03
    /// Task 3.3 measured that the port tracks scipy only where QUADPACK
    /// converges — beyond that Wynn's ε-algorithm is chaotic and the two
    /// can separate without bound — and made "no live shape reaches the
    /// other regime" an obligation each consumer re-checks. This is that
    /// check for the project's first `qagp` consumer.
    ///
    /// The boundary is `E_π ≈ 4e4` MeV (`γ_π ≈ 290`), where a single
    /// photon energy first reports `ier = 5`; below `3e4` the whole
    /// eight-decade photon grid is `ier = 0`. That is 40 GeV against a
    /// library whose domain is **sub-GeV** dark matter, and the corpus's
    /// most boosted block is `10 m_π = 1396` MeV (`γ = 10`), so nothing
    /// hazma computes today is within an order of magnitude of it.
    ///
    /// And the regime is not a cliff here anyway: over an 11 × 8 grid
    /// reaching `E_π = 1e5` MeV the port's `ier` equals the flag scipy
    /// raises on the Cython twin at **all 88 points**, including both
    /// `ier = 4` entries and the non-monotonic pattern in between, with
    /// the values still agreeing to 2.8e-11 on macOS/arm64 — and to
    /// 6.3e-10 on Linux/glibc, which is the *only* place in this kernel
    /// where the platform shows through: in the converged regime the port
    /// tracks the Cython to 1e-12 on both. `test/test_core_photon_pion.py`
    /// carries that comparison, since it needs scipy.
    #[test]
    fn the_live_grid_never_leaves_the_converged_regime() {
        for epi in [MASS_PI, 150.0, 200.0, 500.0, 1500.0, 1e4, 3e4] {
            for egam in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5] {
                let mut integrand = |cl: f64| charged_pion_integrand(cl, egam, epi);
                let outcome =
                    quad(&mut integrand, -1.0, 1.0, &CHARGED_PION_QUAD).expect("valid options");
                assert_eq!(
                    outcome.ier,
                    Ier::Ok,
                    "quad did not converge at E_gam = {egam}, E_pi = {epi}"
                );
            }
        }
    }

    /// At rest the pion's photon spectrum is the pion-rest-frame sum
    /// itself: `γ = 1`, `β = 0`, the Jacobian is `1/2` and the integral
    /// over `cos θ` restores the factor of two. So the closed-form
    /// integrand evaluated at any angle reproduces the quadrature's
    /// answer — a statement about the boost machinery that owes nothing
    /// to the Cython.
    #[test]
    fn a_pion_at_rest_reproduces_its_own_rest_frame_sum() {
        let egam = 20.0;
        let integrated = dnde_photon_charged_pion(egam, MASS_PI);
        let at_one_angle = 2.0 * charged_pion_integrand(0.0, egam, MASS_PI);
        let relative = ((integrated - at_one_angle) / at_one_angle).abs();
        assert!(
            relative < 1e-12,
            "rest-frame spectrum {integrated} vs {at_one_angle} (rel {relative:e})"
        );
    }
}
