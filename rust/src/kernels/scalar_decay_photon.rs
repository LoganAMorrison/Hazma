//! `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx` — the
//! photon spectrum from a boosted scalar mediator's decay.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3). The rest-frame tables, their cache and the mode bit set come
//! from [`crate::kernels::mediator_tables`]; the entry point is
//! [`crate::scalar_mediator`].
//!
//! # Naming
//!
//! The literal transcription of the file name would be
//! `scalar_mediator_decay_spectrum`, whose `scalar_mediator` half is
//! already the PyO3 submodule's name — the same collision
//! [`crate::kernels::scalar_xs`] and [`crate::kernels::vector_xs`]
//! resolved, and resolved the same way.
//!
//! # What the mediator contributes
//!
//! A scalar of mass `ms` and energy `eng_s ≥ ms` decays isotropically in
//! its own rest frame; the lab spectrum is the boost integral over
//! `cos θ ∈ [−1, 1]` of the rest-frame spectrum at the Doppler-shifted
//! energy, weighted by the Jacobian `1/(2γ|1 − β cos θ|)`. Six channels
//! ride inside that integral — FSR off `e⁺e⁻`, `μ⁺μ⁻` and `π⁺π⁻`, and the
//! decay continua of `π⁺π⁻`, `π⁰π⁰` and `μ⁺μ⁻` — and a seventh, the
//! monochromatic `s → γγ` line, is added outside it over the boosted
//! line window.
//!
//! Each channel is gated by a bit of
//! [`mediator_tables::ScalarPhotonModes`], which is this entry point's
//! `modes` list folded once per call rather than re-tested inside the
//! integrand as the `.pyx` did.
//!
//! # Only the charged pion is tabulated
//!
//! The `.pyx` interpolates the charged pion's rest-frame photon spectrum
//! off a 500-point table (`:46`) and calls the muon and neutral-pion
//! kernels point-by-point (`:149`, `:153`), so this module does the same:
//! [`mediator_tables::photon_tables`] serves the pion column and
//! [`photon_muon::dnde_photon_muon`] / [`photon_pion::dnde_photon_neutral_pion`]
//! are called per quadrature node. The muon column that table set also
//! builds belongs to the vector twin and is never read here — see
//! [`mediator_tables::photon_tables`]'s own note on why the set is
//! shared anyway.
//!
//! # `qe**2`
//!
//! The `.pyx` computes `qe = sqrt(4 π α)` once at import (`:27`) and
//! squares it again in both FSR coefficients (`:82`, `:113`). Squaring a
//! rounded square root is not an identity in general, so [`QE_SQUARED`]
//! is the folded `4 π α` **and** carries a test proving the two agree at
//! the legacy `α = 1/137`.
//!
//! # Where the FMAs are
//!
//! `pow(x, n)` is a **call** in the C tree clang contracts on, so
//! `A ± B` fuses only where `A` or `B` is a syntactic multiply — the one
//! rule Phase 05 distilled
//! (`projects/cython-to-rust/learnings/phase-05-mediator-cross-sections.md`).
//! Every [`f64::mul_add`] below is a site where that rule fires, and the
//! plain `+` beside `x * x` in [`dnde_fsr_l_srf`]'s polynomial is a site
//! where it does not: `pow(x, 2.0)` is a call, so the trailing
//! `+ pow(x, 2.0)` has a multiply on neither side.
//!
//! # The complex `**`
//!
//! `dnde_fsr_l_srf`'s `(1 − 4μ²)**1.5` puts its whole coefficient into
//! `double _Complex` — `grep -c SoftComplexToDouble` on the generated C
//! returns 6, of which one is a call site (`:113`) — so that coefficient
//! goes through [`crate::kernels::soft_complex`] rather than through
//! `powf` and `/`. `dnde_fsr_cp_srf`'s sibling spells the same factor
//! `sqrt(1 − 4μ²)` and stays real.

use crate::constants::legacy;
use crate::kernels::mediator_tables::{
    self, PartialWidths, PhotonTables, ScalarPhotonModes, SpectrumError,
};
use crate::kernels::soft_complex::{complex_quotient_real_denominator, soft_complex_pow_1_5};
use crate::kernels::{photon_muon, photon_pion};
use crate::quad::{DEFAULT_LIMIT, QuadOpts, quad};

/// `π²`, as clang folds `pow(M_PI, 2.0)` in both FSR coefficients.
///
/// LLVM rewrites `pow(x, 2.0)` to `x·x` and then constant-folds it, so
/// this is one rounding of `π·π` — `pi_squared_matches_libm` re-derives
/// it.
const PI_SQUARED: f64 = std::f64::consts::PI * std::f64::consts::PI;

/// The `cos θ` quadrature, copied from `:184-186`.
///
/// `points=[-1, 1]` selects `qagpe` even though scipy discards both
/// break points as non-interior — the fifth of the live call sites
/// Phase 03 found doing that, and the reason [`quad`] filters rather
/// than trusting its caller. `limit` is scipy's default, reached by
/// passing no keyword.
const BOOST_QUAD: QuadOpts<'static> = QuadOpts {
    epsabs: 1e-10,
    epsrel: 1e-5,
    limit: DEFAULT_LIMIT,
    points: Some(&[-1.0, 1.0]),
};

/// `qe**2`, where `qe` is the `.pyx`'s module-level `sqrt(4 π α)`.
///
/// The `.pyx` takes the square root at import and squares it again in
/// each FSR coefficient, and squaring a rounded root is **not** an
/// identity in general — at `α = 1` the round trip loses a bit. At the
/// legacy `α = 1/137` it happens to be exact, measured rather than
/// assumed, so this constant is the folded `4 π α` and
/// `qe_squared_is_the_rounded_root_squared` is what pins the two
/// together. Both halves of that test matter: if a future edit changed
/// `α`, the equality is what would break.
const QE_SQUARED: f64 = 4.0 * std::f64::consts::PI * legacy::ALPHA_EM;

/// FSR off the charged pions, in the scalar's rest frame — `:63-84`.
///
/// # Parameters
///
/// * `egam` — photon energy in the mediator rest frame, MeV.
/// * `ms` — mediator mass, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` outside `0 ≤ x ≤ 1 − 4μ_π²` where
/// `x = 2 E_γ / m_s`. A `NaN` `egam` fails both comparisons and
/// propagates through the arithmetic, as the Cython's did.
#[must_use]
pub fn dnde_fsr_cp_srf(egam: f64, ms: f64) -> f64 {
    let mupi = legacy::MASS_PI / ms;
    let x = (2.0 * egam) / ms;
    let mupi2 = mupi * mupi;
    // `1 - 4 mupi**2`: the right operand is a multiply, so clang fuses.
    let xmax = (-4.0_f64).mul_add(mupi2, 1.0);

    if x < 0.0 || x > xmax {
        return 0.0;
    }

    let root_lo = (1.0 - x).sqrt();
    let root_hi = (xmax - x).sqrt();
    // `(1 - x) - root_lo*root_hi` and `(-1 + x) - root_lo*root_hi`: the
    // subtracted operand is a multiply in both, so both fuse.
    let upper = (-root_lo).mul_add(root_hi, 1.0 - x);
    let lower = (-root_lo).mul_add(root_hi, -1.0 + x);
    let log_term = ((upper * upper) / (lower * lower)).ln();
    // `(-1 + 2 mupi**2) + x`: the fused left operand is no longer a
    // multiply, so the `+ x` is a plain add.
    let weight = 2.0_f64.mul_add(mupi2, -1.0) + x;

    // `term1 + term2` with `term1` a multiply: the left operand wins.
    let numerator = (-2.0 * root_lo).mul_add(root_hi, weight * log_term);
    let dynamic = numerator / x;
    let coeff = QE_SQUARED / ((8.0 * xmax.sqrt()) * PI_SQUARED);

    (2.0 * (dynamic * coeff)) / ms
}

/// FSR off a charged lepton pair, in the scalar's rest frame — `:90-115`.
///
/// # Parameters
///
/// * `egam` — photon energy in the mediator rest frame, MeV.
/// * `ml` — lepton mass, MeV: the `.pyx` passes the legacy electron and
///   muon masses.
/// * `ms` — mediator mass, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` outside `0 ≤ x ≤ 1 − 4μ_l²`.
///
/// # Errors
///
/// [`SpectrumError::NonReal`] at `m_s = 2 m_l` and `E_γ = 0` together,
/// where the coefficient's complex division has a zero denominator and
/// `__Pyx_SoftComplexToDouble` raised `TypeError`. Every other energy at
/// that mass fails the `x > xmax` guard first.
pub fn dnde_fsr_l_srf(egam: f64, ml: f64, ms: f64) -> Result<f64, SpectrumError> {
    let mul = ml / ms;
    let x = (2.0 * egam) / ms;
    let mul2 = mul * mul;
    // `pow(mul, 4.0)` stays a libm call: LLVM folds exponent 2 and
    // leaves 4 alone.
    let mul4 = mul.powf(4.0);
    let xmax = (-4.0_f64).mul_add(mul2, 1.0);

    if x < 0.0 || x > xmax {
        return Ok(0.0);
    }

    let root_lo = (1.0 - x).sqrt();
    let root_hi = (xmax - x).sqrt();
    // `-1 + 4 mul**2`, shared by the leading coefficient and the log's
    // inner product — one value in the C after common-subexpression
    // elimination, so one value here.
    let shifted = 4.0_f64.mul_add(mul2, -1.0);
    let lead = (4.0 * shifted) * root_lo;

    let inner = ((-1.0 + x) * (shifted + x)).sqrt();
    let upper = (1.0 - x) + inner;
    let lower = (-1.0 + x) + inner;
    let log_term = ((upper * upper) / (lower * lower)).ln();

    let mut weight = (-12.0_f64).mul_add(mul2, 2.0);
    weight = 16.0_f64.mul_add(mul4, weight);
    weight = (-2.0_f64).mul_add(x, weight);
    weight = (8.0 * mul2).mul_add(x, weight);
    // `+ pow(x, 2.0)`: a call on the right, a fused add on the left, so
    // nothing to contract.
    weight += x * x;

    let numerator = lead.mul_add(root_hi, weight * log_term);
    let dynamic = numerator / x;

    // `qe**2 / (16 (1 - 4 mul**2)**1.5 pi**2)`, in `double _Complex`
    // throughout because of the `1.5` exponent.
    let denominator = (16.0 * soft_complex_pow_1_5(xmax)) * PI_SQUARED;
    let coeff = complex_quotient_real_denominator(QE_SQUARED, denominator)?;

    Ok((2.0 * (dynamic * coeff)) / ms)
}

/// The boost integrand at `cos θ = cl` — `:123-155`.
///
/// `pws` is indexed `[e e, mu mu, pi0 pi0, pi pi, g g]`, the order
/// `hazma/scalar_mediator/_scalar_mediator_spectra.py:74-78` builds. The
/// first four entries are read **unconditionally**, before any mode bit
/// is tested, exactly as the `.pyx` reads them.
///
/// # Errors
///
/// [`SpectrumError::OutOfBounds`] if `pws` is shorter than four
/// elements; [`SpectrumError::NonReal`] from [`dnde_fsr_l_srf`].
fn integrand(
    cl: f64,
    eng_gam: f64,
    eng_s: f64,
    ms: f64,
    pws: PartialWidths<'_>,
    modes: ScalarPhotonModes,
    tables: &PhotonTables,
) -> Result<f64, SpectrumError> {
    let pwee = pws.get(0)?;
    let pwmumu = pws.get(1)?;
    let pwpi0pi0 = pws.get(2)?;
    let pwpipi = pws.get(3)?;

    // `1 - (ms/eng_s)**2` is `1 - <call>`: no multiply on either side.
    let ratio = ms / eng_s;
    let beta = (1.0 - ratio * ratio).sqrt();
    let gamma = eng_s / ms;
    // `1 - beta*cl` fuses; the `.pyx` spells it twice and clang emits the
    // same fused value both times.
    let doppler = (-beta).mul_add(cl, 1.0);
    let jac = 1.0 / ((2.0 * gamma) * doppler.abs());
    let eng_gam_srf = (eng_gam * gamma) * doppler;
    let daughter_energy = ms / 2.0;

    // The channel order is the `.pyx`'s (`:142-153`). It is load-bearing:
    // `result = result + <channel>` is a fused multiply-add per channel,
    // so a different order is a different sum.
    let mut result = 0.0;
    if modes.contains(ScalarPhotonModes::ELECTRON_FSR) {
        result = pwee.mul_add(dnde_fsr_l_srf(eng_gam_srf, legacy::MASS_E, ms)?, result);
    }
    if modes.contains(ScalarPhotonModes::CHARGED_PION_FSR) {
        result = pwpipi.mul_add(dnde_fsr_cp_srf(eng_gam_srf, ms), result);
    }
    if modes.contains(ScalarPhotonModes::CHARGED_PION_DECAY) {
        result = (2.0 * pwpipi).mul_add(tables.charged_pion.lookup(eng_gam_srf), result);
    }
    if modes.contains(ScalarPhotonModes::NEUTRAL_PION_DECAY) {
        result = (2.0 * pwpi0pi0).mul_add(
            photon_pion::dnde_photon_neutral_pion(eng_gam_srf, daughter_energy),
            result,
        );
    }
    if modes.contains(ScalarPhotonModes::MUON_FSR) {
        result = pwmumu.mul_add(dnde_fsr_l_srf(eng_gam_srf, legacy::MASS_MU, ms)?, result);
    }
    if modes.contains(ScalarPhotonModes::MUON_DECAY) {
        result = (2.0 * pwmumu).mul_add(
            photon_muon::dnde_photon_muon(eng_gam_srf, daughter_energy),
            result,
        );
    }

    Ok(jac * result)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ at one photon energy — `:166-191`.
///
/// # Parameters
///
/// * `eng_gam` — photon energy in the lab frame, MeV.
/// * `eng_s` — the mediator's total energy, MeV.
/// * `ms` — the mediator's mass, MeV.
/// * `pws` — normalised partial widths, `[e e, mu mu, pi0 pi0, pi pi, g g]`.
/// * `modes` — which channels to include.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` for `eng_s < ms` — below which the
/// `.pyx` returns before touching `pws`, so a short buffer does not
/// raise there.
///
/// The quadrature's termination flag is discarded because the `.pyx`
/// subscripts `quad(...)[0]`: an `ier != 0` that scipy reported as an
/// `IntegrationWarning` was already invisible to hazma's *value*, and
/// this call site does raise that warning today. The port stops emitting
/// it, as every quad-backed kernel ported in Phase 04 did.
///
/// # Errors
///
/// [`SpectrumError::OutOfBounds`] if `pws` has fewer than four elements,
/// or fewer than five when the `g g` bit is set and `eng_gam` falls
/// inside the boosted line window; [`SpectrumError::NonReal`] from
/// [`dnde_fsr_l_srf`].
pub fn spectrum_point(
    eng_gam: f64,
    eng_s: f64,
    ms: f64,
    pws: PartialWidths<'_>,
    modes: ScalarPhotonModes,
    tables: &PhotonTables,
) -> Result<f64, SpectrumError> {
    if eng_s < ms {
        return Ok(0.0);
    }

    let ratio = ms / eng_s;
    let beta = (1.0 - ratio * ratio).sqrt();
    let eplus = (eng_s * (1.0 + beta)) / 2.0;
    let eminus = (eng_s * (1.0 - beta)) / 2.0;

    // scipy propagates an exception out of the integrand rather than
    // absorbing it, so the first failure is remembered and raised after
    // the integrator finishes; `NaN` keeps QUADPACK's own arithmetic
    // defined until then. Same shape as
    // `crate::kernels::vector_xs::thermal_cross_section`.
    let mut failure: Option<SpectrumError> = None;
    let mut kernel = |cl: f64| match integrand(cl, eng_gam, eng_s, ms, pws, modes, tables) {
        Ok(value) => value,
        Err(error) => {
            failure = failure.or(Some(error));
            f64::NAN
        }
    };
    let mut result = match quad(&mut kernel, -1.0, 1.0, &BOOST_QUAD) {
        Ok(outcome) => outcome.value,
        // Unreachable, and asserted so by
        // `boost_quad_options_are_always_accepted`: `QuadError` is a
        // statement about the options, never about the integrand, and
        // these options are `const`.
        Err(_) => f64::NAN,
    };
    if let Some(error) = failure {
        return Err(error);
    }

    if modes.contains(ScalarPhotonModes::TWO_PHOTON_LINE) && eminus <= eng_gam && eng_gam <= eplus {
        // `pws[4] * 1.` in the `.pyx` (`:189`); multiplying by one is
        // exact, so it is dropped rather than written out.
        result += pws.get(4)? / (eng_s * beta);
    }

    Ok(result)
}

/// The memoized rest-frame tables for a mediator of mass `ms` MeV.
///
/// Re-exported so [`crate::scalar_mediator`] builds them once per call
/// rather than once per photon energy — the whole point of the cache the
/// `.pyx` declared and never populated.
#[must_use]
pub fn tables_for(ms: f64) -> std::sync::Arc<PhotonTables> {
    mediator_tables::photon_tables(ms)
}

#[cfg(test)]
mod tests {
    use super::{
        BOOST_QUAD, PI_SQUARED, QE_SQUARED, dnde_fsr_cp_srf, dnde_fsr_l_srf, spectrum_point,
        tables_for,
    };
    use crate::constants::legacy;
    use crate::kernels::mediator_tables::{PartialWidths, ScalarPhotonModes, SpectrumError};
    use crate::quad::quad;

    /// Every channel open, which is the entry point's default `modes`.
    fn all_modes() -> ScalarPhotonModes {
        ScalarPhotonModes::from_names(ScalarPhotonModes::NAMES)
    }

    /// A partial-width vector with all five entries distinct, so a
    /// channel reading the wrong index cannot pass unnoticed.
    const PWS: [f64; 5] = [0.11, 0.23, 0.31, 0.17, 0.05];

    #[test]
    fn qe_squared_is_the_rounded_root_squared() {
        // The `.pyx` spells this `sqrt(4 pi alpha)` at import and `qe**2`
        // in each coefficient. The two agree bit-for-bit at the legacy
        // alpha...
        let qe = (4.0 * std::f64::consts::PI * legacy::ALPHA_EM).sqrt();
        assert_eq!(QE_SQUARED.to_bits(), (qe * qe).to_bits());
        // ...and the round trip is not an identity in general, which is
        // why the equality above is asserted rather than assumed.
        let counterexample = 4.0 * std::f64::consts::PI;
        let root = counterexample.sqrt();
        assert_ne!((root * root).to_bits(), counterexample.to_bits());
    }

    #[test]
    fn a_power_of_two_coefficient_makes_fusion_unobservable() {
        // Why fourteen of this task's thirty-seven fused sites survived
        // their mutation (see the task note's campaign table): where the
        // coefficient is a power of two, `c * mu2` is exact -- it only
        // shifts an exponent -- so the fused and unfused spellings round
        // exactly once either way and agree bit-for-bit. That is a
        // property of the coefficient, not of the sampled grid, and it is
        // asserted rather than argued because the alternative reading
        // ("the grid is too coarse") would send a later reader hunting.
        let shapes: [(f64, f64); 6] = [
            (-4.0, 1.0), // `1 - 4 mu**2`, both FSR functions
            (2.0, -1.0), // `-1 + 2 mupi**2`
            (4.0, -1.0), // `-1 + 4 mul**2`
            (16.0, 0.0), // `16 mul**4 + <weight>`
            (-2.0, 0.0), // `-2 x + <weight>`
            (-8.0, 2.0), // `2 - 8 mul**4`, the vector twin's
        ];
        for (coeff, addend) in shapes {
            for mass in [212.0, 250.0, 400.0, 550.0, 900.0, 1500.0] {
                for lepton in [legacy::MASS_E, legacy::MASS_MU] {
                    let mu = lepton / mass;
                    let mu2 = mu * mu;
                    assert_eq!(
                        coeff.mul_add(mu2, addend).to_bits(),
                        (coeff * mu2 + addend).to_bits(),
                        "{coeff} * mu2 + {addend} at m={mass}, ml={lepton}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_twelve_coefficient_is_not_in_that_class() {
        // The complement, and the reason the campaign's two remaining
        // survivors are "unobserved on that grid" rather than "provably
        // equivalent": 12 is not a power of two, so `12 mul**2` rounds and
        // the fused sum can differ. It does at 8.5% of mediator masses
        // above the `2 m_mu` threshold that opens the muon FSR channel at
        // all -- measured with exact rational arithmetic over 20,001
        // masses -- and 212 MeV is the lowest whole-MeV one. Below
        // threshold the site is unreachable, which is why a grid anchored
        // at 550 MeV left it alive.
        let mu = legacy::MASS_MU / 212.0;
        let mu2 = mu * mu;
        assert_ne!(
            (-12.0_f64).mul_add(mu2, 2.0).to_bits(),
            (-12.0 * mu2 + 2.0).to_bits()
        );
        // And the electron never reaches it: `12 (m_e/m_s)**2` is so far
        // below the ulp of 2 that the product's rounding cannot survive
        // the addition.
        let mu = legacy::MASS_E / 212.0;
        let mu2 = mu * mu;
        assert_eq!(
            (-12.0_f64).mul_add(mu2, 2.0).to_bits(),
            (-12.0 * mu2 + 2.0).to_bits()
        );
    }

    #[test]
    fn pi_squared_matches_libm() {
        assert_eq!(
            PI_SQUARED.to_bits(),
            std::f64::consts::PI.powf(2.0).to_bits()
        );
    }

    #[test]
    fn boost_quad_options_are_always_accepted() {
        // `spectrum_point` returns NaN on a `QuadError` rather than
        // panicking; this is the assertion that the arm is unreachable.
        let mut integrand = |_: f64| 1.0;
        let outcome = quad(&mut integrand, -1.0, 1.0, &BOOST_QUAD);
        assert!(outcome.is_ok());
        assert!((outcome.unwrap().value - 2.0).abs() < 1e-12);
    }

    #[test]
    fn fsr_vanishes_outside_the_kinematic_window() {
        // `x < 0` and `x > xmax` are the two guards; the endpoint itself
        // is inside the window, so `xmax` exactly must not return zero
        // for the same reason.
        let ms = 550.0;
        assert_eq!(dnde_fsr_cp_srf(-1.0, ms), 0.0);
        assert_eq!(dnde_fsr_cp_srf(ms, ms), 0.0);
        assert_eq!(dnde_fsr_l_srf(-1.0, legacy::MASS_MU, ms), Ok(0.0));
        assert_eq!(dnde_fsr_l_srf(ms, legacy::MASS_MU, ms), Ok(0.0));
    }

    #[test]
    fn fsr_is_positive_and_falling_inside_the_window() {
        // A soft-photon spectrum: 1/E-like, so strictly decreasing away
        // from the soft end. Cheap, but it is the property that would
        // break if a sign or a log argument were inverted.
        let ms = 550.0;
        let mut previous = f64::INFINITY;
        for egam in [1.0, 5.0, 20.0, 60.0, 120.0] {
            let value = dnde_fsr_cp_srf(egam, ms);
            assert!(value > 0.0, "cp FSR non-positive at {egam}");
            assert!(value < previous, "cp FSR not falling at {egam}");
            previous = value;
        }
        let mut previous = f64::INFINITY;
        for egam in [1.0, 5.0, 20.0, 60.0, 120.0] {
            let value = dnde_fsr_l_srf(egam, legacy::MASS_MU, ms).unwrap();
            assert!(value > 0.0, "muon FSR non-positive at {egam}");
            assert!(value < previous, "muon FSR not falling at {egam}");
            previous = value;
        }
    }

    #[test]
    fn the_lepton_fsr_coefficient_raises_at_twice_the_lepton_mass() {
        // `SpectrumError::NonReal`'s only reachable argument: `xmax` is
        // exactly zero, so the complex division's denominator is zero,
        // and only `E_gamma = 0` gets past the `x > xmax` guard to see
        // it.
        let ms = 2.0 * legacy::MASS_MU;
        assert_eq!(
            dnde_fsr_l_srf(0.0, legacy::MASS_MU, ms),
            Err(SpectrumError::NonReal)
        );
        assert_eq!(dnde_fsr_l_srf(1.0, legacy::MASS_MU, ms), Ok(0.0));
    }

    #[test]
    fn a_mediator_below_its_own_mass_contributes_nothing() {
        let tables = tables_for(550.0);
        assert_eq!(
            spectrum_point(
                30.0,
                500.0,
                550.0,
                PartialWidths::new(&PWS),
                all_modes(),
                &tables,
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn a_mediator_below_its_mass_does_not_read_the_partial_widths() {
        // The `.pyx` returns before the buffer is touched, so an empty
        // `pws` is fine there. A port that validated up front would
        // raise instead.
        let tables = tables_for(550.0);
        assert_eq!(
            spectrum_point(
                30.0,
                500.0,
                550.0,
                PartialWidths::new(&[]),
                all_modes(),
                &tables,
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn a_short_partial_width_buffer_is_out_of_bounds() {
        let tables = tables_for(550.0);
        for length in 0..4 {
            let pws = vec![0.1; length];
            assert_eq!(
                spectrum_point(
                    30.0,
                    600.0,
                    550.0,
                    PartialWidths::new(&pws),
                    all_modes(),
                    &tables,
                ),
                Err(SpectrumError::OutOfBounds),
                "length {length} should be out of bounds"
            );
        }
    }

    #[test]
    fn the_fifth_partial_width_is_read_only_inside_the_line_window() {
        // The behaviour a length check would have destroyed, and the
        // shipped extension's: at ms = 550, eng_s = 600 the window is
        // roughly [180, 420] MeV, so 30 MeV succeeds on four entries and
        // 300 MeV does not.
        let tables = tables_for(550.0);
        let four = [0.11, 0.23, 0.31, 0.17];
        assert!(
            spectrum_point(
                30.0,
                600.0,
                550.0,
                PartialWidths::new(&four),
                all_modes(),
                &tables,
            )
            .is_ok()
        );
        assert_eq!(
            spectrum_point(
                300.0,
                600.0,
                550.0,
                PartialWidths::new(&four),
                all_modes(),
                &tables,
            ),
            Err(SpectrumError::OutOfBounds)
        );
    }

    #[test]
    fn no_modes_selected_is_exactly_zero() {
        // `modes=[]` folds to a zero bitflag, the integrand returns
        // `jac * 0`, and QUADPACK sums exact zeros — so this is `0.0`
        // and not merely small.
        let tables = tables_for(550.0);
        assert_eq!(
            spectrum_point(
                30.0,
                600.0,
                550.0,
                PartialWidths::new(&PWS),
                ScalarPhotonModes::default(),
                &tables,
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn the_two_photon_line_is_the_only_channel_outside_the_integral() {
        // Everything else is inside the boost integral, so switching the
        // line on must move the answer by exactly the analytic step
        // `pw / (E_s beta)` inside the window and by nothing outside it.
        let tables = tables_for(550.0);
        let (eng_s, ms): (f64, f64) = (600.0, 550.0);
        let line = ScalarPhotonModes::from_names(["g g"]);
        let ratio = ms / eng_s;
        let beta = (1.0 - ratio * ratio).sqrt();
        let inside =
            spectrum_point(300.0, eng_s, ms, PartialWidths::new(&PWS), line, &tables).unwrap();
        assert_eq!(inside, PWS[4] / (eng_s * beta));
        let outside =
            spectrum_point(30.0, eng_s, ms, PartialWidths::new(&PWS), line, &tables).unwrap();
        assert_eq!(outside, 0.0);
    }

    #[test]
    fn the_channels_sum_to_the_total_within_the_quadrature_budget() {
        // Not bit-equality: each channel is a separate quadrature with
        // its own subdivision, so the sum of six integrals is not the
        // integral of six sums. The declared budget is the integrator's
        // own relative tolerance, 1e-5.
        let tables = tables_for(550.0);
        let pws = PartialWidths::new(&PWS);
        let total = spectrum_point(30.0, 600.0, 550.0, pws, all_modes(), &tables).unwrap();
        let summed: f64 = ScalarPhotonModes::NAMES
            .iter()
            .map(|name| {
                spectrum_point(
                    30.0,
                    600.0,
                    550.0,
                    pws,
                    ScalarPhotonModes::from_names([name]),
                    &tables,
                )
                .unwrap()
            })
            .sum();
        assert!(
            (total - summed).abs() <= 1e-5 * total.abs(),
            "total {total} vs summed {summed}"
        );
    }
}
