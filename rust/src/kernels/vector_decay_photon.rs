//! `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx` — the
//! photon spectrum from a boosted vector mediator's decay.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3). Tables, cache and mode enum come from
//! [`crate::kernels::mediator_tables`]; the entry points are
//! [`crate::vector_mediator`].
//!
//! Named for the same reason as [`crate::kernels::scalar_decay_photon`]:
//! the literal transcription's `vector_mediator` half is already the
//! PyO3 submodule's name.
//!
//! # How it differs from its scalar twin
//!
//! Structurally not at all — the same 500-point log-spaced rest-frame
//! table, the same `1/E` tail below `10⁻¹` MeV, the same `cos θ` QAGP
//! with `epsabs = 1e-10`, `epsrel = 1e-5` and the same discarded break
//! points. Four differences, all of them data:
//!
//! * **two tables, not one.** The vector interpolates the muon's
//!   rest-frame photon spectrum as well as the charged pion's (`:35-36`),
//!   where the scalar calls the muon kernel per node.
//! * **one mode string, not a list.** [`mediator_tables::PhotonMode`]
//!   selects exactly one channel; there is no bit set and no way to ask
//!   for two.
//! * **`π⁰γ` in place of `π⁰π⁰` and `γγ`.** The vector's monochromatic
//!   line is the photon of `V → π⁰γ`, and the accompanying `π⁰`
//!   continuum is evaluated at the two-body energy
//!   `(m_π⁰² + m_V²)/(2 m_V)` rather than at `m_V/2`.
//! * **every channel is evaluated, whatever the mode.** The `.pyx`
//!   computes all six components and then selects one (`:150-179`), so a
//!   single-channel call still pays for the other five — and, more to
//!   the point, still *raises* where any of the six would.
//!   [`integrand`] keeps that.
//!
//! # Where the FMAs are, and the complex `**`
//!
//! Same rule as the scalar twin: `pow(x, n)` is a call in the C tree
//! clang contracts on, so `A ± B` fuses only where one side is a
//! syntactic multiply. Here it is the *charged-pion* FSR coefficient
//! that goes complex — `(1 − 4μ_π²)**1.5` at `:73`, the one live
//! `SoftComplexToDouble` call site of six in the generated C — and the
//! lepton coefficient that stays real, which is the mirror image of the
//! scalar module.

use crate::constants::legacy;
use crate::kernels::mediator_tables::{
    self, PartialWidths, PhotonMode, PhotonTables, SpectrumError,
};
use crate::kernels::photon_pion;
use crate::kernels::soft_complex::{complex_quotient_real_denominator, soft_complex_pow_1_5};
use crate::quad::{DEFAULT_LIMIT, QuadOpts, quad};

/// `π²`, as clang folds `pow(M_PI, 2.0)` in both FSR coefficients.
const PI_SQUARED: f64 = std::f64::consts::PI * std::f64::consts::PI;

/// The `cos θ` quadrature, copied from `:219-221`.
///
/// Identical to the scalar twin's, break points included — and, as
/// there, scipy discards both of them as non-interior while still
/// selecting `qagpe`.
const BOOST_QUAD: QuadOpts<'static> = QuadOpts {
    epsabs: 1e-10,
    epsrel: 1e-5,
    limit: DEFAULT_LIMIT,
    points: Some(&[-1.0, 1.0]),
};

/// `qe**2`, where `qe` is the `.pyx`'s module-level `sqrt(4 π α)` (`:20`).
///
/// Folded, and equal to the `.pyx`'s rounded round trip at the legacy
/// `α`; see [`crate::kernels::scalar_decay_photon`]'s copy for why that
/// is asserted rather than assumed.
const QE_SQUARED: f64 = 4.0 * std::f64::consts::PI * legacy::ALPHA_EM;

/// FSR off the charged pions, in the vector's rest frame — `:61-83`.
///
/// # Parameters
///
/// * `egam` — photon energy in the mediator rest frame, MeV.
/// * `mv` — mediator mass, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` outside `0 ≤ x ≤ 1 − 4μ_π²` where
/// `x = 2 E_γ / m_V`.
///
/// # Errors
///
/// [`SpectrumError::NonReal`] at `m_V = 2 m_π` and `E_γ = 0` together,
/// where the coefficient's complex division has a zero denominator.
pub fn dnde_fsr_cp_vrf(egam: f64, mv: f64) -> Result<f64, SpectrumError> {
    let mupi = legacy::MASS_PI / mv;
    let x = (2.0 * egam) / mv;
    let mupi2 = mupi * mupi;
    let xmax = (-4.0_f64).mul_add(mupi2, 1.0);

    if x < 0.0 || x > xmax {
        return Ok(0.0);
    }

    // `qe**2 / (4 (1 - 4 mupi**2)**1.5 pi**2)`, in `double _Complex`
    // throughout because of the `1.5` exponent. Computed before
    // `dynamic`, as the `.pyx` does, so a failure here is the failure the
    // Cython reported.
    let denominator = (4.0 * soft_complex_pow_1_5(xmax)) * PI_SQUARED;
    let coeff = complex_quotient_real_denominator(QE_SQUARED, denominator)?;

    let root_lo = (1.0 - x).sqrt();
    let root_hi = (xmax - x).sqrt();
    let four_mupi2 = 4.0 * mupi2;

    // `-1 - (4 mupi**2)(-1 + x)`: the subtracted operand is a multiply,
    // so it fuses; the `+ x` and `+ x*x` after it do not, the first
    // because its left operand is now a fused add and the second because
    // `pow(x, 2.0)` is a call.
    let mut weight = (-four_mupi2).mul_add(-1.0 + x, -1.0);
    weight += x;
    weight += x * x;

    // `1 + root_lo*root_hi` and `-1 + root_lo*root_hi` both fuse; the
    // `- x` / `+ x` that follow do not.
    let upper = root_lo.mul_add(root_hi, 1.0) - x;
    let lower = root_lo.mul_add(root_hi, -1.0) + x;
    let log_term = ((upper * upper) / (lower * lower)).ln();

    let leading = ((2.0 * root_hi) * weight) / root_lo;
    let trailing = 4.0_f64.mul_add(mupi2, -1.0) * (2.0_f64.mul_add(mupi2, -1.0) + x);
    // `leading + trailing*log`: the left operand is a *division*, not a
    // multiply, so clang falls through to the right operand and fuses
    // there instead.
    let numerator = trailing.mul_add(log_term, leading);
    let dynamic = numerator / x;

    Ok((2.0 * (dynamic * coeff)) / mv)
}

/// FSR off a charged lepton pair, in the vector's rest frame — `:86-110`.
///
/// # Parameters
///
/// * `egam` — photon energy in the mediator rest frame, MeV.
/// * `ml` — lepton mass, MeV.
/// * `mv` — mediator mass, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` outside `0 ≤ x ≤ 1 − 4μ_l²`. This
/// coefficient is real arithmetic throughout — the `.pyx` spells its
/// `1 − 4μ²` factor `sqrt(...)` rather than `**1.5`, so nothing here can
/// raise.
#[must_use]
pub fn dnde_fsr_l_vrf(egam: f64, ml: f64, mv: f64) -> f64 {
    let mul = ml / mv;
    let x = (2.0 * egam) / mv;
    let mul2 = mul * mul;
    // `pow(mul, 4.0)` stays a libm call.
    let mul4 = mul.powf(4.0);
    let xmax = (-4.0_f64).mul_add(mul2, 1.0);

    if x < 0.0 || x > xmax {
        return 0.0;
    }

    let coeff = -QE_SQUARED / (((8.0 * xmax.sqrt()) * 2.0_f64.mul_add(mul2, 1.0)) * PI_SQUARED);

    let root_lo = (1.0 - x).sqrt();
    let root_hi = (xmax - x).sqrt();
    let four_mul2 = 4.0 * mul2;

    let mut weight = (-four_mul2).mul_add(-1.0 + x, 2.0);
    weight = (-2.0_f64).mul_add(x, weight);
    weight += x * x;

    let upper = root_lo.mul_add(root_hi, -1.0) + x;
    let lower = root_lo.mul_add(root_hi, 1.0) - x;
    let log_term = ((upper * upper) / (lower * lower)).ln();

    let leading = ((2.0 * root_hi) * weight) / root_lo;
    let mut trailing = (-8.0_f64).mul_add(mul4, 2.0);
    trailing = (-four_mul2).mul_add(x, trailing);
    trailing = (-2.0 + x).mul_add(x, trailing);
    let numerator = trailing.mul_add(log_term, leading);
    let dynamic = numerator / x;

    (2.0 * (dynamic * coeff)) / mv
}

/// The boost integrand at `cos θ = cl` — `:115-179`.
///
/// `pws` is indexed `[e e, mu mu, pi0 g, pi pi]`, the order
/// `hazma/vector_mediator/_vector_mediator_spectra.py:87-90` builds, and
/// all four entries are read before anything else happens — including
/// for a `mode` that names nothing.
///
/// Every one of the six components is evaluated whatever the mode, which
/// is what the `.pyx` does and is observable: the charged-pion FSR can
/// raise, so `mode = "e e g"` at `m_V = 2 m_π` and `E_γ = 0` raises there
/// and raises here.
///
/// # Errors
///
/// [`SpectrumError::OutOfBounds`] if `pws` is shorter than four
/// elements; [`SpectrumError::NonReal`] from [`dnde_fsr_cp_vrf`].
fn integrand(
    cl: f64,
    eng_gam: f64,
    eng_v: f64,
    mv: f64,
    pws: PartialWidths<'_>,
    mode: Option<PhotonMode>,
    tables: &PhotonTables,
) -> Result<f64, SpectrumError> {
    let pwee = pws.get(0)?;
    let pwmumu = pws.get(1)?;
    let pwpi0g = pws.get(2)?;
    let pwpipi = pws.get(3)?;

    let ratio = mv / eng_v;
    let beta = (1.0 - ratio * ratio).sqrt();
    let gamma = eng_v / mv;
    let doppler = (-beta).mul_add(cl, 1.0);
    let jac = 1.0 / ((2.0 * gamma) * doppler.abs());
    let eng_gam_vrf = (eng_gam * gamma) * doppler;

    let dnde_ee_f = pwee * dnde_fsr_l_vrf(eng_gam_vrf, legacy::MASS_E, mv);
    let dnde_mu_f = pwmumu * dnde_fsr_l_vrf(eng_gam_vrf, legacy::MASS_MU, mv);
    let dnde_cp_f = pwpipi * dnde_fsr_cp_vrf(eng_gam_vrf, mv)?;
    let dnde_cp_d = (2.0 * pwpipi) * tables.charged_pion.lookup(eng_gam_vrf);
    // The `π⁰` of `V → π⁰ γ` is monochromatic in the vector's rest
    // frame, at `(m_π⁰² + m_V²)/(2 m_V)` (`:158`) — not at `m_V/2`, which
    // is where the two tables above are built.
    let e_pi0 = (0.5 * (legacy::MASS_PI0 * legacy::MASS_PI0 + mv * mv)) / mv;
    let dnde_np_d = pwpi0g * photon_pion::dnde_photon_neutral_pion(eng_gam_vrf, e_pi0);
    let dnde_mu_d = (2.0 * pwmumu) * tables.muon.lookup(eng_gam_vrf);

    let component = match mode {
        // An unrecognised mode falls off the end of the `.pyx`'s
        // `if`-chain, and a C function that does that returns zero. The
        // partial widths above are still read, which is why this arm is
        // here rather than short-circuited by the caller.
        None => return Ok(0.0),
        Some(PhotonMode::Total) => {
            // The `.pyx`'s left-to-right fold (`:163-164`). The order is
            // load-bearing: floating-point addition does not associate.
            ((((dnde_ee_f + dnde_mu_f) + dnde_cp_f) + dnde_cp_d) + dnde_np_d) + dnde_mu_d
        }
        Some(PhotonMode::ElectronFsr) => dnde_ee_f,
        Some(PhotonMode::ChargedPionFsr) => dnde_cp_f,
        Some(PhotonMode::ChargedPionDecay) => dnde_cp_d,
        Some(PhotonMode::NeutralPionLine) => dnde_np_d,
        Some(PhotonMode::MuonFsr) => dnde_mu_f,
        Some(PhotonMode::MuonDecay) => dnde_mu_d,
    };

    Ok(jac * component)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ at one photon energy — `:184-227`.
///
/// # Parameters
///
/// * `eng_gam` — photon energy in the lab frame, MeV.
/// * `eng_v` — the mediator's total energy, MeV.
/// * `mv` — the mediator's mass, MeV.
/// * `pws` — normalised partial widths, `[e e, mu mu, pi0 g, pi pi]`.
/// * `mode` — the selected channel, or `None` for a string that names
///   none.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` for `eng_v < mv` — where the
/// `.pyx` returns before touching `pws`.
///
/// The `π⁰γ` line rides outside the integral and is added for `"pi0 g"`
/// and `"total"` only (`:223`). As in the scalar twin, the quadrature's
/// termination flag is discarded because the `.pyx` subscripts
/// `quad(...)[0]`, so the port no longer raises the `IntegrationWarning`
/// scipy raises here today.
///
/// # Errors
///
/// [`SpectrumError::OutOfBounds`] if `pws` has fewer than three elements
/// and `eng_gam` falls inside the line window, or fewer than four in any
/// case; [`SpectrumError::NonReal`] from [`dnde_fsr_cp_vrf`].
pub fn spectrum_point(
    eng_gam: f64,
    eng_v: f64,
    mv: f64,
    pws: PartialWidths<'_>,
    mode: Option<PhotonMode>,
    tables: &PhotonTables,
) -> Result<f64, SpectrumError> {
    if eng_v < mv {
        return Ok(0.0);
    }

    let ratio = mv / eng_v;
    let beta = (1.0 - ratio * ratio).sqrt();
    let eplus = (eng_v * (1.0 + beta)) / 2.0;
    let eminus = (eng_v * (1.0 - beta)) / 2.0;

    // Computed *before* the integral, as the `.pyx` does, so a short
    // `pws` reports the index the Cython reported.
    let mut lines_contrib = 0.0;
    if eminus <= eng_gam && eng_gam <= eplus {
        lines_contrib = pws.get(2)? / (eng_v * beta);
    }

    let mut failure: Option<SpectrumError> = None;
    let mut kernel = |cl: f64| match integrand(cl, eng_gam, eng_v, mv, pws, mode, tables) {
        Ok(value) => value,
        Err(error) => {
            failure = failure.or(Some(error));
            f64::NAN
        }
    };
    let result = match quad(&mut kernel, -1.0, 1.0, &BOOST_QUAD) {
        Ok(outcome) => outcome.value,
        // Unreachable; see `boost_quad_options_are_always_accepted`.
        Err(_) => f64::NAN,
    };
    if let Some(error) = failure {
        return Err(error);
    }

    if mode.is_some_and(PhotonMode::has_line) {
        return Ok(result + lines_contrib);
    }
    Ok(result)
}

/// The memoized rest-frame tables for a mediator of mass `mv` MeV.
///
/// Both columns are read here, unlike the scalar twin.
#[must_use]
pub fn tables_for(mv: f64) -> std::sync::Arc<PhotonTables> {
    mediator_tables::photon_tables(mv)
}

#[cfg(test)]
mod tests {
    use super::{
        BOOST_QUAD, PI_SQUARED, QE_SQUARED, dnde_fsr_cp_vrf, dnde_fsr_l_vrf, spectrum_point,
        tables_for,
    };
    use crate::constants::legacy;
    use crate::kernels::mediator_tables::{PartialWidths, PhotonMode, SpectrumError};
    use crate::quad::quad;

    /// Four distinct partial widths, so a channel reading the wrong
    /// index cannot pass unnoticed.
    const PWS: [f64; 4] = [0.11, 0.23, 0.31, 0.17];

    const ALL_MODES: [PhotonMode; 7] = [
        PhotonMode::Total,
        PhotonMode::ElectronFsr,
        PhotonMode::ChargedPionFsr,
        PhotonMode::ChargedPionDecay,
        PhotonMode::NeutralPionLine,
        PhotonMode::MuonFsr,
        PhotonMode::MuonDecay,
    ];

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
    fn pi_squared_matches_libm() {
        assert_eq!(
            PI_SQUARED.to_bits(),
            std::f64::consts::PI.powf(2.0).to_bits()
        );
    }

    #[test]
    fn boost_quad_options_are_always_accepted() {
        let mut integrand = |_: f64| 1.0;
        let outcome = quad(&mut integrand, -1.0, 1.0, &BOOST_QUAD);
        assert!(outcome.is_ok());
        assert!((outcome.unwrap().value - 2.0).abs() < 1e-12);
    }

    #[test]
    fn fsr_vanishes_outside_the_kinematic_window() {
        let mv = 550.0;
        assert_eq!(dnde_fsr_cp_vrf(-1.0, mv), Ok(0.0));
        assert_eq!(dnde_fsr_cp_vrf(mv, mv), Ok(0.0));
        assert_eq!(dnde_fsr_l_vrf(-1.0, legacy::MASS_MU, mv), 0.0);
        assert_eq!(dnde_fsr_l_vrf(mv, legacy::MASS_MU, mv), 0.0);
    }

    #[test]
    fn fsr_is_positive_and_falling_inside_the_window() {
        let mv = 550.0;
        let mut previous = f64::INFINITY;
        for egam in [1.0, 5.0, 20.0, 60.0, 120.0] {
            let value = dnde_fsr_cp_vrf(egam, mv).unwrap();
            assert!(value > 0.0, "cp FSR non-positive at {egam}");
            assert!(value < previous, "cp FSR not falling at {egam}");
            previous = value;
        }
        let mut previous = f64::INFINITY;
        for egam in [1.0, 5.0, 20.0, 60.0, 120.0] {
            let value = dnde_fsr_l_vrf(egam, legacy::MASS_MU, mv);
            assert!(value > 0.0, "muon FSR non-positive at {egam}");
            assert!(value < previous, "muon FSR not falling at {egam}");
            previous = value;
        }
    }

    #[test]
    fn the_charged_pion_coefficient_raises_at_twice_the_pion_mass() {
        // The mirror of the scalar module's lepton case: here it is the
        // *pion* coefficient that goes complex, so this is the one
        // reachable `NonReal`.
        let mv = 2.0 * legacy::MASS_PI;
        assert_eq!(dnde_fsr_cp_vrf(0.0, mv), Err(SpectrumError::NonReal));
        assert_eq!(dnde_fsr_cp_vrf(1.0, mv), Ok(0.0));
    }

    #[test]
    fn a_single_channel_still_pays_for_the_pion_coefficient() {
        // The `.pyx` evaluates all six components before selecting one,
        // so a mode that does not name the charged-pion FSR still raises
        // where that coefficient does. A lazy port would return a number
        // here.
        let mv = 2.0 * legacy::MASS_PI;
        let tables = tables_for(mv);
        assert_eq!(
            spectrum_point(
                0.0,
                mv,
                mv,
                PartialWidths::new(&PWS),
                Some(PhotonMode::ElectronFsr),
                &tables,
            ),
            Err(SpectrumError::NonReal)
        );
    }

    #[test]
    fn a_mediator_below_its_own_mass_contributes_nothing() {
        let tables = tables_for(550.0);
        assert_eq!(
            spectrum_point(
                30.0,
                500.0,
                550.0,
                PartialWidths::new(&[]),
                Some(PhotonMode::Total),
                &tables,
            ),
            Ok(0.0)
        );
    }

    #[test]
    fn an_unknown_mode_is_zero_but_still_reads_the_partial_widths() {
        // Both halves matter. The value is the Cython's `0.0` (a C
        // function falling off its end), and the buffer reads happen
        // anyway because they precede the mode chain in the integrand.
        let tables = tables_for(550.0);
        assert_eq!(
            spectrum_point(30.0, 600.0, 550.0, PartialWidths::new(&PWS), None, &tables,),
            Ok(0.0)
        );
        let short = [0.11, 0.23, 0.31];
        assert_eq!(
            spectrum_point(
                30.0,
                600.0,
                550.0,
                PartialWidths::new(&short),
                None,
                &tables,
            ),
            Err(SpectrumError::OutOfBounds)
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
                    Some(PhotonMode::Total),
                    &tables,
                ),
                Err(SpectrumError::OutOfBounds),
                "length {length} should be out of bounds"
            );
        }
    }

    #[test]
    fn only_the_line_bearing_modes_carry_the_line() {
        // `:223` adds `lines_contrib` for `"pi0 g"` and `"total"` and for
        // nothing else, so the difference between "pi0 g" and the bare
        // `π⁰` continuum is exactly the analytic step.
        let (eng_v, mv): (f64, f64) = (600.0, 550.0);
        let tables = tables_for(mv);
        let pws = PartialWidths::new(&PWS);
        let ratio = mv / eng_v;
        let beta = (1.0 - ratio * ratio).sqrt();
        let step = PWS[2] / (eng_v * beta);

        // 300 MeV is inside the boosted line window; 30 MeV is not.
        let with_line = spectrum_point(
            300.0,
            eng_v,
            mv,
            pws,
            Some(PhotonMode::NeutralPionLine),
            &tables,
        )
        .unwrap();
        let without =
            spectrum_point(300.0, eng_v, mv, pws, Some(PhotonMode::MuonDecay), &tables).unwrap();
        assert!(with_line > step);
        assert!(without < step);

        for mode in ALL_MODES {
            let expected = matches!(mode, PhotonMode::Total | PhotonMode::NeutralPionLine);
            assert_eq!(mode.has_line(), expected, "{mode:?}");
        }
    }

    #[test]
    fn the_channels_sum_to_the_total_within_the_quadrature_budget() {
        // Six separate quadratures against one, so the budget is the
        // integrator's own relative tolerance, 1e-5 — not bit-equality.
        let tables = tables_for(550.0);
        let pws = PartialWidths::new(&PWS);
        let total =
            spectrum_point(30.0, 600.0, 550.0, pws, Some(PhotonMode::Total), &tables).unwrap();
        let summed: f64 = ALL_MODES
            .iter()
            .filter(|mode| !matches!(mode, PhotonMode::Total))
            .map(|&mode| spectrum_point(30.0, 600.0, 550.0, pws, Some(mode), &tables).unwrap())
            .sum();
        assert!(
            (total - summed).abs() <= 1e-5 * total.abs(),
            "total {total} vs summed {summed}"
        );
    }
}
