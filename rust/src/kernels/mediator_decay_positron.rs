//! `hazma/{scalar,vector}_mediator/*_positron_spec.pyx` — the positron
//! spectrum from a boosted mediator's decay.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3). Tables, cache and mode enum come from
//! [`crate::kernels::mediator_tables`]; the entry points are
//! [`crate::scalar_mediator`] and [`crate::vector_mediator`].
//!
//! # One module for two `.pyx`
//!
//! Both photon modules got a module each because they genuinely differ.
//! These two do not: normalise `scalar_mediator_positron_spec.pyx`
//! against `vector_mediator_positron_spec.pyx` by rewriting `s`↔`v`,
//! `ms`↔`mv`, `eng_s`↔`eng_v` and "scalar"↔"vector", and `diff` reports
//! only those substitutions and the order of two `import` lines. The
//! arithmetic, the control flow, the mode strings, the quadrature
//! options and the `pws` indexing are the same text. So there is one
//! kernel here and two thin PyO3 pairs on top of it, and the scalar and
//! vector spectra are bit-for-bit identical at equal arguments by
//! construction rather than by coincidence.
//!
//! That makes this module a naming exception of the same kind
//! [`crate::kernels::photon_tables`] is, and it is recorded in
//! [`crate::kernels`]'s own docs.
//!
//! # What the rest frame contributes
//!
//! Only two channels have a continuum: `S/V → π⁺π⁻` and `S/V → μ⁺μ⁻`,
//! each read out of a 500-point log-spaced table built at the daughter
//! energy `m/2` ([`mediator_tables::positron_tables`]). The `e⁺e⁻`
//! channel is a line, and it rides *outside* the integral for every
//! recognised mode — unlike the photon modules, where only two of the
//! seven modes carry one.
//!
//! # The threshold `NaN`, and why it is a compiler artifact
//!
//! `p = sqrt(eng_p * eng_p - m_e * m_e)` is one expression, and clang
//! contracts it to `fma(eng_p, eng_p, -(m_e * m_e))`. Inside the FMA the
//! square is exact while `m_e * m_e` has already been rounded — upward,
//! for the legacy `m_e`, by `1.45e-17` — so at `eng_p == m_e` the
//! radicand is that rounding's negation and `sqrt` answers `NaN`. It is
//! not in the source's semantics: `p` there is a momentum, and the
//! momentum at the threshold is zero. [`momentum`] fuses as clang does,
//! keeping the port's arithmetic on every other energy, and clamps the
//! radicand at zero. See that function's docs for the measurement.

use crate::constants::legacy;
use crate::kernels::mediator_tables::{
    self, PartialWidths, PositronMode, PositronTables, SpectrumError,
};
use crate::quad::{DEFAULT_LIMIT, QuadOpts, quad};

/// The `cos θ` quadrature, copied from
/// `scalar_mediator_positron_spec.pyx:209-211` and its vector clone's
/// `:210-212`.
///
/// Identical to the two photon modules' — break points included, and as
/// there scipy discards both of them as non-interior while still
/// selecting `qagpe`.
const BOOST_QUAD: QuadOpts<'static> = QuadOpts {
    epsabs: 1e-10,
    epsrel: 1e-5,
    limit: DEFAULT_LIMIT,
    points: Some(&[-1.0, 1.0]),
};

/// `m_e²` at the legacy electron mass, rounded once.
///
/// Written out because it is the *rounded* square that [`momentum`]'s
/// FMA subtracts, and the rounding is the whole subject of that
/// function's docs.
const MASS_E_SQUARED: f64 = legacy::MASS_E * legacy::MASS_E;

/// `|p|` in MeV for a positron of total energy `eng_p` MeV.
///
/// The `.pyx` spells this `sqrt(eng_p * eng_p - me * me)` and clang
/// contracts it, which is observable: at `eng_p` exactly equal to the
/// legacy `m_e` the shipped extension returns `NaN` from every mediator
/// positron spectrum, and both neighbouring doubles return `0.0`.
///
/// The mechanism is the FMA's single rounding.
/// `fma(m_e, m_e, -(m_e * m_e))` is `exact(m_e²) − round(m_e²)`, and
/// `round(0.510998928²)` is `1.4517720908119372e-17` *above* the exact
/// square, so the radicand is negative by that much and `sqrt` gives
/// `NaN`. Nothing in the `.pyx` asks for that; the radicand is `|p|²`
/// and a negative one is threshold rounding.
///
/// So the FMA is kept — it is what every other energy's value was
/// computed with — and the radicand is clamped at zero. The comparison
/// is written `radicand < 0.0` rather than `radicand.max(0.0)` so that a
/// `NaN` energy still propagates a `NaN` instead of being silently
/// turned into a momentum of zero.
///
/// Recorded in
/// `docs/followups/done/positron-spectrum-nan-at-legacy-electron-mass.md`,
/// which is where the two ways out of this were weighed.
#[must_use]
pub fn momentum(eng_p: f64) -> f64 {
    let radicand = eng_p.mul_add(eng_p, -MASS_E_SQUARED);
    if radicand < 0.0 {
        return 0.0;
    }
    radicand.sqrt()
}

/// The boost integral's integrand — `:106-161` (scalar), `:107-162`
/// (vector).
///
/// # Parameters
///
/// * `cl` — `cos θ` between the positron and the boost axis.
/// * `eng_p` — positron energy in the lab frame, MeV.
/// * `eng_m` — the mediator's total energy, MeV.
/// * `mass` — the mediator's mass, MeV.
/// * `pws` — normalised partial widths, `[e e, mu mu, pi pi]`.
/// * `mode` — the selected channel, or `None` for a string that names
///   none.
/// * `tables` — the rest-frame tables for `mass`.
///
/// `pws[1]` and `pws[2]` are read before the mode is consulted, exactly
/// as the `.pyx` reads them, so a short buffer raises for *every* mode —
/// `"e e"` included, which then returns `0.0` without using either.
///
/// # Errors
///
/// [`SpectrumError::OutOfBounds`] if `pws` is shorter than three
/// elements. Unlike the photon modules there is no `NonReal` arm:
/// `grep -c SoftComplexToDouble` over the generated C is `0` for both of
/// these `.pyx`, because neither raises anything to a fractional power.
fn integrand(
    cl: f64,
    eng_p: f64,
    eng_m: f64,
    mass: f64,
    pws: PartialWidths<'_>,
    mode: Option<PositronMode>,
    tables: &PositronTables,
) -> Result<f64, SpectrumError> {
    if eng_p < legacy::MASS_E {
        return Ok(0.0);
    }

    let pwmumu = pws.get(1)?;
    let pwpipi = pws.get(2)?;

    let p = momentum(eng_p);
    let gamma = eng_m / mass;
    let ratio = mass / eng_m;
    let beta = (1.0 - ratio * ratio).sqrt();
    let eng_p_rf = gamma * (-(p * beta)).mul_add(cl, eng_p);

    // `:142-145`, term by term. Which of the four `±` clang contracts is
    // measured rather than assumed: every spelling below was rewritten
    // to its opposite, rebuilt and re-measured against the live Cython,
    // and the table is in this task's note. Two results are worth
    // naming here. The `pow(beta * cl, 2)` head stays *unfused* because
    // a `pow` call is not a syntactic multiply for clang's contraction,
    // and fusing it costs 46 bit-equal values; the `m_e²` subtraction
    // also stays unfused, which is the one place the contraction rule
    // would have predicted a fusion and measurement said otherwise.
    let beta_cl = beta * cl;
    let head = (1.0 + beta_cl * beta_cl) * eng_p * eng_p;
    let mass_coefficient = (beta * beta).mul_add(cl.mul_add(cl, -1.0), 1.0) * legacy::MASS_E;
    let momentum_coefficient = ((2.0 * beta) * cl) * eng_p;
    let radicand = (-momentum_coefficient).mul_add(p, head - mass_coefficient * legacy::MASS_E);
    let jac = p / ((2.0 * radicand.sqrt()) * gamma);

    // The `.pyx` initialises both to `0.0` and fills only the ones its
    // mode names, so the sum is over one term or two and the untouched
    // half is an exact zero.
    let mut dnde_cp = 0.0;
    let mut dnde_mu = 0.0;
    match mode {
        // An unrecognised mode falls off the end of the `.pyx`'s
        // `if`-chain, and a C function that does that returns zero. Both
        // partial widths are read above regardless, which is why this
        // arm is here rather than short-circuited by the caller.
        None | Some(PositronMode::ElectronLine) => return Ok(0.0),
        Some(PositronMode::Total) => {
            dnde_cp = pwpipi * tables.charged_pion.lookup(eng_p_rf);
            dnde_mu = pwmumu * tables.muon.lookup(eng_p_rf);
        }
        Some(PositronMode::ChargedPionDecay) => {
            dnde_cp = pwpipi * tables.charged_pion.lookup(eng_p_rf);
        }
        Some(PositronMode::MuonDecay) => {
            dnde_mu = pwmumu * tables.muon.lookup(eng_p_rf);
        }
    }

    Ok(jac * (dnde_cp + dnde_mu))
}

/// The positron spectrum `dN/dE` in MeV⁻¹ at one energy — `:166-215`
/// (scalar), `:167-216` (vector).
///
/// # Parameters
///
/// * `eng_p` — positron energy in the lab frame, MeV.
/// * `eng_m` — the mediator's total energy, MeV.
/// * `mass` — the mediator's mass, MeV.
/// * `pws` — normalised partial widths, `[e e, mu mu, pi pi]`.
/// * `mode` — the selected channel, or `None` for a string that names
///   none.
/// * `tables` — the rest-frame tables for `mass`.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹; exactly `0.0` for `eng_m < mass`, where the `.pyx`
/// returns before touching `pws`, and exactly `0.0` for a mode that
/// names nothing — in that last case *without* the line term, because
/// the `.pyx` falls through to its `return result` with `result` still
/// zero.
///
/// The `e⁺e⁻` line rides outside the integral and is added to every
/// recognised mode. `"e e"` short-circuits before the integral and
/// returns the line alone, so it is the one mode whose value costs no
/// quadrature.
///
/// As in both photon modules the quadrature's termination flag is
/// discarded, because the `.pyx` subscripts `quad(...)[0]`; the port
/// therefore no longer raises the `IntegrationWarning` scipy raises here
/// today.
///
/// # Errors
///
/// [`SpectrumError::OutOfBounds`] if `pws` is empty and `eng_p` falls
/// inside the line window, or has fewer than three elements once the
/// integral runs.
pub fn spectrum_point(
    eng_p: f64,
    eng_m: f64,
    mass: f64,
    pws: PartialWidths<'_>,
    mode: Option<PositronMode>,
    tables: &PositronTables,
) -> Result<f64, SpectrumError> {
    if eng_m < mass {
        return Ok(0.0);
    }

    let ratio = mass / eng_m;
    let beta = (1.0 - ratio * ratio).sqrt();
    // `((4 m_e) m_e) / m²` as the `.pyx` associates it. Multiplying by
    // four only shifts an exponent, so this is the same double as
    // `4 * MASS_E_SQUARED`; it is spelled the `.pyx`'s way anyway, and
    // the subtraction does not fuse because its subtrahend is a
    // division rather than a multiply.
    let r = (1.0 - ((4.0 * legacy::MASS_E) * legacy::MASS_E) / (mass * mass)).sqrt();
    let eplus = (eng_m * r.mul_add(beta, 1.0)) / 2.0;
    let eminus = (eng_m * (-r).mul_add(beta, 1.0)) / 2.0;

    // Computed *before* the integral, as the `.pyx` computes it, so a
    // short `pws` reports the index the Cython reported. `pws[0]` is
    // read only inside the window, which is why an empty buffer can
    // still succeed outside it.
    let mut lines_contrib = 0.0;
    if eminus <= eng_p && eng_p <= eplus {
        lines_contrib = pws.get(0)? / (eng_m * beta);
    }

    if mode == Some(PositronMode::ElectronLine) {
        return Ok(lines_contrib);
    }
    if mode.is_none() {
        return Ok(0.0);
    }

    let mut failure: Option<SpectrumError> = None;
    let mut kernel = |cl: f64| match integrand(cl, eng_p, eng_m, mass, pws, mode, tables) {
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

    Ok(result + lines_contrib)
}

/// The memoized rest-frame tables for a mediator of mass `mass` MeV.
#[must_use]
pub fn tables_for(mass: f64) -> std::sync::Arc<PositronTables> {
    mediator_tables::positron_tables(mass)
}

#[cfg(test)]
mod tests {
    use super::{BOOST_QUAD, MASS_E_SQUARED, momentum, spectrum_point, tables_for};
    use crate::constants::legacy;
    use crate::kernels::mediator_tables::{PartialWidths, PositronMode, SpectrumError};
    use crate::quad::quad;

    /// A mediator light enough that both rest-frame tables are zero.
    ///
    /// The daughter energy is `mass / 2`, so below `2 m_μ` the muon
    /// column and below `2 m_π` the charged-pion column are identically
    /// zero — every entry fails its own kernel's threshold guard. The
    /// boost integral over a zero integrand is then exactly zero, which
    /// makes the whole spectrum equal to its line term and gives the
    /// tests below a closed form to compare against.
    const LIGHT_MASS: f64 = 125.0;

    #[test]
    fn boost_quad_options_are_always_accepted() {
        let mut integrand = |_: f64| 1.0;
        let outcome = quad(&mut integrand, -1.0, 1.0, &BOOST_QUAD);
        assert!(outcome.is_ok());
        assert!((outcome.unwrap().value - 2.0).abs() < 1e-12);
    }

    #[test]
    fn the_legacy_electron_mass_squared_rounds_upward() {
        // The premise of `momentum`'s clamp, asserted rather than
        // assumed: `m_e * m_e` is *above* the exact square, so the FMA's
        // exactly-computed `m_e²` minus it is negative.
        let rounded = MASS_E_SQUARED;
        let radicand = legacy::MASS_E.mul_add(legacy::MASS_E, -rounded);
        assert!(
            radicand < 0.0,
            "expected the rounded square to sit above the exact one, got {radicand:e}"
        );
    }

    #[test]
    fn the_momentum_is_zero_at_the_threshold_and_real_above_it() {
        // The one point where the shipped Cython answered NaN.
        assert_eq!(momentum(legacy::MASS_E), 0.0);
        // Both neighbours were already finite there and stay finite.
        let above = f64::from_bits(legacy::MASS_E.to_bits() + 1);
        let below = f64::from_bits(legacy::MASS_E.to_bits() - 1);
        assert!(momentum(above).is_finite() && momentum(above) > 0.0);
        assert_eq!(momentum(below), 0.0);
    }

    #[test]
    fn the_momentum_keeps_the_compilers_fused_spelling() {
        // Away from the threshold the clamp is inert and the value is
        // the FMA's, which is not always the unfused expression's. The
        // energy below is one where the two spellings differ, so a
        // future edit to the unfused form would fail here rather than
        // silently move 3,340 corpus values.
        let eng_p = 1.0221;
        let fused = momentum(eng_p);
        let unfused = (eng_p * eng_p - MASS_E_SQUARED).sqrt();
        assert_ne!(
            fused.to_bits(),
            unfused.to_bits(),
            "the fused and unfused spellings agree here, so this energy no longer \
             distinguishes them; pick another rather than deleting the test"
        );
        assert_eq!(
            fused.to_bits(),
            eng_p.mul_add(eng_p, -MASS_E_SQUARED).sqrt().to_bits()
        );
    }

    #[test]
    fn a_mediator_at_rest_or_below_its_mass_contributes_nothing() {
        // `:194-195` returns before reading `pws`, so an empty buffer
        // must not raise here.
        let tables = tables_for(LIGHT_MASS);
        let empty: [f64; 0] = [];
        let value = spectrum_point(
            10.0,
            LIGHT_MASS - 1.0,
            LIGHT_MASS,
            PartialWidths::new(&empty),
            Some(PositronMode::Total),
            &tables,
        );
        assert_eq!(value, Ok(0.0));
    }

    #[test]
    fn a_dark_continuum_leaves_the_line_alone() {
        // With both tables identically zero the integral is exactly
        // zero, so the spectrum inside the line window is the closed
        // form `pw_ee / (E β)` — an analytic pin rather than a
        // regression value.
        let tables = tables_for(LIGHT_MASS);
        let eng_m = 250.0;
        let pws = [0.25, 0.5, 0.25];
        let widths = PartialWidths::new(&pws);
        let ratio = LIGHT_MASS / eng_m;
        let beta = (1.0 - ratio * ratio).sqrt();
        let expected = pws[0] / (eng_m * beta);

        // Mid-window: `eminus ≈ 16.7`, `eplus ≈ 233` MeV at these
        // arguments, so 100 MeV is comfortably inside.
        for mode in [
            PositronMode::Total,
            PositronMode::MuonDecay,
            PositronMode::ChargedPionDecay,
            PositronMode::ElectronLine,
        ] {
            let value =
                spectrum_point(100.0, eng_m, LIGHT_MASS, widths, Some(mode), &tables).unwrap();
            assert_eq!(value, expected, "mode {mode:?} moved the line term");
        }
        // Outside the window there is no line and no continuum either.
        let outside = spectrum_point(
            1.0,
            eng_m,
            LIGHT_MASS,
            widths,
            Some(PositronMode::Total),
            &tables,
        );
        assert_eq!(outside, Ok(0.0));
    }

    #[test]
    fn an_unrecognised_mode_is_silently_zero_even_inside_the_line_window() {
        // The `.pyx`'s `if`-chain has no `else`, so it falls through to
        // `return result` with `result` still zero — the line term is
        // computed and then discarded. Filed as
        // `docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`.
        let tables = tables_for(LIGHT_MASS);
        let pws = [0.25, 0.5, 0.25];
        let value = spectrum_point(
            100.0,
            250.0,
            LIGHT_MASS,
            PartialWidths::new(&pws),
            None,
            &tables,
        );
        assert_eq!(value, Ok(0.0));
    }

    #[test]
    fn the_partial_widths_are_read_where_the_cython_read_them() {
        let tables = tables_for(LIGHT_MASS);
        let empty: [f64; 0] = [];
        let two = [0.25, 0.5];

        // `pws[0]` is read only inside the line window, so an empty
        // buffer succeeds outside it and raises inside — the shipped
        // behaviour Task 6.2 measured for the photon pair.
        assert_eq!(
            spectrum_point(
                1.0,
                250.0,
                LIGHT_MASS,
                PartialWidths::new(&empty),
                Some(PositronMode::ElectronLine),
                &tables
            ),
            Ok(0.0)
        );
        assert_eq!(
            spectrum_point(
                100.0,
                250.0,
                LIGHT_MASS,
                PartialWidths::new(&empty),
                Some(PositronMode::ElectronLine),
                &tables
            ),
            Err(SpectrumError::OutOfBounds)
        );

        // `pws[1]` and `pws[2]` are read in the integrand, which only
        // the three continuum modes reach. A two-element buffer is
        // therefore fine for `"e e"` and raises for the rest.
        assert!(
            spectrum_point(
                100.0,
                250.0,
                LIGHT_MASS,
                PartialWidths::new(&two),
                Some(PositronMode::ElectronLine),
                &tables
            )
            .is_ok()
        );
        assert_eq!(
            spectrum_point(
                100.0,
                250.0,
                LIGHT_MASS,
                PartialWidths::new(&two),
                Some(PositronMode::Total),
                &tables
            ),
            Err(SpectrumError::OutOfBounds)
        );
    }

    #[test]
    fn a_boosted_continuum_splits_into_its_two_channels() {
        // Above `2 m_π` both tables are populated, and the `.pyx` builds
        // `"total"` as the plain sum of the two single-channel
        // integrands minus the line it double-counts. Checking the
        // decomposition pins the mode dispatch without pinning a
        // quadrature value.
        let mass = 600.0;
        let eng_m = 900.0;
        let tables = tables_for(mass);
        let pws = [0.2, 0.5, 0.3];
        let widths = PartialWidths::new(&pws);
        // 200 MeV is inside the line window, whose edges here are
        // `eminus = 114.6` and `eplus = 785.4` MeV, so every mode below
        // carries the line and the decomposition has to subtract it.
        let at = |mode| spectrum_point(200.0, eng_m, mass, widths, Some(mode), &tables).unwrap();

        let total = at(PositronMode::Total);
        let muon = at(PositronMode::MuonDecay);
        let pion = at(PositronMode::ChargedPionDecay);
        let line = at(PositronMode::ElectronLine);
        assert!(line > 0.0, "200 MeV should sit inside the line window");
        assert!(
            muon > line && pion > line,
            "both continua must add to the line"
        );
        // The integrand is linear in the two channels, so the continua
        // add exactly once the shared line term is removed. The
        // tolerance is `BOOST_QUAD`'s own `epsrel`, because `"total"`
        // integrates one summed integrand while the two single-channel
        // calls each subdivide on their own — three different node sets,
        // each converged only to that relative error.
        let residual = ((muon - line) + (pion - line) - (total - line)).abs();
        assert!(
            residual < BOOST_QUAD.epsrel * (total - line).abs(),
            "channels did not add: total={total:e} muon={muon:e} pion={pion:e} line={line:e}"
        );
    }
}
