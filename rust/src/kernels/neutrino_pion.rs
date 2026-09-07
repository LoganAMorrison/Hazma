//! The neutrino spectra from charged-pion decay, ported from
//! `hazma/spectra/_neutrino/_pion.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::neutrino`] is the Python-visible half.
//!
//! # The physics
//!
//! A charged pion decays to `μ ν_μ` (BR 0.9998770) and to `e ν_e`
//! (BR 1.230e-4), and the spectrum is the sum of three terms:
//!
//! * the **prompt muon-neutrino line** at [`ENU_MU_PI_RF`], boosted;
//! * the **prompt electron-neutrino line** at [`ENU_E_PI_RF`], boosted,
//!   contributed by [`dnde_e_nue`] alone;
//! * the **muon-decay continuum**, `π → μ ν_μ` followed by
//!   `μ → e ν̄_e ν_μ`, obtained by boosting [`super::neutrino_muon`]'s
//!   spectrum out of the pion frame with the massless flat-boost integral
//!
//!   ```text
//!   dN/dE = 1/(2 β γ) ∫_{γE(1−β)}^{γE(1+β)} dE'  (dN/dE')_μ(E', E_μ^rf) / E' .
//!   ```
//!
//! The two flavors of that continuum are integrated **separately**, in two
//! `quad` calls that differ only in which row of the muon spectrum the
//! integrand returns — so a single evaluation of this kernel runs two
//! adaptive quadratures whose integrand is a closed-form kernel. That is
//! one level shallower than [`super::photon_rho`]'s nesting.
//!
//! # The prompt electron-neutrino line is added once, not twice
//!
//! That line is the one place this module does not reproduce the `.pyx`
//! bit for bit. `_pion.pyx` summed two `cdef`s, `c_dnde_mu_numu_point`
//! and `c_dnde_e_nue_point`, that were meant to partition the pion's two
//! decay modes — and the first of them carried a `π → ν_e e` term as well
//! as its own, so the electron row of the sum came out at `2 BR(π → e
//! ν_e)` where physics wants one. The muon row was never affected:
//! `c_dnde_e_nue_point` writes nothing there, which is what makes the
//! asymmetry a transcription slip rather than a convention. [`dnde_mu_numu`]
//! therefore carries the `π → μ ν_μ` line alone and [`dnde_e_nue`] is the
//! sole source of the electron one; see
//! `docs/followups/todo/neutrino-pion-electron-line-counted-twice.md` for
//! the measurement. The parity corpus still pins the pre-repair arrays;
//! `test/parity/deltas.py` declares how the repaired ones relate to them.
//!
//! # Why there is not a single `mul_add` here
//!
//! `objdump -d hazma/spectra/_neutrino/_pion.cpython-312-darwin.so | grep
//! -c 'fmadd\|fmsub\|fnmadd\|fnmsub'` prints `0` for the whole object, and
//! unlike [`super::photon_rho`] the reason is not untyped locals — every
//! local here is a `cdef double`. It is that the file contains no
//! multiply-add to contract: `1 − β²`, `γE(1∓β)`, `0.5/(γβ)` and the two
//! `two_body_energy` folds are all plain `fmul`/`fdiv`/`fsub` chains, and
//! [`crate::boost`] already documents `1 − (m/E)²` as unfused at every one
//! of its call sites. The FMAs the spectrum does carry live in
//! [`crate::boost::boost_delta_function`] and in
//! [`super::neutrino_muon`], both of which this module calls.
//!
//! # Compile-time constants
//!
//! `hazma/_utils/kinematics.pxd`'s `two_body_energy` is `cdef inline` on
//! `DEF` masses at all three call sites, and clang folds it rather than
//! calling anything: the shipped object materialises
//! `0x405b_71ce_d218_b450`, `0x4051_7231_4f00_a128` and
//! `0x403d_cac9_cbd9_25e7` as immediates.
//! [`tests::the_two_body_energies_match_the_shipped_immediates`] pins all
//! three.

use super::neutrino_flavors::NeutrinoSpectrumPoint;
use super::neutrino_muon;
use crate::boost;
use crate::constants::pdg::{BR_PI_TO_E_NUE, BR_PI_TO_MU_NUMU, MASS_E, MASS_MU, MASS_PI};
use crate::quad::{DEFAULT_EPSABS, DEFAULT_EPSREL, DEFAULT_LIMIT, QuadOpts, quad};

/// The muon's energy in the charged-pion rest frame, MeV.
///
/// `two_body_energy(m_π, m_μ, 0)`. The same physical quantity as
/// [`crate::constants::derived::positron_pion::ENG_MU_PI_RF`], and the same
/// double — both files spell it over the PDG masses.
pub const ENG_MU_PI_RF: f64 = (MASS_PI * MASS_PI + MASS_MU * MASS_MU) / (2.0 * MASS_PI);

/// The electron neutrino's energy in the `π → e ν_e` rest frame, MeV.
///
/// `two_body_energy(m_π, 0, m_e)`. The `.pyx` writes this expression twice
/// — once through the helper and once inline as
/// `(MASS_PI**2 - MASS_E**2) / (2.0 * MASS_PI)` — and the two are the same
/// double.
pub const ENU_E_PI_RF: f64 = (MASS_PI * MASS_PI - MASS_E * MASS_E) / (2.0 * MASS_PI);

/// The muon neutrino's energy in the `π → μ ν_μ` rest frame, MeV.
///
/// `two_body_energy(m_π, 0, m_μ)`.
pub const ENU_MU_PI_RF: f64 = (MASS_PI * MASS_PI - MASS_MU * MASS_MU) / (2.0 * MASS_PI);

/// `scipy.integrate.quad`'s arguments at `_pion.pyx:124` and `:127`.
///
/// Both call sites pass **no** tolerance keywords at all, so every field is
/// scipy's default. No `points` keyword either, so `None` selects
/// [`crate::quad::qagse`]. This is the only ported call site in the crate
/// that runs at the default `1.49e-8` rather than at a tightened
/// `epsabs`/`epsrel` of its own.
const PION_QUAD: QuadOpts<'static> = QuadOpts {
    epsabs: DEFAULT_EPSABS,
    epsrel: DEFAULT_EPSREL,
    limit: DEFAULT_LIMIT,
    points: None,
};

/// Which flavor row the boost integrand returns.
///
/// The `.pyx` passes this as `quad(..., args=(gen,))` with `gen == 1`
/// meaning electron and anything else meaning muon. Named here so the two
/// call sites cannot be transposed silently — a transposition swaps two
/// rows of the published `(3, N)` array and no tolerance would see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    /// The electron-neutrino row — the `.pyx`'s `gen == 1`.
    Electron,
    /// The muon-neutrino row — the `.pyx`'s `else` branch.
    Muon,
}

/// The boost integrand for `π → μ ν_μ` followed by muon decay, MeV⁻².
///
/// # Parameters
///
/// * `e1` — the neutrino energy in the pion rest frame, MeV.
/// * `flavor` — which row of the muon spectrum to integrate.
///
/// # Returns
///
/// `(dN/dE)_μ(e1, E_μ^rf) / e1` in MeV⁻², for the selected flavor. The
/// `1/e1` is the massless flat-boost kernel's, not the spectrum's, which
/// is why this is not itself a spectrum.
#[must_use]
pub fn mu_numu_integrand(e1: f64, flavor: Flavor) -> f64 {
    let point = neutrino_muon::dnde_neutrino_muon(e1, ENG_MU_PI_RF);
    match flavor {
        Flavor::Electron => point.electron / e1,
        Flavor::Muon => point.muon / e1,
    }
}

/// The `π → μ ν_μ` half of the spectrum, MeV⁻¹.
///
/// # Parameters
///
/// * `enu` — the neutrino energy, MeV.
/// * `epi` — the charged pion's total energy, MeV.
///
/// # Returns
///
/// The three flavors' `dN/dE` in MeV⁻¹, the tau row always zero. Exactly
/// zero everywhere for a pion below its own rest mass.
///
/// Three branches, in the `.pyx`'s order:
///
/// 1. `E_π < m_π` → all zeros.
/// 2. `E_π − m_π < DBL_EPSILON` → the muon-decay continuum in the pion
///    rest frame, weighted by `BR(π → μ ν_μ)`, and **without either
///    prompt line** — a pion at rest emits both as `δ` functions, which
///    have no rest-frame representation here. A `NaN` `E_π` fails both
///    comparisons and falls through to the boosted branch, where every
///    quantity is `NaN`, exactly as in the Cython.
/// 3. Otherwise the prompt `π → μ ν_μ` line, boosted, plus the two
///    quadratures.
///
/// Branch 3 carries only the line this function is named for. The `.pyx`
/// added the `π → e ν_e` line here too, on top of the copy
/// [`dnde_e_nue`] contributes; see the module docs.
#[must_use]
// Not a disguised equality test: `epi >= MASS_PI` is already established
// above, so this is the one-sided "within one epsilon MeV of rest"
// threshold the Cython writes, and `.abs()` would change nothing.
#[allow(clippy::float_equality_without_abs)]
pub fn dnde_mu_numu(enu: f64, epi: f64) -> NeutrinoSpectrumPoint {
    if epi < MASS_PI {
        return NeutrinoSpectrumPoint::ZERO;
    }

    if epi - MASS_PI < f64::EPSILON {
        let point = neutrino_muon::dnde_neutrino_muon(enu, ENG_MU_PI_RF);
        return NeutrinoSpectrumPoint {
            electron: point.electron * BR_PI_TO_MU_NUMU,
            muon: point.muon * BR_PI_TO_MU_NUMU,
            tau: point.tau,
        };
    }

    let beta = boost::boost_beta(epi, MASS_PI);
    let (emin, emax, pre) = boost_window(enu, epi);

    let delta_m = BR_PI_TO_MU_NUMU * boost::boost_delta_function(ENU_MU_PI_RF, enu, 0.0, beta);

    let weight = pre * BR_PI_TO_MU_NUMU;
    NeutrinoSpectrumPoint {
        electron: weight * boost_integral(emin, emax, Flavor::Electron),
        muon: delta_m + weight * boost_integral(emin, emax, Flavor::Muon),
        tau: 0.0,
    }
}

/// The boost window `[max(0, γE(1−β)), γE(1+β)]` and the `1/(2βγ)`
/// prefactor.
///
/// Split out of [`dnde_mu_numu`] for the reason [`super::photon_rho`]'s
/// `boost_window` was: it is the module's arithmetic a caller can observe
/// *directly*, and a mutation campaign found it was otherwise
/// unobservable. The mutation in question is a tempting simplification —
/// writing `γ` as [`crate::boost::boost_gamma`] (`E/m`) instead of the
/// `1 / sqrt(1 − β²)` the `.pyx` spells — and it is **wrong**: the two
/// are different doubles, by 5.3e-16 relative (2.4 ulp) at
/// `E_π = 10 m_π` and by **2.9e-11** at `E_π = 10⁵` MeV, which is 29x the
/// corpus's own `PORTED_QUAD_RTOL` for this case. The corpus does not
/// sample past `10 m_π`, so it cannot see it. This function's outputs are pinned bit
/// for bit in
/// [`tests::the_boost_window_uses_the_pyx_s_own_spelling_of_gamma`]
/// instead.
///
/// # Parameters
///
/// * `enu` — the lab-frame neutrino energy, MeV.
/// * `epi` — the pion's total energy, MeV, already known to be `≥ m_π`
///   and outside the near-rest window.
///
/// # Returns
///
/// `(emin, emax, pre)` in MeV, MeV and dimensionless.
#[must_use]
fn boost_window(enu: f64, epi: f64) -> (f64, f64, f64) {
    let beta = boost::boost_beta(epi, MASS_PI);
    // `1.0 / sqrt(1.0 - beta ** 2)`, not `boost_gamma(epi, MASS_PI)` —
    // see the docs above.
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    (
        (enu * gamma * (1.0 - beta)).max(0.0),
        enu * gamma * (1.0 + beta),
        0.5 / (gamma * beta),
    )
}

/// One of the two `quad` calls, MeV⁻¹ before the `1/(2βγ)` prefactor.
///
/// Split out so both flavors go through one call site and one error
/// policy.
fn boost_integral(emin: f64, emax: f64, flavor: Flavor) -> f64 {
    let mut integrand = |e: f64| mu_numu_integrand(e, flavor);
    match quad(&mut integrand, emin, emax, &PION_QUAD) {
        Ok(outcome) => outcome.value,
        // Unreachable, and asserted so by
        // `pion_quad_options_are_always_accepted` below: `QuadError` is a
        // statement about the *options*, never about the integrand or the
        // interval, and these options are `const`. `NaN` rather than a
        // panic, for the reason `crate::boost` gives — `dispatch::map_flavors`
        // evaluates element by element and has no per-element error channel.
        Err(_) => f64::NAN,
    }
}

/// The `π → e ν_e` half of the spectrum, MeV⁻¹.
///
/// # Parameters
///
/// * `enu` — the neutrino energy, MeV.
/// * `epi` — the charged pion's total energy, MeV.
///
/// # Returns
///
/// The boosted `π → e ν_e` line in the electron row, and zeros elsewhere.
/// Exactly zero everywhere for a pion below its own rest mass.
///
/// Unlike [`dnde_mu_numu`] this has no near-rest short circuit: at
/// `E_π = m_π` exactly, `β` is `0` and
/// [`crate::boost::boost_delta_function`] returns zero for it, so the
/// branch is unnecessary rather than missing.
#[must_use]
pub fn dnde_e_nue(enu: f64, epi: f64) -> NeutrinoSpectrumPoint {
    if epi < MASS_PI {
        return NeutrinoSpectrumPoint::ZERO;
    }

    // The `.pyx` spells `beta` inline here rather than calling
    // `boost_beta`, and spells `enu_rf` inline rather than calling
    // `two_body_energy`; both are the same doubles as the named forms.
    let ratio = MASS_PI / epi;
    let beta = (1.0 - ratio * ratio).sqrt();

    NeutrinoSpectrumPoint {
        electron: BR_PI_TO_E_NUE * boost::boost_delta_function(ENU_E_PI_RF, enu, 0.0, beta),
        muon: 0.0,
        tau: 0.0,
    }
}

/// The neutrino spectra `dN/dE` in MeV⁻¹ from charged-pion decay.
///
/// # Parameters
///
/// * `enu` — the neutrino energy, MeV.
/// * `epi` — the charged pion's total energy, MeV.
///
/// # Returns
///
/// The three flavors' `dN/dE` in MeV⁻¹, the tau row always zero.
///
/// The two halves partition the pion's decay modes: [`dnde_mu_numu`]
/// carries the `π → μ ν_μ` line and the muon-decay continuum, and
/// [`dnde_e_nue`] carries the `π → e ν_e` line, so each prompt line
/// appears exactly once. hazma 2.1.0 shipped the electron one twice —
/// the module docs have the history, and `test/parity/deltas.py` declares
/// how the corpus's stored values relate to these.
#[must_use]
pub fn dnde_neutrino_charged_pion(enu: f64, epi: f64) -> NeutrinoSpectrumPoint {
    let mu_nu = dnde_mu_numu(enu, epi);
    let e_nu = dnde_e_nue(enu, epi);

    NeutrinoSpectrumPoint {
        electron: mu_nu.electron + e_nu.electron,
        muon: mu_nu.muon + e_nu.muon,
        tau: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ENG_MU_PI_RF, ENU_E_PI_RF, ENU_MU_PI_RF, Flavor, NeutrinoSpectrumPoint, PION_QUAD,
        boost_window, dnde_e_nue, dnde_mu_numu, dnde_neutrino_charged_pion, mu_numu_integrand,
    };
    use crate::constants::pdg::{BR_PI_TO_E_NUE, BR_PI_TO_MU_NUMU, MASS_MU, MASS_PI};
    use crate::quad::quad;

    /// The three folded immediates read out of the shipped
    /// `_pion.cpython-312-darwin.so`.
    ///
    /// clang folds `two_body_energy` rather than calling it, so these are
    /// the numbers the corpus was captured with. Pinning the bits (not a
    /// tolerance) is the point: an integration separates two of them from
    /// any observable, so a reassociated fold would land elsewhere in the
    /// last ulp and nothing downstream would notice.
    #[test]
    fn the_two_body_energies_match_the_shipped_immediates() {
        assert_eq!(ENG_MU_PI_RF.to_bits(), 0x405b_71ce_d218_b450);
        assert_eq!(ENU_E_PI_RF.to_bits(), 0x4051_7231_4f00_a128);
        assert_eq!(ENU_MU_PI_RF.to_bits(), 0x403d_cac9_cbd9_25e7);
    }

    /// The two-body kinematics balance: each daughter pair sums to the
    /// pion mass.
    ///
    /// True as algebra, and here also exactly in floating point because
    /// the two numerators are the same `2 m_π²` split by one subtraction.
    #[test]
    fn the_daughter_energies_sum_to_the_pion_mass() {
        const { assert!(ENG_MU_PI_RF + ENU_MU_PI_RF == MASS_PI) };
        // The muon takes almost all of it; its neutrino gets 21%.
        const { assert!(ENU_MU_PI_RF < 0.22 * MASS_PI) };
        // The electron channel's neutrino takes almost half instead,
        // which is why the two lines sit a factor of 2.3 apart in energy.
        const { assert!(ENU_E_PI_RF > 0.49 * MASS_PI) };
    }

    /// The integrand selects a row, and the two rows are different
    /// functions.
    ///
    /// A transposed `Flavor` at either `quad` call site swaps two rows of
    /// the published array; this is the assertion that makes the two
    /// distinguishable at all.
    #[test]
    fn the_integrand_rows_are_distinguishable() {
        let e = 20.0;
        let electron = mu_numu_integrand(e, Flavor::Electron);
        let muon = mu_numu_integrand(e, Flavor::Muon);
        assert!(electron > 0.0 && muon > 0.0);
        assert!(electron != muon);
        // And each is the corresponding row of the muon spectrum over `e`.
        let point = super::neutrino_muon::dnde_neutrino_muon(e, ENG_MU_PI_RF);
        assert_eq!(electron.to_bits(), (point.electron / e).to_bits());
        assert_eq!(muon.to_bits(), (point.muon / e).to_bits());
    }

    /// Both halves vanish below the pion's rest mass.
    #[test]
    fn a_pion_below_threshold_gives_three_zeros() {
        let epi = MASS_PI * 0.999_999;
        assert_eq!(dnde_mu_numu(20.0, epi), NeutrinoSpectrumPoint::ZERO);
        assert_eq!(dnde_e_nue(20.0, epi), NeutrinoSpectrumPoint::ZERO);
        assert_eq!(
            dnde_neutrino_charged_pion(20.0, epi),
            NeutrinoSpectrumPoint::ZERO
        );
    }

    /// At rest the muon channel is the rest-frame muon spectrum weighted
    /// by its branching fraction, and the two prompt lines are absent.
    #[test]
    fn a_pion_at_rest_is_the_weighted_muon_spectrum() {
        for enu in [1.0, 10.0, 25.0] {
            let got = dnde_mu_numu(enu, MASS_PI);
            let want = super::neutrino_muon::dnde_neutrino_muon(enu, ENG_MU_PI_RF);
            assert_eq!(
                got.electron.to_bits(),
                (want.electron * BR_PI_TO_MU_NUMU).to_bits()
            );
            assert_eq!(got.muon.to_bits(), (want.muon * BR_PI_TO_MU_NUMU).to_bits());
        }
        // And `dnde_e_nue` contributes nothing at rest, because beta is 0
        // and the boosted delta function refuses a non-positive beta.
        assert_eq!(
            dnde_e_nue(ENU_E_PI_RF, MASS_PI),
            NeutrinoSpectrumPoint::ZERO
        );
    }

    /// The `π → e ν_e` line comes from [`dnde_e_nue`] and nowhere else.
    ///
    /// The half that partitions the modes, asserted rather than described:
    /// at a lab energy inside the electron line's boosted window and
    /// outside the muon line's, `dnde_e_nue` contributes exactly one line
    /// and `dnde_mu_numu` contributes none. The first half of that is a
    /// bit-equal identity. The second cannot be, because the continuum
    /// `dnde_mu_numu` does return there is ~5,000x the line and would
    /// swamp it in any comparison of totals — so it is asserted at the
    /// line's window edge instead, where a hidden copy is a step and the
    /// continuum is smooth.
    #[test]
    fn the_electron_line_comes_from_one_half_only() {
        let epi = 400.0;
        // A lab energy inside the electron line's window: the line is at
        // `ENU_E_PI_RF` in the rest frame, so `gamma * ENU_E_PI_RF` is
        // comfortably inside `[gamma E(1-beta), gamma E(1+beta)]`.
        let enu = ENU_E_PI_RF * epi / MASS_PI;
        let from_mu_half = dnde_mu_numu(enu, epi);
        let from_e_half = dnde_e_nue(enu, epi);
        assert!(from_e_half.electron > 0.0, "the line must be present");
        assert_eq!(
            dnde_neutrino_charged_pion(enu, epi).electron.to_bits(),
            (from_mu_half.electron + from_e_half.electron).to_bits()
        );
        // One copy, from the half named for the channel.
        let beta = crate::boost::boost_beta(epi, MASS_PI);
        let one_line =
            BR_PI_TO_E_NUE * crate::boost::boost_delta_function(ENU_E_PI_RF, enu, 0.0, beta);
        assert_eq!(from_e_half.electron.to_bits(), one_line.to_bits());

        // And the other half carries no line at all, which the sum above
        // cannot see. A boosted line is a plateau that ends where its
        // window stops straddling `ENU_E_PI_RF`, so a copy hiding in
        // `dnde_mu_numu` would show up as a step of `one_line` across that
        // edge; the muon-decay continuum crosses it smoothly.
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let edge = ENU_E_PI_RF / (gamma * (1.0 - beta));
        let (below, above) = (edge * (1.0 - 1e-9), edge * (1.0 + 1e-9));
        assert!(
            dnde_e_nue(below, epi).electron > 0.0 && dnde_e_nue(above, epi).electron == 0.0,
            "the bracket must straddle the line's upper window edge"
        );
        let step = (dnde_mu_numu(above, epi).electron - dnde_mu_numu(below, epi).electron).abs();
        assert!(
            step < 1e-6 * one_line,
            "the mu nu_mu half steps by {step} across the electron line's \
             window edge, so it is carrying a copy of the line ({one_line})"
        );
    }

    /// The muon-neutrino line appears once, and only in the muon row.
    ///
    /// The counterpart to the test above: `dnde_e_nue` writes nothing to
    /// the muon row, so `delta_m` has no second copy.
    #[test]
    fn the_muon_line_is_counted_once() {
        let epi = 400.0;
        let enu = ENU_MU_PI_RF * epi / MASS_PI;
        assert_eq!(dnde_e_nue(enu, epi).muon.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            dnde_neutrino_charged_pion(enu, epi).muon.to_bits(),
            dnde_mu_numu(enu, epi).muon.to_bits()
        );
    }

    /// No kernel here ever writes a tau neutrino.
    #[test]
    fn the_tau_row_is_always_a_positive_zero() {
        for (enu, epi) in [
            (20.0, MASS_PI),
            (20.0, 400.0),
            (1e-4, 1e4),
            (1e9, 400.0),
            (20.0, 1.0),
        ] {
            assert_eq!(
                dnde_neutrino_charged_pion(enu, epi).tau.to_bits(),
                0.0_f64.to_bits(),
                "tau row non-zero at enu = {enu}, epi = {epi}"
            );
        }
    }

    /// The boost window is built from the `.pyx`'s own `γ`, and the
    /// obvious simplification is a different number.
    ///
    /// `_pion.pyx:108` writes `gamma = 1.0 / sqrt(1.0 - beta ** 2)` rather
    /// than reusing `boost_gamma(epi, MASS_PI) = E/m`. Algebraically the
    /// same; in floating point not: `β` has already been rounded by its
    /// own `sqrt`, so squaring it back and inverting loses bits the
    /// division never had. The gap grows with the boost — 3.1e-16 at
    /// `E_π = 200` MeV, 5.3e-16 at `10 m_π`, 2.5e-14 at `10⁴` MeV — and by
    /// `E_π = 10⁵` MeV it is 2.9e-11 relative, **29x the corpus's
    /// `PORTED_QUAD_RTOL` for this case**, which is why swapping the two
    /// is a real error and not a cleanup.
    ///
    /// It is also invisible to every other gate in the tree: the corpus
    /// stops at `10 m_π`, where the gap is two ulp and the resulting
    /// spectrum shift stays inside 1e-14, so a mutation campaign duly
    /// survived it. This is the seam that kills it.
    #[test]
    fn the_boost_window_uses_the_pyx_s_own_spelling_of_gamma() {
        // The two spellings agree near threshold and separate with the
        // boost. Asserting both halves is what makes this a statement
        // about the mechanism rather than about one lucky point.
        for epi in [145.0_f64, 200.0, 1_395.7, 1e5] {
            let beta = crate::boost::boost_beta(epi, MASS_PI);
            let from_beta = 1.0 / (1.0 - beta * beta).sqrt();
            let from_energy = crate::boost::boost_gamma(epi, MASS_PI);
            // `pre` is `0.5 / (gamma beta)`, so it carries gamma exactly.
            let (_, _, pre) = boost_window(1.0, epi);
            assert_eq!(pre.to_bits(), (0.5 / (from_beta * beta)).to_bits());
            if epi >= 200.0 {
                assert_ne!(
                    from_beta.to_bits(),
                    from_energy.to_bits(),
                    "the two spellings of gamma coincide at epi = {epi}"
                );
            }
        }
        // The size of the divergence where the corpus cannot see it.
        let epi = 1e5;
        let beta = crate::boost::boost_beta(epi, MASS_PI);
        let from_beta = 1.0 / (1.0 - beta * beta).sqrt();
        let from_energy = crate::boost::boost_gamma(epi, MASS_PI);
        assert!((from_beta - from_energy).abs() / from_energy > 1e-11);
    }

    /// The window's endpoints, bit for bit, and the `max(0, ...)` floor.
    #[test]
    fn the_boost_window_endpoints_are_the_pyx_s_arithmetic() {
        let (enu, epi) = (20.0, 400.0);
        let beta = crate::boost::boost_beta(epi, MASS_PI);
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let (emin, emax, pre) = boost_window(enu, epi);
        assert_eq!(emin.to_bits(), (enu * gamma * (1.0 - beta)).to_bits());
        assert_eq!(emax.to_bits(), (enu * gamma * (1.0 + beta)).to_bits());
        assert_eq!(pre.to_bits(), (0.5 / (gamma * beta)).to_bits());
        assert!(emin < emax);
        // The floor: a negative neutrino energy would otherwise give a
        // negative lower limit and an inverted interval.
        let (floored, upper, _) = boost_window(-1.0, epi);
        assert_eq!(floored.to_bits(), 0.0_f64.to_bits());
        assert!(upper < 0.0);
    }

    /// The quadrature options are accepted for every interval, so the
    /// `Err` arm in `boost_integral` is unreachable.
    #[test]
    fn pion_quad_options_are_always_accepted() {
        let mut integrand = |e: f64| mu_numu_integrand(e, Flavor::Muon);
        for (lo, hi) in [
            (0.0, ENG_MU_PI_RF),
            (0.0, 0.0),
            (ENG_MU_PI_RF, 0.0),
            (1.0, 1.0 + f64::EPSILON),
        ] {
            assert!(
                quad(&mut integrand, lo, hi, &PION_QUAD).is_ok(),
                "quad refused the options on [{lo}, {hi}]"
            );
        }
    }

    /// The boost conserves neutrino number, per flavor, and each decay
    /// mode contributes its own branching ratio exactly once.
    ///
    /// The statement about this kernel that owes nothing to the Cython.
    /// Per charged pion the yields are:
    ///
    /// * **muon neutrinos** — the prompt `π → μ ν_μ` line (`BR_μ`) plus
    ///   the muon's own `ν_μ` (`BR_μ`, since
    ///   [`super::neutrino_muon`]'s rows each integrate to exactly one);
    /// * **electron neutrinos** — the muon's `ν̄_e` (`BR_μ`) plus the
    ///   prompt `π → e ν_e` line (`BR_e`). The shipped 2.1.0 kernel
    ///   counted that line twice and this row summed to `BR_μ + 2 BR_e`.
    ///
    /// Trapezoid on 4_001 points from just above zero to past the highest
    /// endpoint — the grid starts at one step rather than at zero because
    /// the integrand is `0/0` there, as it is in the Cython. Each point
    /// costs two adaptive quadratures, so the grid is short and the budget
    /// is 3e-3 relative: the spectrum has step discontinuities where each
    /// line's window opens and closes, and a composite rule resolves those
    /// at `O(h)`. That still separates the expected total from any
    /// factor-of-two error by two decades — but it is 24x `BR_e` itself,
    /// so what this test pins is the continuum's normalization, not how
    /// many copies of the electron line the row carries. That is
    /// [`tests::the_electron_line_comes_from_one_half_only`]'s job.
    #[test]
    fn the_boost_conserves_neutrino_number_per_flavor() {
        let epi = 400.0;
        let beta = crate::boost::boost_beta(epi, MASS_PI);
        let gamma = epi / MASS_PI;
        // Above the highest endpoint any term can reach.
        let hi = gamma * (1.0 + beta) * ENU_E_PI_RF * 1.01;
        let n = 4_001_usize;
        let h = hi / n as f64;
        let mut totals = [0.0_f64; 2];
        for index in 1..=n {
            let weight = if index == n { 0.5 } else { 1.0 };
            let point = dnde_neutrino_charged_pion(h * index as f64, epi);
            totals[0] += weight * point.electron;
            totals[1] += weight * point.muon;
        }
        // The electron row: the muon's nu_e_bar, plus the prompt line.
        let expected_electron = BR_PI_TO_MU_NUMU + BR_PI_TO_E_NUE;
        // The muon row: the muon's nu_mu, plus the prompt line once.
        let expected_muon = BR_PI_TO_MU_NUMU + BR_PI_TO_MU_NUMU;
        for (flavor, (total, expected)) in ["electron", "muon"]
            .iter()
            .zip(totals.iter().zip([expected_electron, expected_muon]))
        {
            let integral = total * h;
            assert!(
                (integral - expected).abs() < 3e-3 * expected,
                "the {flavor} row integrates to {integral}, not {expected}"
            );
        }
    }

    /// A neutrino of exactly zero energy gives three exact zeros, and the
    /// reason is [`crate::quad`]'s empty-interval short circuit.
    ///
    /// The boost integrand is `(dN/dE)_μ(E)/E`, which is `0/0` at the
    /// origin, and at `E_ν = 0` the whole integration window collapses
    /// onto it: `emin = emax = 0`. `scipy.integrate.quad` returns
    /// `(0., 0.)` for `a == b` **without evaluating the integrand**, so
    /// the Cython never meets that `NaN` — measured on the live twin at
    /// cython-to-rust Task 4.6, which is what
    /// `dnde_neutrino_charged_pion(0.0, 400.0)` returning `(0, 0, 0)`
    /// there records. The first version of this port handed `[0, 0]` to
    /// [`crate::quad::qagse`] instead and answered `NaN`; the short
    /// circuit in [`crate::quad::quad`] is what closes that gap, and this
    /// is its live call site.
    ///
    /// No corpus grid samples `E_ν = 0` — they start at `1e-5 m_π` — so
    /// this fixes a value nothing pins, which is exactly why it needs its
    /// own test.
    #[test]
    fn a_zero_energy_neutrino_gives_exact_zeros() {
        let point = dnde_neutrino_charged_pion(0.0, 400.0);
        for value in point.to_array() {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }
        // One step up it is finite, positive and negligible.
        let next = dnde_neutrino_charged_pion(1e-12, 400.0);
        assert!(next.electron > 0.0 && next.electron < 1e-25);
        // And the integrand really is `NaN` at the origin, so the zero
        // above is the short circuit rather than a well-behaved integral.
        assert!(mu_numu_integrand(0.0, Flavor::Electron).is_nan());
    }

    /// Well above the highest endpoint every row is exactly zero.
    #[test]
    fn the_spectrum_closes_above_its_endpoint() {
        let epi = 400.0;
        let beta = crate::boost::boost_beta(epi, MASS_PI);
        let endpoint = (epi / MASS_PI) * (1.0 + beta) * ENU_E_PI_RF;
        let point = dnde_neutrino_charged_pion(endpoint * 1.001, epi);
        assert_eq!(point, NeutrinoSpectrumPoint::ZERO);
        assert!(dnde_neutrino_charged_pion(endpoint * 0.5, epi).electron > 0.0);
    }

    /// The muon spectrum this kernel boosts is evaluated at the pion
    /// rest-frame muon energy, not at the muon mass.
    ///
    /// A one-character slip that a tolerance would absorb nowhere: the
    /// muon is boosted in the pion frame (`gamma = 1.039`), so the two
    /// choices differ by 4% in the endpoint. Pinned against the muon
    /// spectrum's own support.
    #[test]
    fn the_inner_muon_is_boosted_not_at_rest() {
        const { assert!(ENG_MU_PI_RF > MASS_MU) };
        let boosted_endpoint = mu_numu_integrand(ENG_MU_PI_RF * 0.49, Flavor::Muon);
        let at_rest_endpoint = super::neutrino_muon::dnde_neutrino_muon(
            ENG_MU_PI_RF * 0.49,
            crate::constants::pdg::MASS_MU,
        );
        assert!(boosted_endpoint > 0.0);
        assert_eq!(at_rest_endpoint.muon, 0.0);
    }
}
