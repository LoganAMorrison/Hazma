//! The positron spectrum from charged-pion decay, ported from
//! `hazma/spectra/_positron/_pion.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::positron`] is the Python-visible half.
//!
//! # The physics
//!
//! A charged pion decays to `μ ν_μ` (BR 0.9998770) and to `e ν_e`
//! (BR 1.230e-4), and both channels put a positron in the final state:
//!
//! * the **muon channel** contributes a continuum — the Michel spectrum
//!   of a muon carrying [`ENG_MU_PI_RF`] in the pion rest frame, boosted
//!   into the lab with the massive-daughter flat-boost integral
//!
//!   ```text
//!   dN/dE = 1/(2 β γ) ∫_{E∓} dE'  (dN/dE')_μ(E', E_μ^rf) / √(E'² − m_e²)
//!   ```
//!
//!   between `E∓ = γ(E ∓ β√(E² − m_e²))`, clipped below at `m_e` and
//!   above at [`EMAX_PI_RF`];
//! * the **electron channel** contributes a line at [`ENG_E_PI_RF`],
//!   boosted by [`crate::boost::boost_delta_function`].
//!
//! The integrand is [`super::positron_muon::dnde_positron_muon`], which
//! this module calls natively — the `.pyx` `cimport`s the same `cdef`
//! from `_positron/_muon.pyx`. So the port inherits Task 4.1's kernel and
//! its declared normalization defect
//! (`docs/followups/todo/positron-muon-spectrum-normalization-inverted.md`)
//! along with it.
//!
//! # Where the FMAs are
//!
//! `objdump -d hazma/spectra/_positron/_pion.cpython-312-darwin.so | grep
//! -c 'fmadd\|fmsub\|fnmadd\|fnmsub'` prints **2** for the whole object,
//! both inside `dnde_positron_charged_pion_point` and both the same
//! expression with opposite signs: `fmsub d0, d9, d12, d8` and `fmadd d0,
//! d9, d12, d8`, i.e. `E ∓ β·k` for the two boost limits. They are written
//! [`f64::mul_add`] here.
//!
//! Three expressions that look fusable and are **not**, each read off the
//! same disassembly rather than guessed:
//!
//! * `e**2 - me**2` — `fmul` then `fadd` against the folded `−m_e²`, in
//!   both the point kernel (`fmul d4, d8, d8`; `fadd d3, d4, d5`) and the
//!   integrand (`fmul d0, d8, d8`; `fadd d0, d0, d1`);
//! * `1 - (mpi/epi)**2` inside `boost_beta`, which [`crate::boost`] already
//!   documents as unfused at all ten of its inlining sites — this file is
//!   one of the three it names;
//! * `gamma * (…)` and `2 * beta * gamma`, which are plain `fmul`/`fdiv`
//!   chains.
//!
//! # Compile-time constants
//!
//! `eng_mu_pi_rf`, `eng_e_pi_rf` and `gamma_mu` are Cython `DEF`s, so they
//! are folded literals in the generated C and live in
//! [`crate::constants::derived::positron_pion`] with the rest of the file's
//! `DEF`s. `beta_mu`, `emax_mu_rf` and `emax_pi_rf` are module-level `cdef
//! double`s instead — computed once at module init, which clang folds
//! entirely: `__pyx_pymod_exec__pion` stores a single immediate,
//! `0x4051_724f_f60e_5ca3`, and the other two never materialise.
//!
//! So [`EMAX_PI_RF`] is the only one this module needs, and it is a
//! literal because it cannot be spelled as a `const` expression — not only
//! because `sqrt` is not `const`, but because clang folds
//! `1.0 + beta_mu * root` **with contraction**, landing one ulp above what
//! the unfused expression gives.
//! [`tests::the_endpoint_constant_matches_the_shipped_object_code`]
//! re-derives it with [`f64::mul_add`] from `beta_mu` and `emax_mu_rf` —
//! which live there, since nothing outside that derivation reads them —
//! and shows the unfused spelling missing.

use crate::boost;
use crate::constants::derived::positron_pion::{ENG_E_PI_RF, ENG_MU_PI_RF, ME, MPI};
use crate::constants::pdg::{BR_PI_TO_E_NUE, BR_PI_TO_MU_NUMU};
use crate::kernels::positron_muon;
use crate::quad::{DEFAULT_LIMIT, QuadOpts, quad};

/// The maximum positron energy in the pion rest frame, MeV, and the upper
/// clip on the boost integral.
///
/// `γ_μ · E_max^μrf · (1 + β_μ √(1 − (m_e/E_max^μrf)²))` — the muon
/// endpoint boosted out of the muon's own frame. See the module docs for
/// why this is a literal and not an expression: the shipped object folds
/// the `1 + β·√…` with an FMA and lands on `0x4051_724f_f60e_5ca3`, one ulp
/// above the unfused value.
const EMAX_PI_RF: f64 = 69.786_130_441_689_74;

/// `scipy.integrate.quad`'s arguments at `_pion.pyx:59`.
///
/// `epsabs`/`epsrel` verbatim from the call site; `limit` is scipy's
/// default, which the site reaches by passing no keyword. No `points`
/// keyword, so `None` selects [`crate::quad::qagse`].
const PION_QUAD: QuadOpts<'static> = QuadOpts {
    epsabs: 1e-10,
    epsrel: 1e-4,
    limit: DEFAULT_LIMIT,
    points: None,
};

/// The boost integrand, MeV⁻² — the muon-channel positron spectrum over
/// the positron's rest-frame momentum.
///
/// # Parameters
///
/// * `e` — the positron energy in the pion rest frame, MeV.
///
/// # Returns
///
/// `(dN/dE)_μ(e, E_μ^rf) / √(e² − m_e²)` in MeV⁻². Below `m_e` the square
/// root is `NaN`; the caller never gets there because the lower limit is
/// clipped at `m_e`, where the numerator is exactly zero and the quotient
/// is `0/0` — see [`dnde_positron_charged_pion`].
#[must_use]
pub fn charged_pion_integrand(e: f64) -> f64 {
    // `fmul` then `fadd` against the folded `−m_e²` — not an `fmsub`.
    positron_muon::dnde_positron_muon(e, ENG_MU_PI_RF) / (e * e - ME * ME).sqrt()
}

/// The positron spectrum `dN/dE` in MeV⁻¹ from charged-pion decay.
///
/// # Parameters
///
/// * `e` — the positron (or electron) energy, MeV.
/// * `epi` — the charged pion's total energy, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` in three cases the `.pyx` guards:
/// a pion below its own rest mass, a positron below its rest mass, and —
/// unlike every other boosted kernel in this crate — a pion **within one
/// `DBL_EPSILON` MeV of rest**, where the others return a rest-frame
/// value. That is the shipped behavior and rule 1 keeps it: at `E_π = m_π`
/// the whole spectrum vanishes rather than collapsing onto the pion
/// rest-frame Michel spectrum, so the corpus's `rest` block for this case
/// is a block of zeros.
///
/// A `NaN` `epi` fails all three comparisons and falls through to the
/// quadrature over `NaN` limits, which is what the Cython does too.
#[must_use]
pub fn dnde_positron_charged_pion(e: f64, epi: f64) -> f64 {
    if epi < MPI || e < ME {
        return 0.0;
    }

    // The `.pyx`'s guard, and note it is two-sided (`fabs`) where the
    // photon and rho kernels write a one-sided `E − m < DBL_EPSILON`. The
    // first comparison above has already excluded `epi < MPI`, so the two
    // spellings select the same pions; it is transcribed as written.
    if (epi - MPI).abs() < f64::EPSILON {
        return 0.0;
    }

    let gamma = boost::boost_gamma(epi, MPI);
    let beta = boost::boost_beta(epi, MPI);

    // `fmul d4, d8, d8` then `fadd d3, d4, d5`: the square is complete
    // before the subtraction.
    let k = (e * e - ME * ME).sqrt();
    // `fmsub d0, d9, d12, d8` and `fmadd d0, d9, d12, d8`: E ∓ β·k, each
    // fused. Rust has no separate spelling for `fmsub`; the product is
    // exact either way.
    let emin = (gamma * (-beta).mul_add(k, e)).max(ME);
    let emax = (gamma * beta.mul_add(k, e)).min(EMAX_PI_RF);

    let mut integrand = charged_pion_integrand;
    let integral = match quad(&mut integrand, emin, emax, &PION_QUAD) {
        Ok(outcome) => outcome.value,
        // Unreachable, and asserted so by
        // `pion_quad_options_are_always_accepted` below: `QuadError` is a
        // statement about the *options*, never about the integrand or the
        // interval, and these options are `const`. `NaN` rather than a
        // panic, for the reason `crate::boost` gives — `dispatch::map_unary`
        // evaluates element by element and has no per-element error channel.
        Err(_) => f64::NAN,
    };

    let dnde_mu = (BR_PI_TO_MU_NUMU * integral) / ((2.0 * beta) * gamma);
    let dnde_e = BR_PI_TO_E_NUE * boost::boost_delta_function(ENG_E_PI_RF, e, ME, beta);

    dnde_mu + dnde_e
}

#[cfg(test)]
mod tests {
    use super::{EMAX_PI_RF, PION_QUAD, charged_pion_integrand, dnde_positron_charged_pion};
    use crate::constants::derived::positron_pion::{
        ENG_E_PI_RF, ENG_MU_PI_RF, GAMMA_MU, ME, MMU, MPI,
    };
    use crate::constants::pdg::{BR_PI_TO_E_NUE, BR_PI_TO_MU_NUMU, MASS_MU};
    use crate::quad::quad;

    /// The muon's velocity in the pion rest frame, dimensionless —
    /// `_pion.pyx`'s `beta_mu`.
    const BETA_MU: f64 = 0.271_384_742_599_515_5;

    /// The maximum positron energy in the muon rest frame, MeV —
    /// `_pion.pyx`'s `emax_mu_rf`. The two-body endpoint the Michel
    /// spectrum's `x = 1 + r²` edge encodes, in energy units.
    const EMAX_MU_RF: f64 = (ME * ME + MMU * MMU) / (2.0 * MMU);

    /// [`EMAX_PI_RF`] against the immediate the shipped
    /// `_pion.cpython-312-darwin.so` stores for it, and against its own
    /// derivation.
    ///
    /// It is the only module-level `cdef double` the `.pyx` keeps in
    /// memory — `str x9, [x8]` in `__pyx_pymod_exec__pion` with
    /// `x9 = 0x4051_724f_f60e_5ca3`, reloaded as `ldr d10, [x8]` at the
    /// `fmin` site. `beta_mu` and `emax_mu_rf` fold into it and never
    /// materialise, which is why they live in this module rather than
    /// beside the constant they build.
    #[test]
    fn the_endpoint_constant_matches_the_shipped_object_code() {
        let ratio = MASS_MU / ENG_MU_PI_RF;
        assert_eq!(BETA_MU.to_bits(), (1.0 - ratio * ratio).sqrt().to_bits());
        assert_eq!(EMAX_MU_RF.to_bits(), 0x404a_6a4b_4c6f_8801);

        // The contraction the module docs describe: clang folds
        // `1.0 + beta_mu * root` as an FMA, and the two spellings are
        // adjacent doubles. Reproducing the *unfused* one here would put
        // `emax_pi_rf` one ulp below what the corpus was captured with.
        let u = ME / EMAX_MU_RF;
        let root = (1.0 - u * u).sqrt();
        let fused = GAMMA_MU * EMAX_MU_RF * BETA_MU.mul_add(root, 1.0);
        let unfused = GAMMA_MU * EMAX_MU_RF * (1.0 + BETA_MU * root);
        assert_eq!(EMAX_PI_RF.to_bits(), 0x4051_724f_f60e_5ca3);
        assert_eq!(EMAX_PI_RF.to_bits(), fused.to_bits());
        assert_eq!(unfused.to_bits(), EMAX_PI_RF.to_bits() - 1);
    }

    /// The muon channel's endpoint and the two-body electron line are the
    /// *same* physical energy, reached by two different routes.
    ///
    /// A statement about the kinematics rather than about the
    /// transcription, and a useful one: the most energetic positron from
    /// `π → μ ν` followed by `μ → e ν ν` is the one emitted forward at
    /// every step, and it carries exactly the energy the two-body
    /// `π → e ν` gives, `(m_π² + m_e²)/(2 m_π)`. So the continuum runs all
    /// the way up to the line rather than stopping short of it.
    ///
    /// The two spellings land on **adjacent doubles**, and the port keeps
    /// them distinct: [`EMAX_PI_RF`] is the boosted chain
    /// ([`GAMMA_MU`]·…, folded by clang with an FMA) and [`ENG_E_PI_RF`]
    /// the closed form, one ulp below. Collapsing them would be a
    /// defensible simplification and a corpus failure, so the difference
    /// is asserted rather than left to chance.
    #[test]
    fn the_muon_endpoint_and_the_electron_line_are_the_same_energy() {
        const { assert!(EMAX_PI_RF.to_bits() == ENG_E_PI_RF.to_bits() + 1) };
        // The two-body electron energy sits *above* half the pion mass by
        // exactly `m_e²/(2 m_π)`: the massless neutrino takes the smaller
        // share. Getting this backwards is the easy error, so it is
        // pinned on the closed form rather than on an inequality alone.
        const EXCESS: f64 = ME * ME / (2.0 * MPI);
        const { assert!(ENG_E_PI_RF - MPI / 2.0 > 0.999 * EXCESS) };
        const { assert!(ENG_E_PI_RF - MPI / 2.0 < 1.001 * EXCESS) };
        // And the muon rest-frame endpoint it is boosted from is well
        // below: the boost out of the pion frame is what closes the gap.
        const { assert!(EMAX_MU_RF < 0.76 * EMAX_PI_RF) };
        const { assert!(GAMMA_MU > 1.0) };
    }

    /// All three thresholds, at and either side of each.
    ///
    /// The middle one is this kernel's oddity: a pion *at* rest returns
    /// zero rather than a rest-frame spectrum, so there is no `E_π = m_π`
    /// value to compare against. See the function's own docs.
    #[test]
    fn the_spectrum_vanishes_at_every_guard() {
        assert_eq!(dnde_positron_charged_pion(10.0, MPI * 0.999_999), 0.0);
        assert_eq!(dnde_positron_charged_pion(ME * 0.5, 500.0), 0.0);
        assert_eq!(dnde_positron_charged_pion(10.0, MPI), 0.0);
        assert_eq!(
            dnde_positron_charged_pion(10.0, MPI + 0.5 * f64::EPSILON),
            0.0
        );
        assert!(dnde_positron_charged_pion(10.0, 500.0) > 0.0);
    }

    /// The near-rest guard admits exactly one pion energy: `m_π` itself.
    ///
    /// The `.pyx` writes it two-sided (`fabs(epi - mpi) < DBL_EPSILON`),
    /// but the preceding `epi < mpi` has already cut the lower half off,
    /// and `DBL_EPSILON` is an *absolute* threshold in MeV against a
    /// 139.57 MeV mass whose ulp is 2.8e-14 — 128 times larger. So the
    /// only double the guard can see is `m_π`, and the `abs()` is
    /// inoperative. Recorded here because it is the sort of thing a reader
    /// assumes is a tolerance band and is not.
    ///
    /// The second assertion is what makes that observable: the very next
    /// representable pion energy falls through to the quadrature and
    /// returns a finite, positive spectrum, so the branch boundary sits
    /// between two adjacent doubles.
    #[test]
    fn the_near_rest_guard_admits_exactly_the_pion_rest_mass() {
        let just_above = f64::from_bits(MPI.to_bits() + 1);
        assert!(just_above - MPI > 100.0 * f64::EPSILON);
        assert_eq!(dnde_positron_charged_pion(10.0, MPI), 0.0);
        assert!(dnde_positron_charged_pion(10.0, just_above) > 0.0);
    }

    /// The integrand is the muon spectrum divided by the positron's
    /// rest-frame momentum, and it vanishes outside the muon spectrum's
    /// own support.
    ///
    /// The upper edge is the assertion that matters: above [`EMAX_PI_RF`]
    /// the integrand is exactly zero, which is why clipping `emax` there
    /// changes no value and why a mutation that drops the clip is invisible
    /// on the corpus. Pinning it here states the fact the clip relies on.
    #[test]
    fn the_integrand_vanishes_above_the_pion_rest_frame_endpoint() {
        assert!(charged_pion_integrand(10.0) > 0.0);
        assert_eq!(charged_pion_integrand(EMAX_PI_RF * 1.000_001), 0.0);
        assert_eq!(charged_pion_integrand(ENG_MU_PI_RF), 0.0);
        // At `m_e` the numerator and the denominator vanish together.
        assert!(charged_pion_integrand(ME).is_nan());
    }

    /// The electron line rides on the muon continuum, and it is a sliver
    /// rather than the spectrum.
    ///
    /// The boosted line is flat across the lab energies whose own boost
    /// window straddles [`ENG_E_PI_RF`], and absent outside them. At
    /// `E_π = 500` MeV that window runs from about 9.9 MeV to about 491
    /// MeV, so 40 MeV is inside it and 5 MeV is below — the seam the sum
    /// offers, and the only way to see the two contributions separately.
    #[test]
    fn the_electron_line_sits_on_top_of_the_muon_continuum() {
        let epi = 500.0;
        let beta = crate::boost::boost_beta(epi, MPI);
        let inside = dnde_positron_charged_pion(40.0, epi);
        let below = dnde_positron_charged_pion(5.0, epi);
        assert!(inside > 0.0 && below > 0.0);

        // Inside, the pedestal's height is `BR_e / (2 γ β k₀)`, positive
        // and small against the continuum it sits on.
        let pedestal =
            BR_PI_TO_E_NUE * crate::boost::boost_delta_function(ENG_E_PI_RF, 40.0, ME, beta);
        assert!(pedestal > 0.0);
        assert!(pedestal < 1e-2 * inside, "the line must not dominate");

        // Below the window it is genuinely absent, so `below` is pure
        // continuum.
        assert_eq!(
            crate::boost::boost_delta_function(ENG_E_PI_RF, 5.0, ME, beta),
            0.0
        );
    }

    /// The branching ratios are the shipped ones and they do not sum to 1.
    ///
    /// `π → μ ν` and `π → e ν` exhaust the charged pion to 1e-7, and the
    /// deficit is the radiative modes the `.pyx` does not carry. Asserted
    /// so that a table edit that "tidied" either figure to make them close
    /// fails here rather than moving the spectrum by 1e-4.
    #[test]
    fn the_two_channels_carry_the_shipped_branching_fractions() {
        const SUM: f64 = BR_PI_TO_MU_NUMU + BR_PI_TO_E_NUE;
        const { assert!(SUM > 1.0 - 1e-6 && SUM < 1.0 + 1e-6) };
        const { assert!(BR_PI_TO_E_NUE < 1.3e-4) };
        const { assert!(BR_PI_TO_MU_NUMU > 0.999) };
    }

    /// The quadrature options are accepted for every interval, so the
    /// `Err` arm above is unreachable.
    ///
    /// [`crate::quad::QuadError`] depends only on the options — `epsabs > 0`
    /// or `epsrel` above the QUADPACK floor, and `limit` above the
    /// surviving break-point count — and `PION_QUAD` is a `const`. The
    /// intervals swept below include the degenerate one the `E = m_e`
    /// corpus anchor produces.
    #[test]
    fn pion_quad_options_are_always_accepted() {
        let mut integrand = charged_pion_integrand;
        for (lo, hi) in [
            (ME, EMAX_PI_RF),
            (ME, ME),
            (EMAX_PI_RF, ME),
            (1.0, 1.0 + f64::EPSILON),
        ] {
            assert!(
                quad(&mut integrand, lo, hi, &PION_QUAD).is_ok(),
                "quad refused the options on [{lo}, {hi}]"
            );
        }
    }

    /// The boost conserves positron number to the accuracy the integrator
    /// claims.
    ///
    /// The statement about this kernel that owes nothing to the Cython:
    /// the flat-boost integral is only right if `∫ dN/dE dE` is the same in
    /// the lab as in the pion rest frame. The rest-frame total is the
    /// muon channel's own norm (`BR_μ / N²`, carrying Task 4.1's inverted
    /// normalization) plus the electron line's `BR_e`.
    ///
    /// Trapezoid on 20_001 points from `m_e` to the endpoint. 2e-4
    /// relative: the boosted spectrum has kinks where the line's window
    /// opens and closes and where the continuum's endpoint lands, and a
    /// composite rule of this order resolves them no better. That still
    /// separates the shipped total from the un-defected one (3.7e-4 away)
    /// and from either channel alone.
    #[test]
    fn the_boost_conserves_positron_number() {
        const R_FACTOR: f64 = crate::constants::derived::positron_muon::R_FACTOR;
        let epi = 400.0;
        let (lo, hi) = (ME, epi);
        let n = 20_001_usize;
        let h = (hi - lo) / (n - 1) as f64;
        let mut total = 0.0;
        for index in 0..n {
            let weight = if index == 0 || index == n - 1 {
                0.5
            } else {
                1.0
            };
            total += weight * dnde_positron_charged_pion(lo + h * index as f64, epi);
        }
        let integral = total * h;
        let expected = BR_PI_TO_MU_NUMU / (R_FACTOR * R_FACTOR) + BR_PI_TO_E_NUE;
        assert!(
            (integral - expected).abs() < 2e-4 * expected,
            "boosted dN/dE integrates to {integral}, not the shipped {expected}"
        );
    }
}
