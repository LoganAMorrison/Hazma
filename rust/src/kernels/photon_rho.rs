//! The photon spectra from charged- and neutral-ρ decay, ported from
//! `hazma/spectra/_photon/_rho.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::photon`] is the Python-visible half.
//!
//! # The physics
//!
//! The ρ(770) decays to pions, and the photons come from the pions. Both
//! entry points are therefore the same shape: take the pion spectrum at
//! the pion energy the two-body decay fixes, and boost it out of the ρ
//! rest frame with the standard flat-boost integral
//!
//! ```text
//! dN/dE = 1/(2 β γ) ∫_{γE(1−β)}^{γE(1+β)} dE'  f(E') / E'
//! ```
//!
//! where `f` is the sum of the daughters' rest-frame spectra.
//!
//! * **Neutral ρ:** `ρ⁰ → π⁺ π⁻` (BR 0.9988), two charged pions each at
//!   `E_π = m_ρ/2`, so `f(E') = 2 · (dN/dE)_{π±}(E', m_ρ/2)`.
//! * **Charged ρ:** `ρ± → π± π⁰`, one charged pion at
//!   [`ENG_PI_CHARGED_RHO`] and one neutral pion at
//!   [`ENG_PI0_CHARGED_RHO`], so `f` is the sum of the two.
//!
//! Neither branching ratio appears: the `.pyx` weights both daughters by
//! 1, so a ρ is treated as decaying to pions with unit probability. That
//! is the shipped behavior and rule 1 (parity discipline) keeps it.
//!
//! # This is the project's nested quadrature
//!
//! `dnde_photon_charged_pion` is itself an adaptive `quad` over `cos θ`
//! (Task 4.4), whose integrand calls `dnde_photon_muon`, which integrates
//! the radiative muon spectrum in closed form. So a single ρ evaluation
//! runs an outer [`crate::quad::qagse`] whose integrand evaluation runs
//! an inner [`crate::quad::qagpe`]. `test/parity/tolerances.py` opens the
//! `NESTED` class at `rtol = 1e-6` for that reason — subdivision is a
//! discontinuous function of the integrand, so a last-ulp change inside
//! can move an outer bisection decision — but the fear was priced too
//! high: Task 4.5 measured this port against the Cython at **1.5e-13**
//! worst over the 1,395 values the corpus pins for the charged ρ and
//! **3.2e-15** for the neutral one, and tightened both cases to
//! `PORTED_NESTED_RTOL` (1e-9).
//!
//! # Why there is not a single `mul_add` here
//!
//! Unlike every other kernel in Phase 04, `_rho.pyx` contracts **nothing**
//! — `objdump -d hazma/spectra/_photon/_rho.cpython-312-darwin.so | grep
//! -c 'fmadd\|fmsub\|fnmadd\|fnmsub'` prints `0` for the whole object.
//!
//! That is not luck, it is the file's untyped locals. Both `*_point`
//! functions declare `cdef beta`, `gamma`, `emin`, `emax`, `pre` with no
//! type, so Cython boxes each into a `PyFloatObject` and evaluates
//! `gamma * e * (1 - beta)` through `PyNumber_Multiply` /
//! `__Pyx_PyFloat_SubtractCObj` — one correctly-rounded IEEE double
//! operation per call, with no expression for clang to contract. The
//! shipped object shows exactly that: `PyFloat_FromDouble` on the
//! `boost_beta`/`boost_gamma` results, then a chain of unboxing loads and
//! scalar `fmul`/`fsub`/`fdiv`.
//!
//! Python float arithmetic *is* `f64` arithmetic, so the port's obligation
//! is simply to keep each operation separate and in source order. Adding
//! an FMA here would be the error; the phase file's exit criterion "the
//! untyped `cdef` locals are ported as plain f64" is discharged by writing
//! the arithmetic out and by
//! [`tests::the_kernel_contracts_nothing_because_the_pyx_boxes_its_locals`].
//!
//! # Compile-time constants
//!
//! `hazma/_utils/kinematics.pxd`'s `two_body_energy` is `cdef inline` on
//! three `DEF` constants at both charged-ρ call sites, and clang folds it:
//! the shipped object materialises `0x4078_4718_126d_6814` and
//! `0x4078_2d10_e355_2748` as immediates rather than calling anything.
//! [`ENG_PI_CHARGED_RHO`] and [`ENG_PI0_CHARGED_RHO`] reproduce those bit
//! patterns from the same expression, pinned in
//! [`tests::the_two_body_energies_match_the_shipped_immediates`].
//!
//! Every mass here comes from `hazma/_utils/constants.pxd`, which the
//! `.pyx` `include`s, so this module reads [`crate::constants::pdg`]
//! throughout — no legacy-table mixing, unlike [`super::photon_pion`].

use crate::boost;
use crate::constants::pdg::{MASS_PI, MASS_PI0, MASS_RHO};
use crate::kernels::photon_pion;
use crate::quad::{DEFAULT_LIMIT, QuadOpts, quad};

/// The charged pion's energy in the charged-ρ rest frame, MeV.
///
/// `two_body_energy(m_ρ, m_π±, m_π⁰)` — see the module docs for the
/// immediate this reproduces.
pub const ENG_PI_CHARGED_RHO: f64 =
    (MASS_RHO * MASS_RHO + MASS_PI * MASS_PI - MASS_PI0 * MASS_PI0) / (2.0 * MASS_RHO);

/// The neutral pion's energy in the charged-ρ rest frame, MeV.
///
/// `two_body_energy(m_ρ, m_π⁰, m_π±)`, the same expression with the two
/// daughter masses exchanged.
pub const ENG_PI0_CHARGED_RHO: f64 =
    (MASS_RHO * MASS_RHO + MASS_PI0 * MASS_PI0 - MASS_PI * MASS_PI) / (2.0 * MASS_RHO);

/// Either charged pion's energy in the neutral-ρ rest frame, MeV.
///
/// `ρ⁰ → π⁺ π⁻` is symmetric, so this is `m_ρ/2` and the `.pyx` writes it
/// that way rather than calling `two_body_energy`.
pub const ENG_PI_NEUTRAL_RHO: f64 = MASS_RHO / 2.0;

/// `scipy.integrate.quad`'s arguments at both of the `.pyx`'s call sites.
///
/// `epsabs`/`epsrel` are copied verbatim from
/// `hazma/spectra/_photon/_rho.pyx`; `limit` is scipy's default, which the
/// call sites reach by passing no keyword. Unlike
/// [`super::photon_pion`]'s site there is no `points` keyword, so `None`
/// selects [`crate::quad::qagse`].
const RHO_QUAD: QuadOpts<'static> = QuadOpts {
    epsabs: 1e-10,
    epsrel: 1e-5,
    limit: DEFAULT_LIMIT,
    points: None,
};

/// The neutral ρ's rest-frame integrand, MeV⁻².
///
/// `2 · (dN/dE)_{π±}(E, m_ρ/2) / E`. The factor 2 is the two charged
/// pions; the `1/E` is the flat-boost kernel's, not the spectrum's, which
/// is why this is not itself a spectrum and why the rest-frame branch of
/// [`dnde_photon_neutral_rho`] returning it is dimensionally odd (see
/// there).
///
/// The shipped object emits the doubling as `fadd d0, d9, d9`, i.e.
/// `x + x`, which is the same double as `2 * x` for every finite `x`; it
/// is written that way here so the object code and the source agree.
#[must_use]
pub fn neutral_rho_integrand(e: f64) -> f64 {
    let dnde = photon_pion::dnde_photon_charged_pion(e, ENG_PI_NEUTRAL_RHO);
    (dnde + dnde) / e
}

/// The charged ρ's rest-frame integrand, MeV⁻².
///
/// `[ (dN/dE)_{π±}(E, E_π) + (dN/dE)_{π⁰}(E, E_π⁰) ] / E`, with the two
/// daughter energies fixed by [`ENG_PI_CHARGED_RHO`] and
/// [`ENG_PI0_CHARGED_RHO`]. Summed first, divided once — the object code
/// shows `fadd` then `fdiv`, and the two orders are not the same double.
#[must_use]
pub fn charged_rho_integrand(e: f64) -> f64 {
    let charged = photon_pion::dnde_photon_charged_pion(e, ENG_PI_CHARGED_RHO);
    let neutral = photon_pion::dnde_photon_neutral_pion(e, ENG_PI0_CHARGED_RHO);
    (charged + neutral) / e
}

/// The boost window `[γE(1−β), γE(1+β)]` and the `1/(2βγ)` prefactor.
///
/// Split out of [`boosted`] for one reason: it is the module's only
/// arithmetic that a caller can observe *directly*, and Task 4.5's
/// mutation campaign found it was otherwise unobservable at all. Fusing
/// `γ·E·(1−β)` into an FMA — the transcription error the module docs warn
/// against, since the `.pyx` boxes its locals and cannot contract — moves
/// `emin` by one ulp and moves the resulting spectrum by **nothing**:
/// every `cargo` test, all 49 tests in `test/test_core_photon_rho.py` and
/// all 10 ρ parity blocks stayed green under it, because the outer
/// integral's own `epsrel` is 1e-5 and one ulp of an endpoint does not
/// reach it.
///
/// So the values are pinned here, bit for bit, where they can be seen.
/// The four other mutations in that campaign were all caught by the
/// ordinary gates; this is the one that needed a seam.
///
/// # Parameters
///
/// * `e` — the lab-frame photon energy, MeV.
/// * `erho` — the ρ's total energy, MeV, already known to be `≥ m_ρ` and
///   outside the rest-frame window.
///
/// # Returns
///
/// `(emin, emax, pre)` in MeV, MeV and dimensionless.
#[must_use]
fn boost_window(e: f64, erho: f64) -> (f64, f64, f64) {
    let beta = boost::boost_beta(erho, MASS_RHO);
    let gamma = boost::boost_gamma(erho, MASS_RHO);
    (
        gamma * e * (1.0 - beta),
        gamma * e * (1.0 + beta),
        0.5 / (beta * gamma),
    )
}

/// The flat boost of a rest-frame integrand out of the ρ's frame.
///
/// Both entry points are this function with a different integrand, which
/// is how the `.pyx` is written too — the two `*_point` `cdef`s are
/// character-for-character identical apart from which integrand they name.
///
/// The three branches, in the `.pyx`'s order:
///
/// 1. `E_ρ < m_ρ` → exactly `0.0`.
/// 2. `E_ρ − m_ρ < DBL_EPSILON` → the rest frame, returned as the bare
///    integrand. A `NaN` `E_ρ` fails both comparisons and falls through to
///    the quadrature, where `β` and `γ` are `NaN`, the limits are `NaN`
///    and the result is `NaN` — the Cython does the same.
/// 3. Otherwise the quadrature between `γE(1∓β)` with the `1/(2βγ)`
///    prefactor.
///
/// Every operation is separate and in source order; see the module docs
/// for why no [`f64::mul_add`] appears.
// Not a disguised equality test, which is what `float_equality_without_abs`
// looks for: `erho >= MASS_RHO` is already established above, so this is the
// one-sided "within one epsilon MeV of rest" threshold the Cython writes,
// and `.abs()` would change nothing.
#[allow(clippy::float_equality_without_abs)]
fn boosted(e: f64, erho: f64, integrand: fn(f64) -> f64) -> f64 {
    if erho < MASS_RHO {
        return 0.0;
    }

    // The `.pyx`'s comment: "If we are sufficiently close to the rho
    // rest-frame, use the rest-frame result." At `E_ρ = m_ρ` exactly this
    // is the only branch that is finite — `β = 0` would make the
    // prefactor infinite and the integration range empty.
    if erho - MASS_RHO < f64::EPSILON {
        return integrand(e);
    }

    let (emin, emax, pre) = boost_window(e, erho);

    let mut integrand = integrand;
    match quad(&mut integrand, emin, emax, &RHO_QUAD) {
        Ok(outcome) => pre * outcome.value,
        // Unreachable, and asserted so by
        // `rho_quad_options_are_always_accepted` below: `QuadError` is a
        // statement about the *options* (`epsabs > 0`, `limit` above the
        // surviving break-point count), never about the integrand or the
        // interval, and these options are `const`. `NaN` rather than a
        // panic, for the reason `crate::boost` gives — `dispatch::map_unary`
        // evaluates element by element and has no per-element error
        // channel, so a panic would take down a whole array where the
        // Cython would have raised once.
        Err(_) => f64::NAN,
    }
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from neutral-ρ decay.
///
/// # Parameters
///
/// * `egam` — the photon energy, MeV.
/// * `erho` — the ρ's total energy, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` for a ρ below its own rest mass.
///
/// **At `E_ρ` within one `DBL_EPSILON` of `m_ρ` the returned quantity is
/// the integrand, not the spectrum** — `2·(dN/dE)_{π±}(E, m_ρ/2)/E`, which
/// carries an extra `1/E` and so is MeV⁻², not MeV⁻¹. The `.pyx` does
/// this, the parity corpus pins it (its `rest_plus_eps` block), and rule 1
/// keeps it. It is recorded as a defect in
/// `docs/followups/todo/rho-rest-frame-branch-returns-the-integrand.md`.
#[must_use]
pub fn dnde_photon_neutral_rho(egam: f64, erho: f64) -> f64 {
    boosted(egam, erho, neutral_rho_integrand)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from charged-ρ decay.
///
/// # Parameters
///
/// * `egam` — the photon energy, MeV.
/// * `erho` — the ρ's total energy, MeV.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹, and exactly `0.0` for a ρ below its own rest mass.
/// The rest-frame branch carries the same units defect as
/// [`dnde_photon_neutral_rho`]; see there.
#[must_use]
pub fn dnde_photon_charged_rho(egam: f64, erho: f64) -> f64 {
    boosted(egam, erho, charged_rho_integrand)
}

#[cfg(test)]
mod tests {
    use super::{
        ENG_PI_CHARGED_RHO, ENG_PI_NEUTRAL_RHO, ENG_PI0_CHARGED_RHO, RHO_QUAD,
        charged_rho_integrand, dnde_photon_charged_rho, dnde_photon_neutral_rho,
        neutral_rho_integrand,
    };
    use crate::constants::pdg::MASS_RHO;
    use crate::quad::quad;

    /// The two folded immediates read out of the shipped
    /// `hazma/spectra/_photon/_rho.cpython-312-darwin.so`.
    ///
    /// `integrand_charged_rho` materialises both with a `mov`/`movk`
    /// quartet and `fmov d0, x8` — clang folds `two_body_energy` rather
    /// than calling it, so these are the numbers the corpus was captured
    /// with. Pinning the bits (not a tolerance) is the point: a
    /// contracted or reassociated fold would land elsewhere in the last
    /// ulp and nothing downstream would notice, because two integrations
    /// separate this constant from any observable.
    #[test]
    fn the_two_body_energies_match_the_shipped_immediates() {
        assert_eq!(ENG_PI_CHARGED_RHO.to_bits(), 0x4078_4718_126d_6814);
        assert_eq!(ENG_PI0_CHARGED_RHO.to_bits(), 0x4078_2d10_e355_2748);
        // The neutral ρ's daughters share the rest mass exactly.
        assert_eq!(ENG_PI_NEUTRAL_RHO.to_bits(), 387.63_f64.to_bits());
    }

    /// The daughter energies balance: `E_π + E_π⁰ = m_ρ`.
    ///
    /// True of `two_body_energy` as algebra, and here it also holds
    /// exactly in floating point because the two numerators are the same
    /// `2 m_ρ²` split by one subtraction.
    #[test]
    fn the_charged_rho_daughter_energies_sum_to_the_rest_mass() {
        assert_eq!(ENG_PI_CHARGED_RHO + ENG_PI0_CHARGED_RHO, MASS_RHO);
        assert_eq!(ENG_PI_NEUTRAL_RHO + ENG_PI_NEUTRAL_RHO, MASS_RHO);
        // The charged pion is the heavier daughter, so it takes *more*
        // energy than the symmetric split — `two_body_energy` grows with
        // the emitting particle's own mass.
        const { assert!(ENG_PI_CHARGED_RHO > ENG_PI_NEUTRAL_RHO) };
    }

    /// The `.pyx` boxes every local, so nothing in it is contractible —
    /// and the choice is load-bearing rather than cosmetic.
    ///
    /// A unit test cannot observe which instructions the module compiled
    /// to, so it pins the thing that makes the disassembly worth reading:
    /// that the un-fused spelling this module uses and the contraction
    /// clang would have emitted for a `cdef double` version are **different
    /// doubles** at a live boost. If a later edit reaches for `mul_add`
    /// here, it is changing the number, not tidying the source.
    /// Where the two spellings separate is **near rest**, not at large
    /// boost: `1 - beta` is where the cancellation lives, and a fused
    /// `ge - ge*beta` skips exactly that rounding. At `E_ρ = 3 m_ρ` the two
    /// agree at every probe tried, which is why this test pins
    /// `E_ρ = 1.05 m_ρ` — the regime the corpus's `rest_plus_eps` and
    /// low-boost blocks actually sample.
    #[test]
    fn the_kernel_contracts_nothing_because_the_pyx_boxes_its_locals() {
        let erho = 1.05 * MASS_RHO;
        let beta = crate::boost::boost_beta(erho, MASS_RHO);
        let gamma = crate::boost::boost_gamma(erho, MASS_RHO);

        for e in [7.0, 13.0, 40.0] {
            // `emin = gamma * e * (1 - beta)`, as the module writes it.
            let separate = gamma * e * (1.0 - beta);
            // The same expression with the outer multiply and the subtract
            // folded into one rounding, which is what `-ffp-contract=on`
            // produces from `ge - ge*beta`.
            let contracted = (gamma * e).mul_add(-beta, gamma * e);

            assert_ne!(
                separate.to_bits(),
                contracted.to_bits(),
                "the two spellings agree at E = {e}, so it proves nothing — pick another"
            );
        }
    }

    /// The boost window, pinned bit for bit at four live arguments.
    ///
    /// This is the seam [`super::boost_window`]'s docs explain: the only
    /// place a transcription error in this module's arithmetic is
    /// *observable*. Fusing `γ·E·(1−β)` moves `emin` by one ulp at the
    /// first two rows and by nothing at the last two — and moves the
    /// spectrum, the 49 Python tests and the 10 ρ parity blocks by
    /// nothing at all, which is why the assertion is on these three
    /// numbers rather than on a spectrum.
    ///
    /// The expected bits come from evaluating the same expression in
    /// Python, whose float arithmetic is IEEE `f64` one operation at a
    /// time — the same thing the `.pyx`'s boxed locals do.
    #[test]
    fn the_boost_window_is_computed_without_contraction() {
        for (factor, e, emin_bits, emax_bits, pre_bits) in [
            (
                1.05_f64,
                7.0_f64,
                0x4014_6f85_30a1_99f3_u64,
                0x4023_2ea3_ce15_996e,
                0x3ff8_fce0_95e0_c6d2,
            ),
            (
                1.05,
                40.0,
                0x403d_3199_b330_007f,
                0x404b_6733_2667_ffc0,
                0x3ff8_fce0_95e0_c6d2,
            ),
            (
                2.0,
                100.0,
                0x403a_cb7f_d3d8_20da,
                0x4077_5348_02c2_7df2,
                0x3fd2_79a7_4590_331d,
            ),
            (
                10.0,
                300.0,
                0x402e_134b_ee41_4326,
                0x40b7_60f6_5a08_df5e,
                0x3fa9_ba9d_a6c7_3588,
            ),
        ] {
            let (emin, emax, pre) = super::boost_window(e, factor * MASS_RHO);
            assert_eq!(emin.to_bits(), emin_bits, "emin at {factor} m_rho, E = {e}");
            assert_eq!(emax.to_bits(), emax_bits, "emax at {factor} m_rho, E = {e}");
            assert_eq!(pre.to_bits(), pre_bits, "prefactor at {factor} m_rho");
        }
    }

    /// The window brackets the lab energy and closes as the boost dies.
    ///
    /// Structural rather than bit-level, and the complement of the test
    /// above: it would survive an ulp-level error and fails loudly on a
    /// swapped sign, a reciprocal `γ`, or `1+β` for `1−β`.
    #[test]
    fn the_boost_window_brackets_the_lab_energy() {
        let e = 137.0;
        let mut previous_width = f64::INFINITY;
        for factor in [10.0_f64, 4.0, 2.0, 1.05] {
            let (emin, emax, pre) = super::boost_window(e, factor * MASS_RHO);
            assert!(emin < e && e < emax, "window at {factor} m_rho excludes E");
            assert!(pre > 0.0);
            // `emin * emax = (gamma E)^2 (1 - beta^2) = E^2`, exactly the
            // statement that makes the rest-frame limit `f(E)`.
            let relative = ((emin * emax).sqrt() - e).abs() / e;
            assert!(relative < 1e-12, "geometric mean at {factor}: {relative:e}");
            let width = emax - emin;
            assert!(width < previous_width, "width shrinks toward rest");
            previous_width = width;
        }
    }

    /// Below its own rest mass a ρ radiates nothing, exactly.
    #[test]
    fn a_rho_below_threshold_is_exactly_zero() {
        for erho in [0.0, 1.0, MASS_RHO * 0.5, MASS_RHO - 1e-9] {
            assert_eq!(dnde_photon_charged_rho(100.0, erho), 0.0);
            assert_eq!(dnde_photon_neutral_rho(100.0, erho), 0.0);
        }
    }

    /// At rest both entry points return their integrand, per the `.pyx`.
    ///
    /// The threshold is `E_ρ − m_ρ < DBL_EPSILON`, an *absolute* window,
    /// so at `m_ρ ≈ 775` MeV it is reached only by `E_ρ` equal to `m_ρ` or
    /// a couple of ulp above.
    #[test]
    fn the_rest_frame_branch_returns_the_bare_integrand() {
        let e = 200.0;
        assert_eq!(
            dnde_photon_neutral_rho(e, MASS_RHO).to_bits(),
            neutral_rho_integrand(e).to_bits()
        );
        assert_eq!(
            dnde_photon_charged_rho(e, MASS_RHO).to_bits(),
            charged_rho_integrand(e).to_bits()
        );
    }

    /// A `NaN` ρ energy reaches the quadrature and comes back `NaN`.
    ///
    /// Both guards compare with `<`, which is false for `NaN`, so neither
    /// short circuit fires. Documented because it is the one input where
    /// "returns 0 below threshold" does not describe the behavior.
    #[test]
    fn a_nan_rho_energy_propagates() {
        assert!(dnde_photon_charged_rho(100.0, f64::NAN).is_nan());
        assert!(dnde_photon_neutral_rho(100.0, f64::NAN).is_nan());
    }

    /// The integrands are positive where the pion spectra are.
    ///
    /// A sign flip in either daughter term would survive the corpus's
    /// relative budget only if it were tiny; this catches the loud version
    /// at the kernel layer, where the outer integral cannot hide it.
    #[test]
    fn the_integrands_are_positive_inside_the_pion_support() {
        for e in [1.0, 10.0, 50.0, 68.0] {
            assert!(neutral_rho_integrand(e) > 0.0, "neutral at {e}");
            assert!(charged_rho_integrand(e) > 0.0, "charged at {e}");
        }
    }

    /// The π⁰ box edge is the sharpest structure either integrand has, and
    /// it separates two regimes with different ratios.
    ///
    /// The charged ρ's π⁰ sits at [`ENG_PI0_CHARGED_RHO`], whose `γγ` box
    /// runs from `E_π⁰(1−β)/2 = 12.157` MeV up. So:
    ///
    /// * **below the edge** only the single charged pion contributes, and
    ///   the neutral ρ's *two* charged pions make it twice as large —
    ///   ratio `≈ 0.5`, and exactly 0.5 in the limit where the two
    ///   daughter energies coincide;
    /// * **above the edge** the box dominates by more than an order of
    ///   magnitude and the inequality reverses.
    ///
    /// Both are structural rather than numerical, so an implementation
    /// that swapped the two daughter energies, dropped the factor of two,
    /// or used the wrong pion mass in the box fails here. Nothing softer
    /// than the reversal itself is asserted, because the ratio depends on
    /// the radiative spectrum's shape.
    #[test]
    fn the_neutral_pion_box_edge_reverses_which_rho_integrand_is_larger() {
        // `E_pi0 (1 - beta) / 2` for `E_pi0 = ENG_PI0_CHARGED_RHO`.
        let box_lower_edge = 12.156_854_062_150_506;

        for e in [1.0, 5.0, 10.0, 12.0] {
            assert!(e < box_lower_edge);
            let ratio = charged_rho_integrand(e) / neutral_rho_integrand(e);
            assert!(
                (0.49..0.51).contains(&ratio),
                "below the box the charged rho is half the neutral one; got {ratio} at {e}"
            );
        }

        for e in [13.0, 20.0, 50.0, 100.0] {
            assert!(e > box_lower_edge);
            assert!(
                charged_rho_integrand(e) > 2.0 * neutral_rho_integrand(e),
                "the pi0 box should dominate at {e}"
            );
        }
    }

    /// The boost integral is normalisation-preserving in the only sense
    /// this kernel can check cheaply: a more boosted ρ spreads the same
    /// photons over a wider range, so the spectrum at a fixed low energy
    /// falls.
    #[test]
    fn boosting_the_rho_lowers_the_spectrum_at_a_fixed_energy() {
        let e = 40.0;
        let mut previous = f64::INFINITY;
        for factor in [1.5_f64, 2.0, 4.0, 8.0] {
            let value = dnde_photon_charged_rho(e, factor * MASS_RHO);
            assert!(value > 0.0, "positive at {factor}");
            assert!(value < previous, "monotone at {factor}");
            previous = value;
        }
    }

    /// [`super::boosted`]'s `Err` arm is unreachable for `RHO_QUAD`.
    ///
    /// `QuadError` depends only on the options, never on the integrand or
    /// the interval, and `RHO_QUAD` is a `const` — so the `NaN` fallback
    /// documents a case the type system forces us to name rather than one
    /// a caller can reach. Asserted rather than assumed, on the widest
    /// range of intervals the kernel can produce.
    #[test]
    fn rho_quad_options_are_always_accepted() {
        for (lo, hi) in [
            (0.0, 1.0),
            (1e-12, 1e12),
            (5.0, 5.0),
            (10.0, 1.0),
            (0.0, f64::INFINITY),
        ] {
            let mut integrand = |x: f64| x;
            assert!(
                quad(&mut integrand, lo, hi, &RHO_QUAD).is_ok(),
                "quad rejected [{lo}, {hi}]"
            );
        }
    }
}
