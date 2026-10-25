//! Resolved prompt photons of the axial-vector mediator's charged final
//! states.
//!
//! The kernels behind `AxialVectorMediator.rest_frame_resolved_photons`,
//! `mediator_decay_resolved_photons` and their counts. Each yield is the
//! real-emission photon spectrum at `O(α)` per Born decay, or annihilation,
//! into its final state, above an explicit infrared cut on the photon
//! energy in the source's rest frame. Virtual corrections, and the `O(α)`
//! recoil and virtual corrections to the decay yields of
//! [`super::axial_yields`], are not included.
//!
//! # Three pions
//!
//! In σ-model coordinates of LO chiral perturbation theory the photon
//! enters only the charged-pion kinetic term. The isovector axial current
//! `σ∂φ₃ − φ₃∂σ` and the four-pion vertex `(φ·∂φ)²/2F²` reach the charged
//! fields only through the gauge-invariant `φ·∂φ ⊃ ∂(π⁺π⁻)` and `φ²`, and
//! the covariant derivative has no axial-photon cross term for this
//! source. Every radiative diagram therefore emits from an external `π±`,
//! and its Born vertex sees only the total charged momentum `q − p₀`. The
//! tree amplitude of `π⁺π⁻π⁰γ` is exactly
//!
//! `M_γ = e (ε_A·H) J·ε_γ*,  J = p₊/(p₊·k) − p₋/(p₋·k)`,
//!
//! with the Born current of [`super::axial_three_pi::ThreePion::current`]
//! evaluated at `s̃ = (q − p₀)²` in place of `(p₊ + p₋)²`, for both the
//! transverse and the source-longitudinal densities and at any photon
//! energy. `J·k = 0` is the Ward identity, and the photon polarization sum
//! is `−J²`. The transverse weight `|p₀|²/3 = λ(q², s̃, m₀²)/(12q²)` and the
//! longitudinal weight `m₀⁴(s̃ − m₀²)²/[q²(q² − m₀²)²]` depend on `s̃`
//! alone; both drop the `(g₃/F)²` that cancels against the Born.
//!
//! At photon energy `ω` in the source frame the four-body measure is
//! `ω dω dΩ/(16π³)` times the three-body measure of the recoil `P = q − k`
//! into the π⁰ and the charged pair of invariant mass `s`. The eikonal
//! integrated over the pair's decay angles is
//! `∫dΩ*(−J²) = (4π/ω_c²)[(1 + β²)ln((1 + β)/(1 − β))/β − 2]`, with `β` each
//! charged pion's speed and `ω_c` the photon energy in the pair frame. A
//! spectrum is then a two-dimensional integral over `s` and the cosine
//! `cos θ₀` of the π⁰ to the photon in the recoil frame, taken by fixed
//! Gauss–Legendre rules. As `ω → 0` it tends to
//! `(α/πω)⟨(1 + β²)L/β − 2⟩` over the Born, the soft eikonal limit.
//!
//! An on-shell mediator decays through the transverse current alone. A
//! photon of energy `ω` then has the angular density
//! `[f + (3F₀ − 1)h P₂(cos θ)]/2` about the mediator's flight direction,
//! the form [`super::polarized_cascade`] gives its pions: `f` is the
//! transverse spectrum, and `h` replaces its weight `|p₀|²/3` with
//! `[(n̂·p₀)² − |p₀|²/3]/2`, where `n̂` is the photon's direction. The
//! current is a polar vector, so there is no `P₁` term.
//!
//! # Lepton pairs
//!
//! `e⁺e⁻` and `μ⁺μ⁻` take the Altarelli–Parisi leading-log spectrum of
//! `hazma/spectra/altarelli_parisi.py`, summed over both leptons:
//! `(2α/πQ)[1 + (1 − x)²]/x [ln(Q²(1 − x)/m²) − 1]` at `x = 2ω/Q`, zero
//! where the bracket is not positive. It is the collinear limit, so each
//! photon moves along its lepton and a moving mediator boosts it with the
//! lepton's angular density: the [`super::polarized_cascade::PairDecay`]
//! analyzer, whose vector powers cancel between the two leptons and leave
//! the tensor power alone. It is an approximation that keeps no term
//! without the collinear logarithm and no spin correlation of the
//! emission itself.
//!
//! Masses and energies are in MeV, densities in MeV⁻¹ and counts are
//! photons per Born decay. The caller validates every input: a source
//! above its threshold, a positive cut, fractions that are a density's
//! diagonal, and ordered bin edges.

use std::f64::consts::PI;

use super::axial_three_pi::{angular_rule, gauss_legendre};
use super::axial_yields::{FOLD_EPSREL, Probe};
use super::polarized_cascade::{Boost, Helicities, legendre2};
use super::polarized_muon::{boost_range, integrate};

/// Gauss–Legendre order of the pair-mass integral, taken after the
/// substitution that removes its endpoint square roots.
const MASS_ORDER: usize = 48;

/// Gauss–Legendre order of the π⁰-direction integral. Its integrand is
/// rational in `cos θ₀`, with the double pole of `1/ω_c²` at
/// `−E_c/p₀ < −1`, `E_c` and `p₀` being the pair's energy and momentum in
/// the recoil frame.
const DIRECTION_ORDER: usize = 32;

/// Return `(1 + β²)ln((1 + β)/(1 − β))/β − 2`, the eikonal integrated over
/// a charged pair's decay angles in units of `4π/ω_c²`.
///
/// Below `β = 1/2` it is the series `Σ_{k≥1} 8kβ^{2k}/(4k² − 1)`, because
/// the closed form cancels to `8β²/3` as `β → 0`.
fn eikonal_radiator(beta: f64) -> f64 {
    if beta < 0.5 {
        let beta2 = beta * beta;
        let mut power = 1.0;
        let mut sum = 0.0;
        // 0.25⁴⁰ is far below an ulp of the leading term.
        for k in 1..=40 {
            power *= beta2;
            let k = f64::from(k);
            sum += 8.0 * k * power / (4.0 * k * k - 1.0);
        }
        sum
    } else {
        (1.0 + beta * beta) * 2.0 * beta.atanh() / beta - 2.0
    }
}

/// A rest-frame photon spectrum and its alignment about a mediator's
/// flight direction.
pub trait RestPhotons {
    /// Return `[f, h]`: the density `dN/dω` and the alignment density, the
    /// weight of `(3F₀ − 1)P₂(cos θ)`, in MeV⁻¹ at rest-frame energy
    /// `omega`. Both vanish outside [`Self::support`].
    fn spectrum(&self, omega: f64) -> [f64; 2];

    /// Return the photon energies `[cut, endpoint]` the spectrum lives on.
    fn support(&self) -> (f64, f64);
}

/// The LO `π⁺π⁻π⁰γ` emission of a source of invariant mass `energy`,
/// above `cut`.
#[derive(Clone, Debug)]
pub struct ThreePionRadiation {
    energy: f64,
    neutral_mass: f64,
    charged_mass: f64,
    transverse_fraction: f64,
    alpha: f64,
    cut: f64,
    /// The Born integrals `∫ds p₀β X(s)` of the transverse and the
    /// longitudinal weight, `p₀` the π⁰ momentum in the source frame.
    born: [f64; 2],
    mass_rule: Vec<(f64, f64)>,
    direction_rule: Vec<(f64, f64)>,
}

impl ThreePionRadiation {
    /// Return the emission of a source of invariant mass `energy` whose
    /// Born rate is `transverse_fraction` transverse and the rest
    /// source-longitudinal, with the π⁰ and π± `masses`, the fine-structure
    /// constant `alpha` and the infrared `cut` on the photon energy.
    ///
    /// The LO current has one pion mass; distinct `masses` evaluate it
    /// with the π⁰ mass in the pion pole and the charged mass in the
    /// kinematics.
    pub fn new(
        energy: f64,
        [neutral_mass, charged_mass]: [f64; 2],
        transverse_fraction: f64,
        alpha: f64,
        cut: f64,
    ) -> Self {
        let mut radiation = Self {
            energy,
            neutral_mass,
            charged_mass,
            transverse_fraction,
            alpha,
            cut,
            born: [0.0; 2],
            mass_rule: angular_rule(MASS_ORDER),
            direction_rule: gauss_legendre(DIRECTION_ORDER),
        };
        radiation.born = radiation.born_integrals();
        radiation
    }

    /// Return the largest photon energy, at which the recoil reaches the
    /// three-pion threshold.
    pub fn endpoint(&self) -> f64 {
        let threshold = self.neutral_mass + 2.0 * self.charged_mass;
        ((self.energy - threshold) * (self.energy + threshold) / (2.0 * self.energy)).max(0.0)
    }

    /// Return the transverse and longitudinal weights at `s̃`.
    fn weights(&self, s_tilde: f64) -> [f64; 2] {
        let (q, m) = (self.energy, self.neutral_mass);
        let (q2, m2) = (q * q, m * m);
        let transverse = ((q - m).powi(2) - s_tilde) * ((q + m).powi(2) - s_tilde) / (12.0 * q2);
        let longitudinal = (m2 * (s_tilde - m2) / (q2 - m2)).powi(2) / q2;
        [transverse.max(0.0), longitudinal]
    }

    /// Return the pair invariant `s` and its Jacobian at the node `theta`
    /// of the substitution `s = 4m² + (upper − 4m²)(1 − cos θ)/2`.
    fn pair_mass(&self, theta: f64, upper: f64) -> (f64, f64) {
        let lower = 4.0 * self.charged_mass * self.charged_mass;
        let span = upper - lower;
        (
            lower + span * (1.0 - theta.cos()) / 2.0,
            span * theta.sin() / 2.0,
        )
    }

    fn pair_speed(&self, s: f64) -> f64 {
        (1.0 - 4.0 * self.charged_mass * self.charged_mass / s)
            .max(0.0)
            .sqrt()
    }

    fn born_integrals(&self) -> [f64; 2] {
        let (q, m0) = (self.energy, self.neutral_mass);
        let upper = (q - m0).powi(2);
        if upper <= 4.0 * self.charged_mass * self.charged_mass {
            return [0.0; 2];
        }
        let mut born = [0.0; 2];
        for &(theta, weight) in &self.mass_rule {
            let (s, jacobian) = self.pair_mass(theta, upper);
            let momentum = (((q - m0).powi(2) - s) * ((q + m0).powi(2) - s))
                .max(0.0)
                .sqrt()
                / (2.0 * q);
            let measure = weight * jacobian * momentum * self.pair_speed(s);
            let [transverse, longitudinal] = self.weights(s);
            born[0] += measure * transverse;
            born[1] += measure * longitudinal;
        }
        born
    }

    /// Return the emission integrals at photon energy `omega`: the
    /// transverse and longitudinal weights and the transverse alignment
    /// weight, each integrated with `p₀β R(β)/(√P² ω_c²)` over `s` and
    /// `cos θ₀`, in the recoil frame's `p₀`.
    fn emission_integrals(&self, omega: f64) -> [f64; 3] {
        let (q, m0) = (self.energy, self.neutral_mass);
        let recoil2 = q * (q - 2.0 * omega);
        if recoil2 <= 0.0 {
            return [0.0; 3];
        }
        let recoil = recoil2.sqrt();
        let upper = (recoil - m0).powi(2);
        if upper <= 4.0 * self.charged_mass * self.charged_mass {
            return [0.0; 3];
        }
        // The photon's energy in the recoil frame.
        let photon = q * omega / recoil;
        let mut integrals = [0.0; 3];
        for &(theta, weight) in &self.mass_rule {
            let (s, jacobian) = self.pair_mass(theta, upper);
            let neutral_energy = (recoil2 + m0 * m0 - s) / (2.0 * recoil);
            let pair_energy = (recoil2 - m0 * m0 + s) / (2.0 * recoil);
            let momentum = (((recoil - m0).powi(2) - s) * ((recoil + m0).powi(2) - s))
                .max(0.0)
                .sqrt()
                / (2.0 * recoil);
            let beta = self.pair_speed(s);
            let outer = weight * jacobian * momentum * beta * eikonal_radiator(beta) / recoil;
            for &(cosine, direction_weight) in &self.direction_rule {
                let pair_k = photon * (pair_energy + momentum * cosine);
                let neutral_k = photon * (neutral_energy - momentum * cosine);
                // `1/ω_c²`, with `ω_c = P_c·k/√s`.
                let eikonal = s / (pair_k * pair_k);
                let [transverse, longitudinal] = self.weights(s + 2.0 * pair_k);
                // `n̂·p₀` in the source frame: `E₀ − k·p₀/ω`.
                let neutral_source_energy = (recoil * neutral_energy + neutral_k) / q;
                let along =
                    neutral_source_energy - q * (neutral_energy - momentum * cosine) / recoil;
                let alignment = (along * along - transverse) / 2.0;
                let measure = outer * direction_weight * eikonal;
                integrals[0] += measure * transverse;
                integrals[1] += measure * longitudinal;
                integrals[2] += measure * alignment;
            }
        }
        integrals
    }

    /// Return, per Born decay in MeV⁻¹, the transverse spectrum, the
    /// longitudinal spectrum and the transverse alignment density at
    /// `omega`, zero outside the support.
    fn spectra(&self, omega: f64) -> [f64; 3] {
        let (cut, endpoint) = self.support();
        if omega < cut || omega >= endpoint {
            return [0.0; 3];
        }
        let [transverse, longitudinal, alignment] = self.emission_integrals(omega);
        let scale = self.alpha * omega * self.energy / (2.0 * PI);
        let per_born = |integral: f64, born: f64| {
            if born > 0.0 {
                scale * integral / born
            } else {
                0.0
            }
        };
        [
            per_born(transverse, self.born[0]),
            per_born(longitudinal, self.born[1]),
            per_born(alignment, self.born[0]),
        ]
    }
}

impl RestPhotons for ThreePionRadiation {
    fn spectrum(&self, omega: f64) -> [f64; 2] {
        let [transverse, longitudinal, alignment] = self.spectra(omega);
        let fraction = self.transverse_fraction;
        [
            fraction.mul_add(transverse, (1.0 - fraction) * longitudinal),
            fraction * alignment,
        ]
    }

    fn support(&self) -> (f64, f64) {
        (self.cut, self.endpoint())
    }
}

/// The Altarelli–Parisi emission of a lepton pair from a source of
/// invariant mass `energy`, above `cut`.
#[derive(Clone, Copy, Debug)]
pub struct PairRadiation {
    /// The source's invariant mass in MeV.
    pub energy: f64,
    /// The lepton mass in MeV.
    pub mass: f64,
    /// The fine-structure constant.
    pub alpha: f64,
    /// The infrared cut on the photon energy in MeV.
    pub cut: f64,
    /// The pair's tensor analyzing power, which the collinear photons
    /// share; it reaches only a moving mediator's yields.
    pub tensor: f64,
}

impl RestPhotons for PairRadiation {
    fn spectrum(&self, omega: f64) -> [f64; 2] {
        let (cut, endpoint) = self.support();
        if omega < cut || omega >= endpoint {
            return [0.0; 2];
        }
        let q = self.energy;
        let x = 2.0 * omega / q;
        let log = (q * (q - 2.0 * omega) / (self.mass * self.mass)).ln() - 1.0;
        let splitting = (1.0 + (1.0 - x).powi(2)) / x;
        let density = 2.0 * self.alpha / (PI * q) * splitting * log;
        [density, self.tensor * density]
    }

    /// The endpoint is where `ln(Q²(1 − x)/m²) = 1`.
    fn support(&self) -> (f64, f64) {
        let q = self.energy;
        let endpoint = (q * q - std::f64::consts::E * self.mass * self.mass) / (2.0 * q);
        (self.cut, endpoint.max(0.0))
    }
}

/// Return the rest-frame yield of `spectrum` at `probe`.
pub fn rest_photons(spectrum: &impl RestPhotons, probe: Probe) -> f64 {
    match probe {
        Probe::Density(omega) => spectrum.spectrum(omega)[0],
        Probe::Bin(low, high) => {
            let (from, to) = spectrum.support();
            let (low, high) = (low.max(from), high.min(to));
            if low >= high {
                return 0.0;
            }
            let mut integrand = |omega: f64| spectrum.spectrum(omega)[0];
            integrate(&mut integrand, low, high, &[], FOLD_EPSREL)
        }
    }
}

/// Return the lab yield at `probe` of `spectrum`, emitted by a mediator
/// with `boost` and `helicities`.
///
/// A photon of rest-frame energy `ω` at `cos θ` to the flight direction
/// reaches `γω(1 + β cos θ)`, so a lab density integrates
/// `[f + (3F₀ − 1)h P₂]/(2γβω)` over the `ω` whose box holds it, and a bin
/// integrates the angular density over the cosines it spans, in closed
/// form, at each `ω`.
pub fn lab_photons(
    spectrum: &impl RestPhotons,
    boost: Boost,
    helicities: Helicities,
    probe: Probe,
) -> f64 {
    let (from, to) = spectrum.support();
    let (gamma, gamma_beta) = (boost.gamma, boost.gamma_beta);
    let weight = helicities.alignment();
    match probe {
        Probe::Density(energy) => {
            let (lower, upper) = boost_range(energy, 0.0, gamma, gamma_beta);
            let (lower, upper) = (lower.max(from), upper.min(to));
            if lower >= upper {
                return 0.0;
            }
            let mut integrand = |omega: f64| {
                let [density, alignment] = spectrum.spectrum(omega);
                let cosine = boost.cosine(energy, omega, omega).clamp(-1.0, 1.0);
                weight.mul_add(alignment * legendre2(cosine), density) / (2.0 * gamma_beta * omega)
            };
            integrate(&mut integrand, lower, upper, &[], FOLD_EPSREL)
        }
        Probe::Bin(low, high) => {
            let (low_lower, low_upper) = boost_range(low, 0.0, gamma, gamma_beta);
            let (high_lower, high_upper) = boost_range(high, 0.0, gamma, gamma_beta);
            let (lower, upper) = (low_lower.max(from), high_upper.min(to));
            if lower >= upper {
                return 0.0;
            }
            // The cosine of either edge leaves [−1, 1] where its box ends.
            let points = [low_upper, high_lower];
            let mut integrand = |omega: f64| {
                let [density, alignment] = spectrum.spectrum(omega);
                let from = boost.cosine(low, omega, omega).clamp(-1.0, 1.0);
                let to = boost.cosine(high, omega, omega).clamp(-1.0, 1.0);
                // The antiderivative of P₂ is (c³ − c)/2.
                let tensor = (to.powi(3) - to - from.powi(3) + from) / 2.0;
                weight.mul_add(alignment * tensor, density * (to - from)) / 2.0
            };
            integrate(&mut integrand, lower, upper, &points, FOLD_EPSREL)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::axial_three_pi::{Channel, ThreePion};
    use crate::kernels::polarized_muon::momentum;

    const M0: f64 = 134.9768;
    const MC: f64 = 139.57039;
    const ALPHA: f64 = 1.0 / 137.04;

    fn assert_close(actual: f64, expected: f64, rtol: f64) {
        let error = (actual - expected).abs() / expected.abs();
        assert!(
            error <= rtol,
            "{actual:e} vs {expected:e}: relative error {error:e} > {rtol:e}"
        );
    }

    fn unpolarized() -> Helicities {
        Helicities {
            plus: 1.0 / 3.0,
            zero: 1.0 / 3.0,
            minus: 1.0 / 3.0,
        }
    }

    fn pure(zero: f64) -> Helicities {
        Helicities {
            plus: (1.0 - zero) / 2.0,
            zero,
            minus: (1.0 - zero) / 2.0,
        }
    }

    fn dot(a: [f64; 4], b: [f64; 4]) -> f64 {
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    }

    /// The radiation as a pure spectrum `[f, h]` with `h` zeroed.
    struct Unaligned<'a>(&'a ThreePionRadiation);

    impl RestPhotons for Unaligned<'_> {
        fn spectrum(&self, omega: f64) -> [f64; 2] {
            [self.0.spectrum(omega)[0], 0.0]
        }

        fn support(&self) -> (f64, f64) {
            self.0.support()
        }
    }

    #[test]
    fn radiator_series_meets_the_closed_form() {
        let closed = |beta: f64| (1.0 + beta * beta) * 2.0 * beta.atanh() / beta - 2.0;
        // At β = 1/2 the closed form has lost about one digit.
        assert_close(eikonal_radiator(0.499_999_999), closed(0.5), 1e-8);
        assert_close(eikonal_radiator(1e-4), 8.0e-8 / 3.0, 1e-7);
    }

    /// The pair-angle integral of the eikonal, by quadrature over explicit
    /// four-vectors, against the closed form the kernel uses.
    #[test]
    fn eikonal_pair_integral_matches_explicit_momenta() {
        let (s, photon_energy, photon_cosine) = (360.0_f64.powi(2), 7.0, 0.3_f64);
        let half = s.sqrt() / 2.0;
        let speed = momentum(half, MC) / half;
        let k = [
            photon_energy,
            photon_energy * (1.0 - photon_cosine * photon_cosine).sqrt(),
            0.0,
            photon_energy * photon_cosine,
        ];
        let rule = gauss_legendre(200);
        let mut integral = 0.0;
        for &(cosine, weight) in &rule {
            for &(azimuth, azimuth_weight) in &rule {
                let azimuth = PI * (azimuth + 1.0);
                let sine = (1.0 - cosine * cosine).sqrt();
                let p = half * speed;
                let direction = [sine * azimuth.cos(), sine * azimuth.sin(), cosine];
                let plus = [half, p * direction[0], p * direction[1], p * direction[2]];
                let minus = [half, -plus[1], -plus[2], -plus[3]];
                let current: [f64; 4] =
                    std::array::from_fn(|mu| plus[mu] / dot(plus, k) - minus[mu] / dot(minus, k));
                integral += weight * azimuth_weight * PI * -dot(current, current);
            }
        }
        let expected = 4.0 * PI / (photon_energy * photon_energy) * eikonal_radiator(speed);
        assert_close(integral, expected, 1e-10);
    }

    /// At one pion mass the Born integrals are the three-pion spectral
    /// functions without their `(g₃/F)²` and measure prefactors.
    #[test]
    fn born_integrals_match_the_spectral_functions() {
        let decay_constant = 92.2;
        let current = ThreePion {
            channel: Channel::Charged,
            pion_mass: M0,
            decay_constant,
        };
        for energy in [420.0, 500.0, 600.0] {
            let radiation = ThreePionRadiation::new(energy, [M0, M0], 1.0, ALPHA, 1.0);
            let (transverse, longitudinal) =
                current.spectral_functions(energy, 1.0, &angular_rule(400));
            let measure = 64.0 * PI.powi(3) * energy * decay_constant * decay_constant;
            assert_close(radiation.born[0] * 16.0 / measure, transverse, 1e-12);
            assert_close(radiation.born[1] * 4.0 / measure, longitudinal, 1e-12);
        }
    }

    /// `ω dN/dω → (α/π)⟨(1 + β²)L/β − 2⟩` over each Born density.
    #[test]
    fn soft_limit_is_the_eikonal_radiator_over_the_born() {
        for (energy, masses) in [(450.0, [M0, MC]), (580.0, [M0, M0])] {
            let radiation = ThreePionRadiation::new(energy, masses, 1.0, ALPHA, 1e-9);
            let [m0, mc] = masses;
            let mean = |index: usize| {
                let integrand = |s: f64| {
                    let weight = radiation.weights(s)[index];
                    let momentum = (((energy - m0).powi(2) - s) * ((energy + m0).powi(2) - s))
                        .max(0.0)
                        .sqrt();
                    let beta = radiation.pair_speed(s);
                    [
                        momentum * beta * weight * eikonal_radiator(beta),
                        momentum * beta * weight,
                    ]
                };
                let (lower, upper) = (4.0 * mc * mc, (energy - m0).powi(2));
                let mut numerator = |s: f64| integrand(s)[0];
                let mut denominator = |s: f64| integrand(s)[1];
                integrate(&mut numerator, lower, upper, &[], 1e-12)
                    / integrate(&mut denominator, lower, upper, &[], 1e-12)
            };
            let omega = 1e-8 * energy;
            let [transverse, longitudinal, _] = radiation.spectra(omega);
            // The approach is linear in ω and steepest near threshold, about
            // 50ω/Q at 450 MeV, so 1e-8 leaves 5e-7.
            assert_close(omega * transverse, ALPHA / PI * mean(0), 1e-5);
            assert_close(omega * longitudinal, ALPHA / PI * mean(1), 1e-5);
        }
    }

    /// The fixed rules are converged: doubling both orders moves a
    /// spectrum by less than its stated precision.
    #[test]
    fn spectra_are_converged_in_the_rule_orders() {
        let radiation = ThreePionRadiation::new(560.0, [M0, M0], 0.7, ALPHA, 0.5);
        let mut fine = radiation.clone();
        fine.mass_rule = angular_rule(2 * MASS_ORDER);
        fine.direction_rule = gauss_legendre(2 * DIRECTION_ORDER);
        fine.born = fine.born_integrals();
        let endpoint = radiation.endpoint();
        for fraction in [1e-3, 0.1, 0.5, 0.9, 0.999] {
            let omega = fraction * endpoint;
            let (coarse, refined) = (radiation.spectra(omega), fine.spectra(omega));
            for (c, f) in coarse.iter().zip(refined) {
                assert!(
                    (c - f).abs() <= 1e-10 * refined[0].abs(),
                    "{c:e} vs {f:e} at {omega}"
                );
            }
        }
    }

    /// The restricted LO fractions of the mediator note's spectra study:
    /// `(9.12 ± 0.28)e-4` at 450 MeV and `(3.14 ± 0.10)e-3` at 500 MeV, for
    /// physical pion masses and `ω > 1 MeV`, each a Monte Carlo estimate
    /// with its one-sigma error; agreement within two sigma.
    #[test]
    fn resolved_fractions_reproduce_the_reference() {
        let alpha = 1.0 / 137.035_999;
        for (energy, reference, error) in [(450.0, 9.12e-4, 0.28e-4), (500.0, 3.14e-3, 0.10e-3)] {
            let radiation = ThreePionRadiation::new(energy, [M0, MC], 1.0, alpha, 1.0);
            let fraction = rest_photons(&radiation, Probe::Bin(1.0, energy));
            assert!(
                (fraction - reference).abs() <= 2.0 * error,
                "{energy}: {fraction:e} vs {reference:e}"
            );
        }
    }

    #[test]
    fn alignment_keeps_every_angular_density_positive() {
        let radiation = ThreePionRadiation::new(590.0, [M0, M0], 1.0, ALPHA, 0.1);
        let endpoint = radiation.endpoint();
        for i in 1..200 {
            let omega = endpoint * f64::from(i) / 200.0;
            let [density, alignment] = radiation.spectrum(omega);
            // P₂ spans [−1/2, 1]; 3F₀ − 1 spans [−1, 2].
            for coefficient in [-1.0, 2.0] {
                for p2 in [-0.5, 1.0] {
                    assert!(density + coefficient * alignment * p2 >= -1e-14 * density);
                }
            }
        }
    }

    #[test]
    fn spectra_vanish_outside_the_support() {
        let radiation = ThreePionRadiation::new(500.0, [M0, M0], 0.5, ALPHA, 2.0);
        let endpoint = radiation.endpoint();
        assert_eq!(radiation.spectrum(1.999), [0.0, 0.0]);
        assert_eq!(radiation.spectrum(endpoint), [0.0, 0.0]);
        // The recoil closes at the endpoint, so the spectrum falls to zero.
        assert!(radiation.spectrum(endpoint * (1.0 - 1e-6))[0] < 1e-6 * radiation.spectrum(2.0)[0]);
    }

    #[test]
    fn rest_counts_round_trip_by_bin_width() {
        let radiation = ThreePionRadiation::new(520.0, [M0, M0], 0.8, ALPHA, 1.0);
        let pair = PairRadiation {
            energy: 520.0,
            mass: 105.6583745,
            alpha: ALPHA,
            cut: 1.0,
            tensor: 0.0,
        };
        for omega in [3.0, 20.0, 40.0] {
            let width = 1e-4;
            let bin = Probe::Bin(omega - width / 2.0, omega + width / 2.0);
            assert_close(
                rest_photons(&radiation, bin) / width,
                radiation.spectrum(omega)[0],
                1e-8,
            );
            assert_close(
                rest_photons(&pair, bin) / width,
                pair.spectrum(omega)[0],
                1e-8,
            );
        }
    }

    #[test]
    fn pair_spectrum_is_twice_the_single_lepton_limit() {
        let pair = PairRadiation {
            energy: 400.0,
            mass: 105.6583745,
            alpha: ALPHA,
            cut: 0.5,
            tensor: 0.0,
        };
        let omega = 30.0_f64;
        let x = 2.0 * omega / 400.0;
        let single = ALPHA / (2.0 * PI) * (1.0 + (1.0 - x).powi(2)) / x
            * ((400.0_f64.powi(2) * (1.0 - x) / 105.6583745_f64.powi(2)).ln() - 1.0)
            * 2.0
            / 400.0;
        assert_close(pair.spectrum(omega)[0], 2.0 * single, 1e-14);
        let (_, endpoint) = pair.support();
        assert_eq!(pair.spectrum(endpoint), [0.0, 0.0]);
        assert!(pair.spectrum(endpoint * (1.0 - 1e-9))[0] >= 0.0);
    }

    /// A pure helicity-zero mediator emits its collinear photons with the
    /// leptons' `sin²θ`-weighted density when the leptons are massless.
    #[test]
    fn pair_photons_follow_the_lepton_alignment() {
        let decay = crate::kernels::polarized_cascade::PairDecay {
            mediator_mass: 400.0,
            mass: 1e-6,
            vector: 1.0,
            axial: 0.0,
        };
        let tensor = decay.analyzer().tensor;
        // Massless leptons from helicity zero: W = 3 sin²θ/4 = (1 − P₂)/2.
        assert_close(tensor, -0.5, 1e-12);
        let pair = PairRadiation {
            energy: 400.0,
            mass: 0.5109989461,
            alpha: ALPHA,
            cut: 1.0,
            tensor,
        };
        let [density, alignment] = pair.spectrum(50.0);
        assert_close(alignment, -0.5 * density, 1e-15);
        let isotropic = PairRadiation {
            tensor: 0.0,
            ..pair
        };
        let boost = Boost::new(400.0, 1000.0);
        let probe = Probe::Bin(20.0, 60.0);
        assert_close(
            lab_photons(&pair, boost, unpolarized(), probe),
            lab_photons(&isotropic, boost, unpolarized(), probe),
            1e-12,
        );
        assert!(
            lab_photons(&pair, boost, pure(1.0), probe)
                != lab_photons(&isotropic, boost, pure(1.0), probe)
        );
    }

    fn lab_total(spectrum: &impl RestPhotons, boost: Boost, helicities: Helicities) -> f64 {
        let (_, endpoint) = spectrum.support();
        let top = boost.gamma * endpoint + boost.gamma_beta * endpoint;
        lab_photons(spectrum, boost, helicities, Probe::Bin(1e-3, top * 1.01))
    }

    #[test]
    fn lab_counts_conserve_the_rest_frame_count() {
        let radiation = ThreePionRadiation::new(480.0, [M0, M0], 1.0, ALPHA, 1.0);
        let rest = rest_photons(&radiation, Probe::Bin(1.0, 480.0));
        for zero in [0.0, 1.0 / 3.0, 1.0] {
            let boost = Boost::new(480.0, 900.0);
            assert_close(lab_total(&radiation, boost, pure(zero)), rest, 1e-7);
        }
    }

    /// With no `P₁` term the mean lab energy is `γ` times the rest mean,
    /// whatever the helicities.
    #[test]
    fn lab_mean_energy_is_the_boosted_rest_mean() {
        let radiation = ThreePionRadiation::new(560.0, [M0, M0], 1.0, ALPHA, 1.0);
        let endpoint = radiation.endpoint();
        let mut rest = |omega: f64| omega * radiation.spectrum(omega)[0];
        let rest_mean = integrate(&mut rest, 1.0, endpoint, &[], 1e-10);
        let boost = Boost::new(560.0, 700.0);
        for zero in [0.0, 1.0] {
            let mut lab = |energy: f64| {
                energy * lab_photons(&radiation, boost, pure(zero), Probe::Density(energy))
            };
            let (up, down) = (
                boost.gamma + boost.gamma_beta,
                boost.gamma - boost.gamma_beta,
            );
            // The density kinks where the cut's and the endpoint's boxes end.
            let kinks = [up, down * endpoint];
            let lab_mean = integrate(&mut lab, down, up * endpoint, &kinks, 1e-9);
            assert_close(lab_mean, boost.gamma * rest_mean, 1e-6);
        }
    }

    #[test]
    fn isotropic_mediator_ignores_the_alignment() {
        let radiation = ThreePionRadiation::new(540.0, [M0, M0], 1.0, ALPHA, 1.0);
        let boost = Boost::new(540.0, 1200.0);
        for energy in [5.0, 40.0, 110.0] {
            let probe = Probe::Density(energy);
            assert_close(
                lab_photons(&radiation, boost, unpolarized(), probe),
                lab_photons(&Unaligned(&radiation), boost, unpolarized(), probe),
                1e-12,
            );
        }
    }

    #[test]
    fn slow_mediator_reproduces_the_rest_frame() {
        let radiation = ThreePionRadiation::new(500.0, [M0, M0], 1.0, ALPHA, 1.0);
        let boost = Boost::new(500.0, 500.0 * (1.0 + 1e-10));
        for omega in [5.0, 25.0, 40.0] {
            assert_close(
                lab_photons(&radiation, boost, pure(1.0), Probe::Density(omega)),
                radiation.spectrum(omega)[0],
                1e-4,
            );
        }
    }

    #[test]
    fn lab_density_and_bins_agree() {
        let radiation = ThreePionRadiation::new(500.0, [M0, M0], 1.0, ALPHA, 1.0);
        let boost = Boost::new(500.0, 800.0);
        for energy in [3.0, 30.0, 70.0] {
            let width = 1e-4;
            let bin = Probe::Bin(energy - width / 2.0, energy + width / 2.0);
            assert_close(
                lab_photons(&radiation, boost, pure(0.0), bin) / width,
                lab_photons(&radiation, boost, pure(0.0), Probe::Density(energy)),
                1e-7,
            );
        }
    }
}
