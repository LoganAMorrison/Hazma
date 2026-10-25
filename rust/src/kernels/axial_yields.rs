//! Rest-frame stable-particle yields of the axial-vector mediator's
//! cascading final states.
//!
//! The kernels behind `AxialVectorMediator.rest_frame_yields` and
//! `rest_frame_counts` for `μ⁺μ⁻`, `π⁺π⁻π⁰` and `π⁰π⁰π⁰` in the χχ̄
//! centre-of-mass frame. Every yield is per annihilation into its final
//! state, with each particle of that state counted, and has no prompt
//! lines: the model's other final states are lines alone, which the
//! Python layer declares.
//!
//! # Parent spectra
//!
//! Each muon of `μ⁺μ⁻` carries half the energy and one effective
//! polarization, the same for both charges because the pair's net
//! helicity sits in its opposite-helicity states (see
//! [`super::polarized_muon`] for the convention).
//!
//! The LO three-pion currents of [`super::axial_three_pi::ThreePion::current`]
//! weight the charged channel's transverse density by `|p₀|²` and its
//! longitudinal density by `(s − m²)²`, with `p₀` the π⁰ momentum and
//! `s = q² + m² − 2qE₀` the π⁺π⁻ invariant mass, both functions of the π⁰
//! energy `E₀` alone; the neutral channel's weight is constant. The
//! three-body measure is flat in two pion energies, and the energy of a
//! partner of a pion of energy `E` spans a length
//! `L(E) = p √(1 − 4m²/s(E))` around `(q − E)/2`. So the π⁰ of the charged
//! channel, and each pion of the neutral one, has the energy density
//! `M(E) L(E)`, while each charged pion has `∫M(E₀) dE₀` over the π⁰
//! energies compatible with its own. The transverse and longitudinal
//! shapes, each normalized, are mixed in the fraction of `σv` each
//! density carries.
//!
//! The currents have one common pion mass, the isospin limit, while the
//! decays use the physical masses. A produced pion of energy `E` decays at
//! the same Lorentz factor `E/m` with its physical mass, so rest stays
//! rest and the decay sees the production velocity; the energy budget is
//! then off by `γ(m_π± − m)` per charged pion.
//!
//! # Decays
//!
//! Muons and charged pions decay through [`super::polarized_muon`], with
//! the muon of `π → μν` fully polarized. Their radiative photons come from
//! Hazma's unpolarized [`super::photon_muon`] and [`super::photon_pion`]
//! kernels. A π⁰ gives the `γγ` box `2 BR(γγ)/(2γβ m/2)`, in full double
//! precision rather than the `f32` of `photon_pion`'s legacy box. The
//! Dalitz decay `π⁰ → γe⁺e⁻` is not included.

use crate::constants::pdg::{BR_PI0_TO_A_A, MASS_PI, MASS_PI0};

use super::axial_three_pi::Channel as PionChannel;
use super::photon_muon::{dnde_photon_muon, photon_endpoint};
use super::photon_pion::{charged_pion_photon_endpoint, dnde_photon_charged_pion};
use super::polarized_muon::{
    Daughter, MuonDecay, PionDecay, box_density, box_fraction, integrate, threshold_gamma,
};

/// Relative tolerance of every integral over a parent's energy and of a
/// photon bin. The integrands are themselves adaptive integrals at
/// 1e-10 or better.
pub(super) const FOLD_EPSREL: f64 = 1e-8;

/// Where a yield is evaluated, in lab energies in MeV.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Probe {
    /// The density `dN/dE` in MeV⁻¹ at one energy.
    Density(f64),
    /// The count in the bin `[low, high)`.
    Bin(f64, f64),
}

impl Probe {
    /// Return the energies at which a parent integral can break.
    pub(super) fn edges(self) -> [f64; 2] {
        match self {
            Self::Density(energy) => [energy, energy],
            Self::Bin(low, high) => [low, high],
        }
    }
}

/// Return a muon's yield of `daughter` at `probe`.
pub(super) fn muon_response(
    daughter: Daughter,
    probe: Probe,
    energy: f64,
    polarization: f64,
) -> f64 {
    let decay = MuonDecay::PDG;
    match probe {
        Probe::Density(e) => decay.dnde(daughter, e, energy, polarization),
        Probe::Bin(low, high) => decay.counts(daughter, low, high, energy, polarization),
    }
}

/// Return a charged pion's yield of `daughter` at `probe`.
pub(super) fn pion_response(daughter: Daughter, probe: Probe, energy: f64) -> f64 {
    let decay = PionDecay::PDG;
    match probe {
        Probe::Density(e) => decay.dnde(daughter, e, energy),
        Probe::Bin(low, high) => decay.counts(daughter, low, high, energy),
    }
}

/// Return a photon yield at `probe` from the density `dnde`, which
/// vanishes above `endpoint`. A bin integrates it.
pub(super) fn photon_response(probe: Probe, dnde: impl Fn(f64) -> f64, endpoint: f64) -> f64 {
    match probe {
        Probe::Density(energy) => dnde(energy),
        Probe::Bin(low, high) => {
            let high = high.min(endpoint);
            if low >= high {
                return 0.0;
            }
            let mut integrand = |energy: f64| dnde(energy);
            integrate(&mut integrand, low, high, &[], FOLD_EPSREL)
        }
    }
}

/// Return the `γγ` yield at `probe` of a π⁰ with Lorentz factor `gamma`
/// and `gamma_beta = γβ`. At rest the two photons are a line, which only
/// a bin can hold.
pub(super) fn neutral_pion_photons(probe: Probe, gamma: f64, gamma_beta: f64) -> f64 {
    let (count, energy) = (2.0 * BR_PI0_TO_A_A, MASS_PI0 / 2.0);
    if gamma_beta == 0.0 {
        return match probe {
            Probe::Density(_) => 0.0,
            Probe::Bin(low, high) => {
                if low <= energy && energy < high {
                    count
                } else {
                    0.0
                }
            }
        };
    }
    count
        * match probe {
            Probe::Density(e) => box_density(e, energy, 0.0, gamma, gamma_beta),
            Probe::Bin(low, high) => box_fraction(low, high, energy, 0.0, gamma, gamma_beta),
        }
}

/// The energy densities of the three pions of one LO final state.
///
/// Every quantity is written in a pion's kinetic energy `u = E − m` and
/// the excess `q − 3m` over threshold, never as a difference of total
/// energies near `m`, so the densities stay normalized arbitrarily close to
/// threshold, where the whole Dalitz plot shrinks to a point.
#[derive(Clone, Copy, Debug)]
pub struct ThreePionParents {
    pub(super) channel: PionChannel,
    /// Invariant energy `q` in MeV, above `3m`.
    energy: f64,
    /// Common production pion mass `m` in MeV, positive.
    pub(super) pion_mass: f64,
    /// `q − 3m` in MeV.
    excess: f64,
    /// Fraction of the channel's `σv` from its transverse density, in
    /// `[0, 1]`; zero for the neutral channel.
    transverse_fraction: f64,
    transverse_norm: f64,
    longitudinal_norm: f64,
}

impl ThreePionParents {
    /// Build the parent densities of `channel` at invariant `energy` with
    /// production `pion_mass`, both in MeV.
    pub fn new(
        channel: PionChannel,
        energy: f64,
        pion_mass: f64,
        transverse_fraction: f64,
    ) -> Self {
        let mut parents = Self {
            channel,
            energy,
            pion_mass,
            excess: energy - 3.0 * pion_mass,
            transverse_fraction,
            transverse_norm: 1.0,
            longitudinal_norm: 1.0,
        };
        let norm = |weight: fn(&Self, f64) -> f64| {
            let mut integrand = |u: f64| weight(&parents, u) * parents.partner_length(u);
            integrate(&mut integrand, 0.0, parents.max_kinetic(), &[], 1e-12)
        };
        let (transverse_norm, longitudinal_norm) = (
            norm(Self::transverse_weight),
            norm(Self::longitudinal_weight),
        );
        parents.transverse_norm = transverse_norm;
        parents.longitudinal_norm = longitudinal_norm;
        parents
    }

    /// Return the largest kinetic energy of any one pion in MeV,
    /// `(q − 3m)(q + m)/(2q)`.
    pub fn max_kinetic(&self) -> f64 {
        let (q, m) = (self.energy, self.pion_mass);
        self.excess * (q + m) / (2.0 * q)
    }

    /// Return the length `L(u)` in MeV of a partner's energy range,
    /// `p √(1 − 4m²/s)` with `p² = u(2m + u)` and
    /// `s − 4m² = 2q(u_max − u)`.
    fn partner_length(&self, kinetic: f64) -> f64 {
        let (q, m) = (self.energy, self.pion_mass);
        let pair_mass2 = (q - m).powi(2) - 2.0 * q * kinetic;
        let gap = (2.0 * q * (self.max_kinetic() - kinetic) / pair_mass2).max(0.0);
        (kinetic * (2.0 * m + kinetic)).max(0.0).sqrt() * gap.sqrt()
    }

    /// Return `s − m² = q(q − 2m − 2u)` in MeV², with `s` the invariant mass
    /// squared of the two partners of a pion of kinetic energy `u`.
    fn pair_excess(&self, kinetic: f64) -> f64 {
        let (q, m) = (self.energy, self.pion_mass);
        q * (q - 2.0 * m - 2.0 * kinetic)
    }

    /// Return the unnormalized transverse weight at π⁰ kinetic energy `u`:
    /// `|p₀|² = u(2m + u)` for the charged channel, and zero for the
    /// neutral one.
    fn transverse_weight(&self, kinetic: f64) -> f64 {
        match self.channel {
            PionChannel::Charged => kinetic * (2.0 * self.pion_mass + kinetic),
            PionChannel::Neutral => 0.0,
        }
    }

    /// Return the unnormalized longitudinal weight at π⁰ kinetic energy
    /// `u`: `(s − m²)²` for the charged channel and one for the neutral one.
    fn longitudinal_weight(&self, kinetic: f64) -> f64 {
        match self.channel {
            PionChannel::Charged => self.pair_excess(kinetic).powi(2),
            PionChannel::Neutral => 1.0,
        }
    }

    /// Return `∫ M dE₀` over the π⁰ kinetic energies compatible with a
    /// charged pion of kinetic energy `u`, `u_a, u_b = (q − 3m − u ∓ L)/2`.
    ///
    /// The cubic primitives are factored through `u_b − u_a = L`:
    /// `∫|p₀|² = L[m(u_a + u_b) + (u_a² + u_a u_b + u_b²)/3]` and, with
    /// `A = s − m²` linear in `u`, `∫A² = L(A_a² + A_a A_b + A_b²)/3`.
    fn weight_integral(&self, kinetic: f64) -> f64 {
        let length = self.partner_length(kinetic);
        let (low, high) = self.partner_range(kinetic, length);
        let transverse = self.transverse_integral(kinetic, length);
        let longitudinal = match self.channel {
            PionChannel::Charged => {
                let (a, b) = (self.pair_excess(low), self.pair_excess(high));
                length * (a * a + a * b + b * b) / 3.0
            }
            PionChannel::Neutral => length,
        };
        self.mix(transverse, longitudinal)
    }

    /// Return the π⁰ kinetic energies `(u_a, u_b)` compatible with a
    /// charged pion of kinetic energy `u`, whose partners span `length`.
    fn partner_range(&self, kinetic: f64, length: f64) -> (f64, f64) {
        let sum = self.excess - kinetic;
        ((sum - length) / 2.0, (sum + length) / 2.0)
    }

    /// Return `∫|p₀|² dE₀` over the π⁰ kinetic energies compatible with a
    /// charged pion of kinetic energy `u`, whose partners span `length`.
    fn transverse_integral(&self, kinetic: f64, length: f64) -> f64 {
        let (low, high) = self.partner_range(kinetic, length);
        let sum = self.excess - kinetic;
        length * (self.pion_mass * sum + (low * low + low * high + high * high) / 3.0)
    }

    /// Return the unnormalized `P₂` moment `∫|p₀|² P₂(cos θ₀) dE₀` of a
    /// charged pion of kinetic energy `u`, with `θ₀` the angle between it
    /// and the π⁰.
    ///
    /// The transverse current `ε·p₀` aligns the π⁰ with the mediator's
    /// polarization, and a charged pion inherits the alignment through
    /// `P₂(cos θ₀)` (the addition theorem). Across the partner range
    /// `E₀ = (q − E)/2 + tL/2` with `t ∈ [−1, 1]`, momentum conservation
    /// gives `p₀ cos θ₀ = −(p + K t)/2` with `p` the charged pion's
    /// momentum and `K = (q − E)v`, `v = √(1 − 4m²/s)`, so
    /// `∫(p₀ cos θ₀)² dE₀ = L(p² + K²/3)/4` with no cancellation as
    /// `p → 0`.
    fn alignment_integral(&self, kinetic: f64) -> f64 {
        let (q, m) = (self.energy, self.pion_mass);
        let length = self.partner_length(kinetic);
        let pair_mass2 = (q - m).powi(2) - 2.0 * q * kinetic;
        let speed2 = (2.0 * q * (self.max_kinetic() - kinetic) / pair_mass2).max(0.0);
        let momentum2 = kinetic * (2.0 * m + kinetic);
        let recoil2 = (q - m - kinetic).powi(2) * speed2;
        let parallel = length * (momentum2 + recoil2 / 3.0) / 4.0;
        (3.0 * parallel - self.transverse_integral(kinetic, length)) / 2.0
    }

    /// Combine unnormalized transverse and longitudinal pieces in the
    /// channel's proportions.
    fn mix(&self, transverse: f64, longitudinal: f64) -> f64 {
        let fraction = self.transverse_fraction;
        let mut total = 0.0;
        if fraction != 0.0 {
            total += fraction * transverse / self.transverse_norm;
        }
        if fraction != 1.0 {
            total += (1.0 - fraction) * longitudinal / self.longitudinal_norm;
        }
        total
    }

    /// Return the normalized density in MeV⁻¹ over kinetic energy of the π⁰
    /// of the charged channel, or of each π⁰ of the neutral one.
    pub fn spectator_density(&self, kinetic: f64) -> f64 {
        let weight = self.mix(
            self.transverse_weight(kinetic),
            self.longitudinal_weight(kinetic),
        );
        weight * self.partner_length(kinetic)
    }

    /// Return the normalized density in MeV⁻¹ over kinetic energy of each
    /// charged pion of the charged channel.
    pub fn charged_density(&self, kinetic: f64) -> f64 {
        self.weight_integral(kinetic)
    }

    /// Return the alignment density in MeV⁻¹ of the π⁰ of the charged
    /// channel, or of each π⁰ of the neutral one, over kinetic energy.
    ///
    /// A pion of a mediator in helicity `λ` has the angular density
    /// `[f + t_λ h P₂(cos θ)]/2` about the quantization axis, with `f` the
    /// energy density, `h` this alignment density, `t₀ = 2` and
    /// `t_± = −1`. Only the transverse current aligns: it is `ε·p₀`, so
    /// the π⁰ has `h = f` within it, while the longitudinal current is
    /// spin zero and leaves every pion isotropic.
    pub fn spectator_alignment(&self, kinetic: f64) -> f64 {
        self.mix(self.transverse_weight(kinetic), 0.0) * self.partner_length(kinetic)
    }

    /// Return the alignment density in MeV⁻¹ of each charged pion of the
    /// charged channel over kinetic energy, as
    /// [`Self::spectator_alignment`] defines it.
    pub fn charged_alignment(&self, kinetic: f64) -> f64 {
        match self.channel {
            PionChannel::Charged => self.mix(self.alignment_integral(kinetic), 0.0),
            PionChannel::Neutral => 0.0,
        }
    }

    /// Integrate `density` times `response` over a pion's kinetic energy,
    /// with the kinetic energies at which `response` breaks for `probe`.
    /// Each feature is a decay daughter's `(energy, mass)` in its parent's
    /// rest frame, whose boost crosses a probe edge at one Lorentz factor.
    fn fold(
        &self,
        density: impl Fn(&Self, f64) -> f64,
        response: impl Fn(f64) -> f64,
        probe: Probe,
        features: &[(f64, f64)],
    ) -> f64 {
        let mut points = Vec::with_capacity(2 * features.len());
        for edge in probe.edges() {
            if edge > 0.0 {
                for &(energy, mass) in features {
                    let gamma = threshold_gamma(edge, energy, mass);
                    points.push(self.pion_mass * (gamma - 1.0));
                }
            }
        }
        let mut integrand = |kinetic: f64| density(self, kinetic) * response(kinetic);
        integrate(
            &mut integrand,
            0.0,
            self.max_kinetic(),
            &points,
            FOLD_EPSREL,
        )
    }

    /// Return the physical-mass energy in MeV at which a charged pion
    /// produced with `kinetic` energy decays, at the same Lorentz factor.
    fn charged_decay_energy(&self, kinetic: f64) -> f64 {
        (self.pion_mass + kinetic) * (MASS_PI / self.pion_mass)
    }

    /// Return the π⁰ photon yield at `probe`, every π⁰ of the state
    /// counted.
    fn neutral_pion_photons(&self, probe: Probe) -> f64 {
        let m = self.pion_mass;
        let response = |kinetic: f64| {
            let gamma_beta = (kinetic * (2.0 * m + kinetic)).sqrt() / m;
            neutral_pion_photons(probe, 1.0 + kinetic / m, gamma_beta)
        };
        let multiplicity = match self.channel {
            PionChannel::Charged => 1.0,
            PionChannel::Neutral => 3.0,
        };
        let features = [(MASS_PI0 / 2.0, 0.0)];
        multiplicity * self.fold(Self::spectator_density, response, probe, &features)
    }

    /// Return the yield of one charged pion's `daughter` at `probe`.
    fn charged_pion_daughters(&self, daughter: Daughter, probe: Probe) -> f64 {
        let decay = PionDecay::PDG;
        let mass = decay.muon.daughter_mass(daughter);
        let features = decay.rest_features(daughter).map(|energy| (energy, mass));
        let response =
            |kinetic: f64| pion_response(daughter, probe, self.charged_decay_energy(kinetic));
        self.fold(Self::charged_density, response, probe, &features)
    }

    /// Return the radiative photon yield of one charged pion at `probe`.
    fn charged_pion_photons(&self, probe: Probe) -> f64 {
        let response = |kinetic: f64| {
            let decay_energy = self.charged_decay_energy(kinetic);
            photon_response(
                probe,
                |photon| dnde_photon_charged_pion(photon, decay_energy),
                charged_pion_photon_endpoint(decay_energy),
            )
        };
        // The photon edge rides on the pion as a massless daughter's box does.
        let features = [(charged_pion_photon_endpoint(MASS_PI), 0.0)];
        self.fold(Self::charged_density, response, probe, &features)
    }
}

/// A source of stable-particle yields, each evaluated at a [`Probe`].
pub trait Yields {
    /// Return the photon yield at `probe`.
    fn photons(&self, probe: Probe) -> f64;

    /// Return the `[e⁺, e⁻]` yields at `probe`.
    fn charged_leptons(&self, probe: Probe) -> [f64; 2];

    /// Return the `ν + ν̄` yields `[e, μ, τ]` per flavor at `probe`.
    fn neutrinos(&self, probe: Probe) -> [f64; 3];
}

/// A final state whose particles decay.
#[derive(Clone, Copy, Debug)]
pub enum Channel {
    /// `μ⁺μ⁻`, each muon of lab `muon_energy` in MeV with effective
    /// `polarization`.
    MuonPair {
        /// Each muon's energy in MeV, at least its mass.
        muon_energy: f64,
        /// Effective polarization in `[−1, 1]`.
        polarization: f64,
    },
    /// `π⁺π⁻π⁰` or `π⁰π⁰π⁰`.
    ThreePion(ThreePionParents),
}

impl Yields for Channel {
    fn photons(&self, probe: Probe) -> f64 {
        match self {
            Self::MuonPair { muon_energy, .. } => {
                let dnde = |photon: f64| dnde_photon_muon(photon, *muon_energy);
                2.0 * photon_response(probe, dnde, photon_endpoint(*muon_energy))
            }
            Self::ThreePion(parents) => match parents.channel {
                PionChannel::Charged => {
                    parents.neutral_pion_photons(probe) + 2.0 * parents.charged_pion_photons(probe)
                }
                PionChannel::Neutral => parents.neutral_pion_photons(probe),
            },
        }
    }

    fn charged_leptons(&self, probe: Probe) -> [f64; 2] {
        let lepton = match self {
            Self::MuonPair {
                muon_energy,
                polarization,
            } => muon_response(Daughter::ChargedLepton, probe, *muon_energy, *polarization),
            Self::ThreePion(parents) => match parents.channel {
                PionChannel::Charged => {
                    parents.charged_pion_daughters(Daughter::ChargedLepton, probe)
                }
                PionChannel::Neutral => 0.0,
            },
        };
        // CP: the μ⁻ (π⁻) gives the e⁻ the μ⁺ (π⁺) gives the e⁺.
        [lepton, lepton]
    }

    fn neutrinos(&self, probe: Probe) -> [f64; 3] {
        let per_parent = |daughter: Daughter| match self {
            Self::MuonPair {
                muon_energy,
                polarization,
            } => muon_response(daughter, probe, *muon_energy, *polarization),
            Self::ThreePion(parents) => match parents.channel {
                PionChannel::Charged => parents.charged_pion_daughters(daughter, probe),
                PionChannel::Neutral => 0.0,
            },
        };
        // Each of the two charged parents gives one flavor and the other
        // the antiflavor.
        [
            2.0 * per_parent(Daughter::ElectronNeutrino),
            2.0 * per_parent(Daughter::MuonNeutrino),
            0.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::kernels::axial_three_pi::{ThreePion, angular_rule};

    fn assert_close(actual: f64, expected: f64, rtol: f64) {
        let error = (actual - expected).abs() / expected.abs();
        assert!(
            error <= rtol,
            "actual {actual:.17e}, expected {expected:.17e}, relative error {error:e} > {rtol:e}"
        );
    }

    fn charged(energy: f64, fraction: f64) -> ThreePionParents {
        ThreePionParents::new(PionChannel::Charged, energy, MASS_PI0, fraction)
    }

    /// The parent weights reproduce `ThreePion::spectral_functions`: with
    /// the measure `L(E₀) dE₀/(32π³)` and the currents' prefactors,
    /// `A_T = (16g²/3F²) ∫|p₀|² L` and
    /// `A_L = 4g²m⁴/(F²q²(q² − m²)²) ∫(s − m²)² L`, and the neutral
    /// `A_L = 4g²q²m⁴/(F²(q² − m²)²) ∫L/3!`. The two quadratures differ,
    /// so 1e-10.
    #[test]
    fn weights_reproduce_the_spectral_functions() {
        let (q, m, f, g) = (520.0, MASS_PI0, 92.2, 0.7);
        let rule = angular_rule(64);
        let charged_parents = charged(q, 0.5);
        let current = ThreePion {
            channel: PionChannel::Charged,
            pion_mass: m,
            decay_constant: f,
        };
        let (transverse, longitudinal) = current.spectral_functions(q, g, &rule);
        let measure = 32.0 * PI.powi(3);
        let off_shell = (q * q - m * m).powi(2);
        assert_close(
            16.0 * g * g / (3.0 * f * f) * charged_parents.transverse_norm / measure,
            transverse,
            1e-10,
        );
        assert_close(
            4.0 * g * g * m.powi(4) / (f * f * q * q * off_shell)
                * charged_parents.longitudinal_norm
                / measure,
            longitudinal,
            1e-10,
        );

        let neutral_parents = ThreePionParents::new(PionChannel::Neutral, q, m, 0.0);
        let neutral = ThreePion {
            channel: PionChannel::Neutral,
            ..current
        };
        let (_, longitudinal) = neutral.spectral_functions(q, g, &rule);
        assert_close(
            4.0 * g * g * q * q * m.powi(4) / (f * f * off_shell)
                * neutral_parents.longitudinal_norm
                / (6.0 * measure),
            longitudinal,
            1e-10,
        );
    }

    /// Assert that every parent density of `parents` integrates to one.
    fn assert_normalized(parents: &ThreePionParents, rtol: f64) {
        let top = parents.max_kinetic();
        let mut spectator = |u: f64| parents.spectator_density(u);
        assert_close(integrate(&mut spectator, 0.0, top, &[], 1e-12), 1.0, rtol);
        if parents.channel == PionChannel::Charged {
            let mut each = |u: f64| parents.charged_density(u);
            assert_close(integrate(&mut each, 0.0, top, &[], 1e-12), 1.0, rtol);
        }
    }

    /// Every parent density is normalized, and the charged pions'
    /// marginal integrates to one through its closed-form primitives.
    #[test]
    fn parent_densities_are_normalized() {
        for parents in [
            charged(450.0, 0.0),
            charged(450.0, 0.37),
            charged(600.0, 1.0),
            ThreePionParents::new(PionChannel::Neutral, 500.0, MASS_PI0, 0.0),
        ] {
            assert_normalized(&parents, 1e-11);
        }
    }

    /// The densities stay normalized as the Dalitz plot shrinks to a
    /// point. Written in total energies, the charged pions' cubic
    /// primitives cancelled catastrophically here and gave 1.0126 e⁺ per
    /// annihilation at `1e-6` MeV above threshold and 2.3e5 at `1e-9` MeV.
    #[test]
    fn parent_densities_are_normalized_at_threshold() {
        for excess in [1e-9, 1e-6, 1e-3] {
            let q = 3.0 * MASS_PI0 + excess;
            for fraction in [0.0, 0.5, 1.0] {
                assert_normalized(&charged(q, fraction), 1e-10);
                let pions = Channel::ThreePion(charged(q, fraction));
                let [positrons, _] = pions.charged_leptons(Probe::Bin(0.0, 1e4));
                assert_close(positrons, 1.0, 1e-9);
            }
            let neutral = ThreePionParents::new(PionChannel::Neutral, q, MASS_PI0, 0.0);
            assert_normalized(&neutral, 1e-10);
        }
    }

    /// The charged pion's alignment is the π⁰-weighted mean of `P₂` of the
    /// angle between the two pions' momenta.
    #[test]
    fn charged_pion_photons_reach_their_endpoint() {
        // Near the endpoint only the fastest charged pions radiate; the oracle
        // folds over them alone. The fold's tolerance is 1e-8.
        let parents = charged(590.0, 0.6);
        let mass = parents.pion_mass;
        let top = parents.charged_decay_energy(parents.max_kinetic());
        let endpoint = charged_pion_photon_endpoint(top);
        let rest_edge = charged_pion_photon_endpoint(MASS_PI);
        for gap in [1e-3, 1e-6] {
            let low = endpoint * (1.0 - gap);
            for probe in [
                Probe::Density(low),
                Probe::Bin(low, endpoint * (1.0 + 1e-3)),
            ] {
                let from = mass * (threshold_gamma(low, rest_edge, 0.0) - 1.0);
                let mut integrand = |kinetic: f64| {
                    let decay = parents.charged_decay_energy(kinetic);
                    let dnde = |photon: f64| dnde_photon_charged_pion(photon, decay);
                    parents.charged_density(kinetic)
                        * photon_response(probe, dnde, charged_pion_photon_endpoint(decay))
                };
                let expected = integrate(&mut integrand, from, parents.max_kinetic(), &[], 1e-10);
                assert!(expected > 0.0, "{probe:?}");
                assert_close(parents.charged_pion_photons(probe), expected, 1e-7);
            }
        }
    }

    #[test]
    fn charged_alignment_is_the_mean_p2_between_pions() {
        // Brute force: the angle between the charged pion and the π⁰ from
        // the third pion's momentum, |p₋|² = |p₊|² + |p₀|² + 2p₊·p₀.
        for energy in [450.0, 600.0] {
            let parents = charged(energy, 1.0);
            let (q, m) = (parents.energy, parents.pion_mass);
            for step in 1..20 {
                let kinetic = parents.max_kinetic() * f64::from(step) / 20.0;
                let length = parents.partner_length(kinetic);
                let (low, high) = parents.partner_range(kinetic, length);
                let charged2 = kinetic * (2.0 * m + kinetic);
                let mut integrand = |neutral: f64| {
                    let neutral2 = neutral * (2.0 * m + neutral);
                    let third = q - 3.0 * m - kinetic - neutral;
                    let third2 = third * (2.0 * m + third);
                    let cosine =
                        (third2 - charged2 - neutral2) / (2.0 * (charged2 * neutral2).sqrt());
                    neutral2 * (3.0 * cosine * cosine - 1.0) / 2.0
                };
                let expected = integrate(&mut integrand, low, high, &[], 1e-12);
                let scale = parents.transverse_integral(kinetic, length);
                assert!((parents.alignment_integral(kinetic) - expected).abs() <= 1e-10 * scale);
                // P₂ lies in [−1/2, 1], and the π⁰ is fully aligned.
                let (density, alignment) = (
                    parents.charged_density(kinetic),
                    parents.charged_alignment(kinetic),
                );
                assert!(-0.5 * density <= alignment && alignment <= density);
                assert_close(
                    parents.spectator_alignment(kinetic),
                    parents.spectator_density(kinetic),
                    1e-15,
                );
            }
        }
    }

    /// The three pions share the invariant energy on average:
    /// `⟨u₀⟩ + 2⟨u±⟩ + 3m = q`.
    #[test]
    fn parent_energies_sum_to_the_invariant_energy() {
        let parents = charged(530.0, 0.6);
        let top = parents.max_kinetic();
        let mut spectator = |u: f64| u * parents.spectator_density(u);
        let mut each = |u: f64| u * parents.charged_density(u);
        let kinetic = integrate(&mut spectator, 0.0, top, &[], 1e-12)
            + 2.0 * integrate(&mut each, 0.0, top, &[], 1e-12);
        assert_close(kinetic + 3.0 * MASS_PI0, 530.0, 1e-12);
    }

    /// A μ⁺μ⁻ pair carries two of each lepton flavor's neutrinos, one e⁺
    /// and one e⁻; π⁺π⁻π⁰ carries one e⁺, one e⁻, two e-flavor and
    /// `4 BR(μν)` μ-flavor neutrinos, and `2 BR(γγ)` π⁰ photons. A bin past
    /// every endpoint counts them all.
    #[test]
    fn channel_multiplicities() {
        let everything = Probe::Bin(0.0, 1e4);
        let pair = Channel::MuonPair {
            muon_energy: 300.0,
            polarization: 0.4,
        };
        for value in pair.charged_leptons(everything) {
            assert_close(value, 1.0, 1e-10);
        }
        let [electron, muon, tau] = pair.neutrinos(everything);
        assert_close(electron, 2.0, 1e-10);
        assert_close(muon, 2.0, 1e-10);
        assert_eq!(tau, 0.0);

        let pions = Channel::ThreePion(charged(500.0, 0.3));
        for value in pions.charged_leptons(everything) {
            assert_close(value, 1.0, 1e-7);
        }
        let br = PionDecay::PDG.br_muon;
        let [electron, muon, _] = pions.neutrinos(everything);
        assert_close(electron, 2.0, 1e-7);
        assert_close(muon, 4.0 * br, 1e-7);

        let neutral = Channel::ThreePion(ThreePionParents::new(
            PionChannel::Neutral,
            500.0,
            MASS_PI0,
            0.0,
        ));
        let gamma_count = neutral.photons(Probe::Bin(1e-3, 1e4));
        assert_close(gamma_count, 6.0 * BR_PI0_TO_A_A, 1e-7);
    }

    /// Narrow bins reduce to the densities, for each product of each
    /// cascading channel. A relative width of 1e-5 leaves a second-order
    /// error near 1e-10, under the folds' 1e-8 tolerance; 1e-6 allows for
    /// the curvature of the folded densities.
    #[test]
    fn counts_round_trip_by_bin_width() {
        let channels = [
            Channel::MuonPair {
                muon_energy: 250.0,
                polarization: -0.5,
            },
            Channel::ThreePion(charged(480.0, 0.8)),
        ];
        for channel in channels {
            for energy in [12.0, 45.0, 88.0] {
                let width = energy * 1e-5;
                let low = energy - width / 2.0;
                let bin = Probe::Bin(low, low + width);
                let density = Probe::Density(energy);
                assert_close(
                    channel.charged_leptons(bin)[0] / width,
                    channel.charged_leptons(density)[0],
                    1e-6,
                );
                assert_close(
                    channel.neutrinos(bin)[1] / width,
                    channel.neutrinos(density)[1],
                    1e-6,
                );
                assert_close(channel.photons(bin) / width, channel.photons(density), 1e-6);
            }
        }
    }
}
