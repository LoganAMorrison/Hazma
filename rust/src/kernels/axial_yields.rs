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
    Daughter, MuonDecay, PionDecay, box_density, box_fraction, integrate, momentum, threshold_gamma,
};

/// Relative tolerance of every integral over a parent's energy and of a
/// photon bin. The integrands are themselves adaptive integrals at
/// 1e-10 or better.
const FOLD_EPSREL: f64 = 1e-8;

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
    fn edges(self) -> [f64; 2] {
        match self {
            Self::Density(energy) => [energy, energy],
            Self::Bin(low, high) => [low, high],
        }
    }
}

/// Return a muon's yield of `daughter` at `probe`.
fn muon_response(daughter: Daughter, probe: Probe, energy: f64, polarization: f64) -> f64 {
    let decay = MuonDecay::PDG;
    match probe {
        Probe::Density(e) => decay.dnde(daughter, e, energy, polarization),
        Probe::Bin(low, high) => decay.counts(daughter, low, high, energy, polarization),
    }
}

/// Return a charged pion's yield of `daughter` at `probe`.
fn pion_response(daughter: Daughter, probe: Probe, energy: f64) -> f64 {
    let decay = PionDecay::PDG;
    match probe {
        Probe::Density(e) => decay.dnde(daughter, e, energy),
        Probe::Bin(low, high) => decay.counts(daughter, low, high, energy),
    }
}

/// Return a photon yield at `probe` from the density `dnde`, which
/// vanishes above `endpoint`. A bin integrates it.
fn photon_response(probe: Probe, dnde: impl Fn(f64) -> f64, endpoint: f64) -> f64 {
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
fn neutral_pion_photons(probe: Probe, gamma: f64, gamma_beta: f64) -> f64 {
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
#[derive(Clone, Copy, Debug)]
pub struct ThreePionParents {
    channel: PionChannel,
    /// Invariant energy `q` in MeV, above `3m`.
    energy: f64,
    /// Common production pion mass `m` in MeV, positive.
    pion_mass: f64,
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
            transverse_fraction,
            transverse_norm: 1.0,
            longitudinal_norm: 1.0,
        };
        let norm = |weight: fn(&Self, f64) -> f64| {
            let mut integrand = |e: f64| weight(&parents, e) * parents.partner_length(e);
            integrate(&mut integrand, pion_mass, parents.max_energy(), &[], 1e-12)
        };
        let (transverse_norm, longitudinal_norm) = (
            norm(Self::transverse_weight),
            norm(Self::longitudinal_weight),
        );
        parents.transverse_norm = transverse_norm;
        parents.longitudinal_norm = longitudinal_norm;
        parents
    }

    /// Return the largest energy of any one pion in MeV, `(q² − 3m²)/(2q)`.
    pub fn max_energy(&self) -> f64 {
        let (q, m) = (self.energy, self.pion_mass);
        (q * q - 3.0 * m * m) / (2.0 * q)
    }

    /// Return `s − m² = q² − 2qE` in MeV² for a pion of `energy`, with `s`
    /// the invariant mass squared of its two partners.
    fn pair_excess(&self, energy: f64) -> f64 {
        self.energy * (self.energy - 2.0 * energy)
    }

    /// Return the length `L(E)` in MeV of a partner's energy range.
    fn partner_length(&self, energy: f64) -> f64 {
        let m2 = self.pion_mass * self.pion_mass;
        let s = self.pair_excess(energy) + m2;
        momentum(energy, self.pion_mass) * (1.0 - 4.0 * m2 / s).max(0.0).sqrt()
    }

    /// Return the partner energy range `(q − E)/2 ∓ L(E)/2` in MeV.
    fn partner_range(&self, energy: f64) -> (f64, f64) {
        let (center, half) = (
            (self.energy - energy) / 2.0,
            self.partner_length(energy) / 2.0,
        );
        (center - half, center + half)
    }

    /// Return the unnormalized transverse weight at π⁰ `energy`: `|p₀|²`
    /// for the charged channel, and zero for the neutral one.
    fn transverse_weight(&self, energy: f64) -> f64 {
        match self.channel {
            PionChannel::Charged => (energy - self.pion_mass) * (energy + self.pion_mass),
            PionChannel::Neutral => 0.0,
        }
    }

    /// Return the unnormalized longitudinal weight at π⁰ `energy`:
    /// `(s − m²)²` for the charged channel and one for the neutral one.
    fn longitudinal_weight(&self, energy: f64) -> f64 {
        match self.channel {
            PionChannel::Charged => self.pair_excess(energy).powi(2),
            PionChannel::Neutral => 1.0,
        }
    }

    /// Return `∫ M dE₀` over `[low, high]`, by the primitives
    /// `E³/3 − m²E` of `|p₀|²` and `−(q² − 2qE)³/(6q)` of `(s − m²)²`.
    fn weight_integral(&self, low: f64, high: f64) -> f64 {
        let m2 = self.pion_mass * self.pion_mass;
        let transverse = |e: f64| e * (e * e / 3.0 - m2);
        let longitudinal = |e: f64| match self.channel {
            PionChannel::Charged => -self.pair_excess(e).powi(3) / (6.0 * self.energy),
            PionChannel::Neutral => e,
        };
        self.mix(
            transverse(high) - transverse(low),
            longitudinal(high) - longitudinal(low),
        )
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

    /// Return the normalized energy density in MeV⁻¹ of the π⁰ of the
    /// charged channel, or of each π⁰ of the neutral one.
    pub fn spectator_density(&self, energy: f64) -> f64 {
        let weight = self.mix(
            self.transverse_weight(energy),
            self.longitudinal_weight(energy),
        );
        weight * self.partner_length(energy)
    }

    /// Return the normalized energy density in MeV⁻¹ of each charged pion
    /// of the charged channel.
    pub fn charged_density(&self, energy: f64) -> f64 {
        let (low, high) = self.partner_range(energy);
        self.weight_integral(low, high)
    }

    /// Integrate `density` times `response` over a pion's energy, with
    /// the parent energies at which `response` breaks for `probe`. Each
    /// feature is a decay daughter's `(energy, mass)` in its parent's rest
    /// frame, whose boost crosses a probe edge at one Lorentz factor.
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
                    points.push(self.pion_mass * threshold_gamma(edge, energy, mass));
                }
            }
        }
        let mut integrand = |energy: f64| density(self, energy) * response(energy);
        integrate(
            &mut integrand,
            self.pion_mass,
            self.max_energy(),
            &points,
            FOLD_EPSREL,
        )
    }

    /// Return the physical-mass energy in MeV at which a charged pion
    /// produced with `energy` decays, at the same Lorentz factor.
    fn charged_decay_energy(&self, energy: f64) -> f64 {
        energy * (MASS_PI / self.pion_mass)
    }

    /// Return the π⁰ photon yield at `probe`, every π⁰ of the state
    /// counted.
    fn neutral_pion_photons(&self, probe: Probe) -> f64 {
        let response = |energy: f64| {
            let gamma = energy / self.pion_mass;
            let gamma_beta = momentum(energy, self.pion_mass) / self.pion_mass;
            neutral_pion_photons(probe, gamma, gamma_beta)
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
            |energy: f64| pion_response(daughter, probe, self.charged_decay_energy(energy));
        self.fold(Self::charged_density, response, probe, &features)
    }

    /// Return the radiative photon yield of one charged pion at `probe`.
    fn charged_pion_photons(&self, probe: Probe) -> f64 {
        let response = |energy: f64| {
            let decay_energy = self.charged_decay_energy(energy);
            photon_response(
                probe,
                |photon| dnde_photon_charged_pion(photon, decay_energy),
                charged_pion_photon_endpoint(decay_energy),
            )
        };
        self.fold(Self::charged_density, response, probe, &[])
    }
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

impl Channel {
    /// Return the photon yield at `probe`.
    pub fn photons(&self, probe: Probe) -> f64 {
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

    /// Return the `[e⁺, e⁻]` yields at `probe`.
    pub fn charged_leptons(&self, probe: Probe) -> [f64; 2] {
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

    /// Return the `ν + ν̄` yields `[e, μ, τ]` per flavor at `probe`.
    pub fn neutrinos(&self, probe: Probe) -> [f64; 3] {
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
            let (low, high) = (parents.pion_mass, parents.max_energy());
            let mut spectator = |e: f64| parents.spectator_density(e);
            assert_close(integrate(&mut spectator, low, high, &[], 1e-12), 1.0, 1e-11);
            if parents.channel == PionChannel::Charged {
                let mut each = |e: f64| parents.charged_density(e);
                assert_close(integrate(&mut each, low, high, &[], 1e-12), 1.0, 1e-11);
            }
        }
    }

    /// The three pions share the invariant energy on average:
    /// `⟨E₀⟩ + 2⟨E±⟩ = q`.
    #[test]
    fn parent_energies_sum_to_the_invariant_energy() {
        let parents = charged(530.0, 0.6);
        let (low, high) = (parents.pion_mass, parents.max_energy());
        let mut spectator = |e: f64| e * parents.spectator_density(e);
        let mut each = |e: f64| e * parents.charged_density(e);
        let total = integrate(&mut spectator, low, high, &[], 1e-12)
            + 2.0 * integrate(&mut each, low, high, &[], 1e-12);
        assert_close(total, 530.0, 1e-11);
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
