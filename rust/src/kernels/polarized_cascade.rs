//! Lab-frame yields of a polarized axial-vector mediator's decays.
//!
//! The kernel behind `AxialVectorMediator.mediator_decay_yields` and
//! `mediator_decay_counts`: a mediator of mass `M`, moving with lab energy
//! `E_A > M`, decays into `e⁺e⁻`, `νν̄`, `μ⁺μ⁻` or `π⁺π⁻π⁰`, and each yield
//! is per decay into that state, with every particle of it counted.
//!
//! # Helicity density and decay matrix
//!
//! The mediator's spin state is a 3×3 helicity density `ρ` along its
//! flight direction `ẑ`, in the basis `λ = +1, 0, −1`. A daughter's lab
//! energy is invariant under rotations about `ẑ`, so the off-diagonal
//! elements, which carry `e^{i(λ−λ′)φ}`, integrate away and only the
//! fractions `F_λ = ρ_λλ` reach these yields. A spin-one parent's decay
//! matrix `D(n̂)` over a daughter's rest-frame direction has rank at most
//! two in `n̂`: with the spin-one matrices `S` and `P_n` the projector onto
//! `n̂`,
//!
//! `D(n̂) = [1 + g S·n̂ + h (3P_n − 1)]/(4π)`,
//!
//! which integrates over solid angle to the identity and, for the
//! diagonal `ρ`, gives the angular density
//! `W(cos θ) = [1 + g (F₊ − F₋) P₁(cos θ) + h (3F₀ − 1) P₂(cos θ)]/2`
//! (an [`Analyzer`]). The vector power `g` needs a parity-violating decay;
//! the tensor power `h` is the alignment, and `F₀ = 1/3` with `F₊ = F₋`
//! gives the isotropic decay whatever the analyzer.
//!
//! # Fermion pairs
//!
//! For `Z′_μ f̄γ^μ(g_V + g_Aγ₅)f` the helicity amplitudes of the fermion
//! along `n̂` and the antifermion along `−n̂` are, in units of `M`,
//! `a = g_V + βg_A` for `(+½, −½)`, `b = g_V − βg_A` for `(−½, +½)` and
//! `c₀ = √2 g_V m/M` for each equal pair, with `β` the fermion's speed in
//! the mediator frame. With Wigner's `d¹`, the fermion of a mediator in
//! helicity `λ` has the rate `d²_{λ1}a² + d²_{λ,−1}b² + 2d²_{λ0}c₀²`, the
//! helicity-weighted rate `d²_{λ1}a² − d²_{λ,−1}b²`, and the rate weighted
//! by its rest-frame spin along `θ̂`, `2c₀d_{λ0}(d_{λ1}a + d_{λ,−1}b)`; its
//! spin normal to the decay plane vanishes. These match explicit Dirac
//! spinors (`test/axial_vector_mediator/spinor_oracle.py`). CP gives the
//! antifermion at `−n̂` the same rest-frame spin vector, so its helicity is
//! minus the fermion's and its spin along `θ̂` is the fermion's.
//!
//! A muon's decay in flight sees only its lab helicity. The boost along
//! `ẑ` is not collinear with the muon, so the helicity mixes with the spin
//! along `θ̂`: a muon of mediator-frame energy `e`, momentum `k`, mass `m`
//! and spin components `s_L`, `s_T` has the lab helicity
//! `γ[s_L(k + βe cos θ) − βm s_T sin θ]/|p_lab|`, the time component of
//! its spin four-vector in the lab times `m/|p_lab|`. Hazma's radiative
//! muon photons stay unpolarized, as in [`super::axial_yields`].
//!
//! # Three pions
//!
//! The on-shell mediator has no longitudinal current, so `π⁺π⁻π⁰` comes
//! from the transverse current `ε·p₀` alone. Each pion has the energy
//! density `f` and the alignment density `h` of
//! [`ThreePionParents::spectator_alignment`] and
//! [`ThreePionParents::charged_alignment`], and the angular density
//! `[f + (3F₀ − 1) h P₂(cos θ)]/2`. A pion's lab energy density is the
//! integral of that over the mediator-frame energies whose boost window
//! holds it, and its decay products follow by folding that density with
//! the decay of a pion of the same lab energy, with the production and
//! decay masses of [`super::axial_yields`].
//!
//! # Boosts
//!
//! A daughter of mediator-frame energy `e` and momentum `k` reaches the lab
//! energy `γe + γβk cos θ`, linear in `cos θ`, so a stable two-body
//! daughter's density is `W(cos θ)/(γβk)` on its box and a bin holds the
//! integral of `W` over the cosines it spans, in closed form. Masses and
//! energies are in MeV, densities in MeV⁻¹ and counts are particles per
//! decay. The caller validates every input: a moving mediator above every
//! decay threshold, fractions that are a density's diagonal, and ordered
//! bin edges.

use std::f64::consts::SQRT_2;

use crate::constants::pdg::{MASS_MU, MASS_PI, MASS_PI0};

use super::axial_three_pi::Channel as PionChannel;
use super::axial_yields::{
    FOLD_EPSREL, Probe, ThreePionParents, Yields, muon_response, neutral_pion_photons,
    photon_response, pion_response,
};
use super::photon_muon::{dnde_photon_muon, photon_endpoint};
use super::photon_pion::{charged_pion_photon_endpoint, dnde_photon_charged_pion};
use super::polarized_muon::{
    Daughter, MuonDecay, PionDecay, boost_range, integrate, momentum, threshold_gamma,
};

/// Relative tolerance of a pion's lab energy density, the inner integral
/// of every three-pion fold, whose integrand is closed form.
const DENSITY_EPSREL: f64 = 1e-11;

/// A pion's mediator-frame energy and alignment densities over kinetic
/// energy, in MeV⁻¹.
type PionDensities = [fn(&ThreePionParents, f64) -> f64; 2];

/// The π⁰ of the charged channel, or each π⁰ of the neutral one.
const SPECTATOR: PionDensities = [
    ThreePionParents::spectator_density,
    ThreePionParents::spectator_alignment,
];

/// Each charged pion of the charged channel.
const CHARGED: PionDensities = [
    ThreePionParents::charged_density,
    ThreePionParents::charged_alignment,
];

/// The diagonal of a mediator's helicity density along its flight
/// direction: nonnegative fractions summing to one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Helicities {
    /// `F₊`, the fraction in helicity +1.
    pub plus: f64,
    /// `F₀`, the fraction in helicity 0.
    pub zero: f64,
    /// `F₋`, the fraction in helicity −1.
    pub minus: f64,
}

impl Helicities {
    /// Return the orientation `F₊ − F₋`, the weight of a daughter's
    /// `P₁(cos θ)`.
    fn orientation(self) -> f64 {
        self.plus - self.minus
    }

    /// Return the alignment `3F₀ − 1`, the weight of a daughter's
    /// `P₂(cos θ)`.
    fn alignment(self) -> f64 {
        3.0 * self.zero - 1.0
    }
}

/// Return the Legendre polynomial `P₂(c) = (3c² − 1)/2`.
fn legendre2(cosine: f64) -> f64 {
    (3.0 * cosine * cosine - 1.0) / 2.0
}

/// A two-body daughter's analyzing powers, which fix its angular density
/// about the mediator's flight direction (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Analyzer {
    /// The vector power `g`, the weight of `F₊ − F₋`.
    pub vector: f64,
    /// The tensor power `h`, the weight of `3F₀ − 1`.
    pub tensor: f64,
}

impl Analyzer {
    /// Return the angular density `W(cos θ)`, normalized over `[−1, 1]`.
    fn density(self, helicities: Helicities, cosine: f64) -> f64 {
        let vector = self.vector * helicities.orientation() * cosine;
        let tensor = self.tensor * helicities.alignment() * legendre2(cosine);
        (1.0 + vector + tensor) / 2.0
    }

    /// Return the integral of [`Self::density`] over `[from, to]`.
    fn integral(self, helicities: Helicities, from: f64, to: f64) -> f64 {
        // The antiderivatives of 1, P₁ and P₂ are c, c²/2 and (c³ − c)/2.
        let linear = (to - from) * (to + from) / 2.0;
        let cubic = (to - from) * ((to * to + to * from + from * from) - 1.0) / 2.0;
        let vector = self.vector * helicities.orientation() * linear;
        let tensor = self.tensor * helicities.alignment() * cubic;
        (to - from + vector + tensor) / 2.0
    }

    /// Return the analyzer of the partner along `−n̂`.
    fn reversed(self) -> Self {
        Self {
            vector: -self.vector,
            tensor: self.tensor,
        }
    }
}

/// The mediator's boost from its rest frame to the lab, along `ẑ`.
#[derive(Clone, Copy, Debug)]
pub struct Boost {
    gamma: f64,
    gamma_beta: f64,
}

impl Boost {
    /// Return the boost of a mediator of `mass` with lab `energy` above it,
    /// both in MeV.
    pub fn new(mass: f64, energy: f64) -> Self {
        Self {
            gamma: energy / mass,
            gamma_beta: momentum(energy, mass) / mass,
        }
    }

    fn beta(self) -> f64 {
        self.gamma_beta / self.gamma
    }

    /// Return the lab energy of a daughter of mediator-frame `energy` and
    /// `momentum` at `cosine` to `ẑ`.
    fn lab_energy(self, energy: f64, momentum: f64, cosine: f64) -> f64 {
        self.gamma
            .mul_add(energy, self.gamma_beta * momentum * cosine)
    }

    /// Return the cosine at which a daughter of mediator-frame `energy`
    /// and `momentum` reaches `lab_energy`, unclipped.
    fn cosine(self, lab_energy: f64, energy: f64, momentum: f64) -> f64 {
        (lab_energy - self.gamma * energy) / (self.gamma_beta * momentum)
    }
}

/// A mediator's decay into a fermion pair coupled as
/// `Z′_μ f̄γ^μ(g_V + g_Aγ₅)f`, with a positive width.
#[derive(Clone, Copy, Debug)]
pub struct PairDecay {
    /// Mediator mass `M` in MeV, above twice the fermion's.
    pub mediator_mass: f64,
    /// Fermion mass `m` in MeV.
    pub mass: f64,
    /// Vector coupling `g_V^f`.
    pub vector: f64,
    /// Axial coupling `g_A^f`.
    pub axial: f64,
}

impl PairDecay {
    /// Return each fermion's mediator-frame energy in MeV.
    fn energy(&self) -> f64 {
        self.mediator_mass / 2.0
    }

    /// Return each fermion's mediator-frame momentum in MeV.
    fn momentum(&self) -> f64 {
        momentum(self.energy(), self.mass)
    }

    /// Return the helicity amplitudes `(a, b, c₀)` in units of `M`.
    fn amplitudes(&self) -> (f64, f64, f64) {
        let speed = self.momentum() / self.energy();
        (
            self.axial.mul_add(speed, self.vector),
            (-self.axial).mul_add(speed, self.vector),
            SQRT_2 * self.vector * self.mass / self.mediator_mass,
        )
    }

    /// Return `∫ rate d(cos θ)` of each helicity, `2(a² + b² + 2c₀²)/3`.
    fn norm(&self) -> f64 {
        let (a, b, c0) = self.amplitudes();
        2.0 * (a * a + b * b + 2.0 * c0 * c0) / 3.0
    }

    /// Return the fermion's analyzer; the antifermion's is its reverse.
    pub fn analyzer(&self) -> Analyzer {
        let (a, b, c0) = self.amplitudes();
        let total = a * a + b * b + 2.0 * c0 * c0;
        Analyzer {
            vector: 1.5 * (a - b) * (a + b) / total,
            tensor: (2.0 * c0 * c0 - (a * a + b * b) / 2.0) / total,
        }
    }

    /// Return the fermion's angular density at `cosine` weighted by its
    /// rest-frame spin, `[W s_L, W s_T]`, along `n̂` and along `θ̂`.
    fn spin(&self, helicities: Helicities, cosine: f64) -> [f64; 2] {
        let (a, b, c0) = self.amplitudes();
        let Helicities { plus, zero, minus } = helicities;
        let (up, down) = ((1.0 + cosine).powi(2) / 4.0, (1.0 - cosine).powi(2) / 4.0);
        let sine2 = ((1.0 - cosine) * (1.0 + cosine)).max(0.0);
        let (a2, b2) = (a * a, b * b);
        let longitudinal = plus * (up * a2 - down * b2)
            + zero * sine2 * (a2 - b2) / 2.0
            + minus * (down * a2 - up * b2);
        let transverse = SQRT_2
            * c0
            * sine2.sqrt()
            * (minus * ((1.0 - cosine) * a + (1.0 + cosine) * b) / 2.0 + zero * cosine * (a - b)
                - plus * ((1.0 + cosine) * a + (1.0 - cosine) * b) / 2.0);
        let norm = self.norm();
        [longitudinal / norm, transverse / norm]
    }
}

/// A decaying mediator's final state.
#[derive(Clone, Copy, Debug)]
pub enum Decay {
    /// `e⁻e⁺`, both stable.
    ElectronPair(PairDecay),
    /// `νν̄` of the flavor `[e, μ, τ][index]`.
    NeutrinoPair(PairDecay, usize),
    /// `μ⁻μ⁺`.
    MuonPair(PairDecay),
    /// `π⁺π⁻π⁰` (or `π⁰π⁰π⁰`) with parents at the mediator mass.
    ThreePion(ThreePionParents),
}

/// One muon of a `μ⁻μ⁺` decay, as a function of its own cosine to `ẑ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Muon {
    Negative,
    Positive,
}

/// A mediator's decay in flight, with its helicity fractions.
#[derive(Clone, Copy, Debug)]
pub struct Cascade {
    /// The final state.
    pub decay: Decay,
    /// The mediator's boost to the lab, with `γβ > 0`.
    pub boost: Boost,
    /// The mediator's helicity fractions.
    pub helicities: Helicities,
}

impl Cascade {
    /// Return the yield at `probe` of a stable daughter of `pair` with the
    /// angular `analyzer`.
    fn stable(&self, pair: &PairDecay, analyzer: Analyzer, probe: Probe) -> f64 {
        let (energy, momentum) = (pair.energy(), pair.momentum());
        let boost = self.boost;
        match probe {
            Probe::Density(lab) => {
                let cosine = boost.cosine(lab, energy, momentum);
                if (-1.0..=1.0).contains(&cosine) {
                    analyzer.density(self.helicities, cosine) / (boost.gamma_beta * momentum)
                } else {
                    0.0
                }
            }
            Probe::Bin(low, high) => {
                let from = boost.cosine(low, energy, momentum).max(-1.0);
                let to = boost.cosine(high, energy, momentum).min(1.0);
                if from < to {
                    analyzer.integral(self.helicities, from, to)
                } else {
                    0.0
                }
            }
        }
    }

    /// Return a muon's angular density, lab energy and effective
    /// polarization (the μ⁻ helicity and minus the μ⁺'s) at `cosine`.
    fn muon_state(&self, pair: &PairDecay, muon: Muon, cosine: f64) -> (f64, f64, f64) {
        let (energy, momentum, mass) = (pair.energy(), pair.momentum(), pair.mass);
        let (analyzer, boost) = (pair.analyzer(), self.boost);
        // The μ⁺ at cos θ partners a μ⁻ at −cos θ with the same rest-frame
        // spin vector, so its helicity reverses and its spin along θ̂ does
        // not.
        let (weight, [longitudinal, transverse]) = match muon {
            Muon::Negative => (
                analyzer.density(self.helicities, cosine),
                pair.spin(self.helicities, cosine),
            ),
            Muon::Positive => {
                let [longitudinal, transverse] = pair.spin(self.helicities, -cosine);
                (
                    analyzer.density(self.helicities, -cosine),
                    [-longitudinal, transverse],
                )
            }
        };
        let lab_energy = boost.lab_energy(energy, momentum, cosine);
        let sine = ((1.0 - cosine) * (1.0 + cosine)).max(0.0).sqrt();
        let lab_momentum = (momentum * sine).hypot(
            boost
                .gamma
                .mul_add(momentum * cosine, boost.gamma_beta * energy),
        );
        let weighted_helicity = boost.gamma * momentum * longitudinal
            + boost.gamma_beta * (energy * cosine * longitudinal - mass * sine * transverse);
        let helicity = if weight > 0.0 && lab_momentum > 0.0 {
            (weighted_helicity / (weight * lab_momentum)).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let polarization = match muon {
            Muon::Negative => helicity,
            Muon::Positive => -helicity,
        };
        (weight, lab_energy, polarization)
    }

    /// Return one muon's yield of `daughter` at `probe`.
    fn muon_daughters(
        &self,
        pair: &PairDecay,
        muon: Muon,
        daughter: Daughter,
        probe: Probe,
    ) -> f64 {
        let decay = MuonDecay::PDG;
        let feature = (decay.max_energy(daughter), decay.daughter_mass(daughter));
        let points = self.muon_points(pair, probe, &[feature]);
        let mut integrand = |cosine: f64| {
            let (weight, lab_energy, polarization) = self.muon_state(pair, muon, cosine);
            if weight == 0.0 {
                return 0.0;
            }
            weight * muon_response(daughter, probe, lab_energy, polarization)
        };
        integrate(&mut integrand, -1.0, 1.0, &points, FOLD_EPSREL)
    }

    /// Return both muons' radiative photon yield at `probe`.
    fn muon_photons(&self, pair: &PairDecay, probe: Probe) -> f64 {
        let (energy, momentum) = (pair.energy(), pair.momentum());
        let analyzer = pair.analyzer();
        // The photon edge rides on the muon as a massless daughter's box does,
        // so near the lab endpoint only a sliver of forward muons contributes.
        let points = self.muon_points(pair, probe, &[(photon_endpoint(MASS_MU), 0.0)]);
        let mut integrand = |cosine: f64| {
            let weight = analyzer.density(self.helicities, cosine)
                + analyzer.density(self.helicities, -cosine);
            let lab_energy = self.boost.lab_energy(energy, momentum, cosine);
            let dnde = |photon: f64| dnde_photon_muon(photon, lab_energy);
            weight * photon_response(probe, dnde, photon_endpoint(lab_energy))
        };
        integrate(&mut integrand, -1.0, 1.0, &points, FOLD_EPSREL)
    }

    /// Return the cosines at which a muon's response to `probe` breaks:
    /// where the boost of a daughter `(energy, mass)` crosses a probe edge.
    fn muon_points(&self, pair: &PairDecay, probe: Probe, features: &[(f64, f64)]) -> Vec<f64> {
        let mut points = Vec::with_capacity(2 * features.len());
        for edge in probe.edges() {
            if edge > 0.0 {
                for &(energy, mass) in features {
                    let lab_energy = pair.mass * threshold_gamma(edge, energy, mass);
                    points.push(
                        self.boost
                            .cosine(lab_energy, pair.energy(), pair.momentum()),
                    );
                }
            }
        }
        points
    }

    /// Return the lab energy density in MeV⁻¹ of a pion with the
    /// mediator-frame `densities`.
    fn pion_lab_density(
        &self,
        parents: &ThreePionParents,
        [density, alignment]: PionDensities,
        lab_energy: f64,
    ) -> f64 {
        let (mass, boost) = (parents.pion_mass, self.boost);
        let (lower, upper) = boost_range(lab_energy, mass, boost.gamma, boost.gamma_beta);
        let from = (lower - mass).max(0.0);
        let to = (upper - mass).min(parents.max_kinetic());
        if from >= to {
            return 0.0;
        }
        let weight = self.helicities.alignment();
        let mut integrand = |kinetic: f64| {
            let momentum = (kinetic * (2.0 * mass + kinetic)).sqrt();
            if momentum == 0.0 {
                return 0.0;
            }
            let cosine = boost
                .cosine(lab_energy, mass + kinetic, momentum)
                .clamp(-1.0, 1.0);
            let angular = weight.mul_add(
                alignment(parents, kinetic) * legendre2(cosine),
                density(parents, kinetic),
            );
            angular / (2.0 * boost.gamma_beta * momentum)
        };
        integrate(&mut integrand, from, to, &[], DENSITY_EPSREL)
    }

    /// Integrate a pion's lab energy density times `response` over its lab
    /// energy, breaking where the density kinks and where a decay daughter
    /// `(energy, mass)` of each feature crosses a probe edge.
    fn pion_fold(
        &self,
        parents: &ThreePionParents,
        densities: PionDensities,
        response: impl Fn(f64) -> f64,
        probe: Probe,
        features: &[(f64, f64)],
    ) -> f64 {
        let (mass, boost) = (parents.pion_mass, self.boost);
        // The fastest pion's box bounds the lab energies; its lower edge
        // falls to the pion mass once some pion moves faster than the
        // mediator in its frame.
        let fastest = mass + parents.max_kinetic();
        let (lower, upper) = boost_range(fastest, mass, boost.gamma, boost.gamma_beta);
        let floor = if momentum(fastest, mass) >= boost.beta() * fastest {
            mass
        } else {
            lower
        };
        let mut points = vec![boost.gamma * mass, lower];
        for edge in probe.edges() {
            if edge > 0.0 {
                for &(energy, daughter_mass) in features {
                    points.push(mass * threshold_gamma(edge, energy, daughter_mass));
                }
            }
        }
        let mut integrand = |lab_energy: f64| {
            let lab_density = self.pion_lab_density(parents, densities, lab_energy);
            if lab_density == 0.0 {
                return 0.0;
            }
            lab_density * response(lab_energy)
        };
        integrate(&mut integrand, floor, upper, &points, FOLD_EPSREL)
    }

    /// Return the π⁰ photon yield at `probe`, every π⁰ of the state
    /// counted.
    fn neutral_pion_photons(&self, parents: &ThreePionParents, probe: Probe) -> f64 {
        let mass = parents.pion_mass;
        let response = |lab_energy: f64| {
            let gamma_beta = momentum(lab_energy, mass) / mass;
            neutral_pion_photons(probe, lab_energy / mass, gamma_beta)
        };
        let multiplicity = match parents.channel {
            PionChannel::Charged => 1.0,
            PionChannel::Neutral => 3.0,
        };
        let features = [(MASS_PI0 / 2.0, 0.0)];
        multiplicity * self.pion_fold(parents, SPECTATOR, response, probe, &features)
    }

    /// Return the energy at which a charged pion of production `lab_energy`
    /// decays with its physical mass, at the same Lorentz factor.
    fn charged_decay_energy(parents: &ThreePionParents, lab_energy: f64) -> f64 {
        lab_energy * (MASS_PI / parents.pion_mass)
    }

    /// Return the yield of one charged pion's `daughter` at `probe`.
    fn charged_pion_daughters(
        &self,
        parents: &ThreePionParents,
        daughter: Daughter,
        probe: Probe,
    ) -> f64 {
        let decay = PionDecay::PDG;
        let mass = decay.muon.daughter_mass(daughter);
        let features = decay.rest_features(daughter).map(|energy| (energy, mass));
        let response = |lab_energy: f64| {
            pion_response(
                daughter,
                probe,
                Self::charged_decay_energy(parents, lab_energy),
            )
        };
        self.pion_fold(parents, CHARGED, response, probe, &features)
    }

    /// Return the radiative photon yield of one charged pion at `probe`.
    fn charged_pion_photons(&self, parents: &ThreePionParents, probe: Probe) -> f64 {
        let response = |lab_energy: f64| {
            let decay_energy = Self::charged_decay_energy(parents, lab_energy);
            photon_response(
                probe,
                |photon| dnde_photon_charged_pion(photon, decay_energy),
                charged_pion_photon_endpoint(decay_energy),
            )
        };
        let features = [(charged_pion_photon_endpoint(MASS_PI), 0.0)];
        self.pion_fold(parents, CHARGED, response, probe, &features)
    }

    fn is_charged(parents: &ThreePionParents) -> bool {
        parents.channel == PionChannel::Charged
    }
}

impl Yields for Cascade {
    fn photons(&self, probe: Probe) -> f64 {
        match &self.decay {
            Decay::ElectronPair(_) | Decay::NeutrinoPair(..) => 0.0,
            Decay::MuonPair(pair) => self.muon_photons(pair, probe),
            Decay::ThreePion(parents) => {
                let neutral = self.neutral_pion_photons(parents, probe);
                if Self::is_charged(parents) {
                    neutral + 2.0 * self.charged_pion_photons(parents, probe)
                } else {
                    neutral
                }
            }
        }
    }

    fn charged_leptons(&self, probe: Probe) -> [f64; 2] {
        match &self.decay {
            Decay::ElectronPair(pair) => {
                let analyzer = pair.analyzer();
                [
                    self.stable(pair, analyzer.reversed(), probe),
                    self.stable(pair, analyzer, probe),
                ]
            }
            Decay::NeutrinoPair(..) => [0.0, 0.0],
            Decay::MuonPair(pair) => [Muon::Positive, Muon::Negative]
                .map(|muon| self.muon_daughters(pair, muon, Daughter::ChargedLepton, probe)),
            Decay::ThreePion(parents) => {
                if Self::is_charged(parents) {
                    // CP: the π⁻ gives the e⁻ the π⁺ gives the e⁺.
                    let lepton =
                        self.charged_pion_daughters(parents, Daughter::ChargedLepton, probe);
                    [lepton, lepton]
                } else {
                    [0.0, 0.0]
                }
            }
        }
    }

    fn neutrinos(&self, probe: Probe) -> [f64; 3] {
        let mut flavors = [0.0; 3];
        match &self.decay {
            Decay::ElectronPair(_) => {}
            Decay::NeutrinoPair(pair, index) => {
                let analyzer = pair.analyzer();
                flavors[*index] = self.stable(pair, analyzer, probe)
                    + self.stable(pair, analyzer.reversed(), probe);
            }
            Decay::MuonPair(pair) => {
                for muon in [Muon::Negative, Muon::Positive] {
                    flavors[0] +=
                        self.muon_daughters(pair, muon, Daughter::ElectronNeutrino, probe);
                    flavors[1] += self.muon_daughters(pair, muon, Daughter::MuonNeutrino, probe);
                }
            }
            Decay::ThreePion(parents) => {
                if Self::is_charged(parents) {
                    // Each of the two charged pions gives one flavor and the
                    // other the antiflavor.
                    flavors[0] = 2.0
                        * self.charged_pion_daughters(parents, Daughter::ElectronNeutrino, probe);
                    flavors[1] =
                        2.0 * self.charged_pion_daughters(parents, Daughter::MuonNeutrino, probe);
                }
            }
        }
        flavors
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use num_complex::Complex64;

    use super::*;
    use crate::boost::boost_delta_function;
    use crate::constants::pdg::{BR_PI0_TO_A_A, MASS_E, MASS_MU};
    use crate::kernels::axial_rates::FermionPair;
    use crate::kernels::axial_yields::Channel;
    use crate::quad::{QuadOpts, quad};

    const ISOTROPIC: Helicities = Helicities {
        plus: 1.0 / 3.0,
        zero: 1.0 / 3.0,
        minus: 1.0 / 3.0,
    };
    const TRANSVERSE: Helicities = Helicities {
        plus: 0.5,
        zero: 0.0,
        minus: 0.5,
    };
    const LONGITUDINAL: Helicities = Helicities {
        plus: 0.0,
        zero: 1.0,
        minus: 0.0,
    };
    const ORIENTED: Helicities = Helicities {
        plus: 0.6,
        zero: 0.25,
        minus: 0.15,
    };
    const FRACTIONS: [Helicities; 4] = [ISOTROPIC, TRANSVERSE, LONGITUDINAL, ORIENTED];
    const MEDIATOR: f64 = 450.0;

    fn assert_close(actual: f64, expected: f64, rtol: f64) {
        let error = (actual - expected).abs();
        assert!(
            error <= rtol * expected.abs(),
            "{actual} vs {expected}: relative error {}",
            error / expected.abs()
        );
    }

    fn pure(helicity: i32) -> Helicities {
        Helicities {
            plus: f64::from(u8::from(helicity == 1)),
            zero: f64::from(u8::from(helicity == 0)),
            minus: f64::from(u8::from(helicity == -1)),
        }
    }

    /// Pairs with and without parity violation, massive and massless.
    fn pairs() -> [PairDecay; 5] {
        let pair = |mass, vector, axial| PairDecay {
            mediator_mass: MEDIATOR,
            mass,
            vector,
            axial,
        };
        [
            pair(MASS_MU, 0.7, -0.4),
            pair(MASS_MU, 0.0, 1.0),
            pair(MASS_MU, 1.0, 0.0),
            pair(MASS_E, 0.3, 0.8),
            pair(0.0, 0.5, -0.5),
        ]
    }

    fn three_pions(energy: f64) -> ThreePionParents {
        ThreePionParents::new(PionChannel::Charged, energy, MASS_PI0, 1.0)
    }

    fn cascade(decay: Decay, mediator_energy: f64, helicities: Helicities) -> Cascade {
        let mass = match decay {
            Decay::ElectronPair(pair) | Decay::NeutrinoPair(pair, _) | Decay::MuonPair(pair) => {
                pair.mediator_mass
            }
            Decay::ThreePion(_) => MEDIATOR,
        };
        Cascade {
            decay,
            boost: Boost::new(mass, mediator_energy),
            helicities,
        }
    }

    // ----- decay matrix -----

    fn polarization(helicity: i32) -> [Complex64; 3] {
        let (one, i, zero) = (
            Complex64::new(1.0, 0.0),
            Complex64::i(),
            Complex64::new(0.0, 0.0),
        );
        match helicity {
            1 => [-one / SQRT_2, -i / SQRT_2, zero],
            0 => [zero, zero, one],
            _ => [one / SQRT_2, -i / SQRT_2, zero],
        }
    }

    /// `D(n̂) = [1 + g S·n̂ + h(3P_n − 1)]/(4π)` in the helicity basis
    /// `+1, 0, −1`, with `⟨λ|S·n̂|λ′⟩ = ε_λ*·(i n̂ × ε_λ′)` and
    /// `⟨λ|P_n|λ′⟩ = (ε_λ*·n̂)(n̂·ε_λ′)`.
    fn decay_matrix(analyzer: Analyzer, n: [f64; 3]) -> [[Complex64; 3]; 3] {
        let dot = |u: &[Complex64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let mut matrix = [[Complex64::new(0.0, 0.0); 3]; 3];
        for (row, &left) in [1, 0, -1].iter().enumerate() {
            for (column, &right) in [1, 0, -1].iter().enumerate() {
                let (e, f) = (polarization(left).map(|x| x.conj()), polarization(right));
                let cross = [
                    n[1] * f[2] - n[2] * f[1],
                    n[2] * f[0] - n[0] * f[2],
                    n[0] * f[1] - n[1] * f[0],
                ];
                let spin = Complex64::i() * (e[0] * cross[0] + e[1] * cross[1] + e[2] * cross[2]);
                let projector = dot(&e, n) * dot(&f, n);
                let identity = f64::from(u8::from(row == column));
                matrix[row][column] = (identity
                    + analyzer.vector * spin
                    + analyzer.tensor * (3.0 * projector - identity))
                    / (4.0 * PI);
            }
        }
        matrix
    }

    fn direction(cosine: f64, azimuth: f64) -> [f64; 3] {
        let sine = (1.0 - cosine * cosine).sqrt();
        [sine * azimuth.cos(), sine * azimuth.sin(), cosine]
    }

    /// The analyzers of every daughter here: fermion pairs, the π⁰ and
    /// charged pions across their spectrum.
    fn physical_analyzers() -> Vec<Analyzer> {
        let mut analyzers: Vec<Analyzer> = pairs().iter().map(PairDecay::analyzer).collect();
        analyzers.push(Analyzer {
            vector: 0.0,
            tensor: 1.0,
        });
        let parents = three_pions(MEDIATOR);
        for step in 1..20 {
            let kinetic = parents.max_kinetic() * f64::from(step) / 20.0;
            analyzers.push(Analyzer {
                vector: 0.0,
                tensor: parents.charged_alignment(kinetic) / parents.charged_density(kinetic),
            });
        }
        analyzers
    }

    /// The angular average of the decay matrix is the identity for any
    /// analyzing powers, since `∫n̂ dΩ = ∫(3P_n − 1) dΩ = 0`, so this pins
    /// the matrix's structure and not `g` or `h`; those are pinned by
    /// `pair_spin_sums_match_the_helicity_amplitudes`,
    /// `pair_endpoints_at_rest_and_ultra_relativistic`, and the Python
    /// `test_muon_cascades_match_explicit_spinors`.
    #[test]
    fn decay_matrix_integrates_to_the_identity() {
        for analyzer in physical_analyzers() {
            for row in 0..3 {
                for column in 0..3 {
                    for real in [true, false] {
                        // An 8-point azimuthal rule is exact for e^{ikφ}, |k| ≤ 2.
                        let mut integrand = |cosine: f64| {
                            (0..8)
                                .map(|step| {
                                    let azimuth = PI * f64::from(step) / 4.0;
                                    let element =
                                        decay_matrix(analyzer, direction(cosine, azimuth))[row]
                                            [column];
                                    if real { element.re } else { element.im }
                                })
                                .sum::<f64>()
                                * (PI / 4.0)
                        };
                        // Most elements integrate to zero, so the tolerance is absolute.
                        let opts = QuadOpts {
                            epsabs: 1e-15,
                            epsrel: 0.0,
                            limit: 50,
                            points: None,
                        };
                        let integral = quad(&mut integrand, -1.0, 1.0, &opts).unwrap().value;
                        let expected = f64::from(u8::from(real && row == column));
                        assert!((integral - expected).abs() < 1e-13, "{analyzer:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn decay_matrix_diagonal_is_the_angular_density() {
        for analyzer in physical_analyzers() {
            for (index, helicity) in [1, 0, -1].into_iter().enumerate() {
                for cosine in [-0.95, -0.3, 0.0, 0.4, 0.99] {
                    for azimuth in [0.0, 1.0, 4.0] {
                        let element =
                            decay_matrix(analyzer, direction(cosine, azimuth))[index][index];
                        assert!(element.im.abs() < 1e-15);
                        let density = analyzer.density(pure(helicity), cosine);
                        assert!((2.0 * PI * element.re - density).abs() < 1e-14);
                    }
                }
            }
        }
    }

    #[test]
    fn physical_decay_matrices_are_positive() {
        let states = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.2, 0.9, 0.1, -0.5, 0.4],
            [-0.7, 0.6, 0.2, -0.9, 0.3, 0.8],
            [0.1, 0.1, -0.4, 0.7, 0.6, -0.2],
        ];
        for analyzer in physical_analyzers() {
            for cosine in [-1.0, -0.6, 0.0, 0.5, 1.0] {
                let matrix = decay_matrix(analyzer, direction(cosine, 0.7));
                for state in states {
                    let psi = [0, 2, 4].map(|k| Complex64::new(state[k], state[k + 1]));
                    let mut value = Complex64::new(0.0, 0.0);
                    for row in 0..3 {
                        for column in 0..3 {
                            value += psi[row].conj() * matrix[row][column] * psi[column];
                        }
                    }
                    assert!(value.re >= -1e-15, "{analyzer:?} {cosine} {}", value.re);
                    assert!(value.im.abs() < 1e-15);
                }
            }
        }
    }

    // ----- fermion pairs -----

    #[test]
    fn pair_spin_sums_match_the_helicity_amplitudes() {
        let s = |c: f64| (1.0 - c * c).sqrt();
        for pair in pairs() {
            let (a, b, c0) = pair.amplitudes();
            for helicity in [1, 0, -1] {
                for cosine in [-0.9, -0.2, 0.3, 0.8] {
                    // Wigner d¹_{λ1}, d¹_{λ,−1}, d¹_{λ0} at θ.
                    let (up, down, middle) = match helicity {
                        1 => (
                            (1.0 + cosine) / 2.0,
                            (1.0 - cosine) / 2.0,
                            -s(cosine) / SQRT_2,
                        ),
                        0 => (s(cosine) / SQRT_2, -s(cosine) / SQRT_2, cosine),
                        _ => (
                            (1.0 - cosine) / 2.0,
                            (1.0 + cosine) / 2.0,
                            s(cosine) / SQRT_2,
                        ),
                    };
                    let rate =
                        up * up * a * a + down * down * b * b + 2.0 * middle * middle * c0 * c0;
                    let fractions = pure(helicity);
                    let density = pair.analyzer().density(fractions, cosine);
                    assert_close(density, rate / pair.norm(), 1e-13);
                    let [longitudinal, transverse] = pair.spin(fractions, cosine);
                    assert!(
                        (longitudinal - (up * up * a * a - down * down * b * b) / pair.norm())
                            .abs()
                            < 1e-14
                    );
                    let expected = 2.0 * c0 * middle * (up * a + down * b) / pair.norm();
                    assert!((transverse - expected).abs() < 1e-14);
                    assert!(longitudinal.hypot(transverse) <= density * (1.0 + 1e-14));
                }
            }
        }
    }

    #[test]
    fn mean_pair_helicity_matches_the_rates_model() {
        for pair in pairs() {
            let rates = FermionPair {
                mass: pair.mass,
                vector: pair.vector,
                axial: pair.axial,
                colors: 1.0,
            };
            let expected = rates.helicity_density(MEDIATOR) / rates.densities(MEDIATOR).0;
            for fractions in FRACTIONS {
                let mut integrand = |cosine: f64| pair.spin(fractions, cosine)[0];
                let mean = integrate(&mut integrand, -1.0, 1.0, &[], 1e-13);
                assert!((mean - expected).abs() < 1e-13, "{pair:?} {fractions:?}");
            }
        }
    }

    #[test]
    fn muon_lab_helicity_is_the_frame_helicity_at_rest_and_bounded() {
        let pair = pairs()[0];
        let slow = cascade(Decay::MuonPair(pair), MEDIATOR * (1.0 + 1e-18), ORIENTED);
        let fast = cascade(Decay::MuonPair(pair), 5000.0, ORIENTED);
        for cosine in [-0.9, -0.4, 0.1, 0.7] {
            let [longitudinal, _] = pair.spin(ORIENTED, cosine);
            let density = pair.analyzer().density(ORIENTED, cosine);
            let (_, _, negative) = slow.muon_state(&pair, Muon::Negative, cosine);
            assert_close(negative, longitudinal / density, 1e-12);
            // The μ⁺ at cos θ has minus the helicity of the μ⁻ at −cos θ.
            let [longitudinal, _] = pair.spin(ORIENTED, -cosine);
            let density = pair.analyzer().density(ORIENTED, -cosine);
            let (_, _, positive) = slow.muon_state(&pair, Muon::Positive, cosine);
            assert_close(positive, longitudinal / density, 1e-12);
            for muon in [Muon::Negative, Muon::Positive] {
                let (_, _, polarization) = fast.muon_state(&pair, muon, cosine);
                assert!(polarization.abs() < 1.0);
            }
        }
    }

    #[test]
    fn pair_daughters_are_normalized_and_share_the_energy() {
        let everything = Probe::Bin(0.0, 1e5);
        for fractions in FRACTIONS {
            for pair in pairs() {
                let electrons = cascade(Decay::ElectronPair(pair), 700.0, fractions);
                for count in electrons.charged_leptons(everything) {
                    assert_close(count, 1.0, 1e-14);
                }
                let neutrinos = cascade(Decay::NeutrinoPair(pair, 2), 700.0, fractions);
                assert_close(neutrinos.neutrinos(everything)[2], 2.0, 1e-14);
                // The two daughters' mean lab energies sum to the mediator's,
                // which needs the vector power's P₁ to cancel between them.
                let (low, high) = boost_range(pair.energy(), pair.mass, 700.0 / MEDIATOR, {
                    momentum(700.0, MEDIATOR) / MEDIATOR
                });
                let mut integrand = |energy: f64| {
                    let [positron, electron] = electrons.charged_leptons(Probe::Density(energy));
                    energy * (positron + electron)
                };
                let total = integrate(&mut integrand, low, high, &[], 1e-13);
                assert_close(total, 700.0, 1e-12);
            }
            let muons = cascade(Decay::MuonPair(pairs()[0]), 700.0, fractions);
            for count in muons.charged_leptons(everything) {
                assert_close(count, 1.0, 1e-8);
            }
            let [electron, muon, tau] = muons.neutrinos(everything);
            assert_close(electron, 2.0, 1e-8);
            assert_close(muon, 2.0, 1e-8);
            assert_eq!(tau, 0.0);
        }
    }

    #[test]
    fn transverse_pairs_give_equal_charges() {
        // F₊ = F₋ is CP-even, so the e⁺ and e⁻ spectra agree; F₊ ≠ F₋ with
        // parity violation separates them.
        let bin = Probe::Bin(150.0, 160.0);
        for pair in [pairs()[0], pairs()[3]] {
            let decay = if pair.mass == MASS_MU {
                Decay::MuonPair(pair)
            } else {
                Decay::ElectronPair(pair)
            };
            let [positron, electron] = cascade(decay, 700.0, TRANSVERSE).charged_leptons(bin);
            // The two charges share each quadrature's integrand, so they
            // agree to the fold tolerance (1e-8) with margin.
            assert_close(positron, electron, 1e-9);
            let [positron, electron] = cascade(decay, 700.0, ORIENTED).charged_leptons(bin);
            assert!((positron - electron).abs() > 1e-3 * electron);
        }
    }

    #[test]
    fn isotropic_pair_is_the_boosted_line() {
        let pair = pairs()[3];
        let electrons = cascade(Decay::ElectronPair(pair), 700.0, ISOTROPIC);
        let beta = electrons.boost.beta();
        for energy in [60.0, 100.0, 300.0, 500.0, 600.0] {
            let [positron, electron] = electrons.charged_leptons(Probe::Density(energy));
            let expected = boost_delta_function(pair.energy(), energy, pair.mass, beta);
            assert_close(positron, expected, 1e-12);
            assert_close(electron, expected, 1e-12);
        }
    }

    #[test]
    fn pair_endpoints_at_rest_and_ultra_relativistic() {
        let pair = pairs()[3];
        let (energy, momentum) = (pair.energy(), pair.momentum());
        // Nearly at rest the whole box sits in the bin around M/2.
        let slow = cascade(
            Decay::ElectronPair(pair),
            MEDIATOR * (1.0 + 1e-12),
            ORIENTED,
        );
        let [positron, electron] = slow.charged_leptons(Probe::Bin(energy - 1e-3, energy + 1e-3));
        assert_close(positron, 1.0, 1e-14);
        assert_close(electron, 1.0, 1e-14);
        // At γ = 10⁴, dN/dx with x = E/E_A tends to 2W(2x − 1); the
        // corrections are O(1/γ²) and O(m_e²/M²) ≈ 5e-6, hence 1e-5.
        let analyzer = pair.analyzer();
        let lab = 1e4 * MEDIATOR;
        let fast = cascade(Decay::ElectronPair(pair), lab, ORIENTED);
        for x in [0.05, 0.3, 0.5, 0.8, 0.97] {
            let [_, electron] = fast.charged_leptons(Probe::Density(x * lab));
            assert_close(
                electron * lab,
                2.0 * analyzer.density(ORIENTED, 2.0 * x - 1.0),
                1e-5,
            );
        }
        // The upper edge carries the forward density W(1)/(γβk).
        let boost = fast.boost;
        let edge = boost.lab_energy(energy, momentum, 1.0);
        let [_, electron] = fast.charged_leptons(Probe::Density(edge));
        assert_close(
            electron,
            analyzer.density(ORIENTED, 1.0) / (boost.gamma_beta * momentum),
            1e-12,
        );
    }

    // ----- three pions -----

    #[test]
    fn pion_lab_densities_are_normalized_and_share_the_energy() {
        let parents = three_pions(MEDIATOR);
        for fractions in [TRANSVERSE, LONGITUDINAL, ORIENTED] {
            for lab in [460.0, 700.0, 3000.0] {
                let pions = cascade(Decay::ThreePion(parents), lab, fractions);
                let moments = |densities: PionDensities, power: i32| {
                    pions.pion_fold(
                        &parents,
                        densities,
                        |e: f64| e.powi(power),
                        Probe::Density(0.0),
                        &[],
                    )
                };
                assert_close(moments(SPECTATOR, 0), 1.0, 1e-9);
                assert_close(moments(CHARGED, 0), 1.0, 1e-9);
                let total = moments(SPECTATOR, 1) + 2.0 * moments(CHARGED, 1);
                assert_close(total, lab, 1e-9);
            }
        }
    }

    /// The lab second moment `⟨E²⟩` of each pion matches Dalitz-plot
    /// moments integrated by brute force from momentum conservation, which
    /// pins the charged pions' alignment wiring (`CHARGED`) at the lab
    /// level. A pion with the mediator-frame angular density
    /// `[f + (3F₀ − 1) h P₂(c)]/2` has no `P₁` term, so
    /// `⟨E²⟩ = γ² ∫e² f du + (γβ)² ∫k² (f/3 + 2κh/15) du` with `κ = 3F₀ − 1`.
    #[test]
    fn pion_lab_second_moments_match_dalitz_moments() {
        let parents = three_pions(MEDIATOR);
        let (mass, top) = (MASS_PI0, parents.max_kinetic());
        let excess = MEDIATOR - 3.0 * mass;
        let momentum2 = |kinetic: f64| kinetic * (2.0 * mass + kinetic);
        // The cosine of the angle between pions with kinetic energies `a`
        // and `b`, from the third pion's momentum.
        let cosine = |a: f64, b: f64| {
            let third = excess - a - b;
            (momentum2(third) - momentum2(a) - momentum2(b))
                / (2.0 * (momentum2(a) * momentum2(b)).sqrt())
        };
        let feasible = |a: f64, b: f64| (-1.0..=1.0).contains(&cosine(a, b));
        // The partner's physical kinetic energies at a fixed pion's: the
        // symmetric point (equal partner energies) is always physical, so
        // each edge is bisected outward from it.
        let span = |fixed: f64| {
            let edge = |outer: f64| {
                if feasible(fixed, outer) {
                    return outer;
                }
                let (mut inside, mut outside) = ((excess - fixed) / 2.0, outer);
                for _ in 0..100 {
                    let middle = 0.5 * (inside + outside);
                    if feasible(fixed, middle) {
                        inside = middle;
                    } else {
                        outside = middle;
                    }
                }
                inside
            };
            (edge(0.0), edge(excess - fixed))
        };
        let dalitz_integral = |fixed: f64, weight: &dyn Fn(f64) -> f64| {
            let (low, high) = span(fixed);
            integrate(&mut |partner: f64| weight(partner), low, high, &[], 1e-12)
        };
        let mut area = |neutral: f64| momentum2(neutral) * dalitz_integral(neutral, &|_| 1.0);
        let norm = integrate(&mut area, 0.0, top, &[], 1e-12);
        // The energy density `f` and alignment density `h` of the π⁰ and of
        // each charged pion, in MeV⁻¹.
        let spectator = |neutral: f64| {
            let density = momentum2(neutral) * dalitz_integral(neutral, &|_| 1.0) / norm;
            (density, density)
        };
        let charged = |charged: f64| {
            let density = dalitz_integral(charged, &momentum2) / norm;
            let alignment = dalitz_integral(charged, &|neutral| {
                let c = cosine(charged, neutral);
                momentum2(neutral) * legendre2(c)
            }) / norm;
            (density, alignment)
        };
        let brute = |pions: &Cascade, densities: &dyn Fn(f64) -> (f64, f64)| {
            let kappa = pions.helicities.alignment();
            let (gamma, gamma_beta) = (pions.boost.gamma, pions.boost.gamma_beta);
            let mut integrand = |kinetic: f64| {
                let (density, alignment) = densities(kinetic);
                let energy = mass + kinetic;
                let along = density / 3.0 + 2.0 * kappa * alignment / 15.0;
                gamma * gamma * energy * energy * density
                    + gamma_beta * gamma_beta * momentum2(kinetic) * along
            };
            integrate(&mut integrand, 0.0, top, &[], 1e-10)
        };
        let fold = |pions: &Cascade, densities: PionDensities| {
            pions.pion_fold(&parents, densities, |e| e * e, Probe::Density(0.0), &[])
        };
        // The two sides agree to about 5e-12; the allowance covers the
        // fold's 1e-8 request and the brute-force nested quadratures
        // converging well inside theirs, while the alignment moves the
        // moment by about 2e-2.
        let rtol = 1e-10;
        for lab in [900.0, 3000.0] {
            let mut charged_moments = Vec::new();
            for fractions in [TRANSVERSE, LONGITUDINAL] {
                let pions = cascade(Decay::ThreePion(parents), lab, fractions);
                assert_close(fold(&pions, SPECTATOR), brute(&pions, &spectator), rtol);
                assert_close(fold(&pions, CHARGED), brute(&pions, &charged), rtol);
                charged_moments.push(fold(&pions, CHARGED));
            }
            // The alignment moves the moment far beyond the tolerance.
            assert!(
                (charged_moments[1] - charged_moments[0]).abs() > 1e6 * rtol * charged_moments[0]
            );
        }
    }

    #[test]
    fn three_pion_counts_round_trip_by_bin_width() {
        let pions = cascade(Decay::ThreePion(three_pions(MEDIATOR)), 600.0, TRANSVERSE);
        for energy in [40.0, 120.0, 250.0] {
            let width = 1e-4;
            let bin = Probe::Bin(energy - width / 2.0, energy + width / 2.0);
            let density = Probe::Density(energy);
            // A 1e-4 MeV bin averages the density to 1e-5: the density's
            // curvature over the bin and the fold tolerance set the margin.
            assert_close(pions.photons(bin) / width, pions.photons(density), 1e-5);
            assert_close(
                pions.charged_leptons(bin)[0] / width,
                pions.charged_leptons(density)[0],
                1e-5,
            );
            assert_close(
                pions.neutrinos(bin)[1] / width,
                pions.neutrinos(density)[1],
                1e-5,
            );
        }
    }

    /// Return density and bin probes the relative `gaps` below an
    /// `endpoint`, the bins reaching past it.
    fn endpoint_probes(endpoint: f64, gaps: &[f64]) -> Vec<Probe> {
        gaps.iter()
            .flat_map(|gap| {
                let low = endpoint * (1.0 - gap);
                [
                    Probe::Density(low),
                    Probe::Bin(low, endpoint * (1.0 + 1e-3)),
                ]
            })
            .collect()
    }

    #[test]
    fn muon_photons_reach_their_lab_endpoint() {
        // Near the endpoint only a sliver of forward muons radiates; the oracle
        // integrates over that sliver alone. The fold's tolerance is 1e-8.
        // Hazma's unpolarized radiative-muon density dips to about −3e-13
        // MeV⁻¹ within 1e-3 of its edge, so the oracle is held to be nonzero,
        // not positive.
        let pair = pairs()[0];
        let muons = cascade(Decay::MuonPair(pair), 600.0, ORIENTED);
        let (energy, momentum) = (pair.energy(), pair.momentum());
        let analyzer = pair.analyzer();
        let endpoint = photon_endpoint(muons.boost.lab_energy(energy, momentum, 1.0));
        for probe in endpoint_probes(endpoint, &[1e-2, 1e-3]) {
            let threshold = threshold_gamma(probe.edges()[0], photon_endpoint(MASS_MU), 0.0);
            let from = muons.boost.cosine(pair.mass * threshold, energy, momentum);
            let mut integrand = |cosine: f64| {
                let lab = muons.boost.lab_energy(energy, momentum, cosine);
                let weight =
                    analyzer.density(ORIENTED, cosine) + analyzer.density(ORIENTED, -cosine);
                let dnde = |photon: f64| dnde_photon_muon(photon, lab);
                weight * photon_response(probe, dnde, photon_endpoint(lab))
            };
            let expected = integrate(&mut integrand, from, 1.0, &[], 1e-10);
            assert!(expected != 0.0, "{probe:?}");
            assert_close(muons.photons(probe), expected, 1e-7);
        }
        assert_eq!(muons.photons(Probe::Density(endpoint * (1.0 + 1e-9))), 0.0);
    }

    #[test]
    fn charged_pion_photons_reach_their_lab_endpoint() {
        // As for muons: only the fastest charged pions radiate near the edge.
        let parents = three_pions(MEDIATOR);
        let pions = cascade(Decay::ThreePion(parents), 600.0, TRANSVERSE);
        let mass = parents.pion_mass;
        let boost = pions.boost;
        let fastest = mass + parents.max_kinetic();
        let (_, top) = boost_range(fastest, mass, boost.gamma, boost.gamma_beta);
        let endpoint = charged_pion_photon_endpoint(Cascade::charged_decay_energy(&parents, top));
        let rest_edge = charged_pion_photon_endpoint(MASS_PI);
        for probe in endpoint_probes(endpoint, &[1e-3, 1e-6]) {
            let from = mass * threshold_gamma(probe.edges()[0], rest_edge, 0.0);
            let mut integrand = |lab: f64| {
                let decay = Cascade::charged_decay_energy(&parents, lab);
                let dnde = |photon: f64| dnde_photon_charged_pion(photon, decay);
                pions.pion_lab_density(&parents, CHARGED, lab)
                    * photon_response(probe, dnde, charged_pion_photon_endpoint(decay))
            };
            let expected = integrate(&mut integrand, from, top, &[], 1e-10);
            assert!(expected > 0.0, "{probe:?}");
            assert_close(pions.charged_pion_photons(&parents, probe), expected, 1e-7);
        }
    }

    #[test]
    fn slow_three_pions_reproduce_the_rest_frame() {
        let parents = three_pions(MEDIATOR);
        let rest = Channel::ThreePion(parents);
        let slow = cascade(
            Decay::ThreePion(parents),
            MEDIATOR * (1.0 + 1e-10),
            LONGITUDINAL,
        );
        // The physical change is O(β²) ≈ 1e-10, so the compounding fold
        // tolerances of the two sides set the 1e-6.
        for low in [20.0, 100.0] {
            let bin = Probe::Bin(low, low + 5.0);
            assert_close(slow.photons(bin), rest.photons(bin), 1e-6);
            assert_close(
                slow.charged_leptons(bin)[0],
                rest.charged_leptons(bin)[0],
                1e-6,
            );
            assert_close(slow.neutrinos(bin)[0], rest.neutrinos(bin)[0], 1e-6);
        }
    }

    /// The research reference, re-run with Hazma's common pion mass: π⁰
    /// photons per decay with `BR(π⁰ → γγ) = 1` in 500 bins, transverse
    /// (`F₀ = 0`) and longitudinal (`F₀ = 1`).
    fn reference() -> Vec<(f64, f64, f64, f64, f64, f64)> {
        include_str!("testdata/three_pion_cascade.csv")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.starts_with('m'))
            .map(|line| {
                let v: Vec<f64> = line.split(',').map(|x| x.parse().unwrap()).collect();
                (v[0], v[1], v[2], v[3], v[4], v[5])
            })
            .collect()
    }

    /// Return the total variation between two spectra normalized to one.
    fn total_variation(first: &[f64], second: &[f64]) -> f64 {
        let (a, b): (f64, f64) = (first.iter().sum(), second.iter().sum());
        first
            .iter()
            .zip(second)
            .map(|(x, y)| (x / a - y / b).abs())
            .sum::<f64>()
            / 2.0
    }

    #[test]
    fn three_pion_cascades_reproduce_the_reference_spectra() {
        let rows = reference();
        for (dark, mediator, expected_tv) in [(600.0, 450.0, 0.0538), (900.0, 500.0, 0.0585)] {
            let parents = three_pions(mediator);
            let case: Vec<_> = rows
                .iter()
                .filter(|row| row.0 == dark && row.1 == mediator)
                .collect();
            assert_eq!(case.len(), 500);
            let peak = case.iter().map(|row| row.4.max(row.5)).fold(0.0, f64::max);
            let mut spectra = [Vec::new(), Vec::new(), Vec::new()];
            for (fractions, column) in [(TRANSVERSE, 0), (LONGITUDINAL, 1)] {
                let pions = Cascade {
                    decay: Decay::ThreePion(parents),
                    boost: Boost::new(mediator, dark),
                    helicities: fractions,
                };
                for row in &case {
                    let bin = Probe::Bin(row.2, row.3);
                    let count = pions.neutral_pion_photons(&parents, bin) / BR_PI0_TO_A_A;
                    let expected = if column == 0 { row.4 } else { row.5 };
                    // The reference converged to 2.1e-7 per bin, 1e-5 of the peak;
                    // the plan's acceptance for binned cascades is 1e-3.
                    assert!(
                        (count - expected).abs() <= 1e-5 * peak,
                        "{dark}/{mediator} F0={} [{}, {}): {count} vs {expected}",
                        column,
                        row.2,
                        row.3
                    );
                    spectra[column].push(count);
                }
            }
            spectra[2] = spectra[0]
                .iter()
                .zip(&spectra[1])
                .map(|(t, l)| (2.0 * t + l) / 3.0)
                .collect();
            let tv = total_variation(&spectra[0], &spectra[2]);
            assert!((tv - expected_tv).abs() < 5e-5, "{tv}");
        }
    }
}
