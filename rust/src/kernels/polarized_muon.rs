//! Polarized muon decay and the charged-pion chain that feeds it.
//!
//! The visible-axial-spectra project's `polarized_muon` kernel. Hazma's
//! [`super::positron_muon`] and [`super::neutrino_muon`] decay an
//! unpolarized muon, but the muon of `π → μν` is fully polarized and one
//! from a chiral `μ⁺μ⁻` pair partly so. The spin moves the mean e⁺ energy
//! per π⁺ at rest from 38.430221 MeV to 36.940693 MeV (the mediator note's
//! decay-kernel section, `sec:k22`).
//!
//! # The decay density
//!
//! In the rest frame of a μ⁻ with spin along `ẑ`, the Born V−A decay with
//! massless neutrinos and a finite electron mass gives each daughter the
//! density `d²N/(dx dcos θ) = a(x) + b(x) cos θ`, with `x = 2E/m_μ` its
//! total energy, `θ` its angle to the spin, `r = (m_e/m_μ)²` and
//! `F = 1 − 8r + 8r³ − r⁴ − 12r² ln r`:
//!
//! * e⁻: `a = √(x² − 4r)[x(3 − 2x) + r(3x − 4)]/F` and
//!   `b = (x² − 4r)(1 + 3r − 2x)/F`, for `2√r ≤ x ≤ 1 + r`;
//! * ν̄_e: `a = b = 6x²(q − r)²/(qF)` with `q = 1 − x`, for `0 ≤ x ≤ 1 − r`;
//! * ν_μ: `a = K[A(1 − x/2) + B]` and `b = K[B − Ax/2]`, with
//!   `A = (q − r)(q + 2r)/(6q²)`, `B = (q − r)²/(12q)` and
//!   `K = 12x²(1 − r/q)/F`, for `0 ≤ x ≤ 1 − r`.
//!
//! Each density integrates to one daughter per decay. CP maps a μ⁺ onto
//! the same densities for e⁺, ν_e and ν̄_μ with the spin reversed, so one
//! *effective polarization* `P` serves both charges: about the muon's
//! flight direction the density is `a + P b cos θ`, where `P` is the
//! helicity of a μ⁻ and minus the helicity of a μ⁺. The muon of `π → μν`
//! has `P = 1` for either charge.
//!
//! # Boosts
//!
//! A daughter of rest-frame energy `e` and momentum `k` reaches the lab
//! energies `γ(e + β k cos θ)`, linear in `cos θ`. A muon in flight
//! therefore has `dN/dE = ∫dx (a + P b cos θ)/(γβk)` over the rest-frame
//! energies whose boost window contains `E`, and a charged pion, whose
//! rest-frame spectrum is isotropic, has
//! `dN/dE = ∫dt f(t)/(2γβk)` over the same kind of window. Both integrals
//! are taken over the rest-frame momentum `k`, which removes the square
//! root at a massive daughter's threshold. Count bins replace the
//! `cos θ` density by its integral over the part of the window inside the
//! bin, so a bin of width `ΔE` holds `dN/dE ΔE` as `ΔE → 0`.
//!
//! # Closed forms for the neutrinos
//!
//! A neutrino's `a` and `b` are `x²` times Laurent polynomials in
//! `q = 1 − x` with powers −3 to 3: `a = b = 6x²(r²/q − 2r + q)/F` for
//! ν̄_e, and for ν_μ `KA = 2x²(1 − 3r²/q² + 2r³/q³)/F` and
//! `KB = x²(q − 3r + 3r²/q − r³/q²)/F`. Along a boost window
//! `cos θ = α + β'/x` is linear in `1/x`, so once the `x²` pays for the
//! `1/e` and `1/e²` of the angular factors every muon-level integral above
//! is a Laurent polynomial in `q`, whose antiderivative is a polynomial
//! plus `ln q`. [`MuonDecay::dnde`] and [`MuonDecay::counts`] take that
//! closed form for the neutrinos and keep the adaptive quadrature for the
//! massive charged lepton, whose `√(x² − 4r)` has no such form. The
//! quadrature stays available to the tests as the closed form's oracle.
//! Both forms know a window's edges only to the machine epsilon `ε` of the
//! lab energies, so a bin of relative width `w` from a muon of speed `β`
//! carries a relative error of at most about `ε/(βw)`. For `w = 10⁻⁶` the
//! closed forms and the quadrature differ by 7e-11 at `γβ = 10⁻³`, 6e-8
//! at `10⁻⁵` and 9e-5 at `10⁻⁷`, and a muon at rest agrees to 1.4e-10 on
//! the bin from 10⁻⁶ MeV to 2·10⁻⁶ MeV.
//!
//! Lines are kept apart from the continuum. A prompt line of a pion in
//! flight is a flat box; at rest it is a delta function, which only
//! [`PionDecay::counts`] can represent, so [`PionDecay::dnde`] at rest is
//! the continuum alone. Masses and energies are in MeV, densities in
//! MeV⁻¹ and counts are particles per decay. The caller validates every
//! input: parent energies at or above their masses and bin edges ordered.

use crate::constants::pdg::{BR_PI_TO_E_NUE, BR_PI_TO_MU_NUMU, MASS_E, MASS_MU, MASS_PI};
use crate::quad::{QuadOpts, quad};

/// Relative tolerance of the integral over the muon's rest-frame
/// daughter momentum. Its integrand is smooth, so a single 21-point
/// Kronrod panel usually meets it.
const MUON_EPSREL: f64 = 1e-12;

/// Relative tolerance of the boost integral of a pion's rest-frame
/// continuum, whose integrand is itself a [`MUON_EPSREL`] integral.
const PION_EPSREL: f64 = 1e-10;

/// Subinterval limit of every adaptive integral here.
const QUAD_LIMIT: usize = 200;

/// Integrate `f` over `[a, b]` with break points `points`, returning
/// `NaN` if the options are rejected. They never are: the tolerance and
/// limit are constants that `quad` accepts, and break points outside
/// `[a, b]` are filtered.
pub(super) fn integrate<F>(f: &mut F, a: f64, b: f64, points: &[f64], epsrel: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    let opts = QuadOpts {
        epsabs: 0.0,
        epsrel,
        limit: QUAD_LIMIT,
        points: Some(points),
    };
    quad(f, a, b, &opts).map_or(f64::NAN, |outcome| outcome.value)
}

/// Return the momentum of a particle of `energy` and `mass`, zero at or
/// below rest.
pub(super) fn momentum(energy: f64, mass: f64) -> f64 {
    ((energy - mass) * (energy + mass)).max(0.0).sqrt()
}

fn energy_from_momentum(momentum: f64, mass: f64) -> f64 {
    momentum.hypot(mass)
}

/// Return the lab energies `[γ(e − βk), γ(e + βk)]` of a daughter with
/// rest-frame `energy` and `mass` from a parent with Lorentz factor
/// `gamma` and `gamma_beta = γβ`.
///
/// The lower edge is rationalized as `(e² + (γβ)²m²)/(γe + γβk)` so it
/// keeps full precision when it nearly cancels. The same interval is the
/// set of rest-frame energies that boost to a lab `energy`, since the
/// inverse boost differs only in the sign of `β`. A massless daughter of
/// zero energy has the empty interval at zero.
pub(super) fn boost_range(energy: f64, mass: f64, gamma: f64, gamma_beta: f64) -> (f64, f64) {
    let upper = gamma * energy + gamma_beta * momentum(energy, mass);
    if upper == 0.0 {
        return (0.0, 0.0);
    }
    let lower = (energy * energy + (gamma_beta * mass).powi(2)) / upper;
    (lower, upper)
}

/// Return the fraction of an isotropic daughter's flat box, from a
/// daughter of rest-frame `energy` and `mass` boosted by `gamma`,
/// `gamma_beta > 0`, that lands in the lab bin `[low, high]`.
pub(super) fn box_fraction(
    low: f64,
    high: f64,
    energy: f64,
    mass: f64,
    gamma: f64,
    gamma_beta: f64,
) -> f64 {
    let (lower, upper) = boost_range(energy, mass, gamma, gamma_beta);
    let overlap = (high.min(upper) - low.max(lower)).max(0.0);
    overlap / (2.0 * gamma_beta * momentum(energy, mass))
}

/// Return the density in MeV⁻¹ at lab `probe` of the same flat box, with
/// both edges included.
pub(super) fn box_density(probe: f64, energy: f64, mass: f64, gamma: f64, gamma_beta: f64) -> f64 {
    let (lower, upper) = boost_range(energy, mass, gamma, gamma_beta);
    if lower <= probe && probe <= upper {
        1.0 / (2.0 * gamma_beta * momentum(energy, mass))
    } else {
        0.0
    }
}

/// Return the smallest parent Lorentz factor at which a daughter of
/// rest-frame `energy` and `mass` reaches the lab energy `probe > 0`.
///
/// It is the root of `|probe − γe| = γβk` below `probe/e`, written
/// `(E² + k²)/(Ee + k p_E)` so that it holds for a massless daughter.
/// Integrals over a parent's energy break at it, where a box edge or a
/// continuum endpoint crosses `probe`.
pub(super) fn threshold_gamma(probe: f64, energy: f64, mass: f64) -> f64 {
    let k = momentum(energy, mass);
    (probe * probe + k * k) / (probe * energy + k * momentum(probe, mass))
}

/// Return the rest-frame energies in MeV whose boxes, from a daughter of
/// `mass` boosted by `gamma`, `gamma_beta > 0`, can reach the lab bin
/// `[low, high]`, as `(start, stop)`, with the three energies inside at
/// which a box edge crosses a bin edge.
///
/// A box's upper edge rises from `γm` with the rest-frame energy, so every
/// box reaches a `low` below `γm`, and above it the boxes from the lower
/// root of [`boost_range`] up do. A box's lower edge falls from `γm` to
/// `m` and rises again, so it crosses `low` at both roots and `high` at
/// its lower root.
fn bin_window(low: f64, high: f64, mass: f64, gamma: f64, gamma_beta: f64) -> (f64, f64, [f64; 3]) {
    let (low_lower, low_upper) = boost_range(low, mass, gamma, gamma_beta);
    let (high_lower, high_upper) = boost_range(high, mass, gamma, gamma_beta);
    let start = if low > gamma * mass { low_lower } else { mass };
    (start, high_upper, [low_lower, low_upper, high_lower])
}

/// `Σₙ cₙ qⁿ + Σₖ pₖ uᵏ` with `q = 1 − x` and `u = x − x₀`, `n` from −3 to
/// −1 and `k` from 0 to 3: a Laurent polynomial in `q` whose polynomial
/// part is expanded about a center `x₀`.
///
/// Every neutrino integrand of this module has this form once the `x²`
/// of its density is spent on the `1/e` and `1/e²` of the boost (see the
/// module docs), so its integrals are closed forms. The poles are the
/// endpoint layer at `q → r`, where `q` is the small variable. The
/// polynomial part is expanded about the middle of the interval it is
/// integrated over: the angular factors `x cos θ = β' + αx` have `α` and
/// `β'` of order `1/β` that cancel to order one along the window, so
/// about `x = 0` their square would carry coefficients of order `1/β²`
/// cancelling against one another, while about `x₀` every coefficient is
/// of the order of the values it describes.
#[derive(Clone, Copy, Debug)]
struct Laurent {
    /// The center `x₀` of the polynomial part.
    center: f64,
    /// The coefficients of `q⁻³`, `q⁻²` and `q⁻¹`.
    poles: [f64; 3],
    /// The coefficients of `1`, `u`, `u²` and `u³`.
    polynomial: [f64; 4],
}

/// `C(n, k)` for `n` and `k` at most 3.
const BINOMIAL: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 2.0, 1.0, 0.0],
    [1.0, 3.0, 3.0, 1.0],
];

impl Laurent {
    /// Zero, expanded about `center`.
    fn zero(center: f64) -> Self {
        Self {
            center,
            poles: [0.0; 3],
            polynomial: [0.0; 4],
        }
    }

    /// Add the pole `coefficient · q^power` for `power` from −3 to −1.
    fn with_pole(mut self, power: i32, coefficient: f64) -> Self {
        self.poles[(power + 3) as usize] += coefficient;
        self
    }

    /// Add the polynomial `Σ pₖ xᵏ` in `x` itself, re-expanded about the
    /// center.
    fn with_polynomial(mut self, polynomial: &[f64]) -> Self {
        for (k, &coefficient) in polynomial.iter().enumerate() {
            self.add_shifted_power(coefficient, self.center, k);
        }
        self
    }

    /// Add `coefficient · (u + shift)ᵏ` to the polynomial part.
    fn add_shifted_power(&mut self, coefficient: f64, shift: f64, k: usize) {
        for (i, term) in self.polynomial.iter_mut().enumerate().take(k + 1) {
            *term += coefficient * BINOMIAL[k][i] * shift.powi((k - i) as i32);
        }
    }

    fn add(mut self, other: Self) -> Self {
        debug_assert!(
            self.center == other.center,
            "expanded about different centers"
        );
        for (term, addend) in self.poles.iter_mut().zip(other.poles) {
            *term += addend;
        }
        for (term, addend) in self.polynomial.iter_mut().zip(other.polynomial) {
            *term += addend;
        }
        self
    }

    fn scale(mut self, factor: f64) -> Self {
        for term in self.poles.iter_mut().chain(&mut self.polynomial) {
            *term *= factor;
        }
        self
    }

    /// Multiply by the polynomial `Σ pₖ uᵏ` in `u`, whose degree must keep
    /// the product within the cubic.
    fn times(self, polynomial: &[f64]) -> Self {
        let mut product = Self::zero(self.center);
        for (k, &factor) in polynomial.iter().enumerate() {
            if factor == 0.0 {
                continue;
            }
            for (index, &term) in self.polynomial.iter().enumerate() {
                if term != 0.0 {
                    product.polynomial[index + k] += term * factor;
                }
            }
            for (index, &term) in self.poles.iter().enumerate() {
                if term != 0.0 {
                    product.add_pole_times_power(index as i32 - 3, term * factor, k);
                }
            }
        }
        product
    }

    /// Add `coefficient · qⁿ uᵏ`, with `uᵏ = (q₀ − q)ᵏ` for `q₀ = 1 − x₀`
    /// expanded in `q`, and any power of `q` that comes out nonnegative
    /// returned to `u` as `(q₀ − u)^m`.
    fn add_pole_times_power(&mut self, n: i32, coefficient: f64, k: usize) {
        let q0 = 1.0 - self.center;
        for (j, &binomial) in BINOMIAL[k].iter().enumerate().take(k + 1) {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            let term = coefficient * binomial * sign * q0.powi((k - j) as i32);
            let power = n + j as i32;
            if power < 0 {
                self.poles[(power + 3) as usize] += term;
                continue;
            }
            let m = power as usize;
            for (i, polynomial) in self.polynomial.iter_mut().enumerate().take(m + 1) {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                *polynomial += term * BINOMIAL[m][i] * sign * q0.powi((m - i) as i32);
            }
        }
    }

    /// Return `∫ dx` from `x_from` to `x_to`, both in `[0, 1]`, with the
    /// poles integrated in `q`. A pole whose coefficient is zero is
    /// skipped, so the massless-electron limit, where `q` reaches zero,
    /// never forms `0 · ∞`.
    fn integral(&self, x_from: f64, x_to: f64) -> f64 {
        let (u_from, u_to) = (x_from - self.center, x_to - self.center);
        // The length is formed once from the `x` edges, which carry the
        // caller's precision; differencing `u` or `q` values near 1 would
        // add an absolute error of the order of the machine epsilon that
        // the poles, whose coefficients grow as `1/β²`, amplify.
        let length = x_to - x_from;
        let polynomial: f64 = self
            .polynomial
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| **coefficient != 0.0)
            .map(|(k, coefficient)| coefficient * power_integral(k as i32, u_from, u_to, length))
            .sum();
        // `dx = −dq` turns the range `[x_from, x_to]` into `[q(x_to), q(x_from)]`.
        let poles: f64 = self
            .poles
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| **coefficient != 0.0)
            .map(|(index, coefficient)| {
                coefficient * power_integral(index as i32 - 3, 1.0 - x_to, 1.0 - x_from, length)
            })
            .sum();
        polynomial + poles
    }
}

/// Return `∫ tⁿ dt` from `t_from` to `t_to`, both positive when `n` is
/// negative.
///
/// The caller supplies `length = t_to − t_from`, because two edges near 1
/// carry an absolute rounding error, and a length recomputed from them
/// would carry it too. The difference of the two antiderivatives is
/// factored through `length`, so a narrow window loses no more precision
/// than that length does.
fn power_integral(power: i32, t_from: f64, t_to: f64, length: f64) -> f64 {
    if power == -1 {
        return (length / t_from).ln_1p();
    }
    // `∫ qⁿ = q^m / m` with `m = n + 1`, and
    // `t_to^m − t_from^m = (t_to − t_from) Σⱼ t_toʲ t_from^(m−1−j)` for
    // `m > 0`; a negative `m` divides the same sum by `(t_to t_from)^|m|`.
    let m = power + 1;
    let degree = m.abs();
    let sum: f64 = (0..degree)
        .map(|j| t_to.powi(j) * t_from.powi(degree - 1 - j))
        .sum();
    let power_difference = if m > 0 {
        length * sum
    } else {
        -length * sum / (t_to * t_from).powi(degree)
    };
    power_difference / f64::from(m)
}

/// The unclamped `cos θ = alpha + beta_prime / x` at which a neutrino of
/// rest energy `x m_μ/2` reaches one lab energy, as coefficients in `x`.
#[derive(Clone, Copy, Debug)]
struct CosineEdge {
    alpha: f64,
    beta_prime: f64,
}

impl CosineEdge {
    const FORWARD: Self = Self {
        alpha: 1.0,
        beta_prime: 0.0,
    };
    const BACKWARD: Self = Self {
        alpha: -1.0,
        beta_prime: 0.0,
    };

    /// The edge of lab energy `lab` from a muon of `gamma` and
    /// `gamma_beta > 0`: `(lab − γe)/(γβe)` with `e = x m_μ/2`.
    fn at(lab: f64, muon_mass: f64, gamma: f64, gamma_beta: f64) -> Self {
        Self {
            alpha: -gamma / gamma_beta,
            beta_prime: 2.0 * lab / (gamma_beta * muon_mass),
        }
    }

    /// The edge of `lab` as it applies between the rest energies `from`
    /// and `to`, where it is either clamped throughout or unclamped
    /// throughout: the crossings are the interval's ends.
    fn between(lab: f64, from: f64, to: f64, muon_mass: f64, gamma: f64, gamma_beta: f64) -> Self {
        let middle = (from + to) / 2.0;
        let cosine = (lab - gamma * middle) / (gamma_beta * middle);
        if cosine >= 1.0 {
            Self::FORWARD
        } else if cosine <= -1.0 {
            Self::BACKWARD
        } else {
            Self::at(lab, muon_mass, gamma, gamma_beta)
        }
    }

    /// Return `x cos θ = β' + αx` at `x`, which is of order one along the
    /// window even where `α` and `β'` are of order `1/β`.
    fn scaled_cosine(&self, x: f64) -> f64 {
        self.alpha.mul_add(x, self.beta_prime)
    }
}

/// A muon daughter, named for the μ⁻ decay; a μ⁺ gives its antiparticle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Daughter {
    /// The charged lepton: e⁻ from μ⁻, e⁺ from μ⁺.
    ChargedLepton,
    /// The electron-flavor neutrino: ν̄_e from μ⁻, ν_e from μ⁺.
    ElectronNeutrino,
    /// The muon-flavor neutrino: ν_μ from μ⁻, ν̄_μ from μ⁺.
    MuonNeutrino,
}

impl Daughter {
    /// Return the neutrino this daughter is, if it is one.
    fn neutrino(self) -> Option<Neutrino> {
        match self {
            Self::ChargedLepton => None,
            Self::ElectronNeutrino => Some(Neutrino::Electron),
            Self::MuonNeutrino => Some(Neutrino::Muon),
        }
    }
}

/// The two massless daughters, whose integrals are closed forms.
#[derive(Clone, Copy, Debug)]
enum Neutrino {
    Electron,
    Muon,
}

impl Neutrino {
    fn daughter(self) -> Daughter {
        match self {
            Self::Electron => Daughter::ElectronNeutrino,
            Self::Muon => Daughter::MuonNeutrino,
        }
    }
}

/// The Born decay of a polarized muon.
#[derive(Clone, Copy, Debug)]
pub struct MuonDecay {
    /// Muon mass in MeV, positive.
    pub muon_mass: f64,
    /// Electron mass in MeV: zero, or at least about 1e-5, and below half
    /// the muon mass. A smaller nonzero mass makes `r = (m_e/m_μ)²`
    /// comparable to the machine epsilon, so the neutrino closed forms
    /// reach `q = 0` with nonzero pole coefficients and return infinity or
    /// NaN.
    pub electron_mass: f64,
}

impl MuonDecay {
    /// The decay with `hazma.parameters`' masses.
    pub const PDG: Self = Self {
        muon_mass: MASS_MU,
        electron_mass: MASS_E,
    };

    fn mass_ratio(&self) -> f64 {
        (self.electron_mass / self.muon_mass).powi(2)
    }

    /// The phase-space factor `F` that normalizes every density.
    fn width_factor(&self) -> f64 {
        let r = self.mass_ratio();
        let logarithm = if r == 0.0 { 0.0 } else { r * r * r.ln() };
        1.0 - 8.0 * r + 8.0 * r.powi(3) - r.powi(4) - 12.0 * logarithm
    }

    /// Return the mass of `daughter` in MeV.
    pub fn daughter_mass(&self, daughter: Daughter) -> f64 {
        match daughter {
            Daughter::ChargedLepton => self.electron_mass,
            Daughter::ElectronNeutrino | Daughter::MuonNeutrino => 0.0,
        }
    }

    /// Return the largest rest-frame energy of `daughter` in MeV.
    pub fn max_energy(&self, daughter: Daughter) -> f64 {
        let r = self.mass_ratio();
        match daughter {
            Daughter::ChargedLepton => self.muon_mass * (1.0 + r) / 2.0,
            Daughter::ElectronNeutrino | Daughter::MuonNeutrino => self.muon_mass * (1.0 - r) / 2.0,
        }
    }

    /// Return break points, as rest-frame momenta in MeV, that resolve the
    /// neutrino densities' layer of width `r` at their endpoint
    /// `q = 1 − x → r`, where factors such as `(q − r)/q` vary on the
    /// scale `r` rather than one. They sit at `q = 2r, 10r, 100r, 1000r`;
    /// the charged lepton has no such layer.
    fn endpoint_points(&self, daughter: Daughter) -> Vec<f64> {
        if daughter == Daughter::ChargedLepton {
            return Vec::new();
        }
        let r = self.mass_ratio();
        [2.0, 10.0, 100.0, 1000.0]
            .iter()
            .map(|multiple| self.muon_mass * (1.0 - multiple * r) / 2.0)
            .collect()
    }

    /// Return the dimensionless `(a, b)` of `d²N/(dx dcos θ) = a + b cos θ`
    /// at `x = 2E/m_μ`, zero outside the daughter's support. See the module
    /// docs for the formulas.
    pub fn rest_coefficients(&self, daughter: Daughter, x: f64) -> (f64, f64) {
        let r = self.mass_ratio();
        let f = self.width_factor();
        match daughter {
            Daughter::ChargedLepton => {
                let momentum2 = x * x - 4.0 * r;
                if momentum2 < 0.0 || x > 1.0 + r {
                    return (0.0, 0.0);
                }
                let a = momentum2.sqrt() * (x * (3.0 - 2.0 * x) + r * (3.0 * x - 4.0)) / f;
                let b = momentum2 * (1.0 + 3.0 * r - 2.0 * x) / f;
                (a, b)
            }
            Daughter::ElectronNeutrino => {
                let q = 1.0 - x;
                if x < 0.0 || q <= r {
                    return (0.0, 0.0);
                }
                let a = 6.0 * x * x * (q - r).powi(2) / (q * f);
                (a, a)
            }
            Daughter::MuonNeutrino => {
                let q = 1.0 - x;
                if x < 0.0 || q <= r {
                    return (0.0, 0.0);
                }
                let big_a = (q - r) * (q + 2.0 * r) / (6.0 * q * q);
                let big_b = (q - r).powi(2) / (12.0 * q);
                let k = 12.0 * x * x * (1.0 - r / q) / f;
                (
                    k * (big_a * (1.0 - x / 2.0) + big_b),
                    k * (big_b - big_a * x / 2.0),
                )
            }
        }
    }

    /// Return `(ã, b̃)` with `a = x² ã` and `b = x² b̃` for `neutrino`,
    /// whose coefficients are Laurent polynomials in `q = 1 − x` once the
    /// `x²` is set aside (see the module docs), expanded about `center`.
    fn neutrino_coefficients(&self, neutrino: Neutrino, center: f64) -> (Laurent, Laurent) {
        let r = self.mass_ratio();
        let f = self.width_factor();
        match neutrino {
            Neutrino::Electron => {
                // `6(q − r)²/q = 6(r²/q + (1 − 2r) − x)`.
                let a = Laurent::zero(center)
                    .with_pole(-1, r * r)
                    .with_polynomial(&[1.0 - 2.0 * r, -1.0])
                    .scale(6.0 / f);
                (a, a)
            }
            Neutrino::Muon => {
                // `K A = 2x²(2r³/q³ − 3r²/q² + 1)/F` and
                // `K B = x²(−r³/q² + 3r²/q + (1 − 3r) − x)/F`.
                let ka = Laurent::zero(center)
                    .with_pole(-3, 2.0 * r.powi(3))
                    .with_pole(-2, -3.0 * r * r)
                    .with_polynomial(&[1.0])
                    .scale(2.0 / f);
                let kb = Laurent::zero(center)
                    .with_pole(-2, -r.powi(3))
                    .with_pole(-1, 3.0 * r * r)
                    .with_polynomial(&[1.0 - 3.0 * r, -1.0])
                    .scale(1.0 / f);
                // `1 − x/2 = (1 − x₀/2) − u/2` and `x/2 = x₀/2 + u/2`.
                let a = ka.times(&[1.0 - center / 2.0, -0.5]).add(kb);
                let b = kb.add(ka.times(&[-center / 2.0, -0.5]));
                (a, b)
            }
        }
    }

    /// Return `x = 2E/m_μ` of a rest-frame `energy`.
    fn x(&self, energy: f64) -> f64 {
        2.0 * energy / self.muon_mass
    }

    /// Return `dN/dE` in MeV⁻¹ of `daughter` at lab `energy` from a muon
    /// of lab `muon_energy` and effective `polarization` in `[−1, 1]`.
    pub fn dnde(
        &self,
        daughter: Daughter,
        energy: f64,
        muon_energy: f64,
        polarization: f64,
    ) -> f64 {
        match daughter.neutrino() {
            Some(neutrino) => self.dnde_neutrino(neutrino, energy, muon_energy, polarization),
            None => self.dnde_quadrature(daughter, energy, muon_energy, polarization),
        }
    }

    /// [`Self::dnde`] for a neutrino in closed form: on the boost window
    /// `cos θ = α + β'/x`, so `(a + P b cos θ)/e` is
    /// `(2/m_μ)[x ã + P b̃ (β' + αx)]`, a [`Laurent`] form.
    fn dnde_neutrino(
        &self,
        neutrino: Neutrino,
        energy: f64,
        muon_energy: f64,
        polarization: f64,
    ) -> f64 {
        let daughter = neutrino.daughter();
        let gamma_beta = momentum(muon_energy, self.muon_mass) / self.muon_mass;
        if gamma_beta == 0.0 {
            let (a, _) = self.rest_coefficients(daughter, self.x(energy));
            return 4.0 * a / self.muon_mass;
        }
        if energy < 0.0 {
            return 0.0;
        }
        let gamma = muon_energy / self.muon_mass;
        let (lower, upper) = boost_range(energy, 0.0, gamma, gamma_beta);
        let (low, high) = (lower.max(0.0), upper.min(self.max_energy(daughter)));
        if low >= high {
            return 0.0;
        }
        let (x_low, x_high) = (self.x(low), self.x(high));
        let center = (x_low + x_high) / 2.0;
        let (a, b) = self.neutrino_coefficients(neutrino, center);
        let edge = CosineEdge::at(energy, self.muon_mass, gamma, gamma_beta);
        // `x = x₀ + u` and `β' + αx = (β' + αx₀) + αu`.
        let density = a.times(&[center, 1.0]).add(
            b.times(&[edge.scaled_cosine(center), edge.alpha])
                .scale(polarization),
        );
        // `∫ de = (m_μ/2) ∫ dx` cancels the `2/m_μ`, and the boost divides
        // by `(m_μ/2) γβ` as in quadrature.
        2.0 * density.integral(x_low, x_high) / (self.muon_mass * gamma_beta)
    }

    /// [`Self::dnde`] by adaptive quadrature over the rest-frame momentum,
    /// the only path for the massive charged lepton and the oracle of
    /// [`Self::dnde_neutrino`].
    fn dnde_quadrature(
        &self,
        daughter: Daughter,
        energy: f64,
        muon_energy: f64,
        polarization: f64,
    ) -> f64 {
        let mass = self.daughter_mass(daughter);
        let gamma_beta = momentum(muon_energy, self.muon_mass) / self.muon_mass;
        if gamma_beta == 0.0 {
            // At rest the angular term integrates away.
            let (a, _) = self.rest_coefficients(daughter, 2.0 * energy / self.muon_mass);
            return 4.0 * a / self.muon_mass;
        }
        if energy < mass {
            return 0.0;
        }
        let gamma = muon_energy / self.muon_mass;
        let (lower, upper) = boost_range(energy, mass, gamma, gamma_beta);
        let (low, high) = (lower.max(mass), upper.min(self.max_energy(daughter)));
        if low >= high {
            return 0.0;
        }
        let mut integrand = |k: f64| {
            let e = energy_from_momentum(k, mass);
            let (a, b) = self.rest_coefficients(daughter, 2.0 * e / self.muon_mass);
            let cos = ((energy - gamma * e) / (gamma_beta * k)).clamp(-1.0, 1.0);
            polarization.mul_add(b * cos, a) / e
        };
        let integral = integrate(
            &mut integrand,
            momentum(low, mass),
            momentum(high, mass),
            &self.endpoint_points(daughter),
            MUON_EPSREL,
        );
        2.0 * integral / (self.muon_mass * gamma_beta)
    }

    /// Return the number of `daughter`s per decay in the lab bin
    /// `[low, high]` from a muon of lab `muon_energy` and effective
    /// `polarization`.
    pub fn counts(
        &self,
        daughter: Daughter,
        low: f64,
        high: f64,
        muon_energy: f64,
        polarization: f64,
    ) -> f64 {
        match daughter.neutrino() {
            Some(neutrino) => self.counts_neutrino(neutrino, low, high, muon_energy, polarization),
            None => self.counts_quadrature(daughter, low, high, muon_energy, polarization),
        }
    }

    /// [`Self::counts`] for a neutrino in closed form. Between two
    /// consecutive crossings each bin edge's `cos θ` is clamped or
    /// `α + β'/x` throughout, so with `c = x cos θ` the angular integral
    /// `a Δcos θ + P b Δ(cos² θ)/2 = x ã Δc + P b̃ Δ(c²)/2` is a
    /// [`Laurent`] form, and the bin is a sum of closed forms.
    fn counts_neutrino(
        &self,
        neutrino: Neutrino,
        low: f64,
        high: f64,
        muon_energy: f64,
        polarization: f64,
    ) -> f64 {
        let max_energy = self.max_energy(neutrino.daughter());
        let gamma_beta = momentum(muon_energy, self.muon_mass) / self.muon_mass;
        if gamma_beta == 0.0 {
            let (low, high) = (low.max(0.0), high.min(max_energy));
            if low >= high {
                return 0.0;
            }
            let (x_low, x_high) = (self.x(low), self.x(high));
            let center = (x_low + x_high) / 2.0;
            let (a, _) = self.neutrino_coefficients(neutrino, center);
            // `∫ 2a dx` with `a = x² ã` and `x² = x₀² + 2x₀u + u²`.
            return a
                .times(&[2.0 * center * center, 4.0 * center, 2.0])
                .integral(x_low, x_high);
        }
        let gamma = muon_energy / self.muon_mass;
        let (start, stop, crossings) = bin_window(low, high, 0.0, gamma, gamma_beta);
        let stop = stop.min(max_energy);
        if start >= stop {
            return 0.0;
        }
        // The window starts at the lower crossing of `low`; the other two
        // crossings fall in either order.
        let [_, low_upper, high_lower] = crossings;
        let points = [
            start,
            low_upper.min(high_lower).clamp(start, stop),
            low_upper.max(high_lower).clamp(start, stop),
            stop,
        ];
        let mut total = 0.0;
        for pair in points.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            if from >= to {
                continue;
            }
            let (x_from, x_to) = (self.x(from), self.x(to));
            let center = (x_from + x_to) / 2.0;
            let (a, b) = self.neutrino_coefficients(neutrino, center);
            let lower_edge = CosineEdge::between(low, from, to, self.muon_mass, gamma, gamma_beta);
            let upper_edge = CosineEdge::between(high, from, to, self.muon_mass, gamma, gamma_beta);
            // With `c = c₀ + αu` for each edge, `x Δc = (x₀ + u)(Δc₀ + Δα u)`
            // and `Δ(c²) = Δ(c₀²) + 2Δ(c₀α) u + Δ(α²) u²`.
            let (lower_c, upper_c) = (
                lower_edge.scaled_cosine(center),
                upper_edge.scaled_cosine(center),
            );
            let delta_alpha = upper_edge.alpha - lower_edge.alpha;
            let delta_c = upper_c - lower_c;
            let delta_c2 = delta_c * (upper_c + lower_c);
            let delta_cross = upper_c * upper_edge.alpha - lower_c * lower_edge.alpha;
            let delta_alpha2 = delta_alpha * (upper_edge.alpha + lower_edge.alpha);
            let angular = a
                .times(&[
                    center * delta_c,
                    delta_c + center * delta_alpha,
                    delta_alpha,
                ])
                .add(
                    b.times(&[delta_c2, 2.0 * delta_cross, delta_alpha2])
                        .scale(polarization / 2.0),
                );
            // `(2/m_μ) ∫ de = ∫ dx`.
            total += angular.integral(x_from, x_to);
        }
        total
    }

    /// [`Self::counts`] by adaptive quadrature over the rest-frame
    /// momentum, the only path for the massive charged lepton and the
    /// oracle of [`Self::counts_neutrino`].
    fn counts_quadrature(
        &self,
        daughter: Daughter,
        low: f64,
        high: f64,
        muon_energy: f64,
        polarization: f64,
    ) -> f64 {
        let mass = self.daughter_mass(daughter);
        let max_energy = self.max_energy(daughter);
        let scale = 2.0 / self.muon_mass;
        let gamma_beta = momentum(muon_energy, self.muon_mass) / self.muon_mass;
        if gamma_beta == 0.0 {
            let (low, high) = (low.max(mass), high.min(max_energy));
            if low >= high {
                return 0.0;
            }
            let mut integrand = |k: f64| {
                let e = energy_from_momentum(k, mass);
                let (a, _) = self.rest_coefficients(daughter, scale * e);
                2.0 * a * scale * k / e
            };
            return integrate(
                &mut integrand,
                momentum(low, mass),
                momentum(high, mass),
                &self.endpoint_points(daughter),
                MUON_EPSREL,
            );
        }
        let gamma = muon_energy / self.muon_mass;
        let (start, stop, crossings) = bin_window(low, high, mass, gamma, gamma_beta);
        let stop = stop.min(max_energy);
        if start >= stop {
            return 0.0;
        }
        let mut integrand = |k: f64| {
            let e = energy_from_momentum(k, mass);
            let (a, b) = self.rest_coefficients(daughter, scale * e);
            let edge = |lab: f64| ((lab - gamma * e) / (gamma_beta * k)).clamp(-1.0, 1.0);
            let (cos_low, cos_high) = (edge(low), edge(high));
            let angular = a * (cos_high - cos_low)
                + polarization * b * (cos_high * cos_high - cos_low * cos_low) / 2.0;
            angular * scale * k / e
        };
        let mut points = self.endpoint_points(daughter);
        points.extend(crossings.map(|energy| momentum(energy, mass)));
        integrate(
            &mut integrand,
            momentum(start, mass),
            momentum(stop, mass),
            &points,
            MUON_EPSREL,
        )
    }
}

/// The decay of a charged pion through `μν`, with its polarized muon,
/// and through `eν`.
#[derive(Clone, Copy, Debug)]
pub struct PionDecay {
    /// Charged-pion mass in MeV, above the muon mass.
    pub pion_mass: f64,
    /// The decay of the daughter muon.
    pub muon: MuonDecay,
    /// Branching fraction into `μν`.
    pub br_muon: f64,
    /// Branching fraction into `eν`.
    pub br_electron: f64,
}

impl PionDecay {
    /// The decay with `hazma.parameters`' masses and the branching
    /// fractions of [`crate::constants::pdg`].
    pub const PDG: Self = Self {
        pion_mass: MASS_PI,
        muon: MuonDecay::PDG,
        br_muon: BR_PI_TO_MU_NUMU,
        br_electron: BR_PI_TO_E_NUE,
    };

    /// Return the muon's energy in MeV in the pion rest frame.
    pub fn muon_energy(&self) -> f64 {
        let (pion, muon) = (self.pion_mass, self.muon.muon_mass);
        (pion * pion + muon * muon) / (2.0 * pion)
    }

    /// Return the prompt line of `daughter` in the pion rest frame as
    /// `(energy in MeV, count per decay)`: the e and ν_e of `π → eν` and
    /// the ν_μ of `π → μν`.
    pub fn line(&self, daughter: Daughter) -> (f64, f64) {
        let pion = self.pion_mass;
        let two_body = |mass: f64, sign: f64| sign.mul_add(mass * mass, pion * pion) / (2.0 * pion);
        let (electron, muon) = (self.muon.electron_mass, self.muon.muon_mass);
        match daughter {
            Daughter::ChargedLepton => (two_body(electron, 1.0), self.br_electron),
            Daughter::ElectronNeutrino => (two_body(electron, -1.0), self.br_electron),
            Daughter::MuonNeutrino => (two_body(muon, -1.0), self.br_muon),
        }
    }

    /// Return the muon-decay continuum `dN/dE` in MeV⁻¹ of `daughter` at
    /// `energy` in the pion rest frame.
    pub fn rest_continuum(&self, daughter: Daughter, energy: f64) -> f64 {
        self.br_muon * self.muon.dnde(daughter, energy, self.muon_energy(), 1.0)
    }

    /// Return the `(kink, end)` of the rest-frame continuum in MeV: the
    /// muon's endpoint boosted backward and forward. Above the kink the
    /// integration window loses its upper edge.
    fn continuum_edges(&self, daughter: Daughter) -> (f64, f64) {
        let muon = self.muon.muon_mass;
        let gamma = self.muon_energy() / muon;
        let gamma_beta = momentum(self.muon_energy(), muon) / muon;
        let mass = self.muon.daughter_mass(daughter);
        boost_range(self.muon.max_energy(daughter), mass, gamma, gamma_beta)
    }

    /// Return the rest-frame energies in MeV at which a pion's lab
    /// spectrum of `daughter` has a kink or a step: the prompt line and
    /// both continuum edges.
    pub fn rest_features(&self, daughter: Daughter) -> [f64; 3] {
        let (kink, end) = self.continuum_edges(daughter);
        [self.line(daughter).0, kink, end]
    }

    /// Return `dN/dE` in MeV⁻¹ of `daughter` at lab `energy` from a pion
    /// of lab `pion_energy`. At rest the prompt line is a delta function
    /// this density omits.
    pub fn dnde(&self, daughter: Daughter, energy: f64, pion_energy: f64) -> f64 {
        let gamma_beta = momentum(pion_energy, self.pion_mass) / self.pion_mass;
        if gamma_beta == 0.0 {
            return self.rest_continuum(daughter, energy);
        }
        let gamma = pion_energy / self.pion_mass;
        let mass = self.muon.daughter_mass(daughter);
        let (line_energy, line_count) = self.line(daughter);
        let line = line_count * box_density(energy, line_energy, mass, gamma, gamma_beta);
        if energy < mass {
            return line;
        }
        let end = self.continuum_edges(daughter).1;
        let (lower, upper) = boost_range(energy, mass, gamma, gamma_beta);
        let (low, high) = (lower.max(mass), upper.min(end));
        if low >= high {
            return line;
        }
        let mut integrand = |k: f64| {
            let t = energy_from_momentum(k, mass);
            self.rest_continuum(daughter, t) / t
        };
        let continuum = integrate(
            &mut integrand,
            momentum(low, mass),
            momentum(high, mass),
            &[momentum(self.continuum_edges(daughter).0, mass)],
            PION_EPSREL,
        );
        line + continuum / (2.0 * gamma_beta)
    }

    /// Return the number of `daughter`s per decay in the lab bin
    /// `[low, high)` from a pion of lab `pion_energy`, prompt line
    /// included.
    pub fn counts(&self, daughter: Daughter, low: f64, high: f64, pion_energy: f64) -> f64 {
        let mass = self.muon.daughter_mass(daughter);
        let (line_energy, line_count) = self.line(daughter);
        let end = self.continuum_edges(daughter).1;
        let gamma_beta = momentum(pion_energy, self.pion_mass) / self.pion_mass;
        if gamma_beta == 0.0 {
            let line = if low <= line_energy && line_energy < high {
                line_count
            } else {
                0.0
            };
            let (start, stop) = (low.max(mass), high.min(end));
            if start >= stop {
                return line;
            }
            let mut integrand = |t: f64| self.rest_continuum(daughter, t);
            let kink = self.continuum_edges(daughter).0;
            return line + integrate(&mut integrand, start, stop, &[kink], PION_EPSREL);
        }
        let gamma = pion_energy / self.pion_mass;
        let line = line_count * box_fraction(low, high, line_energy, mass, gamma, gamma_beta);
        let (start, stop, crossings) = bin_window(low, high, mass, gamma, gamma_beta);
        let stop = stop.min(end);
        if start >= stop {
            return line;
        }
        let mut integrand = |k: f64| {
            let t = energy_from_momentum(k, mass);
            let fraction = box_fraction(low, high, t, mass, gamma, gamma_beta);
            self.rest_continuum(daughter, t) * fraction * k / t
        };
        let kink = self.continuum_edges(daughter).0;
        let points =
            [kink, crossings[0], crossings[1], crossings[2]].map(|energy| momentum(energy, mass));
        let continuum = integrate(
            &mut integrand,
            momentum(start, mass),
            momentum(stop, mass),
            &points,
            PION_EPSREL,
        );
        line + continuum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAUGHTERS: [Daughter; 3] = [
        Daughter::ChargedLepton,
        Daughter::ElectronNeutrino,
        Daughter::MuonNeutrino,
    ];

    /// The masses of the mediator note's decay kernel (`sec:k22`), which
    /// its reference moments were computed with: PDG 2022's muon mass,
    /// one unit in the seventh decimal above `hazma.parameters`'.
    const NOTE_PION: PionDecay = PionDecay {
        pion_mass: 139.57039,
        muon: MuonDecay {
            muon_mass: 105.6583755,
            electron_mass: 0.51099895,
        },
        br_muon: 0.999877,
        br_electron: 0.000123,
    };

    fn assert_close(actual: f64, expected: f64, rtol: f64) {
        // An expected zero, such as the massless `b` at `x = 1/2`, is
        // compared absolutely.
        let scale = if expected == 0.0 { 1.0 } else { expected.abs() };
        let error = (actual - expected).abs() / scale;
        assert!(
            error <= rtol,
            "actual {actual:.17e}, expected {expected:.17e}, relative error {error:e} > {rtol:e}"
        );
    }

    /// Return `∫ (1, x) a(x) dx` over the support, the zeroth and first
    /// moments of the rest-frame density, by a fine composite rule in the
    /// daughter's momentum.
    fn rest_moments(decay: &MuonDecay, daughter: Daughter) -> (f64, f64) {
        let mass = decay.daughter_mass(daughter);
        let scale = 2.0 / decay.muon_mass;
        let mut count = |k: f64| {
            let e = energy_from_momentum(k, mass);
            2.0 * decay.rest_coefficients(daughter, scale * e).0 * scale * k / e
        };
        let top = momentum(decay.max_energy(daughter), mass);
        let zeroth = integrate(&mut count, 0.0, top, &[], 1e-13);
        let mut first = |k: f64| energy_from_momentum(k, mass) * count(k);
        (zeroth, integrate(&mut first, 0.0, top, &[], 1e-13))
    }

    /// Every density integrates to one daughter per decay, at the physical
    /// electron mass and in the massless limit. Quadrature only, so 1e-12.
    #[test]
    fn michel_densities_are_normalized() {
        for decay in [
            MuonDecay::PDG,
            MuonDecay {
                electron_mass: 0.0,
                ..MuonDecay::PDG
            },
        ] {
            for daughter in DAUGHTERS {
                assert_close(rest_moments(&decay, daughter).0, 1.0, 1e-12);
            }
        }
    }

    /// The three rest-frame mean energies of a muon decay sum to its mass,
    /// and the massless e and ν_μ share the Michel mean `7m_μ/20`.
    #[test]
    fn rest_frame_energies_sum_to_the_muon_mass() {
        let decay = MuonDecay::PDG;
        let total: f64 = DAUGHTERS
            .iter()
            .map(|&daughter| rest_moments(&decay, daughter).1)
            .sum();
        assert_close(total, decay.muon_mass, 1e-12);

        let massless = MuonDecay {
            electron_mass: 0.0,
            ..decay
        };
        for daughter in [Daughter::ChargedLepton, Daughter::MuonNeutrino] {
            assert_close(
                rest_moments(&massless, daughter).1,
                0.35 * massless.muon_mass,
                1e-12,
            );
        }
    }

    /// The massless coefficients are the textbook Michel forms,
    /// `a = x²(3 − 2x)`, `b = x²(1 − 2x)` for e and ν_μ and
    /// `a = b = 6x²(1 − x)` for ν̄_e.
    #[test]
    fn massless_coefficients_are_michel() {
        let decay = MuonDecay {
            electron_mass: 0.0,
            ..MuonDecay::PDG
        };
        for x in [0.1, 0.37, 0.5, 0.82, 0.99] {
            let michel = (x * x * (3.0 - 2.0 * x), x * x * (1.0 - 2.0 * x));
            for daughter in [Daughter::ChargedLepton, Daughter::MuonNeutrino] {
                let (a, b) = decay.rest_coefficients(daughter, x);
                assert_close(a, michel.0, 1e-14);
                assert_close(b, michel.1, 1e-14);
            }
            let (a, b) = decay.rest_coefficients(Daughter::ElectronNeutrino, x);
            assert_close(a, 6.0 * x * x * (1.0 - x), 1e-14);
            assert_eq!(a, b);
        }
    }

    /// A muon in flight keeps one daughter of each kind, whatever its
    /// polarization: the count over a window that covers every boosted
    /// energy is one. A polarized muon moves the mean lab energy by
    /// `γβ P ⟨k cos θ⟩`, which the bins reproduce.
    #[test]
    fn muon_counts_are_normalized_in_flight() {
        let decay = MuonDecay::PDG;
        for daughter in DAUGHTERS {
            for polarization in [-1.0, -0.3, 0.0, 1.0] {
                let count = decay.counts(daughter, 0.0, 1e4, 400.0, polarization);
                assert_close(count, 1.0, 1e-11);
            }
        }
    }

    /// Log-spaced bins from the electron's rest energy tile a moving
    /// muon's e⁺ exactly. Their low edges fall between `m_e` and `γm_e`,
    /// where every box reaches the edge, so this pins the window's start
    /// there. Quadrature only, so 1e-12.
    #[test]
    fn bins_starting_below_the_boosted_rest_energy_tile_the_spectrum() {
        let decay = MuonDecay::PDG;
        let pion = PionDecay::PDG;
        let edges: Vec<f64> = (0..=20)
            .map(|i| MASS_E * (400.0 / MASS_E).powf(f64::from(i) / 20.0))
            .collect();
        let muon: f64 = edges
            .windows(2)
            .map(|bin| decay.counts(Daughter::ChargedLepton, bin[0], bin[1], 250.0, 0.7))
            .sum();
        assert_close(muon, 1.0, 1e-12);
        let pions: f64 = edges
            .windows(2)
            .map(|bin| pion.counts(Daughter::ChargedLepton, bin[0], bin[1], 300.0))
            .sum();
        assert_close(pions, 1.0, 1e-9);
    }

    /// Bins of a moving muon reduce to the density as they shrink: a
    /// relative width of 1e-6 leaves a second-order error near 1e-12, so
    /// 1e-9 covers it and the quadrature.
    #[test]
    fn muon_counts_round_trip_by_bin_width() {
        let decay = MuonDecay::PDG;
        for daughter in DAUGHTERS {
            for energy in [5.0, 31.0, 77.0, 140.0] {
                let width = energy * 1e-6;
                let low = energy - width / 2.0;
                let bin = decay.counts(daughter, low, low + width, 250.0, -0.6) / width;
                assert_close(bin, decay.dnde(daughter, energy, 250.0, -0.6), 1e-9);
            }
        }
    }

    const NEUTRINOS: [Daughter; 2] = [Daughter::ElectronNeutrino, Daughter::MuonNeutrino];

    /// Assert that the neutrino closed forms of `decay` agree with the
    /// quadrature they replace at `muon_energy`, for densities and for
    /// wide, narrow and endpoint-reaching bins, to `rtol`.
    fn assert_closed_forms_match_quadrature(decay: &MuonDecay, muon_energy: f64, rtol: f64) {
        let check = |label: String, actual: f64, expected: f64, rtol: f64| {
            let scale = if expected == 0.0 { 1.0 } else { expected.abs() };
            let error = (actual - expected).abs() / scale;
            assert!(
                error <= rtol,
                "{label}: closed form {actual:.17e}, quadrature {expected:.17e}, \
                 relative error {error:e} > {rtol:e}"
            );
        };
        for daughter in NEUTRINOS {
            for polarization in [-1.0, 0.4, 1.0] {
                let label = |what: String| {
                    format!(
                        "{daughter:?} from a muon of {muon_energy} MeV, P = {polarization}, {what}"
                    )
                };
                for energy in [5e-3, 0.5, 5.0, 31.0, 52.0, 77.0, 140.0, 900.0] {
                    check(
                        label(format!("dnde at {energy} MeV")),
                        decay.dnde(daughter, energy, muon_energy, polarization),
                        decay.dnde_quadrature(daughter, energy, muon_energy, polarization),
                        rtol,
                    );
                    // A bin of relative width 1e-3 costs the quadrature three
                    // digits, since it forms each edge's `cos θ` before
                    // differencing them, where the closed form differences
                    // the lab energies; the pinned 0.5 MeV reference below
                    // shows the closed form is the accurate side in that
                    // case.
                    for (width, rtol) in [(0.3 * energy, rtol), (1e-3 * energy, rtol.max(1e-9))] {
                        let (low, high) = (energy - width / 2.0, energy + width / 2.0);
                        check(
                            label(format!("counts in [{low}, {high}] MeV")),
                            decay.counts(daughter, low, high, muon_energy, polarization),
                            decay.counts_quadrature(daughter, low, high, muon_energy, polarization),
                            rtol,
                        );
                    }
                }
                let top = 2.0 * muon_energy;
                check(
                    label(format!("counts in [0, {top}] MeV")),
                    decay.counts(daughter, 0.0, top, muon_energy, polarization),
                    decay.counts_quadrature(daughter, 0.0, top, muon_energy, polarization),
                    rtol,
                );
            }
        }
    }

    /// Bins pinned against a 40-digit `mpmath` integration of the same
    /// densities with the same clamped `cos θ` windows: the slow muon's
    /// endpoint bin and wide bin, a fast muon's narrow bin, a moderate case
    /// and a bin of relative width 1e-3 near 5.5e-3 MeV from the slow muon,
    /// whose pole integrals are the most sensitive to their interval's
    /// length. The
    /// reference's kinks sit at crossings it knows only to
    /// double precision, so 1e-10 covers both sides.
    #[test]
    fn neutrino_bins_match_a_high_precision_reference() {
        let decay = MuonDecay::PDG;
        let slow = decay.muon_mass / (1.0_f64 - 1e-6).sqrt();
        let cases = [
            (
                Daughter::ElectronNeutrino,
                51.974,
                52.026,
                slow,
                -1.0,
                0.0001755283927069126,
            ),
            (
                Daughter::MuonNeutrino,
                51.974,
                52.026,
                slow,
                -1.0,
                0.001964288682611424,
            ),
            (
                Daughter::ElectronNeutrino,
                44.2,
                59.8,
                slow,
                0.4,
                0.12745390189578906,
            ),
            (
                Daughter::MuonNeutrino,
                44.2,
                59.8,
                slow,
                0.4,
                0.3184732214239274,
            ),
            (
                Daughter::ElectronNeutrino,
                0.49975,
                0.50025,
                2000.0,
                1.0,
                2.3680743028145054e-10,
            ),
            (
                Daughter::MuonNeutrino,
                0.49975,
                0.50025,
                2000.0,
                1.0,
                6.423714503561537e-08,
            ),
            (
                Daughter::MuonNeutrino,
                30.0,
                31.0,
                250.0,
                -0.6,
                0.006297247697720865,
            ),
            (
                Daughter::MuonNeutrino,
                5.505_365_311_236_44e-3,
                5.510_873_430_607_361e-3,
                slow,
                1.0,
                6.798856735042118e-15,
            ),
        ];
        for (daughter, low, high, muon_energy, polarization, expected) in cases {
            assert_close(
                decay.counts(daughter, low, high, muon_energy, polarization),
                expected,
                1e-10,
            );
        }
    }

    /// The neutrinos' closed forms agree with the quadrature they replace
    /// from rest to `γ ≈ 19`. The quadrature meets its 1e-12 target and
    /// the closed form is exact up to the window edges, so 1e-10.
    #[test]
    fn neutrino_closed_forms_match_the_quadrature() {
        let decay = MuonDecay::PDG;
        for muon_energy in [decay.muon_mass, 110.0, 250.0, 2000.0] {
            assert_closed_forms_match_quadrature(&decay, muon_energy, 1e-10);
        }
    }

    /// A muon with `β = 1e-3` has boost windows of relative width 2e-3,
    /// whose edges carry three fewer digits than the energies; both forms
    /// inherit that, and they still agree to 1e-9.
    #[test]
    fn neutrino_closed_forms_hold_for_a_slow_muon() {
        let decay = MuonDecay::PDG;
        let muon_energy = decay.muon_mass / (1.0_f64 - 1e-6).sqrt();
        assert_closed_forms_match_quadrature(&decay, muon_energy, 1e-9);
    }

    /// With a massless electron `r = 0`, so the Laurent coefficients of
    /// the negative powers and of `ln q` vanish while `q` reaches zero at
    /// the endpoint. The closed forms skip those terms rather than form
    /// `0 · ∞`, and still count one neutrino per decay.
    #[test]
    fn neutrino_closed_forms_survive_the_massless_limit() {
        let decay = MuonDecay {
            electron_mass: 0.0,
            ..MuonDecay::PDG
        };
        for daughter in NEUTRINOS {
            let endpoint = decay.max_energy(daughter);
            assert_close(
                decay.counts(daughter, 0.0, endpoint, decay.muon_mass, 0.0),
                1.0,
                1e-13,
            );
            assert_close(decay.counts(daughter, 0.0, 1e4, 300.0, 0.5), 1.0, 1e-11);
            let near = decay.dnde(daughter, 0.999 * endpoint, 300.0, 1.0);
            assert!(near.is_finite());
            assert_close(
                near,
                decay.dnde_quadrature(daughter, 0.999 * endpoint, 300.0, 1.0),
                1e-10,
            );
        }
        assert_closed_forms_match_quadrature(&decay, 250.0, 1e-10);
    }

    /// A muon at the next representable energy above rest decays like one
    /// at rest: the lab density is continuous in the boost.
    #[test]
    fn muon_density_is_continuous_at_rest() {
        let decay = MuonDecay::PDG;
        let barely = decay.muon_mass * (1.0 + 1e-12);
        for daughter in DAUGHTERS {
            let rest = decay.dnde(daughter, 30.0, decay.muon_mass, 1.0);
            assert_close(decay.dnde(daughter, 30.0, barely, 1.0), rest, 1e-5);
        }
    }

    /// Return the zeroth and first moments of a pion's rest-frame spectrum
    /// of `daughter`, prompt line included.
    fn pion_rest_moments(pion: &PionDecay, daughter: Daughter) -> (f64, f64) {
        let mass = pion.muon.daughter_mass(daughter);
        let (kink, end) = pion.continuum_edges(daughter);
        let (line_energy, line_count) = pion.line(daughter);
        let mut count = |t: f64| pion.rest_continuum(daughter, t);
        let zeroth = integrate(&mut count, mass, end, &[kink], 1e-12);
        let mut first = |t: f64| t * pion.rest_continuum(daughter, t);
        let first = integrate(&mut first, mass, end, &[kink], 1e-12);
        (zeroth + line_count, first + line_count * line_energy)
    }

    /// The note's reference: the mean e⁺ energy per π⁺ at rest is
    /// 36.94069288635647 MeV with the muon spin, with `π → eν` included.
    /// The closed form it came from agrees to 1e-14; the nested
    /// quadratures here are good to about 1e-11.
    #[test]
    fn pion_rest_mean_positron_energy_is_the_note_value() {
        let (count, energy) = pion_rest_moments(&NOTE_PION, Daughter::ChargedLepton);
        assert_close(count, 1.0, 1e-11);
        assert_close(energy, 36.940_692_886_356_47, 1e-11);
    }

    /// With `hazma.parameters`' masses the same closed form gives
    /// 36.94069258356754 MeV, which the Python suite pins through the
    /// public counts.
    #[test]
    fn pion_rest_mean_positron_energy_with_hazma_masses() {
        let (count, energy) = pion_rest_moments(&PionDecay::PDG, Daughter::ChargedLepton);
        assert_close(count, 1.0, 1e-11);
        assert_close(energy, 36.940_692_583_567_54, 1e-11);
    }

    /// Without the spin the same mean is the note's 38.430221 MeV, quoted
    /// to the microelectronvolt, so 1e-8.
    #[test]
    fn pion_rest_mean_positron_energy_without_spin() {
        let pion = NOTE_PION;
        let mass = pion.muon.daughter_mass(Daughter::ChargedLepton);
        let (kink, end) = pion.continuum_edges(Daughter::ChargedLepton);
        let mut first = |t: f64| {
            t * pion.br_muon
                * pion
                    .muon
                    .dnde(Daughter::ChargedLepton, t, pion.muon_energy(), 0.0)
        };
        let (line_energy, line_count) = pion.line(Daughter::ChargedLepton);
        let energy = integrate(&mut first, mass, end, &[kink], 1e-12) + line_count * line_energy;
        assert_close(energy, 38.430_221, 1e-8);
    }

    /// Per π⁺ the four neutrino species carry the rest of the pion's
    /// energy, and the per-flavor counts are one ν_e and `2 BR(μν)` muon
    /// neutrinos. The four rest means are the note's values.
    #[test]
    fn pion_rest_neutrinos_close_the_energy() {
        let pion = NOTE_PION;
        let (electron_count, electron) = pion_rest_moments(&pion, Daughter::ElectronNeutrino);
        let (muon_count, muon) = pion_rest_moments(&pion, Daughter::MuonNeutrino);
        let (_, lepton) = pion_rest_moments(&pion, Daughter::ChargedLepton);
        assert_close(electron_count, 1.0, 1e-11);
        assert_close(muon_count, 2.0 * pion.br_muon, 1e-11);
        assert_close(electron, 35.915_176_455_710_046, 1e-11);
        assert_close(muon, 36.926_044_181_387_95 + 29.788_476_476_545_533, 1e-11);
        assert_close(lepton + electron + muon, pion.pion_mass, 1e-11);
    }

    /// A moving pion keeps its multiplicities, and its bins reduce to its
    /// density, prompt boxes included, at the same 1e-9 as the muon's.
    #[test]
    fn pion_counts_are_normalized_and_round_trip() {
        let pion = PionDecay::PDG;
        let pion_energy = 300.0;
        for daughter in DAUGHTERS {
            let (_, line_count) = pion.line(daughter);
            let expected = pion.br_muon + line_count;
            assert_close(pion.counts(daughter, 0.0, 1e4, pion_energy), expected, 1e-9);
            for energy in [3.0, 41.0, 102.0, 230.0] {
                let width = energy * 1e-6;
                let low = energy - width / 2.0;
                let bin = pion.counts(daughter, low, low + width, pion_energy) / width;
                assert_close(bin, pion.dnde(daughter, energy, pion_energy), 1e-8);
            }
        }
    }

    /// At rest a pion's prompt line is a delta function, counted once in
    /// the half-open bin that starts at its energy and never in the one
    /// that ends there.
    #[test]
    fn rest_line_lands_in_one_bin() {
        let pion = PionDecay::PDG;
        let daughter = Daughter::MuonNeutrino;
        let (line_energy, line_count) = pion.line(daughter);
        let below = pion.counts(daughter, 29.0, line_energy, pion.pion_mass);
        let above = pion.counts(daughter, line_energy, 30.0, pion.pion_mass);
        let whole = pion.counts(daughter, 29.0, 30.0, pion.pion_mass);
        assert_close(below + above, whole, 1e-10);
        assert!(below < 0.1 && above > line_count);
    }

    /// The boosted continuum of `π⁺ → μ⁺ → e⁺` against Hazma's
    /// unpolarized kernel: switching the spin off must reproduce
    /// `positron_pion` to its own `1e-4` tolerance, while the polarized
    /// spectrum differs from it at the percent level.
    #[test]
    fn unpolarized_limit_matches_the_existing_pion_kernel() {
        use super::super::positron_pion::dnde_positron_charged_pion;
        let pion = PionDecay::PDG;
        let pion_energy = 250.0;
        let gamma = pion_energy / pion.pion_mass;
        let gamma_beta = momentum(pion_energy, pion.pion_mass) / pion.pion_mass;
        let mass = MASS_E;
        for energy in [10.0, 40.0, 90.0] {
            let (lower, upper) = boost_range(energy, mass, gamma, gamma_beta);
            let (kink, end) = pion.continuum_edges(Daughter::ChargedLepton);
            let mut integrand = |k: f64| {
                let t = energy_from_momentum(k, mass);
                pion.br_muon
                    * pion
                        .muon
                        .dnde(Daughter::ChargedLepton, t, pion.muon_energy(), 0.0)
                    / t
            };
            let unpolarized = integrate(
                &mut integrand,
                momentum(lower.max(mass), mass),
                momentum(upper.min(end), mass),
                &[momentum(kink, mass)],
                1e-10,
            ) / (2.0 * gamma_beta)
                + pion.br_electron
                    * box_density(
                        energy,
                        pion.line(Daughter::ChargedLepton).0,
                        mass,
                        gamma,
                        gamma_beta,
                    );
            assert_close(
                unpolarized,
                dnde_positron_charged_pion(energy, pion_energy),
                2e-4,
            );
            let polarized = pion.dnde(Daughter::ChargedLepton, energy, pion_energy);
            assert!((polarized / unpolarized - 1.0).abs() > 1e-2);
        }
    }

    #[test]
    fn threshold_gamma_is_where_the_box_reaches_the_probe() {
        let (energy, mass) = (40.0, 10.0);
        for probe in [12.0, 30.0, 90.0] {
            let gamma = threshold_gamma(probe, energy, mass);
            let gamma_beta = (gamma * gamma - 1.0).sqrt();
            let (lower, upper) = boost_range(energy, mass, gamma, gamma_beta);
            let edge = if probe > energy { upper } else { lower };
            assert_close(edge, probe, 1e-13);
        }
    }
}
