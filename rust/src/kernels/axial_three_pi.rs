//! Leading-order, exact-isospin axial three-pion currents and their
//! spectral functions.
//!
//! The kernels behind `hazma/form_factors/axial/_three_pi.py`, whose module
//! docstring derives the currents. For the source `G_A = g3 diag(1, −1, 0)`
//! the charged channel `π⁺π⁻π⁰` has
//! `H = (2 g3/F) (−2 p₀ + q (q² − s)/(q² − m²))` with `s = (p₊ + p₋)²`, and
//! the neutral channel `3π⁰` has `H = −(2 g3/F) q m²/(q² − m²)`. Momenta are
//! in MeV with metric `diag(1, −1, −1, −1)`, the currents are dimensionless
//! and the spectral functions are in MeV².
//!
//! The caller validates every input: momenta on shell, `q` timelike and
//! inside the declared support, and the coupling a pure isovector source.
//! Arithmetic is written in the order the NumPy oracle in
//! `test/form_factors/axial/axial_oracle.py` evaluates it, so the two agree to a
//! few units in the last place.

use std::f64::consts::PI;

use crate::boost::pairwise_sum;

/// Which three-pion final state a current describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// `π⁺ π⁻ π⁰`, in that particle order.
    Charged,
    /// `π⁰ π⁰ π⁰`, which carries the identical-particle factor `1/3!`.
    Neutral,
}

/// Three on-shell momenta of one event: `momenta[μ][i]` is component `μ`
/// of particle `i`, in MeV.
pub type Momenta = [[f64; 3]; 4];

/// The LO three-pion current of one channel.
#[derive(Clone, Copy, Debug)]
pub struct ThreePion {
    pub channel: Channel,
    /// Common pion mass in MeV; zero is the chiral limit.
    pub pion_mass: f64,
    /// Pion decay constant `F` in MeV.
    pub decay_constant: f64,
}

fn dot(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] - (a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
}

fn particle(momenta: &Momenta, i: usize) -> [f64; 4] {
    momenta.map(|row| row[i])
}

fn total(momenta: &Momenta) -> [f64; 4] {
    momenta.map(|row| row[0] + row[1] + row[2])
}

impl ThreePion {
    /// Return the contravariant current `H^μ` of one event, dimensionless.
    ///
    /// It includes both the contact and the single-pion-pole terms, and is
    /// identically zero for `g3 = 0` and for massless neutral pions.
    pub fn current(&self, momenta: &Momenta, g3: f64) -> [f64; 4] {
        let q = total(momenta);
        let q2 = dot(&q, &q);
        let m2 = self.pion_mass * self.pion_mass;
        if g3 == 0.0 || (self.channel == Channel::Neutral && self.pion_mass == 0.0) {
            return [0.0; 4];
        }
        match self.channel {
            Channel::Neutral => {
                let prefactor = -2.0 * g3 / self.decay_constant;
                let pole = m2 / (q2 - m2);
                q.map(|x| prefactor * x * pole)
            }
            Channel::Charged => {
                let pair: [f64; 4] = momenta.map(|row| row[0] + row[1]);
                let s = dot(&pair, &pair);
                let prefactor = 2.0 * g3 / self.decay_constant;
                let pole = (q2 - s) / (q2 - m2);
                let neutral = particle(momenta, 2);
                std::array::from_fn(|mu| prefactor * (-2.0 * neutral[mu] + q[mu] * pole))
            }
        }
    }

    /// Return the physical spin-averaged weight `Σ|ε·H|²/3` of one event,
    /// dimensionless.
    ///
    /// This is the rest-frame transverse projection with no phase-space
    /// measure or identical-particle factor. The neutral current is
    /// parallel to q, so its weight is exactly zero.
    pub fn differential_weight(&self, momenta: &Momenta, g3: f64) -> f64 {
        match self.channel {
            Channel::Neutral => 0.0,
            Channel::Charged => {
                let q = total(momenta);
                let q2 = dot(&q, &q);
                let q_dot_p = dot(&q, &particle(momenta, 2));
                let momentum2 = (q_dot_p * q_dot_p / q2 - self.pion_mass * self.pion_mass).max(0.0);
                16.0 * (g3 * g3) * momentum2 / (3.0 * (self.decay_constant * self.decay_constant))
            }
        }
    }

    /// Return `(A_T, A_L)` in MeV² at invariant energy `energy` (MeV).
    ///
    /// `A_T = ∫dΦ₃ |H_spatial|²/3` and `A_L = ∫dΦ₃ |H_time|²` in the q rest
    /// frame, with the angular integrals done analytically and the
    /// remaining integral over the pair invariant `s ∈ [4m², (q − m)²]`
    /// taken with `rule` from [`angular_rule`] after the substitution
    /// `s = 4m² + [(q − m)² − 4m²](1 − cos θ)/2`, which removes the
    /// endpoint square roots. The measure is
    /// `dΦ₃/ds = √λ(q², s, m²) √(1 − 4m²/s)/(128 π³ q²)`, the full `dΦ`
    /// with no Gram-matrix factor `1/2`, evaluated in ratios to `q` so that
    /// tiny MeV scales stay representable. Both vanish at and below the
    /// threshold `3m`; the neutral channel's `1/3!` is applied here.
    pub fn spectral_functions(&self, energy: f64, g3: f64, rule: &[(f64, f64)]) -> (f64, f64) {
        if self.pion_mass == 0.0 {
            return match self.channel {
                Channel::Neutral => (0.0, 0.0),
                Channel::Charged => {
                    let scaled = g3 * energy / self.decay_constant;
                    let transverse = scaled * scaled * (energy * energy) / (384.0 * PI.powf(3.0));
                    (transverse, 0.0)
                }
            };
        }
        if energy <= 3.0 * self.pion_mass {
            return (0.0, 0.0);
        }
        let ratio = self.pion_mass / energy;
        let ratio2 = ratio * ratio;
        let span = ((1.0 - 3.0 * ratio) * (1.0 + ratio)).max(0.0);
        let below = (1.0 - ratio) * (1.0 - ratio);
        let above = (1.0 + ratio) * (1.0 + ratio);
        let normalization = 128.0 * PI.powf(3.0);
        let scaled = g3 * energy / self.decay_constant;
        let prefactor = 4.0 * (scaled * scaled);
        let off_shell = (1.0 - ratio2) * (1.0 - ratio2);

        let mut transverse = Vec::with_capacity(rule.len());
        let mut longitudinal = Vec::with_capacity(rule.len());
        for &(theta, weight) in rule {
            let s = 4.0 * ratio2 + span * (1.0 - theta.cos()) / 2.0;
            let jacobian = span * theta.sin() / 2.0;
            let lambda = (below - s) * (above - s);
            let measure = energy
                * energy
                * weight
                * jacobian
                * lambda.max(0.0).sqrt()
                * (1.0 - 4.0 * ratio2 / s).max(0.0).sqrt()
                / normalization;
            match self.channel {
                Channel::Neutral => {
                    longitudinal.push(measure * (prefactor * (ratio2 * ratio2) / off_shell));
                }
                Channel::Charged => {
                    let pole = (s - ratio2) * (s - ratio2);
                    transverse.push(measure * (prefactor * lambda / 3.0));
                    longitudinal.push(measure * (prefactor * (ratio2 * ratio2) * pole / off_shell));
                }
            }
        }
        match self.channel {
            Channel::Neutral => (0.0, pairwise_sum(&longitudinal) / 6.0),
            Channel::Charged => (pairwise_sum(&transverse), pairwise_sum(&longitudinal)),
        }
    }
}

/// Return the Gauss–Legendre rule of `order` points mapped to `θ ∈ [0, π]`,
/// as `(θ, weight)` pairs in ascending `θ`.
pub fn angular_rule(order: usize) -> Vec<(f64, f64)> {
    gauss_legendre(order)
        .into_iter()
        .map(|(x, w)| ((x + 1.0) * PI / 2.0, w * PI / 2.0))
        .collect()
}

/// Return the Gauss–Legendre nodes on `[−1, 1]`, ascending, with their
/// weights.
///
/// Each positive root is found by Newton's method on the three-term
/// recurrence from the asymptotic guess `cos(π(i + 3/4)/(n + 1/2))`, and
/// mirrored; an odd order's middle node is exactly zero. Against a 30-digit
/// mpmath evaluation at order 511, the nodes are accurate to about an ulp
/// and the weights to about 3e-12 relative; NumPy's `leggauss` has the same
/// node accuracy but weights good to only 2e-10.
fn gauss_legendre(order: usize) -> Vec<(f64, f64)> {
    let n = order as f64;
    // (P_n(x), P_n'(x)).
    let legendre = |x: f64| {
        let (mut current, mut previous) = (x, 1.0);
        for k in 1..order {
            let k = k as f64;
            let next = ((2.0 * k + 1.0) * x * current - k * previous) / (k + 1.0);
            previous = current;
            current = next;
        }
        (current, n * (x * current - previous) / (x * x - 1.0))
    };
    let weight = |x: f64| {
        let derivative = legendre(x).1;
        2.0 / ((1.0 - x * x) * derivative * derivative)
    };

    let mut rule = vec![(0.0, 0.0); order];
    for i in 0..order / 2 {
        let mut x = (PI * (i as f64 + 0.75) / (n + 0.5)).cos();
        for _ in 0..100 {
            let (value, derivative) = legendre(x);
            let step = value / derivative;
            x -= step;
            // The rounding of `value / derivative` keeps a converged step
            // a couple of ulps wide, so the stop is a few ulps; Newton's
            // quadratic convergence puts the one step taken afterwards at
            // the rounding floor.
            if step.abs() <= 4.0 * f64::EPSILON * x {
                let (value, derivative) = legendre(x);
                x -= value / derivative;
                break;
            }
        }
        let w = weight(x);
        rule[order - 1 - i] = (x, w);
        rule[i] = (-x, w);
    }
    if order % 2 == 1 {
        rule[order / 2] = (0.0, weight(0.0));
    }
    rule
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::axial_tensors::current_projections;
    use num_complex::Complex64;

    const M: f64 = 134.9768;
    const F: f64 = 92.2;

    fn charged(pion_mass: f64) -> ThreePion {
        ThreePion {
            channel: Channel::Charged,
            pion_mass,
            decay_constant: F,
        }
    }

    fn neutral(pion_mass: f64) -> ThreePion {
        ThreePion {
            channel: Channel::Neutral,
            pion_mass,
            decay_constant: F,
        }
    }

    fn close(actual: f64, expected: f64, rtol: f64) -> bool {
        (actual - expected).abs() <= rtol * expected.abs()
    }

    /// An on-shell rest-frame event of three particles of mass `m`:
    /// particle 2 has momentum `p` along z and the other two recoil with
    /// `∓k` along x, then the whole event is boosted along z with `beta`.
    fn event(m: f64, k: f64, p: f64, beta: f64) -> Momenta {
        let energy = |p2: f64| (m * m + p2).sqrt();
        let pair = energy(k * k + p * p / 4.0);
        let rest = [
            [pair, pair, energy(p * p)],
            [k, -k, 0.0],
            [0.0; 3],
            [-p / 2.0, -p / 2.0, p],
        ];
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        [
            std::array::from_fn(|i| gamma * (rest[0][i] + beta * rest[3][i])),
            rest[1],
            rest[2],
            std::array::from_fn(|i| gamma * (rest[3][i] + beta * rest[0][i])),
        ]
    }

    /// PCAC: `q·H = 2 g3 m² (m² − s)/(F (q² − m²))` for the charged current
    /// and `−2 g3 m² q²/(F (q² − m²))` for the neutral one, both vanishing
    /// in the chiral limit.
    #[test]
    fn ward_identities() {
        let g3 = 0.7;
        for beta in [0.0, 0.6] {
            let momenta = event(M, 90.0, 60.0, beta);
            let q = total(&momenta);
            let q2 = dot(&q, &q);
            let pair = momenta.map(|row| row[0] + row[1]);
            let s = dot(&pair, &pair);
            let m2 = M * M;

            let divergence = dot(&q, &charged(M).current(&momenta, g3));
            let expected = 2.0 * g3 * m2 * (m2 - s) / (F * (q2 - m2));
            assert!(
                close(divergence, expected, 1e-12),
                "{divergence} vs {expected}"
            );

            let divergence = dot(&q, &neutral(M).current(&momenta, g3));
            let expected = -2.0 * g3 * m2 * q2 / (F * (q2 - m2));
            assert!(
                close(divergence, expected, 1e-12),
                "{divergence} vs {expected}"
            );

            let massless = event(0.0, 90.0, 60.0, beta);
            let q = total(&massless);
            let current = charged(0.0).current(&massless, g3);
            let scale = q[0] * current.iter().map(|h| h.abs()).sum::<f64>();
            assert!(dot(&q, &current).abs() <= 1e-14 * scale);
            assert_eq!(neutral(0.0).current(&massless, g3), [0.0; 4]);
        }
    }

    /// The weight is the transverse projection of the current itself.
    #[test]
    fn differential_weight_is_the_transverse_projection() {
        let g3 = -1.3;
        let momenta = event(M, 90.0, 60.0, 0.6);
        let q = total(&momenta);
        let current = charged(M)
            .current(&momenta, g3)
            .map(|h| Complex64::new(h, 0.0));
        let (transverse, _) = current_projections(&current, &q);
        let weight = charged(M).differential_weight(&momenta, g3);
        assert!(close(weight, transverse, 1e-12), "{weight} vs {transverse}");
        assert_eq!(neutral(M).differential_weight(&momenta, g3), 0.0);
        assert_eq!(charged(M).current(&momenta, 0.0), [0.0; 4]);
    }

    #[test]
    fn spectral_functions_vanish_at_and_below_threshold() {
        let rule = angular_rule(64);
        for model in [charged(M), neutral(M)] {
            for energy in [0.0, M, 3.0 * M] {
                assert_eq!(model.spectral_functions(energy, 1.0, &rule), (0.0, 0.0));
            }
            let (transverse, longitudinal) =
                model.spectral_functions(3.0 * M * (1.0 + 1e-9), 1.0, &rule);
            assert!((0.0..1e-20).contains(&transverse), "{transverse}");
            assert!(longitudinal > 0.0, "{longitudinal}");
            assert!(longitudinal < 1e-10, "{longitudinal}");
        }
        assert_eq!(charged(0.0).spectral_functions(0.0, 1.0, &rule), (0.0, 0.0));
        assert_eq!(
            neutral(0.0).spectral_functions(450.0, 1.0, &rule),
            (0.0, 0.0)
        );
    }

    /// The massive integral approaches the chiral limit
    /// `A_T = (g3 q/F)² q²/(384 π³)`, `A_L = 0`.
    #[test]
    fn massive_integral_reaches_the_chiral_limit() {
        let rule = angular_rule(64);
        let energy = 500.0;
        let (limit, zero) = charged(0.0).spectral_functions(energy, 1.0, &rule);
        assert_eq!(zero, 0.0);
        assert!(close(
            limit,
            (energy / F).powi(2) * energy * energy / (384.0 * PI.powi(3)),
            1e-15
        ));
        let (transverse, longitudinal) = charged(1e-7).spectral_functions(energy, 1.0, &rule);
        assert!(close(transverse, limit, 1e-8), "{transverse} vs {limit}");
        assert!(longitudinal < 1e-30 * limit);
    }

    #[test]
    fn spectral_functions_scale_with_the_coupling_squared() {
        let rule = angular_rule(32);
        for model in [charged(M), neutral(M)] {
            let (t1, l1) = model.spectral_functions(550.0, 1.0, &rule);
            let (t3, l3) = model.spectral_functions(550.0, -3.0, &rule);
            assert!(close(t3, 9.0 * t1, 1e-15) && close(l3, 9.0 * l1, 1e-15));
        }
    }

    /// The rule integrates polynomials of degree `2n − 1` exactly.
    #[test]
    fn gauss_legendre_is_exact_to_degree_2n_minus_1() {
        for order in [8, 9, 64, 511, 512] {
            let rule = gauss_legendre(order);
            assert!(rule.windows(2).all(|pair| pair[0].0 < pair[1].0));
            for (node, mirror) in rule.iter().zip(rule.iter().rev()) {
                assert_eq!(node.0, -mirror.0);
                assert_eq!(node.1, mirror.1);
            }
            for degree in [0, 2, 2 * order - 2] {
                let integral: f64 = rule.iter().map(|&(x, w)| w * x.powi(degree as i32)).sum();
                let exact = 2.0 / (degree as f64 + 1.0);
                // x^(2n−2) amplifies a node's rounding by 2n − 2: 1e-12.
                assert!(
                    close(integral, exact, 1e-12),
                    "order {order} degree {degree}"
                );
            }
        }
    }
}
