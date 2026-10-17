//! Axial current tensors and their projections onto the q rest frame.
//!
//! The kernels behind `hazma/form_factors/axial/_tensors.py`. Conventions
//! are the visible-axial-spectra project's ADR-0001: metric
//! `diag(1, -1, -1, -1)`, contravariant components, and the source-
//! longitudinal direction proportional to q, which is not the helicity-zero
//! polarization of an on-shell spin-one particle.
//!
//! Every function takes one sample; the PyO3 layer maps them over a batch.
//! Arithmetic is written in the order the NumPy oracle in
//! `test/form_factors/axial/axial_oracle.py` evaluates it, and sums over more
//! than a few terms use [`pairwise_sum`], so the two agree to a few units in
//! the last place. They still differ by rounding, because NumPy fuses the
//! products of a complex multiply, divides a complex number by a real one as
//! a multiply by the reciprocal, and reduces some axes sequentially.

use num_complex::Complex64;

use crate::boost::pairwise_sum;

/// The metric signature, applied to a contravariant index to lower it.
const METRIC: [f64; 4] = [1.0, -1.0, -1.0, -1.0];

/// Why [`tensor_projections`] refused a tensor.
///
/// The variants are ordered by the precedence the Python surface reports
/// them in when a batch fails in more than one way, so the batch error is
/// the maximum over its samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProjectionError {
    /// The roundoff bound exceeds `rtol` times the rest-frame norm.
    IllConditioned,
    /// A projected density is negative by more than the roundoff bound.
    NegativeDensity,
    /// The contraction overflowed.
    Overflow,
}

/// The invariant `q·q` of a contravariant four-vector.
fn invariant(q: &[f64; 4]) -> f64 {
    q[0] * q[0] - (q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
}

/// Return the Hermitian tensor `H^μ H^ν*`.
pub fn hadronic_tensor(current: &[Complex64; 4]) -> [[Complex64; 4]; 4] {
    current.map(|h_mu| current.map(|h_nu| h_mu * h_nu.conj()))
}

/// Return `(|H_spatial|² / 3, |H_time|²)` with `H` boosted to the q rest
/// frame.
///
/// `q` must be future-directed and timelike. The boost evaluates both
/// projectors as sums of positive terms, so the cancellation of the
/// covariant form `|q·H|²/s − H·H*` never happens and the relative error
/// grows only like `γ²` in the boost factor `γ = q⁰/√s`.
pub fn current_projections(current: &[Complex64; 4], q: &[f64; 4]) -> (f64, f64) {
    let mass = invariant(q).sqrt();
    let spatial_dot = current[1] * q[1] + current[2] * q[2] + current[3] * q[3];
    let time = (current[0] * q[0] - spatial_dot) / mass;
    let shift = spatial_dot / (mass * (q[0] + mass)) - current[0] / mass;
    let spatial = [1, 2, 3].map(|i| current[i] + shift * q[i]);
    let norms = spatial.map(|h| {
        let modulus = h.norm();
        modulus * modulus
    });
    let transverse = (norms[0] + norms[1] + norms[2]) / 3.0;
    let modulus = time.norm();
    (transverse, modulus * modulus)
}

/// Project a Hermitian tensor `W^μν` onto `(T, L)` with the invariant
/// projectors `(−g_μν + q_μ q_ν/s)/3` and `q_μ q_ν/s`.
///
/// The contraction cancels terms of order `γ⁴` times the result, so each
/// projection carries a roundoff bound of `128 ε` times the sum of the
/// magnitudes it adds. A density below minus that bound is an error, one
/// within it is clamped to zero, and a bound above `rtol` times the
/// rest-frame norm `L + 3T` is [`ProjectionError::IllConditioned`]. The
/// caller has already checked that `W` is finite and Hermitian and that `q`
/// is future-directed and timelike.
///
/// # Errors
///
/// See [`ProjectionError`].
pub fn tensor_projections(
    tensor: &[[Complex64; 4]; 4],
    q: &[f64; 4],
    rtol: f64,
) -> Result<(f64, f64), ProjectionError> {
    let s = invariant(q);
    let covariant: [f64; 4] = std::array::from_fn(|mu| q[mu] * METRIC[mu]);
    // Both sums run over (μ, ν) in row-major order, as NumPy reduces a
    // batched `(4, 4, N)` array.
    let mut longitudinal = 0.0;
    let mut magnitude = 0.0;
    for (mu, row) in tensor.iter().enumerate() {
        for (nu, &entry) in row.iter().enumerate() {
            let term = entry * (covariant[mu] * covariant[nu]) / s;
            longitudinal += term.re;
            magnitude += term.norm();
        }
    }
    let trace = tensor[0][0].re - (tensor[1][1].re + tensor[2][2].re + tensor[3][3].re);
    let transverse = (longitudinal - trace) / 3.0;
    let diagonal =
        tensor[0][0].norm() + tensor[1][1].norm() + tensor[2][2].norm() + tensor[3][3].norm();
    let budget = 128.0 * f64::EPSILON * (magnitude + diagonal);

    // A finite budget bounds every term it sums, and so both projections.
    if !budget.is_finite() {
        return Err(ProjectionError::Overflow);
    }
    if transverse < -budget || longitudinal < -budget {
        return Err(ProjectionError::NegativeDensity);
    }
    let (transverse, longitudinal) = (transverse.max(0.0), longitudinal.max(0.0));
    if budget > rtol * (longitudinal + 3.0 * transverse) {
        return Err(ProjectionError::IllConditioned);
    }
    Ok((transverse, longitudinal))
}

/// Return `Σ density·weight` over one row of samples.
///
/// This is the phase-space integral of a projected density against a
/// supplied measure; the weights carry the measure, its normalization and
/// any symmetry factor.
pub fn integrate(densities: &[f64], weights: &[f64]) -> f64 {
    let terms: Vec<f64> = densities
        .iter()
        .zip(weights)
        .map(|(density, weight)| density * weight)
        .collect();
    pairwise_sum(&terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// A boost of `p` along z with factor `gamma`.
    fn boost_z(p: [f64; 4], gamma: f64) -> [f64; 4] {
        let beta_gamma = (gamma * gamma - 1.0).sqrt();
        [
            gamma * p[0] + beta_gamma * p[3],
            p[1],
            p[2],
            beta_gamma * p[0] + gamma * p[3],
        ]
    }

    fn close(actual: f64, expected: f64, rtol: f64) -> bool {
        (actual - expected).abs() <= rtol * expected.abs()
    }

    const Q: [f64; 4] = [700.0, 120.0, -80.0, 210.0];

    #[test]
    fn tensor_is_hermitian_outer_product() {
        let current = [c(1.0, 2.0), c(-0.5, 0.0), c(0.0, 3.0), c(2.0, -1.0)];
        let tensor = hadronic_tensor(&current);
        for (mu, row) in tensor.iter().enumerate() {
            for (nu, entry) in row.iter().enumerate() {
                assert_eq!(*entry, tensor[nu][mu].conj());
            }
            assert_eq!(row[mu].im, 0.0);
        }
        assert_eq!(tensor[0][2], c(1.0, 2.0) * c(0.0, -3.0));
    }

    /// The Ward identity in its two limits: a current along q is purely
    /// source-longitudinal with `L = |q·H|²/s`, and a conserved current
    /// (`q·H = 0`) has no source-longitudinal part.
    #[test]
    fn ward_identity_limits_of_both_projections() {
        let s = invariant(&Q);
        let phase = c(0.6, -0.8);
        let along_q = Q.map(|x| phase * x);
        let (t, l) = current_projections(&along_q, &Q);
        assert!(t <= 1e-12 * s, "transverse {t}");
        assert!(close(l, s, 1e-14));
        let (t, l) = tensor_projections(&hadronic_tensor(&along_q), &Q, 1e-6).unwrap();
        assert!(t <= 1e-12 * s);
        assert!(close(l, s, 1e-13));

        // q·H = q⁰H⁰ − q·H_spatial = 0 for H = (q_z, 0, 0, q⁰) (1, i, …).
        let conserved = [c(Q[3], 1.0 * Q[3]), c(0.0, 0.0), c(0.0, 0.0), c(Q[0], Q[0])];
        let dot = conserved[0] * Q[0] - conserved[3] * Q[3];
        assert_eq!(dot, c(0.0, 0.0));
        let (t, l) = current_projections(&conserved, &Q);
        assert!(l <= 1e-12 * t, "longitudinal {l}");
        // −H·H* is the rest-frame |H_spatial|², all of it transverse.
        let norm = conserved[3].norm_sqr() - conserved[0].norm_sqr();
        assert!(close(t, norm / 3.0, 1e-13));
        let (t2, l2) = tensor_projections(&hadronic_tensor(&conserved), &Q, 1e-6).unwrap();
        assert!(close(t2, t, 1e-12));
        assert!(l2 <= 1e-12 * t);
    }

    /// Projections are Lorentz invariant: the rest-frame values of a
    /// current equal those of its boosted image.
    #[test]
    fn projections_are_boost_invariant() {
        let rest = [500.0, 0.0, 0.0, 0.0];
        let current = [c(0.3, 0.1), c(1.0, -0.2), c(0.0, 0.7), c(-0.4, 0.0)];
        let (t0, l0) = current_projections(&current, &rest);
        assert!(close(t0, (1.04 + 0.49 + 0.16) / 3.0, 1e-15));
        assert!(close(l0, 0.1, 1e-15));
        let gamma = 3.0;
        let re = boost_z(current.map(|h| h.re), gamma);
        let im = boost_z(current.map(|h| h.im), gamma);
        let boosted: [Complex64; 4] = std::array::from_fn(|mu| c(re[mu], im[mu]));
        let q = boost_z(rest, gamma);
        let (t, l) = current_projections(&boosted, &q);
        assert!(close(t, t0, 1e-14) && close(l, l0, 1e-14));
        let (t, l) = tensor_projections(&hadronic_tensor(&boosted), &q, 1e-6).unwrap();
        assert!(close(t, t0, 1e-12) && close(l, l0, 1e-12));
    }

    /// The conditioning guard rejects a projection whose `γ⁴ ε` roundoff
    /// bound passes `rtol`, and the rest-frame tensor is always accepted.
    #[test]
    fn conditioning_guard_tracks_the_boost() {
        let current = [c(0.3, 0.1), c(1.0, -0.2), c(0.0, 0.7), c(-0.4, 0.0)];
        let project = |gamma: f64| {
            let re = boost_z(current.map(|h| h.re), gamma);
            let im = boost_z(current.map(|h| h.im), gamma);
            let boosted: [Complex64; 4] = std::array::from_fn(|mu| c(re[mu], im[mu]));
            let q = boost_z([500.0, 0.0, 0.0, 0.0], gamma);
            tensor_projections(&hadronic_tensor(&boosted), &q, 1e-6)
        };
        assert!(project(1.0).is_ok());
        assert!(project(10.0).is_ok());
        assert_eq!(project(1e3), Err(ProjectionError::IllConditioned));
        assert_eq!(project(1e4), Err(ProjectionError::IllConditioned));
    }

    /// The bound scales with `rtol`, so a tighter tolerance rejects a
    /// projection a looser one accepts.
    #[test]
    fn conditioning_guard_follows_rtol() {
        let rest = [500.0, 0.0, 0.0, 0.0];
        let tensor = hadronic_tensor(&[c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
        assert!(tensor_projections(&tensor, &rest, 1e-6).is_ok());
        assert_eq!(
            tensor_projections(&tensor, &rest, 1e-16),
            Err(ProjectionError::IllConditioned)
        );
    }

    /// Pins the `128 ε` factor and the diagonal term of the roundoff bound.
    ///
    /// In the rest frame a tensor with only `W⁰⁰ = 1` has `L = 1`, `T = 0`,
    /// a magnitude sum of 1 and a diagonal of 1, so the bound is exactly
    /// `256 ε` against a rest-frame norm of 1, and `rtol` decides the
    /// verdict at that edge.
    #[test]
    fn conditioning_bound_is_128_epsilon_times_magnitude_and_diagonal() {
        let rest = [1.0, 0.0, 0.0, 0.0];
        let mut tensor = [[c(0.0, 0.0); 4]; 4];
        tensor[0][0] = c(1.0, 0.0);
        assert_eq!(
            tensor_projections(&tensor, &rest, 256.0 * f64::EPSILON),
            Ok((0.0, 1.0))
        );
        assert_eq!(
            tensor_projections(&tensor, &rest, 255.0 * f64::EPSILON),
            Err(ProjectionError::IllConditioned)
        );
    }

    #[test]
    fn negative_and_overflowing_tensors_are_refused() {
        let rest = [1.0, 0.0, 0.0, 0.0];
        let zero = c(0.0, 0.0);
        let mut tensor = [[zero; 4]; 4];
        tensor[1][1] = c(-1.0, 0.0);
        assert_eq!(
            tensor_projections(&tensor, &rest, 1e-6),
            Err(ProjectionError::NegativeDensity)
        );
        tensor[1][1] = c(f64::MAX, 0.0);
        tensor[2][2] = c(f64::MAX, 0.0);
        assert_eq!(
            tensor_projections(&tensor, &rest, 1e-6),
            Err(ProjectionError::Overflow)
        );
        assert!(ProjectionError::Overflow > ProjectionError::NegativeDensity);
        assert!(ProjectionError::NegativeDensity > ProjectionError::IllConditioned);
    }

    /// A density negative by less than its roundoff bound is clamped to
    /// zero rather than refused.
    #[test]
    fn roundoff_negative_density_is_clamped() {
        let rest = [1.0, 0.0, 0.0, 0.0];
        let zero = c(0.0, 0.0);
        let mut tensor = [[zero; 4]; 4];
        tensor[0][0] = c(1.0, 0.0);
        // T = (L − trace)/3 = (1 − (1 + ε))/3 = −ε/3, inside 128 ε (2 + 2ε).
        tensor[1][1] = c(-f64::EPSILON, 0.0);
        assert_eq!(tensor_projections(&tensor, &rest, 1e-6), Ok((0.0, 1.0)));
    }

    #[test]
    fn integrate_is_a_weighted_sum() {
        let densities: Vec<f64> = (0..200).map(f64::from).collect();
        let weights = vec![0.5; 200];
        assert_eq!(integrate(&densities, &weights), 0.5 * 199.0 * 200.0 / 2.0);
        assert_eq!(integrate(&[], &[]), 0.0);
    }
}
