//! The interval and break points of both mediators' thermal averages.
//!
//! [`super::scalar_xs::thermal_cross_section`] and
//! [`super::vector_xs::thermal_cross_section`] integrate
//!
//! ```text
//!   ∫₂^Z dz  σ_all(m_x z) z² (z² − 4) K₁(x z)
//! ```
//!
//! and the Boltzmann weight `K₁(x z) ~ e^{−x z}` confines each channel's
//! contribution to a few decay lengths `1/x` past the `z` at which it
//! opens. So the partition is built from those openings, the
//! *features*: the pair-production threshold `z = 2`, every channel
//! threshold and the mediator resonance.
//!
//! - **The upper limit** is [`DECAY_LENGTHS_PAST_LAST`] decay lengths past
//!   the last feature. Measuring it from `z = 2` alone drops a channel
//!   that opens beyond that window: `HiggsPortal(mx=200, ms=550,
//!   stheta=0)` has only `S S`, which opens at `z = 5.5`, and at `x = 30`
//!   a limit of `2 + 100/x` returns `0.0`. A fixed limit such as the
//!   `.pyx`'s `max(50/x, 150)` covers the features only while they lie
//!   below it.
//! - **The break points** are every feature and [`SPLITS`] decay lengths
//!   past each. A piece many decay lengths long whose integrand sits
//!   within `1/x` of its left end is where QUADPACK's first Gauss–Kronrod
//!   nodes all land in the tail and its error estimate misses the peak.
//!   On the `.pyx`'s `[2, 150]` that put `KineticMixing(mx=300, mv=200)`'s
//!   average 1.9e-4 high at `x = 300`.
//! - **A feature within [`MIN_THRESHOLD_PIECE`] of `z = 2`** is moved onto
//!   it. `σ_all` is infinite at `z = 2` itself, where the integrand's
//!   `(z² − 4)` makes it `NaN`, and the Gauss–Kronrod nodes of a sliver
//!   `[2, 2 + ε]` round to that endpoint. With `m_x` one ulp below the
//!   vector kernel's muon mass, the `μ μ` threshold sits at
//!   `z = 2 (1 + ε)`, and as its own break point it makes
//!   `KineticMixing(mv=550)`'s average `NaN` at `x = 20`.
//!
//! The pure-Python sites apply the same rule in
//! `hazma.relic_density._thermal_functions.thermal_cross_section_partition`,
//! fed from each model's `annihilation_thresholds()`, except that they
//! bracket a resonance with a ladder of break points scaled by its width
//! rather than splitting at the peak.

/// How far past the last feature the integral runs, in decay lengths
/// `1/x`. The Bessel kernel's tail beyond it is at most 3.0e-38 of its
/// integral from the feature, for `x` from 0.01 to 300.
pub const DECAY_LENGTHS_PAST_LAST: f64 = 100.0;

/// Break points past each feature, in decay lengths `1/x`. The piece
/// after the last split starts where the weight has fallen by `e^{−50}`,
/// so no feature's peak can hide inside a long piece.
pub const SPLITS: [f64; 4] = [1.0, 4.0, 16.0, 50.0];

/// The shortest piece allowed to start at the threshold `z = 2`. The
/// outermost 21-point Kronrod node sits 2.17e-3 of a piece's length
/// inside it, so on a piece this long the first evaluation lands 2.2e-12
/// past `z = 2`, about 4900 ulps, where `σ_all` is finite.
pub const MIN_THRESHOLD_PIECE: f64 = 1e-9;

/// The upper limit and break points for the thermal average at `x`.
///
/// `features` are the `z = e_cm / m_x` at which a channel opens or the
/// mediator resonates; any below the threshold `z = 2`, or within
/// [`MIN_THRESHOLD_PIECE`] above it, are moved onto it. The points are returned unsorted and
/// may repeat or fall outside `[2, upper]`, all of which
/// [`crate::quad::quad`] filters as scipy does.
#[must_use]
pub fn partition(x: f64, features: &[f64]) -> (f64, Vec<f64>) {
    let mut openings = vec![2.0];
    openings.extend(features.iter().map(|&z| {
        if z < 2.0 + MIN_THRESHOLD_PIECE {
            2.0
        } else {
            z
        }
    }));
    let last = openings.iter().copied().fold(2.0, f64::max);
    let upper = last + DECAY_LENGTHS_PAST_LAST / x;
    let points = openings
        .iter()
        .flat_map(|&z| std::iter::once(z).chain(SPLITS.iter().map(move |k| z + k / x)))
        .collect();
    (upper, points)
}

#[cfg(test)]
mod tests {
    use super::{DECAY_LENGTHS_PAST_LAST, MIN_THRESHOLD_PIECE, SPLITS, partition};

    /// The limit runs from the last feature, not from threshold.
    #[test]
    fn the_upper_limit_follows_the_last_feature() {
        let x = 30.0;
        let (upper, _) = partition(x, &[0.5, 2.75, 5.5]);
        assert_eq!(upper, 5.5 + DECAY_LENGTHS_PAST_LAST / x);
        let (upper, _) = partition(x, &[0.5, 1.5]);
        assert_eq!(upper, 2.0 + DECAY_LENGTHS_PAST_LAST / x);
    }

    /// Every feature at or above threshold, and each split past it, is a
    /// break point; a feature below threshold contributes threshold's.
    #[test]
    fn every_feature_is_split_past() {
        let x = 10.0;
        let (_, points) = partition(x, &[1.0, 3.0]);
        for z in [2.0, 3.0] {
            assert!(points.contains(&z));
            for k in SPLITS {
                assert!(points.contains(&(z + k / x)), "{z} + {k}/x");
            }
        }
        assert!(!points.contains(&1.0));
    }

    /// A feature a sliver above threshold is moved onto it, so no piece
    /// starting at `z = 2` is shorter than [`MIN_THRESHOLD_PIECE`].
    #[test]
    fn a_feature_just_above_threshold_moves_onto_it() {
        let x = 20.0;
        let sliver = 2.0 * (1.0 + f64::EPSILON);
        let (_, points) = partition(x, &[sliver, 2.0 + MIN_THRESHOLD_PIECE]);
        assert!(!points.contains(&sliver));
        assert!(points.contains(&(2.0 + MIN_THRESHOLD_PIECE)));
        let shortest = points
            .iter()
            .filter(|&&z| z > 2.0)
            .fold(f64::INFINITY, |a, &z| a.min(z - 2.0));
        assert!(shortest >= MIN_THRESHOLD_PIECE);
    }
}
