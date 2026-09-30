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
//!
//! The pure-Python sites apply the same upper limit through
//! `hazma.relic_density._thermal_functions.thermal_cross_section_upper_limit`,
//! whose docstring bounds the tail it drops.

/// How far past the last feature the integral runs, in decay lengths
/// `1/x`. The Bessel kernel's tail beyond it is at most 3.0e-38 of its
/// integral from the feature, for `x` from 0.01 to 300.
pub const DECAY_LENGTHS_PAST_LAST: f64 = 100.0;

/// Break points past each feature, in decay lengths `1/x`. The piece
/// after the last split starts where the weight has fallen by `e^{−50}`,
/// so no feature's peak can hide inside a long piece.
pub const SPLITS: [f64; 4] = [1.0, 4.0, 16.0, 50.0];

/// The upper limit and break points for the thermal average at `x`.
///
/// `features` are the `z = e_cm / m_x` at which a channel opens or the
/// mediator resonates; any below the threshold `z = 2` are raised to it,
/// since the integral starts there. The points are returned unsorted and
/// may repeat or fall outside `[2, upper]`, all of which
/// [`crate::quad::quad`] filters as scipy does.
#[must_use]
pub fn partition(x: f64, features: &[f64]) -> (f64, Vec<f64>) {
    let mut openings = vec![2.0];
    openings.extend(features.iter().map(|&z| z.max(2.0)));
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
    use super::{DECAY_LENGTHS_PAST_LAST, SPLITS, partition};

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
}
