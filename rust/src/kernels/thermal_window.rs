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
//! - **The break points** are every threshold and [`SPLITS`] decay
//!   lengths past each. A piece many decay lengths long whose integrand sits
//!   within `1/x` of its left end is where QUADPACK's first Gauss–Kronrod
//!   nodes all land in the tail and its error estimate misses the peak.
//!   On the `.pyx`'s `[2, 150]` that put `KineticMixing(mx=300, mv=200)`'s
//!   average 1.9e-4 high at `x = 300`.
//! - **A resonance** of peak `z_r` and width `g`, both in units of
//!   `m_x`, contributes the ladder `z_r ± g·4^k` for `k = 0, 1, …` up to
//!   the length of the interval, and counts toward the last feature. The
//!   peak itself is never a break point: there it would sit on a piece's
//!   endpoint, where no Gauss–Kronrod node samples it, and once the
//!   resonance is narrower than the node spacing QUADPACK reports
//!   convergence on a partition that never saw it. With the peak as a
//!   break point, `HiggsPortal(mx=200, ms=550, gsxx=1e-2, stheta=1e-3)`,
//!   whose width is 3.5e-6 of `m_x`, was 87% low at `x = 2`, and
//!   `KineticMixing(mx=200, mv=550, gvxx=1e-2, eps=1e-3)`, at 6.3e-6,
//!   kept 2.6e-4 of its average at `x = 1`. With the ladder, each kernel
//!   holds both within its own `epsrel` from `x = 0.1` to 300.
//! - **A threshold within [`MIN_THRESHOLD_PIECE`] of `z = 2`** is moved
//!   onto it, and no break point is kept that close above it. `σ_all` is
//!   infinite at `z = 2` itself, where the integrand's `(z² − 4)` makes
//!   it `NaN`, and the Gauss–Kronrod nodes of a sliver `[2, 2 + ε]` round
//!   to that endpoint. With `m_x` one ulp below the vector kernel's muon
//!   mass, the `μ μ` threshold sits at `z = 2 (1 + ε)`, and as its own
//!   break point it makes `KineticMixing(mv=550)`'s average `NaN` at
//!   `x = 20`.
//!
//! The pure-Python sites apply the same rule in
//! `hazma.relic_density._thermal_functions.thermal_cross_section_partition`,
//! fed from each model's `annihilation_thresholds()` and
//! `annihilation_resonances()`.

/// How far past the last feature the integral runs, in decay lengths
/// `1/x`. The Bessel kernel's tail beyond it is at most 3.0e-38 of its
/// integral from the feature, for `x` from 0.01 to 300.
pub const DECAY_LENGTHS_PAST_LAST: f64 = 100.0;

/// Break points past each feature, in decay lengths `1/x`. The piece
/// after the last split starts where the weight has fallen by `e^{−50}`,
/// so no feature's peak can hide inside a long piece.
pub const SPLITS: [f64; 4] = [1.0, 4.0, 16.0, 50.0];

/// Ratio between successive break points bracketing a resonance, in
/// units of its width. The ladder adds `2 ⌈log₄((Z − 2)/g)⌉` points for
/// an interval `[2, Z]` and a width `g`: 70 for a width of 1e-17 of
/// `m_x` at `x = 0.01`.
pub const RESONANCE_LADDER_RATIO: f64 = 4.0;

/// The shortest piece allowed to start at the threshold `z = 2`. The
/// outermost 21-point Kronrod node sits 2.17e-3 of a piece's length
/// inside it, so on a piece this long the first evaluation lands 2.2e-12
/// past `z = 2`, about 4900 ulps, where `σ_all` is finite.
pub const MIN_THRESHOLD_PIECE: f64 = 1e-9;

/// The upper limit and break points for the thermal average at `x`.
///
/// `thresholds` are the `z = e_cm / m_x` at which a channel opens; any
/// below the threshold `z = 2`, or within [`MIN_THRESHOLD_PIECE`] above
/// it, are moved onto it. `resonances` are each mediator's `(z_r, g)`,
/// its mass and width divided by `m_x`; a zero width contributes no
/// ladder. The points are returned sorted and distinct, and only those
/// more than [`MIN_THRESHOLD_PIECE`] above `z = 2` and below the upper
/// limit are kept, so their count is the number of interior break points
/// [`crate::quad::quad`] sees.
#[must_use]
pub fn partition(x: f64, thresholds: &[f64], resonances: &[(f64, f64)]) -> (f64, Vec<f64>) {
    let mut openings = vec![2.0];
    openings.extend(thresholds.iter().map(|&z| {
        if z < 2.0 + MIN_THRESHOLD_PIECE {
            2.0
        } else {
            z
        }
    }));
    let last = openings
        .iter()
        .chain(resonances.iter().map(|(z_r, _)| z_r))
        .copied()
        .fold(2.0, f64::max);
    let upper = last + DECAY_LENGTHS_PAST_LAST / x;

    let mut points: Vec<f64> = openings
        .iter()
        .flat_map(|&z| std::iter::once(z).chain(SPLITS.iter().map(move |k| z + k / x)))
        .collect();
    for &(z_r, width) in resonances {
        let mut offset = width;
        while 0.0 < offset && offset < upper - 2.0 {
            points.extend([z_r - offset, z_r + offset]);
            offset *= RESONANCE_LADDER_RATIO;
        }
    }
    points.retain(|&z| 2.0 + MIN_THRESHOLD_PIECE < z && z < upper);
    points.sort_by(f64::total_cmp);
    points.dedup();
    (upper, points)
}

#[cfg(test)]
mod tests {
    use super::{
        DECAY_LENGTHS_PAST_LAST, MIN_THRESHOLD_PIECE, RESONANCE_LADDER_RATIO, SPLITS, partition,
    };

    /// The limit runs from the last feature, not from threshold.
    #[test]
    fn the_upper_limit_follows_the_last_feature() {
        let x = 30.0;
        let (upper, _) = partition(x, &[0.5, 2.75, 5.5], &[]);
        assert_eq!(upper, 5.5 + DECAY_LENGTHS_PAST_LAST / x);
        let (upper, _) = partition(x, &[0.5, 1.5], &[]);
        assert_eq!(upper, 2.0 + DECAY_LENGTHS_PAST_LAST / x);
    }

    /// Every threshold above `z = 2`, and each split past it, is a break
    /// point; a threshold below `z = 2` contributes only `z = 2`'s splits.
    #[test]
    fn every_feature_is_split_past() {
        let x = 10.0;
        let (_, points) = partition(x, &[1.0, 3.0], &[]);
        assert!(points.contains(&3.0));
        for z in [2.0, 3.0] {
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
        let kept = 2.0 + 2.0 * MIN_THRESHOLD_PIECE;
        let (_, points) = partition(x, &[sliver, kept], &[]);
        assert!(!points.contains(&sliver));
        assert!(points.contains(&kept));
        let shortest = points
            .iter()
            .filter(|&&z| z > 2.0)
            .fold(f64::INFINITY, |a, &z| a.min(z - 2.0));
        assert!(shortest >= MIN_THRESHOLD_PIECE);
    }

    /// A resonance past the last threshold sets the upper limit.
    #[test]
    fn the_upper_limit_follows_a_resonance() {
        let x = 10.0;
        let (upper, _) = partition(x, &[2.5], &[(5.5, 1e-3)]);
        assert_eq!(upper, 5.5 + DECAY_LENGTHS_PAST_LAST / x);
    }

    /// A resonance is bracketed by the width ladder on both sides, up to
    /// the interval's length, and its peak is never a break point.
    #[test]
    fn a_resonance_is_bracketed_by_its_width_ladder() {
        let x = 5.0;
        let (z_r, width) = (2.75, 3.5e-6);
        let (upper, points) = partition(x, &[], &[(z_r, width)]);
        assert!(!points.contains(&z_r));
        let mut offset = width;
        let mut rungs = 0;
        while offset < upper - 2.0 {
            for z in [z_r - offset, z_r + offset] {
                if 2.0 + MIN_THRESHOLD_PIECE < z && z < upper {
                    assert!(points.contains(&z), "rung at {z}");
                }
            }
            offset *= RESONANCE_LADDER_RATIO;
            rungs += 1;
        }
        // log₄((upper − 2) / width) = log₄(20 / 3.5e-6) = 11.2.
        assert_eq!(rungs, 12);
        let nearest = points
            .iter()
            .fold(f64::INFINITY, |a, &z| a.min((z - z_r).abs()));
        // `z_r ± width` rounds at the ulp of `z_r`, 4.4e-16.
        assert!((nearest - width).abs() < 1e-15);
    }

    /// A zero-width resonance adds no break point, and the returned points
    /// are sorted, distinct and strictly inside `(2 + MIN_THRESHOLD_PIECE,
    /// upper)`, so their count is what the integrator sees.
    #[test]
    fn the_points_are_sorted_distinct_and_interior() {
        let x = 1.0;
        let (_, bare) = partition(x, &[0.5, 3.0, 3.0], &[]);
        let (upper, points) = partition(x, &[0.5, 3.0, 3.0], &[(4.0, 0.0)]);
        assert_eq!(bare, points);
        assert!(points.windows(2).all(|w| w[0] < w[1]));
        assert!(
            points
                .iter()
                .all(|&z| 2.0 + MIN_THRESHOLD_PIECE < z && z < upper)
        );
    }
}
