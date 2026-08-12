//! Pure numerical kernels.
//!
//! Everything in this module is plain Rust: `fn(f64, ...) -> f64`, no
//! PyO3 types, no GIL. That is `rules.md` rule 3 of the cython-to-rust
//! project — it keeps `cargo test` free of the interpreter and keeps the
//! math readable next to the Cython it replaces. The PyO3 layer lives in
//! [`crate::dispatch`] and the per-domain submodules.

/// Return `x` unchanged.
///
/// The scaffold's only kernel. It exists so Phase 02 can prove the
/// Python → Rust → Python path end to end before any physics depends on
/// it, and it is deliberately value-preserving so the plumbing tests can
/// assert bit-equality with no tolerance to argue about.
///
/// Identity alone would not prove the Rust code ran — a dispatch layer
/// that returned its argument untouched would pass the same assertion.
/// What proves it is that the array path allocates a *new* array, so the
/// result is never the input object. Phase 02 Task 2.3 asserts that.
#[must_use]
pub fn roundtrip(x: f64) -> f64 {
    x
}

/// Return `x` as three distinguishable flavor components.
///
/// The plumbing probe for the neutrino return shape
/// ([`crate::dispatch::map_flavors`]), which is the one non-uniform shape
/// in hazma's public surface: a 3-tuple for a scalar, a `(3, N)` array for
/// a grid. Like [`roundtrip`] it computes no physics, and like it the
/// point is what a wrong implementation would still pass.
///
/// The three components are deliberately **distinct functions of `x`**
/// rather than three copies of it: with equal rows, a transposed result,
/// a reversed row order or a row written twice would all satisfy a
/// value-by-value assertion. `-x` and `1.0/x` are both exactly
/// reproducible in Python — IEEE negation and division are correctly
/// rounded — so the test still argues about no tolerance.
#[must_use]
pub fn roundtrip_flavors(x: f64) -> [f64; 3] {
    [x, -x, 1.0 / x]
}

#[cfg(test)]
mod tests {
    use super::{roundtrip, roundtrip_flavors};

    #[test]
    fn roundtrip_flavors_rows_are_pairwise_distinguishable() {
        // The property the probe exists for: at a generic argument no two
        // rows collide, so a row permutation cannot pass unnoticed.
        let [electron, muon, tau] = roundtrip_flavors(4.0);
        assert_eq!(electron.to_bits(), 4.0_f64.to_bits());
        assert_eq!(muon.to_bits(), (-4.0_f64).to_bits());
        assert_eq!(tau.to_bits(), 0.25_f64.to_bits());
        assert!(electron != muon && muon != tau && electron != tau);
    }

    #[test]
    fn roundtrip_flavors_is_defined_at_the_non_finite_edges() {
        // Zero and infinity swap under the third row, so both directions
        // of the reciprocal are exercised rather than assumed.
        assert_eq!(roundtrip_flavors(0.0)[2], f64::INFINITY);
        assert_eq!(roundtrip_flavors(-0.0)[2], f64::NEG_INFINITY);
        assert_eq!(
            roundtrip_flavors(f64::INFINITY)[2].to_bits(),
            0.0_f64.to_bits()
        );
        assert!(
            roundtrip_flavors(f64::NAN)
                .iter()
                .all(|value| value.is_nan())
        );
    }

    #[test]
    fn roundtrip_is_the_identity_on_ordinary_values() {
        for x in [0.0, -0.0, 1.0, -1.5, 1e-300, 1e300, f64::MIN_POSITIVE] {
            assert_eq!(roundtrip(x).to_bits(), x.to_bits());
        }
    }

    #[test]
    fn roundtrip_preserves_non_finite_values() {
        assert!(roundtrip(f64::NAN).is_nan());
        assert_eq!(roundtrip(f64::INFINITY), f64::INFINITY);
        assert_eq!(roundtrip(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }
}
