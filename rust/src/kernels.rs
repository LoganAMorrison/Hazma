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

#[cfg(test)]
mod tests {
    use super::roundtrip;

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
