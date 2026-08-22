//! The two pieces of C complex arithmetic Cython's `**` operator drags
//! in, reproduced exactly.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3), and shared by both mediator kernel modules:
//! [`crate::kernels::vector_xs`] found this in Task 5.1 and
//! [`crate::kernels::scalar_xs`] needed the identical pair in Task 5.2.
//!
//! # Why any of this exists
//!
//! Cython 3's default `cpow` semantics say a `double ** double` *may* be
//! complex, so a `.pyx` expression containing one is compiled in
//! `double _Complex` **in its entirety** and converted back with
//! `__Pyx_SoftComplexToDouble`, which raises `TypeError` when the
//! imaginary part is non-zero. `grep -c SoftComplexToDouble` over the
//! generated C is how you find out: two call sites in the vector module
//! (`__sigma_xx_to_v_to_pipi`, `__sigma_xx_to_v_to_pi0v`) and **one** in
//! the scalar module (`__sigma_xx_to_s_to_ff`, whose
//! `(-4 mf**2 + e_cm**2) ** 1.5` is easy to miss). Each such expression
//! reaches libm's `cpow` and compiler-rt's `__divdc3` rather than `pow`
//! and `/`, and neither agrees with its real-arithmetic spelling.
//!
//! Measured on the capturing platform (macOS/arm64, Apple clang,
//! libSystem libm) over 3.7M logarithmically spaced arguments during
//! Task 5.1:
//!
//! | comparison | fraction differing | worst relative |
//! | --- | --- | --- |
//! | `cpow(t + 0i, 1.5 + 0i)` vs `pow(t, 1.5)` | 90% | 9.0e-15 |
//! | `cpow(t + 0i, 1.5 + 0i)` vs `t·√t` | 90% | 9.1e-15 |
//! | `cpow(t + 0i, 1.5 + 0i)` vs `exp(1.5·log t)` | **0** | **0** |
//! | `(a + 0i)/(c + 0i)` vs `a/c` | 32% | 4.0e-16 |
//!
//! So [`soft_complex_pow_1_5`] and
//! [`complex_quotient_real_denominator`] below are what a
//! bit-equality-class corpus budget actually requires; ignoring either
//! puts every kernel that reaches them outside it.

/// The `**`-operator result was complex, so the Cython raised.
///
/// Carries nothing: `__Pyx_SoftComplexToDouble`'s only observable is the
/// `TypeError` it sets, and the parity corpus compares exception *types*
/// (`test/parity/generate.py`'s `_sweep_pointwise` records
/// `type(err).__name__` and nothing else). [`crate::vector_mediator`]
/// turns this into that `TypeError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonRealResult;

/// `logb(x)` truncated to an `i32`, for finite non-zero `x`.
///
/// `__divdc3` calls `(int)logb(fmax(|c|, |d|))`, which for a finite
/// non-zero argument is its unbiased binary exponent. Read off the bits
/// rather than via `log2().floor()`, which is not exact at a power of
/// two.
pub(crate) fn ilogb_finite(x: f64) -> i32 {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: the exponent is set by the leading mantissa bit.
        // `x = m · 2⁻¹⁰⁷⁴` with `m` the 52-bit significand, so
        // `⌊log₂ x⌋ = (63 − m.leading_zeros()) − 1074`.
        let significand = bits & 0x000f_ffff_ffff_ffff;
        63 - (significand.leading_zeros() as i32) - 1074
    } else {
        biased - 1023
    }
}

/// `x · 2ⁿ`, correctly rounded — C's `scalbn`.
///
/// Written as up to three multiplications by powers of two rather than
/// one so that an intermediate cannot overflow or fall subnormal while
/// the final answer is representable, which is the whole reason
/// `__divdc3` scales in the first place. Each step is exact except the
/// last, which rounds once.
pub(crate) fn scalbn(x: f64, n: i32) -> f64 {
    /// `2¹⁰²³`, the largest power of two a single step may apply.
    const HUGE_STEP: f64 = 8.988_465_674_311_58e307;
    /// `2⁻¹⁰²²⁺⁵³`, one downward step. The `2⁵³` keeps the intermediate
    /// normal, so no step but the last can round.
    const TINY_STEP: f64 = 2.004_168_360_008_973e-292;

    let mut y = x;
    let mut n = n;
    if n > 1023 {
        y *= HUGE_STEP;
        n -= 1023;
        if n > 1023 {
            y *= HUGE_STEP;
            n -= 1023;
            n = n.min(1023);
        }
    } else if n < -1022 {
        y *= TINY_STEP;
        n += 1022 - 53;
        if n < -1022 {
            y *= TINY_STEP;
            n += 1022 - 53;
            n = n.max(-1022);
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

/// `cpow(t + 0i, 1.5 + 0i)`, real part, for the `t ≥ 0` this file reaches.
///
/// `cexp(w · clog(z))` at zero imaginary part collapses to
/// `exp(1.5 · log(t))`: `clog(t + 0i)` is `log(t) + 0i` for `t > 0`, the
/// complex product with `1.5 + 0i` leaves both parts alone, and
/// `cexp(y + 0i)` is `exp(y) + 0i`. Verified bit-for-bit against
/// libSystem's `cpow` at 3.7M arguments spanning `[1e-8, 1e14]` — see
/// the module docs.
///
/// At `t = 0` — reachable, and the corpus samples it: `e_cm = 2 m_π`
/// exactly is a grid anchor — `log(0)` is `−∞`, `1.5 · −∞` is `−∞`, and
/// `exp(−∞)` is `0`, which is what `cpow(0, 1.5)` gives.
#[must_use]
pub fn soft_complex_pow_1_5(t: f64) -> f64 {
    (1.5 * t.ln()).exp()
}

/// Real part of `(a + 0i) / (c + 0i)`, as compiler-rt's `__divdc3` gives
/// it, plus whether the imaginary part came back non-zero.
///
/// Not `a / c`: `__divdc3` scales the denominator into `[1, 2)`, forms
/// `denom = c′²`, and returns `scalbn((a·c′ + b·d)/denom, −ilogb)`. With
/// `b = d = 0` — which is every call this file makes — that is two extra
/// roundings, and it disagrees with a single division at a third of all
/// arguments.
///
/// The imaginary part is `(b·c′ − a·d)/denom`, i.e. `0/denom`, which is a
/// signed zero for every finite non-zero denominator. It is non-zero only
/// through the NaN-recovery clause C99 Annex G requires, and in this file
/// only one input reaches it: `c == 0`, where the recovery sets
/// `imag = copysign(∞, c)·b = ∞·0 = NaN`. That is the `e_cm = 2 m_x`
/// threshold, where `√(e_cm² − 4 m_x²)` is exactly zero and the whole
/// denominator with it — and it is why two of hazma's cross sections
/// raise `TypeError` there while the other four return `inf`
/// (`docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md`).
///
/// The algorithm is Smith's (1962), which C99 Annex G §G.5.1 gives as the
/// recommended practice for complex division; it is transcribed from that
/// specification rather than from any implementation's source.
///
/// # Errors
///
/// [`NonRealResult`] when the imaginary part is non-zero, which is what
/// `__Pyx_SoftComplexToDouble` raises `TypeError` on.
pub fn complex_quotient_real_denominator(a: f64, c: f64) -> Result<f64, NonRealResult> {
    let logbw = if c == 0.0 {
        f64::NEG_INFINITY
    } else if c.is_finite() {
        f64::from(ilogb_finite(c))
    } else {
        f64::INFINITY
    };

    let (scaled, ilogbw) = if logbw.is_finite() {
        let ilogbw = logbw as i32;
        (scalbn(c, -ilogbw), ilogbw)
    } else {
        (c, 0)
    };

    // `b` and `d` are zero, so `a·c + b·d` is `a·c` and `b·c − a·d` is
    // `0·c − a·0`. Both are written out rather than simplified, because
    // the zeros are what turn `0/0` into the NaN the recovery clause
    // keys on.
    let denom = scaled * scaled;
    let real = scalbn((a * scaled) / denom, -ilogbw);
    let imag = scalbn((0.0 * scaled - a * 0.0) / denom, -ilogbw);

    // C99 Annex G's recovery, restricted to the clauses `b = d = 0` can
    // reach. The `denom == 0` clause is the live one; the infinite-`a`
    // clause is unreachable from this file's finite numerators but is
    // part of the same specification and costs nothing to keep.
    let (real, imag) = if real.is_nan() && imag.is_nan() {
        if denom == 0.0 && !a.is_nan() {
            (
                f64::INFINITY.copysign(scaled) * a,
                f64::INFINITY.copysign(scaled) * 0.0,
            )
        } else if a.is_infinite() && scaled.is_finite() {
            let a = (if a.is_infinite() { 1.0 } else { 0.0_f64 }).copysign(a);
            (f64::INFINITY * (a * scaled), f64::INFINITY * (0.0 * scaled))
        } else {
            (real, imag)
        }
    } else {
        (real, imag)
    };

    // `if (unlikely(__Pyx_CIMAG(value)))` — C truthiness, so both signed
    // zeros pass and a NaN does not.
    if imag != 0.0 {
        return Err(NonRealResult);
    }
    Ok(real)
}

#[cfg(test)]
mod tests {
    use super::{
        NonRealResult, complex_quotient_real_denominator, ilogb_finite, scalbn,
        soft_complex_pow_1_5,
    };

    fn bits(x: f64) -> u64 {
        x.to_bits()
    }

    // -- The complex-arithmetic shims ------------------------------------

    /// `soft_complex_pow_1_5` is `cpow`'s answer, and `pow`'s is a
    /// different number.
    ///
    /// The equality against libSystem `cpow` itself is measured in C (see
    /// the module docs — 3.7M arguments, no disagreement); what a Rust
    /// test can hold is the half that would silently break here, namely
    /// that `t.powf(1.5)` and `t * t.sqrt()` are *not* substitutes. Both
    /// alternatives are checked because both are the obvious
    /// simplification.
    #[test]
    fn the_three_halves_power_is_not_powf() {
        let mut powf_differs = 0;
        let mut sqrt_differs = 0;
        let mut samples = 0;
        let mut t = 1e-3;
        while t < 1e12 {
            let got = soft_complex_pow_1_5(t);
            samples += 1;
            if bits(got) != bits(t.powf(1.5)) {
                powf_differs += 1;
            }
            if bits(got) != bits(t * t.sqrt()) {
                sqrt_differs += 1;
            }
            // Agrees to a few ulp with both, which is what makes the
            // difference a rounding question rather than a wrong formula.
            assert!((got - t.powf(1.5)).abs() <= 1e-14 * t.powf(1.5));
            t *= 1.037;
        }
        assert!(samples > 900, "sweep collapsed to {samples} samples");
        assert!(
            powf_differs * 2 > samples,
            "only {powf_differs}/{samples} differ from powf; the two \
             spellings have become interchangeable"
        );
        assert!(sqrt_differs * 2 > samples);
    }

    /// `cpow(0 + 0i, 1.5 + 0i)` is zero, and the corpus samples it.
    ///
    /// `e_cm = 2 m_pi` exactly is a grid anchor, and there the kinematic
    /// factor is exactly zero: `log(0)` is `-inf`, `1.5 * -inf` is `-inf`,
    /// `exp(-inf)` is `+0`.
    #[test]
    fn the_three_halves_power_is_zero_at_zero() {
        assert_eq!(bits(soft_complex_pow_1_5(0.0)), bits(0.0));
        assert_eq!(bits(soft_complex_pow_1_5(-0.0)), bits(0.0));
    }

    /// `ilogb` reads the exponent field, including through the subnormals.
    #[test]
    fn ilogb_is_the_unbiased_exponent() {
        assert_eq!(ilogb_finite(1.0), 0);
        assert_eq!(ilogb_finite(1.999_999), 0);
        assert_eq!(ilogb_finite(2.0), 1);
        assert_eq!(ilogb_finite(0.5), -1);
        assert_eq!(ilogb_finite(-8.0), 3);
        assert_eq!(ilogb_finite(f64::MIN_POSITIVE), -1022);
        // The largest subnormal and the smallest, both below MIN_POSITIVE.
        assert_eq!(ilogb_finite(f64::from_bits(0x000f_ffff_ffff_ffff)), -1023);
        assert_eq!(ilogb_finite(f64::from_bits(1)), -1074);
        assert_eq!(ilogb_finite(f64::MAX), 1023);
    }

    /// `scalbn` is exact where a plain multiply is, and survives where it
    /// is not.
    #[test]
    fn scalbn_scales_by_powers_of_two() {
        for x in [1.0, -3.5, 1.234_567_890_123_45e77, f64::MIN_POSITIVE] {
            for n in [-60, -1, 0, 1, 53, 200] {
                assert_eq!(bits(scalbn(x, n)), bits(x * 2.0_f64.powi(n)), "{x} << {n}");
            }
        }
        // Past what one multiply can express. 2^2000 is not a double, so
        // the scale-up saturates -- the point of the stepping is that a
        // scale-*down* through the same magnitude does not, which the
        // subnormal round-trip below covers.
        assert_eq!(scalbn(1.5, 2000), f64::INFINITY);
        assert_eq!(scalbn(1.0, 3000), f64::INFINITY);
        assert_eq!(scalbn(1.0, -3000), 0.0);
        // Through the subnormals and back.
        assert_eq!(bits(scalbn(scalbn(1.0, -1060), 1060)), bits(1.0));
    }

    /// The complex quotient is not a division, and the difference is real.
    ///
    /// `__divdc3` forms `(a·c′)/(c′·c′)` with `c′ ∈ [1, 2)`, which rounds
    /// three times where `a/c` rounds once. If a later reader replaces the
    /// call with `a / c`, this fails.
    #[test]
    fn the_complex_quotient_differs_from_plain_division() {
        let mut differs = 0;
        let mut samples = 0;
        let mut a = 1e-20;
        while a < 1e20 {
            let c = 3.718_281_828 * a * 1.4e7 + 1.0;
            let got = complex_quotient_real_denominator(a, c).expect("finite denominator");
            samples += 1;
            if bits(got) != bits(a / c) {
                differs += 1;
            }
            // Never more than two ulp apart: it is the same quotient,
            // rounded three times instead of once.
            assert!((got - a / c).abs() <= 2.0 * f64::EPSILON * (a / c).abs());
            a *= 1.031;
        }
        assert!(samples > 1000);
        assert!(
            differs > samples / 10,
            "only {differs}/{samples} differ from a/c"
        );
    }

    /// A zero denominator is the raise, and nothing else here is.
    #[test]
    fn the_complex_quotient_fails_only_on_a_vanishing_denominator() {
        assert_eq!(
            complex_quotient_real_denominator(1.0, 0.0),
            Err(NonRealResult)
        );
        assert_eq!(
            complex_quotient_real_denominator(0.0, 0.0),
            Err(NonRealResult)
        );
        // Ordinary denominators, including subnormal and enormous ones,
        // come back real.
        for c in [1e-300, 1e-30, 1.0, 1e30, 1e300, -4.5] {
            assert!(complex_quotient_real_denominator(7.0, c).is_ok(), "{c}");
        }
    }
}
