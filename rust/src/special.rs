//! The three `scipy.special` functions hazma's compiled layer uses.
//!
//! A PyO3-free shim over the [`spec_math`] crate, plus two routines that
//! deliberately do not use it
//! (`projects/cython-to-rust/rules.md`, Rust conventions rule 3 — flat
//! `fn(f64, ..) -> f64`, no PyO3 types, so `cargo test` needs no GIL).
//! `crate::special_probe` is the Python-visible half.
//!
//! # Sources and licensing
//!
//! Upstream: `spec_math` 0.1.6 (crates.io, **MIT OR Apache-2.0**), a
//! Rust re-implementation of the **cephes** library, Cephes Math Library
//! Release 2.1 (1989), Copyright 1984–1992 Stephen L. Moshier. The
//! coefficient tables [`SPENCE_A`] / [`SPENCE_B`] are transcribed from
//! the same cephes release, which ADR-0002 lists as permitted
//! provenance. Nothing here is GSL-derived, which is what
//! `projects/cython-to-rust/adrs/ADR-0002-license-clean-numerics.md`
//! requires (`rules.md` rule 5 / Licensing 1).
//!
//! That lineage is the point rather than a coincidence: scipy's own
//! `spence`, `k0` and `k1` are cephes wrappers too, so the port is
//! algorithm-for-algorithm rather than merely value-for-value against
//! them.
//!
//! **Two of the three do not go through `spec_math` in the end**, and
//! each says why at its own definition. [`bessel_kn`] does not because
//! `scipy.special.kn` is not cephes `kn` (Task 3.2). [`spence`] does not
//! because `spec_math` evaluates cephes' Horner unfused while the C
//! scipy ships is contracted, and the one kernel that calls it amplifies
//! the difference by `1/β` (Task 4.3). Same algorithm, same
//! coefficients, different roundings — and the second is measurably the
//! wrong answer for hazma.
//!
//! # What calls these, and with what
//!
//! | Function | Cython call site | Argument range |
//! | --- | --- | --- |
//! | [`spence`] | `hazma/spectra/_photon/_muon.pyx:113` | `xm`, `xp` ∈ (0, 1) |
//! | [`bessel_k1`] | `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1361`, `hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:606` | `x·z`, `z ≥ 2` |
//! | [`bessel_kn`] | scalar `:1404`, vector `:650` (always `n = 2`) | `x ∈ (0, 300]` |
//!
//! The `.pyx` reach them through
//! `from scipy.special.cython_special cimport ...`, which is what pins
//! `scipy>=1.13` in `pyproject.toml`'s build requirements. Those
//! `cython_special` entry points were measured to return bit-identical
//! values to the `scipy.special` ufuncs of the same name (Task 3.2), so
//! the Python-level oracle in `test/test_core_special.py` is the same
//! function the Cython actually calls.

use spec_math::Bessel;

/// `π²/6`, the value of [`spence`] at `0` and the reflection constant.
///
/// Folded the way cephes' C folds `M_PI * M_PI / 6.0`; both round to the
/// same double, which `the_reflection_constant_is_pi_squared_over_six`
/// checks against the literal the closed-form tests use.
const PI_SQ_OVER_SIX: f64 = std::f64::consts::PI * std::f64::consts::PI / 6.0;

/// Numerator coefficients of cephes' rational approximation on
/// `(0.5, 1.5)`. Cephes Math Library Release 2.1 (January 1989),
/// `spence.c`, Copyright 1985–1989 Stephen L. Moshier — the same
/// lineage the module docs cite, transcribed rather than depended upon
/// for the reason [`spence`] gives.
#[allow(clippy::excessive_precision)]
const SPENCE_A: [f64; 8] = [
    4.651_285_860_739_900_452_78E-5,
    7.315_890_452_380_947_110_71E-3,
    1.338_476_395_783_090_186_50E-1,
    8.796_913_117_545_303_153_41E-1,
    2.711_498_511_965_534_699_20E0,
    4.256_971_560_081_217_557_24E0,
    3.297_713_409_852_251_069_36E0,
    1.000_000_000_000_000_001_26E0,
];

/// Denominator coefficients; see [`SPENCE_A`].
#[allow(clippy::excessive_precision)]
const SPENCE_B: [f64; 8] = [
    6.909_904_889_125_532_769_99E-4,
    2.540_437_639_325_443_791_13E-2,
    2.829_748_606_025_680_899_43E-1,
    1.411_725_977_518_310_696_17E0,
    3.638_005_333_451_370_754_18E0,
    5.032_788_801_433_169_903_90E0,
    3.547_713_409_852_250_962_17E0,
    9.999_999_999_999_999_987_40E-1,
];

/// Cephes' `polevl`: Horner, **fused**.
///
/// The fusion is the whole reason this function is in the tree — see
/// [`spence`].
fn polevl(x: f64, coefficients: &[f64; 8]) -> f64 {
    let mut answer = coefficients[0];
    for coefficient in &coefficients[1..] {
        answer = answer.mul_add(x, *coefficient);
    }
    answer
}

/// Spence's integral, in **scipy's** argument convention.
///
/// Returns
///
/// ```text
///        ⌠ x  ln(t)
///   −    ⎮    ───── dt   =   Li₂(1 − x)
///        ⌡ 1  t − 1
/// ```
///
/// for `x ≥ 0`. This is `scipy.special.spence(x)`, i.e. the dilogarithm
/// evaluated at `1 − x` — **not** `Li₂(x)`. Getting that backwards is the
/// single most likely way to break `dnde_photon_muon`, which subtracts
/// two of these (`spence(xm) - spence(xp)`), so the convention is pinned
/// against scipy in `test/test_core_special.py` rather than trusted.
///
/// # Why this is transcribed and not `spec_math::Polylog::li2`
///
/// `spec_math` does have it, and its body *is* `cephes64::spence` — the
/// convention trap one level down, since the method is named `li2`. What
/// it does not have is the C build's **contraction**. scipy ships cephes
/// compiled by clang with `-ffp-contract=on`, so `polevl`'s
/// `ans = ans*x + c` and the reflection's `π²/6 − ln(x)·ln(1−x)` are
/// fused multiply-adds; `spec_math` writes them unfused, and Rust does
/// not contract. Measured against `scipy.special.spence` on 13,000
/// points across all four branches (cython-to-rust Task 4.3):
/// `spec_math` misses by up to **2.0e-15** relative, this transcription
/// by **0** — bit-identical everywhere sampled.
///
/// That distinction would be cosmetic anywhere else and is not here.
/// `crate::kernels::photon_muon` forms `(5/β)·(spence(x₋) − spence(x₊))`,
/// and the parity corpus samples `β = 1.4e-6`, so the `1/β` turns a
/// two-ulp difference in `spence` into a **3.2e-11** relative difference
/// in the spectrum — 320x the `SPECFUN` budget. Bit-equality is the only
/// tolerance that survives that amplification, so it is what this
/// function delivers.
///
/// Fusing is also the *more* accurate Horner, so nothing is traded for
/// the agreement. It is scoped to a platform only in the sense that the
/// comparison is: a scipy built without contraction would move, and the
/// `test/test_core_special.py` sweep is what would say so.
///
/// Edge behavior, matching cephes and therefore scipy: `x < 0` and
/// `x = ∞` give `NaN`, `x = 0` gives `π²/6`, `x = 1` gives `0`, and
/// `NaN` propagates.
pub fn spence(x: f64) -> f64 {
    if x < 0.0 {
        // Cephes raises a domain error and returns NaN. A `NaN` argument
        // fails this comparison and falls through to the arithmetic
        // below, where it propagates — as it does in cephes.
        return f64::NAN;
    }
    if x == 1.0 {
        return 0.0;
    }
    if x == 0.0 {
        return PI_SQ_OVER_SIX;
    }

    let mut x = x;
    // `flag` is cephes' own two-bit reflection record: bit 0 for the
    // `1 − x` map, bit 1 for the `1/x` map. Both can be set.
    let mut flag = 0_u8;

    if x > 2.0 {
        x = 1.0 / x;
        flag |= 2;
    }

    let w = if x > 1.5 {
        flag |= 2;
        1.0 / x - 1.0
    } else if x < 0.5 {
        flag |= 1;
        -x
    } else {
        x - 1.0
    };

    // The division breaks contraction on both sides, so this line is
    // plain arithmetic in the C too.
    let mut y = -w * polevl(w, &SPENCE_A) / polevl(w, &SPENCE_B);

    if flag & 1 != 0 {
        // `fnmsub`: the product of the two logarithms is fused into the
        // subtraction from π²/6.
        y = (-x.ln()).mul_add((1.0 - x).ln(), PI_SQ_OVER_SIX) - y;
    }

    if flag & 2 != 0 {
        let z = x.ln();
        // Likewise `-0.5·z·z − y`: the outer multiply-add is fused, the
        // inner `0.5 * z` is not.
        y = (-0.5 * z).mul_add(z, -y);
    }

    y
}

/// Modified Bessel function of the second kind, order one — `K₁(x)`.
///
/// `scipy.special.k1(x)`. Used as the Maxwell–Boltzmann weight
/// `k1(x·z)` in both mediator models' thermal-⟨σv⟩ integrand.
///
/// Edge behavior, matching cephes and therefore scipy: `x = 0` gives
/// `+∞`, `x < 0` gives `NaN`, `x = ∞` gives `0`, and the `exp(-x)` in
/// the large-argument branch underflows to `0` on its own a little past
/// `x = 710` (no explicit cutoff, in either implementation).
pub fn bessel_k1(x: f64) -> f64 {
    x.bessel_k1()
}

/// Modified Bessel function of the second kind, integer order —
/// `Kₙ(x)`.
///
/// `scipy.special.kn(n, x)`, argument order included. Hazma calls it at
/// exactly one order, `n = 2`, in the `x/(2·kn(2, x))²` prefactor of
/// both thermal cross sections. Negative `n` folds to `|n|`
/// (`K₋ₙ = Kₙ`).
///
/// # Why this is a recurrence and not `spec_math`'s `bessel_kn`
///
/// `spec_math` does have `Bessel::bessel_kn`, a faithful translation of
/// cephes `kn.c` — and it is the wrong function, because **cephes `kn`
/// is not what scipy runs**. `scipy.special.kn` dispatches to
/// `scipy.special.kv` (the AMOS-lineage real-order routine) and only
/// `k0`/`k1` are still cephes there. Measured against scipy 1.18.0 over
/// `x ∈ [1e-8, 300]` (Task 3.2), cephes `kn(2, ·)` misses scipy by up to
/// **5.1e-9 relative** (at `x = 9.531`), peaking just below that
/// routine's own `x = 9.55` branch switch. That is four orders of
/// magnitude past this task's 1e-13 gate, and it would land squarely in
/// the parity corpus's 1e-8 budget for `thermal_cross_section`, whose
/// prefactor squares this value.
///
/// So `Kₙ` is instead built from the upward recurrence
///
/// ```text
///   K_{m+1}(x) = K_{m-1}(x) + (2m/x)·K_m(x)          (DLMF 10.29.1)
/// ```
///
/// seeded on cephes `k0`/`k1` — the two routines scipy *does* still take
/// from cephes. Upward recurrence is the stable direction for `K`, which
/// grows with order. Measured agreement with `scipy.special.kn` over the
/// same grid: `≤ 3.4e-15` relative for every order `n = 0..5`, and
/// `≤ 8.9e-16` at the `n = 2` hazma uses.
///
/// # Divergence from scipy in the underflow tail
///
/// The seeds decay as `exp(-x)`, so the recurrence keeps returning
/// subnormals until `x ≈ 742`, whereas scipy's `kn` has flushed to `0`
/// from `x ≈ 698`. The two therefore disagree wholesale on
/// `698 ≲ x ≲ 742` — where scipy's answer is `0` and this one is around
/// `1e-311`.
///
/// That region is unreachable from hazma: `thermal_cross_section`
/// short-circuits to `0.0` above `x = 300`, where `K₂ ≈ 3.7e-132`. The
/// boundary is pinned in `test/test_core_special.py` so a future caller
/// that widens the domain finds the divergence in a test rather than in
/// a spectrum.
pub fn bessel_kn(n: i32, x: f64) -> f64 {
    // `Kₙ(0) = +∞` at every order, and cephes' `k0`/`k1` already say so
    // — but the recurrence below cannot reach that answer from them at
    // **negative** zero. IEEE puts `-0.0` here rather than in the
    // negative branch (`-0.0 < 0.0` is false), so the seeds are `+∞`
    // while `2m/x` is `-∞`, and `∞ + -∞` is `NaN`. scipy returns `+∞`
    // for `kn(n, -0.0)` at every order; without this guard every order
    // from 2 up returned `NaN` (PR #59 review).
    if x == 0.0 {
        return f64::INFINITY;
    }

    match n.unsigned_abs() {
        0 => x.bessel_k0(),
        1 => x.bessel_k1(),
        order => {
            // The remaining boundary inputs need no special-casing: the
            // seeds carry cephes' answers (NaN below zero, 0 at +∞) and
            // the recurrence propagates each one, because `2m/x` is
            // finite or zero for every `x` that reaches here.
            let mut lower = x.bessel_k0();
            let mut current = x.bessel_k1();
            for m in 1..order {
                let next = lower + 2.0 * f64::from(m) * current / x;
                lower = current;
                current = next;
            }
            current
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spec_math::Bessel;

    /// π²/6 and π²/12 to full double precision, for the two closed-form
    /// `spence` values. Written out rather than computed from
    /// `std::f64::consts::PI` so the expected value cannot drift with a
    /// refactor of the expression that produces it.
    const PI_SQ_OVER_6: f64 = 1.644_934_066_848_226_4;
    const PI_SQ_OVER_12: f64 = 0.822_467_033_424_113_2;

    /// Relative tolerance for the analytic identities below.
    ///
    /// cephes advertises ~1e-16 relative accuracy for these routines on
    /// their principal domains, and each identity here combines two or
    /// three of them, so a few ulp of accumulation is expected and
    /// anything at 1e-13 is a real defect rather than rounding.
    const RTOL: f64 = 1e-13;

    fn assert_close(got: f64, want: f64, what: &str) {
        let err = (got - want).abs() / want.abs();
        assert!(
            err <= RTOL,
            "{what}: got {got:e}, want {want:e}, relative error {err:e} > {RTOL:e}"
        );
    }

    #[test]
    fn spence_matches_its_closed_forms() {
        // Li₂(1 − 0) = Li₂(1) = π²/6, Li₂(1 − 1) = Li₂(0) = 0, and
        // Li₂(1 − 2) = Li₂(−1) = −π²/12. The third is the one that
        // would move if the wrapper exposed Li₂(x) instead: Li₂(2) is
        // complex, and Li₂(0.5) = π²/12 − ln²2/2 ≈ 0.5822 would land on
        // spence(0.5) by coincidence of value, not of convention.
        assert_eq!(spence(0.0), PI_SQ_OVER_6);
        assert_eq!(spence(1.0), 0.0);
        assert_close(spence(2.0), -PI_SQ_OVER_12, "spence(2)");

        // Li₂(1 − 0.5) = Li₂(0.5) = π²/12 − ln(2)²/2.
        let want = PI_SQ_OVER_12 - 0.5 * std::f64::consts::LN_2.powi(2);
        assert_close(spence(0.5), want, "spence(0.5)");
    }

    #[test]
    fn spence_reflects_about_the_branch_point() {
        // Li₂(z) + Li₂(1 − z) = π²/6 − ln(z)·ln(1 − z), which in scipy's
        // convention reads spence(x) + spence(1 − x)
        //   = π²/6 − ln(1 − x)·ln(x).
        for &x in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            let lhs = spence(x) + spence(1.0 - x);
            let rhs = PI_SQ_OVER_6 - (1.0 - x).ln() * x.ln();
            assert_close(lhs, rhs, "spence reflection");
        }
    }

    /// The rational approximation is evaluated **fused**, and that is
    /// not cosmetic.
    ///
    /// The whole reason `spence` is transcribed here rather than taken
    /// from `spec_math` (see its docs): scipy's cephes is compiled with
    /// contraction on, so `polevl` is a chain of `fmadd`. A mutation to
    /// plain `ans * x + c` leaves cephes' own accuracy claim intact and
    /// would pass every identity above — but it moves `spence` by up to
    /// 2.0e-15 relative against scipy, which
    /// `crate::kernels::photon_muon` amplifies by `1/β` into a 3.2e-11
    /// spectrum shift and the parity corpus then rejects. This test is
    /// what keeps the fusion from being "simplified" away without that
    /// failure being the first thing anyone sees.
    ///
    /// Asserted as a *difference* from the unfused evaluation, searched
    /// for rather than assumed, so it cannot be satisfied by a lucky
    /// argument.
    #[test]
    fn the_rational_approximation_is_evaluated_fused() {
        fn unfused(x: f64, coefficients: &[f64; 8]) -> f64 {
            let mut answer = coefficients[0];
            for coefficient in &coefficients[1..] {
                answer = answer * x + *coefficient;
            }
            answer
        }

        let mut differs = false;
        let mut w: f64 = -0.4;
        for _ in 0..4096 {
            differs |= polevl(w, &SPENCE_A).to_bits() != unfused(w, &SPENCE_A).to_bits()
                || polevl(w, &SPENCE_B).to_bits() != unfused(w, &SPENCE_B).to_bits();
            w = f64::from_bits(w.to_bits() + 1);
        }
        assert!(
            differs,
            "fused and unfused Horner agreed at every sampled w, so this \
             test can no longer tell the two evaluations apart"
        );
    }

    /// `π²/6` folds to the same double as the literal the closed-form
    /// tests use.
    #[test]
    fn the_reflection_constant_is_pi_squared_over_six() {
        assert_eq!(PI_SQ_OVER_SIX.to_bits(), PI_SQ_OVER_6.to_bits());
    }

    #[test]
    fn spence_edges_follow_cephes() {
        assert!(spence(-1.0).is_nan(), "negative argument is a domain error");
        assert!(spence(f64::INFINITY).is_nan());
        assert!(spence(f64::NAN).is_nan());
    }

    #[test]
    fn bessel_k1_satisfies_the_wronskian() {
        // I₁(x)·K₀(x) + I₀(x)·K₁(x) = 1/x, exactly (DLMF 10.28.2). An
        // identity rather than a table lookup, so it pins K₁ against
        // three independent cephes routines at once.
        for &x in &[0.1, 0.5, 1.0, 2.0, 2.5, 5.0, 20.0, 100.0] {
            let lhs = x.bessel_i1() * x.bessel_k0() + x.bessel_i0() * bessel_k1(x);
            assert_close(lhs, 1.0 / x, "K1 Wronskian");
        }
    }

    #[test]
    fn bessel_k1_edges_follow_cephes() {
        assert_eq!(bessel_k1(0.0), f64::INFINITY);
        assert!(bessel_k1(-1.0).is_nan());
        assert_eq!(bessel_k1(f64::INFINITY), 0.0);
        assert!(bessel_k1(f64::NAN).is_nan());
        // No explicit underflow cutoff: the large-argument branch is
        // exp(-x)·(…)/√x, so it decays into the subnormals and reaches
        // zero on its own.
        assert!(bessel_k1(700.0) > 0.0);
        assert_eq!(bessel_k1(10_000.0), 0.0);
    }

    /// `Iₙ(x)` from its defining power series,
    /// `Σ_{k≥0} (x/2)^{2k+n} / (k!·(k+n)!)` (DLMF 10.25.2).
    ///
    /// Independent of cephes on purpose — it is what makes
    /// [`bessel_kn_satisfies_the_wronskian`] a real check on `bessel_kn`
    /// rather than a restatement of the recurrence the implementation
    /// already runs. Converges to machine precision well inside 60 terms
    /// for the `x ≤ 5` used there.
    fn bessel_in_series(n: u32, x: f64) -> f64 {
        let half = 0.5 * x;
        // k = 0: (x/2)^n / (0!·n!).
        let mut term = half.powi(n as i32);
        let mut factorials = (1..=n).map(f64::from).product::<f64>().max(1.0);
        let mut sum = term / factorials;
        for k in 1..60 {
            let kf = f64::from(k);
            term *= half * half;
            factorials *= kf * (kf + f64::from(n));
            sum += term / factorials;
        }
        sum
    }

    #[test]
    fn bessel_kn_satisfies_the_wronskian() {
        // I_ν(x)·K_{ν+1}(x) + I_{ν+1}(x)·K_ν(x) = 1/x, exactly
        // (DLMF 10.28.2). The recurrence in `bessel_kn` cannot satisfy
        // this by construction: a wrong seed, a wrong number of steps,
        // or an off-by-one in the order all break it, because the `I`
        // here come from their own series and not from cephes.
        //
        // ν = 2 is not redundant with ν = 1. Only order 2 is live, but
        // the recurrence's `2m/x` factor is `2/x` at the single step
        // that produces K₂ — so dropping `m` is invisible until a third
        // step runs, and this is where it shows.
        for order in 1..=2_u32 {
            for &x in &[0.5, 1.0, 2.0, 3.5, 5.0] {
                let lhs = bessel_in_series(order, x) * bessel_kn(order as i32 + 1, x)
                    + bessel_in_series(order + 1, x) * bessel_kn(order as i32, x);
                assert_close(lhs, 1.0 / x, "K Wronskian");
            }
        }
    }

    #[test]
    fn bessel_kn_agrees_with_k0_and_k1_at_their_orders() {
        // The two seeds are returned directly, so this pins the match
        // arms: a swapped pair would still satisfy the Wronskian above
        // only by accident.
        for &x in &[0.1, 1.0, 9.5, 50.0, 300.0] {
            assert_eq!(bessel_kn(0, x), x.bessel_k0());
            assert_eq!(bessel_kn(1, x), bessel_k1(x));
        }
    }

    #[test]
    fn bessel_kn_edges_follow_cephes() {
        // K₋ₙ = Kₙ.
        assert_eq!(bessel_kn(-2, 3.0), bessel_kn(2, 3.0));
        assert_eq!(bessel_kn(-1, 3.0), bessel_kn(1, 3.0));
        // The seeds carry these; the recurrence has to propagate them.
        assert_eq!(bessel_kn(2, 0.0), f64::INFINITY);
        assert!(bessel_kn(2, -1.0).is_nan());
        assert_eq!(bessel_kn(2, f64::INFINITY), 0.0);
        assert!(bessel_kn(2, f64::NAN).is_nan());
    }

    #[test]
    fn negative_zero_is_zero_and_not_a_negative_argument() {
        // `-0.0 < 0.0` is false, so IEEE puts negative zero in the
        // *zero* branch of every one of these — which is where scipy
        // puts it too. Checked at every order because only the
        // recurrence arm (n ≥ 2) ever got it wrong: the seeds are `+∞`
        // there while `2m/x` is `-∞`, and `∞ + -∞` is `NaN` (PR #59
        // review).
        assert_eq!(spence(-0.0), spence(0.0));
        assert_eq!(bessel_k1(-0.0), f64::INFINITY);
        for order in 0..6_i32 {
            assert_eq!(
                bessel_kn(order, -0.0),
                f64::INFINITY,
                "kn({order}, -0.0) must be +inf, as at +0.0"
            );
            assert_eq!(bessel_kn(order, -0.0), bessel_kn(order, 0.0));
        }
    }
}
