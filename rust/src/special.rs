//! The three `scipy.special` functions hazma's compiled layer uses.
//!
//! A thin, PyO3-free shim over the [`spec_math`] crate
//! (`projects/cython-to-rust/rules.md`, Rust conventions rule 3 — flat
//! `fn(f64, ..) -> f64`, no PyO3 types, so `cargo test` needs no GIL).
//! `crate::special_probe` is the Python-visible half.
//!
//! # Sources and licensing
//!
//! Upstream: `spec_math` 0.1.6 (crates.io, **MIT OR Apache-2.0**), a
//! Rust re-implementation of the **cephes** library, Cephes Math Library
//! Release 2.1 (1989), Copyright 1984–1992 Stephen L. Moshier. Nothing
//! here is GSL-derived, which is what
//! `projects/cython-to-rust/adrs/ADR-0002-license-clean-numerics.md`
//! requires (`rules.md` rule 5 / Licensing 1).
//!
//! That lineage is the point rather than a coincidence: scipy's own
//! `spence`, `k0` and `k1` are cephes wrappers too, so the port is
//! algorithm-for-algorithm rather than merely value-for-value against
//! them. `scipy.special.kn` is the exception — it is `kv`, not cephes,
//! and [`bessel_kn`] documents what this module does about that.
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

use spec_math::{Bessel, Polylog};

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
/// The upstream spelling is `spec_math`'s `Polylog::li2`, whose *name*
/// says `Li₂` and whose *body* is `cephes64::spence` — the convention
/// trap one level down. Wrapping it under scipy's name here means no
/// kernel has to remember which one it got.
///
/// Edge behavior, matching cephes and therefore scipy: `x < 0` and
/// `x = ∞` give `NaN`, `x = 0` gives `π²/6`, `x = 1` gives `0`, and
/// `NaN` propagates.
pub fn spence(x: f64) -> f64 {
    x.li2()
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
    match n.unsigned_abs() {
        0 => x.bessel_k0(),
        1 => x.bessel_k1(),
        order => {
            // Non-finite and boundary inputs need no special-casing: the
            // seeds already carry cephes' answers (+∞ at x = 0, NaN
            // below it, 0 at +∞) and the recurrence propagates each one.
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
}
