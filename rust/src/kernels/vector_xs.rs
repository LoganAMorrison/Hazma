//! The vector-mediator annihilation cross sections, ported from
//! `hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::vector_mediator`] is the Python-visible half.
//!
//! # The physics
//!
//! Six entry points, all in MeV⁻² and all spin-averaged. Five are
//! closed-form `σ(e_cm)` for one final state — `f f̄`, `π⁺π⁻`, `π⁰γ`,
//! `π⁰V` through an s-channel mediator, and `VV` through the t/u
//! channels — and each opens at its own threshold, returning exactly
//! `0.0` below it. The sixth, [`thermal_cross_section`], is the
//! Maxwell–Boltzmann average
//!
//! ```text
//!   ⟨σv⟩(x) = x / (2 K₂(x))² · ∫₂^Z dz  σ_all(m_x z) z² (z² − 4) K₁(x z)
//! ```
//!
//! over the sum of all six channels, with `Z = max(50/x, 150)` and QAGP
//! break points at `[2, m_v/m_x, 2 m_v/m_x]`.
//!
//! `sigma_xx_to_all` — the sum the integrand needs — was also a public
//! Cython `def`. Nothing imported it, so the plan drops it rather than
//! porting it (`phase-05-mediator-cross-sections.md`, Task 5.1); it
//! survives here as the private [`sigma_xx_to_all`] the integrand calls,
//! which is the only consumer it ever had.
//!
//! # The `**` operator went through **complex** arithmetic
//!
//! The single most important fact about this file, and it is invisible in
//! the `.pyx`. Two expressions raise a double to the power `1.5`, so
//! Cython compiled the *whole enclosing expression* in `double _Complex`
//! and converted back with `__Pyx_SoftComplexToDouble`, which raises
//! `TypeError` if the imaginary part is non-zero. `grep -c
//! SoftComplexToDouble` over the generated C finds exactly two call
//! sites, `__sigma_xx_to_v_to_pipi` and `__sigma_xx_to_v_to_pi0v`, and
//! the shipped object code confirms it: both call `cpow` and both call
//! compiler-rt's `___divdc3`, while the other three kernels call neither.
//!
//! Neither routine agrees with its real-arithmetic spelling, so this is
//! not a detail that can be transcribed away. The measurements, the two
//! reproductions ([`soft_complex_pow_1_5`],
//! [`complex_quotient_real_denominator`]) and their tests live in
//! [`crate::kernels::soft_complex`], which cython-to-rust Task 5.2 split
//! out of this file when the scalar module turned out to need the
//! identical pair.
//!
//! With both, all five closed-form kernels reproduce the Cython
//! **bit-for-bit at every one of the 5,811 values the parity corpus
//! compares them on** — 5,814 stored (5,670 on the swept grids, 144 on
//! the scalar probes) less the 3 positions that stand in for a pinned
//! `TypeError` rather than for a number.
//!
//! # Where the FMAs are
//!
//! `objdump -d hazma/vector_mediator/_c_vector_mediator_cross_sections
//! .cpython-313-darwin.so | grep -cE '\bfn?m(add|sub)\b'` prints **84**,
//! of which 9 are inside `___divdc3` itself and the rest divide as
//! 5 per copy in `ff`, 4 in `pi0g`, 4 in `pipi`, 3 in `pi0v` and 12 in
//! `vv` (the `def` wrappers inline the scalar and array paths, so the
//! per-symbol counts are doubled). Each is written [`f64::mul_add`]
//! below. Four shapes recur in every kernel and are fused in all of them:
//! `−4 m² + e_cm²`, `2 m² + e_cm²`, `m_v² Γ_v² + (m_v² − e_cm²)²`, and
//! the `(m_v² − e_cm²)` subtraction's *absence* from that list.
//!
//! Three expressions that look fusable and are **not**, each read off the
//! disassembly rather than guessed:
//!
//! * `mv**2 - e_cm**2` — `fmul d3, d9, d9` then `fsub d2, d3, d2`, in all
//!   five kernels. `mv²` feeds both this subtraction and the
//!   `mv²·width_v²` product beside it, so the multiply has two uses and
//!   clang leaves it alone. `e_cm²` and `mx²` are likewise multi-use
//!   throughout and never fuse as the multiplicand.
//! * `-mpi0**2 + e_cm**2` in [`sigma_xx_to_v_to_pi0g`] — `fmul d0, d1, d1`
//!   then `fsub d0, d11, d0`, unfused, even though `mpi0²` has one use.
//!   Its `pipi` sibling `-4·mpi² + e_cm²` **is** fused (`fmadd d0, d0,
//!   d1, d12`), so the two spellings are not interchangeable: a
//!   coefficient of `-1` gives clang a plain negation to fold into the
//!   subtraction and no multiply-add to form.
//! * `e_cm**4 + (...)` in [`sigma_xx_to_vv`] — `fadd d0, d0, d9`. The
//!   left operand is a `pow` return, not a multiply.
//!
//! # Association, not operand order
//!
//! Several expressions below are built up with `*=` and `+=`, which puts
//! the accumulator on the *left* where the disassembly has it on the
//! right. That is a rewrite `cargo clippy` asks for and it is safe:
//! IEEE-754 `+` and `*` are commutative to the bit, so only the
//! **grouping** is load-bearing. The grouping is the `.pyx`'s throughout
//! — clang commuted operands freely and reassociated nothing, and neither
//! does this file. The corpus is what proves it: all five kernels stayed
//! bit-equal across that rewrite.
//!
//! # `pow` is a libm call except at exponent 2
//!
//! clang folds `pow(x, 2.0)` to `x·x` (exact either way) and folds
//! `pow(M_PI, 4.0)` / `pow(M_PI, 5.0)` to immediates, but leaves
//! `pow(x, 3.0)` and `pow(x, 4.0)` as calls — `_pow` is in the shipped
//! object's lazy-bind table, reached 8 times from `vv`, 4 from `pi0g` and
//! 1 from `pi0v`. `x.powf(3.0)` and `x.powf(4.0)` below are therefore
//! **not** interchangeable with `x·x·x`: a correctly rounded `pow` is a
//! different number.
//!
//! # Constants
//!
//! The `.pyx` declares its own six module-level `cdef double`s rather
//! than `include`-ing either shared header, so they live here rather than
//! in [`crate::constants::derived`] — which is scored against surviving
//! `.pyx` files and this file's is deleted by the same task that adds
//! this module (Phase 04's learnings, §5). Four of the six happen to
//! equal their [`crate::constants::legacy`] counterparts and two do not;
//! all six are transcribed from the `.pyx` verbatim under `rules.md`
//! rule 4, which forbids reconciling them here.

use crate::kernels::soft_complex::{
    NonRealResult, complex_quotient_real_denominator, soft_complex_pow_1_5,
};
use crate::quad::{DEFAULT_EPSABS, DEFAULT_EPSREL, DEFAULT_LIMIT, QuadOpts, quad};
use crate::special::{bessel_k1, bessel_kn};

/// Electron mass in MeV — `_c_vector_mediator_cross_sections.pyx:9`.
pub const ME: f64 = 0.510998928;
/// Muon mass in MeV — `:10`.
pub const MMU: f64 = 105.6583715;
/// Neutral pion mass in MeV — `:11`.
pub const MPI0: f64 = 134.9766;
/// Charged pion mass in MeV — `:12`.
pub const MPI: f64 = 139.57018;
/// Pion decay constant in MeV — `:13`. Not in either shared table.
pub const FPI: f64 = 92.2138;
/// Fine-structure constant — `:14`. A third value again: `crate::
/// constants::pdg` uses `1/137.035999084` and `legacy` uses `1/137`.
pub const ALPHA_EM: f64 = 1.0 / 137.04;

/// `π⁴`, as clang folds `pow(M_PI, 4.0)`.
///
/// A literal because `powf` is not `const`, and **not** `PI*PI*PI*PI`,
/// which is a different double. `pi_powers_match_libm` re-derives both
/// from [`std::f64::consts::PI`] at run time.
const PI_4: f64 = 97.40909103400242;
/// `π⁵`, as clang folds `pow(M_PI, 5.0)`. See [`PI_4`].
const PI_5: f64 = 306.0196847852814;

/// The quadrature settings `thermal_cross_section` inherits from
/// `scipy.integrate.quad`'s defaults — the `.pyx` passes neither
/// `epsabs` nor `epsrel` (`:656-660`). `points` is supplied per call
/// because two of the three break points depend on `m_v/m_x`.
const THERMAL_EPSABS: f64 = DEFAULT_EPSABS;
/// See [`THERMAL_EPSABS`].
const THERMAL_EPSREL: f64 = DEFAULT_EPSREL;

/// `σ(x x̄ → V* → f f̄)` in MeV⁻², for a lepton of mass `mf`.
///
/// Zero below `e_cm = max(2 m_f, 2 m_x)`. `gvll` is the mediator's
/// coupling to that lepton and `width_v` its full decay width in MeV.
#[must_use]
pub fn sigma_xx_to_v_to_ff(
    e_cm: f64,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvll: f64,
    width_v: f64,
    mf: f64,
) -> f64 {
    if e_cm < 2.0 * mf || e_cm < 2.0 * mx {
        return 0.0;
    }

    let e2 = e_cm * e_cm;
    let mf2 = mf * mf;
    let mx2 = mx * mx;
    let mv2 = mv * mv;

    let mut numerator = (gvll * gvll) * (gvxx * gvxx);
    numerator *= mf2.mul_add(2.0, e2);
    numerator *= (mf2.mul_add(-4.0, e2) / mx2.mul_add(-4.0, e2)).sqrt();
    numerator *= mx2.mul_add(2.0, e2);

    numerator / (((12.0 * std::f64::consts::PI) * e2) * propagator(mv2, e2, width_v))
}

/// `σ(x x̄ → V* → π⁺π⁻)` in MeV⁻².
///
/// Zero below `e_cm = max(2 m_x, 2 m_π)`.
///
/// # Errors
///
/// [`NonRealResult`] at `e_cm = 2 m_x`, where the denominator is exactly
/// zero and `__divdc3`'s recovery clause returns a NaN imaginary part —
/// see [`complex_quotient_real_denominator`].
#[allow(clippy::too_many_arguments)]
pub fn sigma_xx_to_v_to_pipi(
    e_cm: f64,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    width_v: f64,
) -> Result<f64, NonRealResult> {
    if e_cm < 2.0 * mx || e_cm < 2.0 * MPI {
        return Ok(0.0);
    }

    let e2 = e_cm * e_cm;
    let mx2 = mx * mx;
    let mv2 = mv * mv;

    let isospin = gvdd - gvuu;
    let couplings = (gvxx * gvxx) * (isospin * isospin);

    let mut numerator = couplings * soft_complex_pow_1_5((MPI * MPI).mul_add(-4.0, e2));
    numerator *= mx2.mul_add(2.0, e2);

    let mut denominator = ((48.0 * std::f64::consts::PI) * e2) * mx2.mul_add(-4.0, e2).sqrt();
    denominator *= propagator(mv2, e2, width_v);

    complex_quotient_real_denominator(numerator, denominator)
}

/// `σ(x x̄ → V* → π⁰γ)` in MeV⁻².
///
/// Zero below `e_cm = max(m_π⁰, 2 m_x)` — the threshold the `.pyx`
/// writes is `m_π⁰`, not `2 m_π⁰`, and it is transcribed as written.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn sigma_xx_to_v_to_pi0g(
    e_cm: f64,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    width_v: f64,
) -> f64 {
    if e_cm < MPI0 || e_cm < 2.0 * mx {
        return 0.0;
    }

    let e2 = e_cm * e_cm;
    let mx2 = mx * mx;
    let mv2 = mv * mv;

    let isospin = gvuu.mul_add(2.0, gvdd);
    let couplings = (gvxx * gvxx) * (ALPHA_EM * (isospin * isospin));

    // Unfused, unlike its `pipi` sibling — see the module docs.
    let mut numerator = (e2 - MPI0 * MPI0).powf(3.0) * couplings;
    numerator *= mx2.mul_add(2.0, e2);

    let mut denominator = ((FPI * FPI) * 3456.0) * PI_4;
    denominator *= e_cm.powf(3.0);
    denominator *= mx2.mul_add(-4.0, e2).sqrt();
    denominator *= propagator(mv2, e2, width_v);

    numerator / denominator
}

/// `σ(x x̄ → V* → π⁰ V)` in MeV⁻².
///
/// Zero below `e_cm = max(m_π⁰ + m_v, 2 m_x)`. The `.pyx` carries a
/// `# TODO: UPDATE THIS!` above this expression; the port transcribes it
/// unchanged, because changing a published number is not this project's
/// business (`rules.md` rule 1).
///
/// # Errors
///
/// As [`sigma_xx_to_v_to_pipi`].
#[allow(clippy::too_many_arguments)]
pub fn sigma_xx_to_v_to_pi0v(
    e_cm: f64,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    width_v: f64,
) -> Result<f64, NonRealResult> {
    if e_cm < MPI0 + mv || e_cm < 2.0 * mx {
        return Ok(0.0);
    }

    let e2 = e_cm * e_cm;
    let mx2 = mx * mx;
    let mv2 = mv * mv;

    let isospin = gvdd - gvuu;
    let isoscalar = gvuu + gvdd;
    let couplings = (gvxx * gvxx) * ((isospin * isospin) * (isoscalar * isoscalar));

    // The Källén-like product, in the `.pyx`'s own association order:
    // ((mpi0 - mv - e) (mpi0 + mv - e)) then × (mpi0 - mv + e) then
    // × (mpi0 + mv + e).
    let minus = MPI0 - mv;
    let plus = mv + MPI0;
    let mut kallen = (minus - e_cm) * (plus - e_cm);
    kallen *= e_cm + minus;
    kallen *= e_cm + plus;

    let mut numerator = couplings * soft_complex_pow_1_5(kallen);
    numerator *= mx2.mul_add(2.0, e2);

    let mut denominator = ((FPI * FPI) * 1536.0) * PI_5;
    denominator *= e_cm.powf(3.0);
    denominator *= mx2.mul_add(-4.0, e2).sqrt();
    denominator *= propagator(mv2, e2, width_v);

    complex_quotient_real_denominator(numerator, denominator)
}

/// `σ(x x̄ → V V)` in MeV⁻², through the t and u channels.
///
/// Zero below `e_cm = max(2 m_v, 2 m_x)`. The only kernel here with no
/// s-channel propagator and the only one that calls `log`.
#[must_use]
pub fn sigma_xx_to_vv(e_cm: f64, mx: f64, mv: f64, gvxx: f64) -> f64 {
    if e_cm < 2.0 * mv || e_cm < 2.0 * mx {
        return 0.0;
    }

    let gvxx4 = gvxx.powf(4.0);
    let e2 = e_cm * e_cm;
    let mv2 = mv * mv;
    let mx2 = mx * mx;

    let root_v = mv2.mul_add(-4.0, e2).sqrt();
    let root_x = mx2.mul_add(-4.0, e2).sqrt();
    // `-2 · root_v · root_x`, with the sign carried by the subtraction at
    // the end: clang forms `root_x · (root_v + root_v)` and negates by
    // reversing that subtraction, and both steps are exact.
    let roots = root_x * (root_v + root_v);

    let mv4 = mv.powf(4.0);
    let mx4 = mx.powf(4.0);

    let t_channel = mx2.mul_add(e2, mv4.mul_add(2.0, mx4 * 4.0)) * roots
        / mx2.mul_add(e2, (mv2 * -4.0).mul_add(mx2, mv4));

    let mut polynomial = mv4.mul_add(4.0, mv2 * -8.0 * mx2);
    polynomial = mx4.mul_add(-8.0, polynomial);
    polynomial = (mx2 * 4.0).mul_add(e2, polynomial);
    polynomial += e_cm.powf(4.0);
    polynomial += polynomial;

    let shifted = mv2.mul_add(-2.0, e2);
    let ratio = root_v.mul_add(root_x, shifted) / (-root_v).mul_add(root_x, shifted);
    let s_channel = polynomial * ratio.ln() / shifted;

    gvxx4 * (s_channel - t_channel) / (((16.0 * std::f64::consts::PI) * e2) * mx2.mul_add(-4.0, e2))
}

/// The Breit–Wigner denominator `(m_v² − e_cm²)² + m_v² Γ_v²`, shared by
/// the four s-channel kernels.
///
/// Factored out because all four spell it identically and clang emits the
/// identical two instructions for each: `fmul`/`fsub` for
/// `m_v² − e_cm²` (unfused — `m_v²` has a second use) and `fmadd` for the
/// sum.
fn propagator(mv2: f64, e2: f64, width_v: f64) -> f64 {
    let detuning = mv2 - e2;
    mv2.mul_add(width_v * width_v, detuning * detuning)
}

/// The sum of all six channels, `σ_all(e_cm)` in MeV⁻².
///
/// The Cython exported this as a public `def` that nothing imported, so
/// Task 5.1 drops it from the public surface and keeps it here as the
/// [`thermal_cross_section`] integrand's own helper — its only real
/// consumer. Summation order is the `.pyx`'s:
/// `e + μ + π⁺π⁻ + π⁰γ + π⁰V + VV`.
///
/// # Errors
///
/// As [`sigma_xx_to_v_to_pipi`], from either complex-valued channel.
#[allow(clippy::too_many_arguments)]
fn sigma_xx_to_all(
    e_cm: f64,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    gvee: f64,
    gvmumu: f64,
    width_v: f64,
) -> Result<f64, NonRealResult> {
    let sig_e = sigma_xx_to_v_to_ff(e_cm, mx, mv, gvxx, gvee, width_v, ME);
    let sig_mu = sigma_xx_to_v_to_ff(e_cm, mx, mv, gvxx, gvmumu, width_v, MMU);
    let sig_pi = sigma_xx_to_v_to_pipi(e_cm, mx, mv, gvxx, gvuu, gvdd, width_v)?;
    let sig_pi0g = sigma_xx_to_v_to_pi0g(e_cm, mx, mv, gvxx, gvuu, gvdd, width_v);
    let sig_pi0v = sigma_xx_to_v_to_pi0v(e_cm, mx, mv, gvxx, gvuu, gvdd, width_v)?;
    let sig_v = sigma_xx_to_vv(e_cm, mx, mv, gvxx);

    Ok(sig_e + sig_mu + sig_pi + sig_pi0g + sig_pi0v + sig_v)
}

/// The thermally averaged `⟨σv⟩` in MeV⁻², at `x = m_x / T`.
///
/// `x` is clipped at 300 rather than short-circuited to zero — the scalar
/// model does the opposite at the same boundary, and the corpus pins both
/// (`test/parity/cases.py`'s `_thermal_blocks`). Above `x = 300` this
/// therefore keeps returning the value at 300.
///
/// # Errors
///
/// As [`sigma_xx_to_v_to_pipi`], if the integrand's `σ_all` hits the
/// `e_cm = 2 m_x` threshold. Unreachable in practice: that needs `z = 2`,
/// which is the integration's *lower limit*, and Gauss–Kronrod evaluates
/// only strictly inside each subinterval.
#[allow(clippy::too_many_arguments)]
pub fn thermal_cross_section(
    x: f64,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    gvee: f64,
    gvmumu: f64,
    width_v: f64,
) -> Result<f64, NonRealResult> {
    // "If x is really large, we will get divide by zero errors; we clip x
    // since the thermal cross section should tend to a constant."
    let xnew = if x < 300.0 { x } else { 300.0 };
    let two_k2 = 2.0 * bessel_kn(2, xnew);
    let prefactor = xnew / (two_k2 * two_k2);

    // `max(50.0 / xnew, 150.0)`, in Python's evaluation order.
    let floor = 50.0 / xnew;
    let upper = if 150.0 > floor { 150.0 } else { floor };

    // "points at which integrand may have trouble are: 1. endpoint;
    // 2. when ss final state is accessible => z = 2 mv / mx;
    // 3. when we hit mediator resonance => z = mv / mx"
    let ratio = mv / mx;
    let points = [2.0, ratio, 2.0 * ratio];

    let mut nonreal = false;
    let mut integrand = |z: f64| {
        match sigma_xx_to_all(mx * z, mx, mv, gvxx, gvuu, gvdd, gvee, gvmumu, width_v) {
            // `xnew`, not `x`: the `.pyx` passes the *clipped* value into
            // the integrand's args tuple (`:658`), so above `x = 300`
            // the Boltzmann weight saturates along with the prefactor.
            // With the unclipped `x` here, `K₁(x z)` underflows to zero
            // past `x ≈ 350` and the whole average collapses to `0.0`
            // instead of tending to a constant — which is the behavior
            // the clip exists to produce.
            Ok(sigma) => sigma * (z * z) * ((z * z) - 4.0) * bessel_k1(xnew * z),
            Err(NonRealResult) => {
                // scipy's `quad` propagates an exception out of the
                // integrand rather than absorbing it, so the flag makes
                // this call raise the same way. `NaN` keeps the
                // integrator's own arithmetic defined until then.
                nonreal = true;
                f64::NAN
            }
        }
    };

    let options = QuadOpts {
        epsabs: THERMAL_EPSABS,
        epsrel: THERMAL_EPSREL,
        limit: DEFAULT_LIMIT,
        points: Some(&points),
    };
    let integral = match quad(&mut integrand, 2.0, upper, &options) {
        Ok(outcome) => outcome.value,
        // Unreachable, and asserted so by
        // `thermal_quad_options_are_always_accepted`: `QuadError` is a
        // statement about the options, never about the integrand, and
        // `limit` is a const comfortably above the three break points.
        Err(_) => f64::NAN,
    };

    if nonreal {
        return Err(NonRealResult);
    }
    Ok(prefactor * integral)
}

#[cfg(test)]
mod tests {
    use super::{
        ALPHA_EM, FPI, ME, MMU, MPI, MPI0, NonRealResult, PI_4, PI_5, propagator, sigma_xx_to_all,
        sigma_xx_to_v_to_ff, sigma_xx_to_v_to_pi0g, sigma_xx_to_v_to_pi0v, sigma_xx_to_v_to_pipi,
        sigma_xx_to_vv, thermal_cross_section,
    };
    use crate::quad::{QuadOpts, quad};

    /// A representative model point: the parity corpus's `open_resonance`
    /// `KineticMixing(mx=100, mv=300, gvxx=1, eps=1e-1)` couplings, rounded
    /// to values a reader can check by eye. Nothing here depends on them
    /// being that model's exact numbers — the corpus is where the port is
    /// compared against the Cython.
    const MX: f64 = 100.0;
    const MV: f64 = 300.0;
    const GVXX: f64 = 1.0;
    const GVUU: f64 = 0.3;
    const GVDD: f64 = -0.15;
    // No `GVSS`: the strange-quark coupling reaches none of the five
    // kernels. The `.pyx` declared it in four of their signatures and
    // marked it `CYTHON_UNUSED` in every one, so it survives only at the
    // PyO3 boundary, where the public signature keeps it —
    // `test/test_core_vector_xs.py` is where that is pinned, because it
    // is the only layer that can see the argument.
    const GVEE: f64 = 0.1;
    const GVMUMU: f64 = 0.1;
    const WIDTH_V: f64 = 2.5;

    fn bits(x: f64) -> u64 {
        x.to_bits()
    }

    // -- The two constants clang folded ---------------------------------

    /// `PI_4` and `PI_5` are `pow`'s answers, and a product of `PI`s is
    /// not always the same number.
    ///
    /// The reason they are literals: `powf` is not `const`. The reason
    /// they are not spelled as products is measured here rather than
    /// assumed, and the measurement is not uniform — on this platform
    /// `pow(π, 4)` happens to equal the left-associated `π·π·π·π` while
    /// balancing the product as `(π·π)·(π·π)` moves it by an ulp, and
    /// `pow(π, 5)` equals neither product. So the guard is stated for the
    /// spellings that do differ, and the equalities above it are what
    /// actually pins the constants.
    #[test]
    fn pi_powers_match_libm_pow() {
        let pi = std::f64::consts::PI;
        assert_eq!(bits(PI_4), bits(pi.powf(4.0)));
        assert_eq!(bits(PI_5), bits(pi.powf(5.0)));
        assert_ne!(bits(PI_4), bits((pi * pi) * (pi * pi)));
        assert_ne!(bits(PI_5), bits(pi * pi * pi * pi * pi));
    }

    /// The six module-level `cdef double`s, against the `.pyx` digits.
    ///
    /// Four coincide with `crate::constants::legacy` and two do not, and
    /// `rules.md` rule 4 forbids reconciling either group here. Written as
    /// literals on both sides so the check is a transcription check.
    #[test]
    fn the_module_constants_are_the_pyx_digits() {
        assert_eq!(bits(ME), bits(0.510_998_928));
        assert_eq!(bits(MMU), bits(105.658_371_5));
        assert_eq!(bits(MPI0), bits(134.976_6));
        assert_eq!(bits(MPI), bits(139.570_18));
        assert_eq!(bits(FPI), bits(92.213_8));
        assert_eq!(bits(ALPHA_EM), bits(1.0 / 137.04));

        // The four that agree with the legacy table, and the two that do
        // not: alpha_em here is 1/137.04 where legacy is 1/137, and fpi
        // is in neither shared table at all.
        use crate::constants::legacy;
        assert_eq!(bits(ME), bits(legacy::MASS_E));
        assert_eq!(bits(MMU), bits(legacy::MASS_MU));
        assert_eq!(bits(MPI0), bits(legacy::MASS_PI0));
        assert_eq!(bits(MPI), bits(legacy::MASS_PI));
        assert_ne!(bits(ALPHA_EM), bits(legacy::ALPHA_EM));
    }

    // -- Thresholds -------------------------------------------------------

    /// Every kernel returns exactly `0.0` below its own threshold, and
    /// something non-zero just above it.
    ///
    /// `0.0` exactly, not "small": the corpus compares the sub-threshold
    /// region with `atol = 0`, so a port that returned 1e-300 there would
    /// fail — which is the intended answer
    /// (`test/parity/tolerances.py`, "`atol` is 0.0 everywhere").
    #[test]
    fn each_channel_opens_at_its_own_threshold() {
        let below = |e: f64| e * (1.0 - 1e-12);
        let above = |e: f64| e * (1.0 + 1e-12);

        // f fbar: max(2 m_f, 2 m_x). With m_x = 100 the dark-matter
        // threshold dominates both leptons.
        let ff = |e| sigma_xx_to_v_to_ff(e, MX, MV, GVXX, GVEE, WIDTH_V, ME);
        assert_eq!(bits(ff(below(2.0 * MX))), bits(0.0));
        assert!(ff(above(2.0 * MX)) > 0.0);

        // pi+ pi-: max(2 m_x, 2 m_pi). With m_x = 100 the *pion* wins,
        // at 279.14 MeV, so the two clauses are checked separately --
        // below 2 m_pi it is zero even though 2 m_x is already cleared.
        let pipi = |e| sigma_xx_to_v_to_pipi(e, MX, MV, GVXX, GVUU, GVDD, WIDTH_V);
        const { assert!(2.0 * MPI > 2.0 * MX) };
        assert_eq!(bits(pipi(above(2.0 * MX)).unwrap()), bits(0.0));
        assert_eq!(bits(pipi(below(2.0 * MPI)).unwrap()), bits(0.0));
        assert!(pipi(above(2.0 * MPI)).unwrap() > 0.0);

        // pi0 gamma: max(m_pi0, 2 m_x).
        let pi0g = |e| sigma_xx_to_v_to_pi0g(e, MX, MV, GVXX, GVUU, GVDD, WIDTH_V);
        assert_eq!(bits(pi0g(below(2.0 * MX))), bits(0.0));
        assert!(pi0g(above(2.0 * MX)) > 0.0);

        // pi0 V: max(m_pi0 + m_v, 2 m_x) -- the mediator wins here.
        let pi0v = |e| sigma_xx_to_v_to_pi0v(e, MX, MV, GVXX, GVUU, GVDD, WIDTH_V);
        assert_eq!(bits(pi0v(below(MPI0 + MV)).unwrap()), bits(0.0));
        assert!(pi0v(above(MPI0 + MV)).unwrap() > 0.0);
        assert_eq!(bits(pi0v(2.0 * MX + 1.0).unwrap()), bits(0.0));

        // V V: max(2 m_v, 2 m_x) -- the mediator again.
        let vv = |e| sigma_xx_to_vv(e, MX, MV, GVXX);
        assert_eq!(bits(vv(below(2.0 * MV))), bits(0.0));
        assert!(vv(above(2.0 * MV)) > 0.0);
        assert_eq!(bits(vv(2.0 * MX + 1.0)), bits(0.0));
    }

    /// The `pi0 gamma` threshold really is `m_pi0`, not `2 m_pi0`.
    ///
    /// A one-body-plus-photon final state needs only `m_pi0` of
    /// center-of-mass energy, so the `.pyx`'s asymmetric guard is right
    /// and not a typo for its four `2 m` siblings. Pinned with a light
    /// dark matter so the `2 m_x` clause cannot mask it.
    #[test]
    fn the_pi0_gamma_threshold_is_one_pion_mass() {
        let mx = 1.0;
        let sigma = |e| sigma_xx_to_v_to_pi0g(e, mx, MV, GVXX, GVUU, GVDD, WIDTH_V);
        assert_eq!(bits(sigma(MPI0 * (1.0 - 1e-12))), bits(0.0));
        assert!(sigma(MPI0 * (1.0 + 1e-12)) > 0.0);
    }

    /// Both complex kernels raise at `e_cm = 2 m_x` exactly, and only
    /// there.
    ///
    /// The behavior the parity corpus pins as a `TypeError` in three
    /// blocks. It is a defect rather than a design
    /// (`docs/followups/todo/vector-cross-sections-raise-at-the-two-mx-threshold.md`),
    /// reproduced under `rules.md` rule 1.
    #[test]
    fn the_complex_kernels_raise_at_the_dark_matter_threshold() {
        // m_x chosen so 2 m_x clears both channels' own thresholds.
        let mx = 400.0;
        let e_cm = 2.0 * mx;
        assert_eq!(
            sigma_xx_to_v_to_pipi(e_cm, mx, MV, GVXX, GVUU, GVDD, WIDTH_V),
            Err(NonRealResult)
        );
        assert_eq!(
            sigma_xx_to_v_to_pi0v(e_cm, mx, MV, GVXX, GVUU, GVDD, WIDTH_V),
            Err(NonRealResult)
        );
        // One ulp away in either direction it is an ordinary number.
        for neighbour in [
            f64::from_bits(e_cm.to_bits() + 1),
            f64::from_bits(e_cm.to_bits() + 2),
        ] {
            assert!(sigma_xx_to_v_to_pipi(neighbour, mx, MV, GVXX, GVUU, GVDD, WIDTH_V).is_ok());
            assert!(sigma_xx_to_v_to_pi0v(neighbour, mx, MV, GVXX, GVUU, GVDD, WIDTH_V).is_ok());
        }
        // And the four real kernels return infinities there instead of
        // raising, which is the asymmetry the follow-up records.
        assert!(sigma_xx_to_v_to_ff(e_cm, mx, MV, GVXX, GVEE, WIDTH_V, ME).is_infinite());
        assert!(sigma_xx_to_v_to_pi0g(e_cm, mx, MV, GVXX, GVUU, GVDD, WIDTH_V).is_infinite());
    }

    // -- Statements the Cython never made ---------------------------------

    /// `σ(x x̄ → V* → f f̄) · s → g_ll² g_xx² / 12π` far above every scale.
    ///
    /// The high-energy limit of the closed form: with `s ≫ m_f², m_x²,
    /// m_v²` the two `(2m² + s)` factors go to `s`, the square root goes
    /// to 1, and the propagator to `s²`, leaving
    /// `g_ll² g_xx² s² / (12π s · s²)`. Owes nothing to the Cython —
    /// it is the analytic content of the expression, and it fails if any
    /// coefficient or power in the numerator or denominator is wrong.
    #[test]
    fn the_lepton_channel_has_the_right_high_energy_limit() {
        let expected = (GVEE * GVEE * GVXX * GVXX) / (12.0 * std::f64::consts::PI);
        // 1e8 MeV is six decades above m_v; the leading correction is
        // O(m_v²/s) ~ 1e-11, so 1e-9 is loose enough to be a limit check
        // and tight enough to fail on a wrong constant.
        let s = 1e8_f64 * 1e8;
        let got = sigma_xx_to_v_to_ff(1e8, MX, MV, GVXX, GVEE, WIDTH_V, ME) * s;
        assert!(
            (got - expected).abs() < 1e-9 * expected,
            "sigma*s -> {got}, expected {expected}"
        );
    }

    /// The Breit–Wigner denominator is `m_v² Γ_v²` exactly on resonance.
    ///
    /// The one place the propagator's two terms are separable, so it pins
    /// both: a sign error in `m_v² − e_cm²` survives everywhere else.
    #[test]
    fn the_propagator_is_the_width_term_on_resonance() {
        let mv2 = MV * MV;
        assert_eq!(
            bits(propagator(mv2, mv2, WIDTH_V)),
            bits(mv2 * (WIDTH_V * WIDTH_V))
        );
        // And the resonance is a maximum of the cross section: half a
        // width away it is already smaller.
        let on = sigma_xx_to_v_to_ff(MV, MX, MV, GVXX, GVEE, WIDTH_V, ME);
        let off = sigma_xx_to_v_to_ff(MV + WIDTH_V, MX, MV, GVXX, GVEE, WIDTH_V, ME);
        assert!(on > off, "on-shell {on} is not above off-shell {off}");
    }

    /// `sigma_xx_to_all` is the sum of the six channels, and only the open
    /// ones contribute.
    ///
    /// At `e_cm` between `2 m_x` and `m_pi0 + m_v` the two heavy channels
    /// are shut, so the sum has to equal the four that are not — which
    /// catches a channel wired in twice or left out.
    #[test]
    fn the_total_is_the_sum_of_the_open_channels() {
        let e_cm = 2.0 * MX + 50.0;
        let total = sigma_xx_to_all(e_cm, MX, MV, GVXX, GVUU, GVDD, GVEE, GVMUMU, WIDTH_V).unwrap();

        let parts = sigma_xx_to_v_to_ff(e_cm, MX, MV, GVXX, GVEE, WIDTH_V, ME)
            + sigma_xx_to_v_to_ff(e_cm, MX, MV, GVXX, GVMUMU, WIDTH_V, MMU)
            + sigma_xx_to_v_to_pipi(e_cm, MX, MV, GVXX, GVUU, GVDD, WIDTH_V).unwrap()
            + sigma_xx_to_v_to_pi0g(e_cm, MX, MV, GVXX, GVUU, GVDD, WIDTH_V);

        assert_eq!(bits(total), bits(parts));
        // The two that are shut, named rather than assumed.
        assert_eq!(
            bits(sigma_xx_to_v_to_pi0v(e_cm, MX, MV, GVXX, GVUU, GVDD, WIDTH_V).unwrap()),
            bits(0.0)
        );
        assert_eq!(bits(sigma_xx_to_vv(e_cm, MX, MV, GVXX)), bits(0.0));
        assert!(total > 0.0);
    }

    // -- The thermal average ----------------------------------------------

    /// Above `x = 300` the average saturates rather than vanishing.
    ///
    /// The `.pyx` clips `x` and passes the *clipped* value into the
    /// integrand's argument tuple, so both the prefactor and the
    /// Boltzmann weight freeze. Reproducing the clip in the prefactor
    /// alone leaves `K₁(x z)` underflowing to zero past `x ≈ 350` and the
    /// whole average collapsing to `0.0` — which is what this port did
    /// until the corpus's `x = 1000` grid point caught it.
    #[test]
    fn the_thermal_average_saturates_above_three_hundred() {
        let at_300 =
            thermal_cross_section(300.0, MX, MV, GVXX, GVUU, GVDD, GVEE, GVMUMU, WIDTH_V).unwrap();
        assert!(at_300 > 0.0);
        for x in [300.0, 301.0, 1_000.0, 1e6] {
            let got =
                thermal_cross_section(x, MX, MV, GVXX, GVUU, GVDD, GVEE, GVMUMU, WIDTH_V).unwrap();
            assert_eq!(bits(got), bits(at_300), "x = {x} did not saturate");
        }
        // And below the clip it is a genuine function of x.
        let at_299 =
            thermal_cross_section(299.0, MX, MV, GVXX, GVUU, GVDD, GVEE, GVMUMU, WIDTH_V).unwrap();
        assert_ne!(bits(at_299), bits(at_300));
    }

    /// The thermal integral, against an independent composite rule — and
    /// the shipped tolerances stop **1% short of it**.
    ///
    /// Nothing here reuses `crate::quad`'s algorithm: the same integrand
    /// is summed with Simpson's rule on a uniform grid. What it does
    /// reuse is a change of variable, and that is the whole reason the
    /// reference is accurate. Every cross section carries a
    /// `1/√(e_cm² − 4 m_x²)` factor and the weight in front of it a
    /// `(z² − 4)`, so the integrand behaves like `√(z² − 4)` at the lower
    /// limit — a branch point with an infinite derivative, on which a
    /// uniform composite rule converges like `h^{3/2}` rather than `h⁴`.
    /// Substituting `z = √(4 + w²)` (so `dz = w/z · dw`) turns that into
    /// `w²`, which Simpson handles to rounding.
    ///
    /// The entry point then lands **0.79% away** from that reference, and
    /// that is shipped behavior rather than a port defect: the `.pyx`
    /// passes neither `epsabs` nor `epsrel`, so the integral runs at
    /// scipy's default `epsabs = 1.49e-8` against an integrand whose
    /// integral is of order `1e-27`. The absolute criterion is met by the
    /// very first Gauss–Kronrod pass, QUADPACK returns on its initial
    /// three-interval partition (63 evaluations), and no subdivision ever
    /// happens. `test/test_core_quad.py` records the same partition from
    /// the other side.
    ///
    /// So this test asserts two things at two standards, and the pairing
    /// is the point: the entry point is within 2% of the true integral,
    /// and the *same integrand through the same integrator* at a
    /// convergent tolerance reproduces the reference to 4.2e-8. Together
    /// they say the formula and the transcription are right and the gap
    /// is the tolerance —
    /// `docs/followups/todo/thermal-cross-section-quadrature-never-converges.md`.
    #[test]
    fn the_thermal_integral_matches_a_composite_rule() {
        let x = 20.0;
        let upper: f64 = 150.0_f64.max(50.0 / x);

        // In `w = √(z² − 4)`: the integrand times the Jacobian.
        let transformed = |w: f64| {
            // `w = 0` is `z = 2`, where every cross section divides by
            // zero and the `w²` in front of it is zero: `inf · 0 = NaN`
            // in arithmetic, `0` in the limit, since `σ ~ C/w` there and
            // the product goes as `C z w² K₁ · (1/z) → 0`.
            if w == 0.0 {
                return 0.0;
            }
            let z = (4.0 + w * w).sqrt();
            let sigma =
                sigma_xx_to_all(MX * z, MX, MV, GVXX, GVUU, GVDD, GVEE, GVMUMU, WIDTH_V).unwrap();
            sigma * (z * z) * (w * w) * crate::special::bessel_k1(x * z) * (w / z)
        };

        // Panel boundaries at every kink: the two mediator break points
        // the `.pyx` names and the four channel thresholds, mapped into
        // `w`. A kink inside a panel would cost more accuracy than the
        // branch point did.
        let to_w = |z: f64| (z * z - 4.0).max(0.0).sqrt();
        let ratio = MV / MX;
        let w_max = to_w(upper);
        let mut edges = vec![0.0, w_max];
        for z in [
            ratio,
            2.0 * ratio,
            2.0 * MPI / MX,
            MPI0 / MX,
            2.0 * MV / MX,
            (MPI0 + MV) / MX,
        ] {
            if z > 2.0 && z < upper {
                edges.push(to_w(z));
            }
        }
        edges.sort_by(f64::total_cmp);
        edges.dedup();

        let mut reference = 0.0;
        for window in edges.windows(2) {
            let (lo, hi) = (window[0], window[1]);
            if hi <= lo {
                continue;
            }
            // Even panel count, so Simpson's rule is well defined.
            let panels = 40_000_usize;
            let h = (hi - lo) / panels as f64;
            let mut sum = transformed(lo) + transformed(hi);
            for i in 1..panels {
                let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
                sum += weight * transformed(lo + h * i as f64);
            }
            reference += sum * h / 3.0;
        }

        let prefactor = {
            let two_k2 = 2.0 * crate::special::bessel_kn(2, x);
            x / (two_k2 * two_k2)
        };
        let expected = prefactor * reference;
        let got =
            thermal_cross_section(x, MX, MV, GVXX, GVUU, GVDD, GVEE, GVMUMU, WIDTH_V).unwrap();

        // What the entry point ships: the right integral, resolved to
        // about a percent. 2e-2 is one measurement (7.9e-3) plus room,
        // not a fitted bound -- the convergent comparison below is where
        // the precision claim lives.
        assert!(
            (got - expected).abs() < 2e-2 * expected.abs(),
            "shipped tolerances gave {got}, Simpson {expected}"
        );

        // The same integrand, the same integrator, a convergent
        // tolerance: now the two independent quadratures agree to
        // 4.2e-8 relative (measured), which is Simpson's own residual
        // error -- the channel thresholds put a `(z - z_open)^{3/2}` on
        // panel boundaries but the rule still only sees `h^{5/2}` across
        // them. 1e-6 is 24x that, and five decades tighter than the 2e-2
        // the shipped tolerances leave above.
        let mut integrand = |z: f64| {
            let sigma =
                sigma_xx_to_all(MX * z, MX, MV, GVXX, GVUU, GVDD, GVEE, GVMUMU, WIDTH_V).unwrap();
            sigma * (z * z) * ((z * z) - 4.0) * crate::special::bessel_k1(x * z)
        };
        let ratio = MV / MX;
        let points = [2.0, ratio, 2.0 * ratio];
        let converged = quad(
            &mut integrand,
            2.0,
            upper,
            &QuadOpts {
                epsabs: 0.0,
                epsrel: 1e-11,
                limit: 500,
                points: Some(&points),
            },
        )
        .expect("convergent options are valid options")
        .value;
        let converged = prefactor * converged;
        assert!(
            (converged - expected).abs() < 1e-6 * expected.abs(),
            "converged {converged} vs Simpson {expected}"
        );
    }

    /// The quadrature options are accepted at every `x` the entry point
    /// admits, so `thermal_cross_section`'s `Err(_) => NaN` arm is
    /// unreachable.
    ///
    /// `crate::quad::QuadError` depends only on the options — `epsabs > 0`
    /// and `limit` above the surviving break-point count — and all three
    /// break points can survive, so `limit = 50` is the binding claim.
    #[test]
    fn thermal_quad_options_are_always_accepted() {
        for x in [1e-6, 0.1, 1.0 / 3.0, 1.0, 20.0, 300.0, 1e6] {
            let got =
                thermal_cross_section(x, MX, MV, GVXX, GVUU, GVDD, GVEE, GVMUMU, WIDTH_V).unwrap();
            assert!(got.is_finite(), "x = {x} gave {got}");
        }
    }
}
