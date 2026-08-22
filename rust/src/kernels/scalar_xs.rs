//! The scalar-mediator cross sections, ported from
//! `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx`.
//!
//! PyO3-free (`projects/cython-to-rust/rules.md`, Rust conventions
//! rule 3); [`crate::scalar_mediator`] is the Python-visible half.
//!
//! # The physics
//!
//! Twelve entry points, all spin-averaged and all in MeV⁻². Five are
//! annihilation cross sections `σ(e_cm)` through an s-channel scalar —
//! `f f̄`, `γγ`, `π⁰π⁰`, `π⁺π⁻` — plus `SS` through the t and u
//! channels; one is the crossed `S S → x x̄`; five are elastic
//! `x + target → x + target` for a lepton, a charged pion, a neutral
//! pion, a photon and the mediator itself; and the twelfth,
//! [`thermal_cross_section`], is the Maxwell–Boltzmann average
//!
//! ```text
//!   ⟨σv⟩(x) = x / (2 K₂(x))² · ∫₂^Z dz  σ_all(m_x z) z² (z² − 4) K₁(x z)
//! ```
//!
//! with `Z = max(50/x, 100)` and QAGP break points at
//! `[2, m_s/m_x, 2 m_s/m_x]`.
//!
//! Every kernel returns exactly `0.0` below its own threshold, which the
//! corpus compares at `atol = 0`.
//!
//! `sigma_xx_to_all` — the sum the integrand needs — was also a public
//! Cython `def` that nothing imported, so the plan drops it rather than
//! porting it (`phase-05-mediator-cross-sections.md`, Task 5.2, the same
//! rule Task 5.1 applied to the vector twin). It survives here as the
//! private [`sigma_xx_to_all`] the integrand calls, which is the only
//! consumer it ever had.
//!
//! # How this file was written
//!
//! Not by hand. `rules.md` calls the mediator cross sections Mathematica
//! dumps "where a dropped digit is silent", and two of them
//! ([`sigma_xpi_to_xpi`], [`sigma_xpi0_to_xpi0`]) are 90-line
//! expressions. Every expression below was emitted by a transliterator
//! that parses the `.pyx` and prints Rust, then checked point-for-point
//! against the live Cython before that Cython was deleted. The
//! transliterator and its checks are in the task note
//! (`projects/cython-to-rust/task-notes/phase-05/task-5.2-scalar-xs.md`);
//! what matters for reading this file is the three rules it encodes.
//!
//! **1. `x ** n` is a `pow` call, and that is what decides fusion.**
//! clang contracts `A + B` into an FMA when `A` is a multiply, else when
//! `B` is (`CGExprScalar.cpp`, `EmitFMulAdd`) — a decision taken on the C
//! tree Cython emits, where `x ** 2` is `pow(x, 2.0)`, a **call**. So
//! `-4 * mx**2 + e_cm**2` fuses (its left operand is the multiply
//! `-4.0 * pow(mx, 2.0)`) and the otherwise identical-looking
//! `ms**2 - e_cm**2` does not (both operands are calls). A leading unary
//! minus breaks it too: `-mpi0**2 + e_cm**2` is an `FNeg` feeding an
//! `FAdd`, with no multiply for either operand. [`sq`] below is
//! `pow(x, 2.0)`, which clang folds to `x·x`; `powf(3.0)`, `powf(4.0)`
//! and `powf(6.0)` stay libm calls and are **not** interchangeable with
//! repeated multiplication.
//!
//! **2. `np.log(4)` boxes half of [`sigma_xl_to_xl`] into Python.** A
//! single Python-level call at `:283` makes every operation between it
//! and the root a `PyObject` operation — `PyNumber_Multiply`,
//! `PyNumber_Add` — so clang never sees an expression to contract there.
//! The pure-C operands of those operations are still doubles and still
//! fuse *inside* themselves. That is why the `atan` half of that kernel
//! is full of `mul_add` and the `log` tail has almost none. It is the
//! same effect Phase 04 found in `_photon/_rho.pyx`, from a different
//! cause (untyped `cdef` locals there, one stray `np.` here).
//!
//! **3. One expression went through complex arithmetic.**
//! `__sigma_xx_to_s_to_ff` raises a kinematic factor to the power `1.5`,
//! so Cython compiled that whole expression in `double _Complex`:
//! `grep -c SoftComplexToDouble` on the generated C finds one call site,
//! and the shipped object calls `cpow` and compiler-rt's `___divdc3`.
//! [`crate::kernels::soft_complex`] holds both reproductions and the
//! evidence for them. Phase 05's handoff predicted this file had no such
//! expression; it has one, and without the shims
//! [`sigma_xx_to_s_to_ff`] misses bit-equality at 355 of 935 corpus
//! points on the electron block alone.
//!
//! # Constants
//!
//! The `.pyx` declares its own nine module-level `cdef double`s rather
//! than `include`-ing either shared header, so they live here rather than
//! in [`crate::constants::derived`] — which is scored against surviving
//! `.pyx` files and this file's is deleted by the same task that adds
//! this module (Phase 04's learnings §5, and Task 5.1's precedent). They
//! are transcribed verbatim under `rules.md` rule 4, which forbids
//! reconciling them with the shared tables here. `ALPHA_EM` is
//! `1/137.04`, a third value beside `crate::constants::pdg`'s and
//! `legacy`'s — the vector module's own copy has the same one.

use crate::kernels::soft_complex::{
    NonRealResult, complex_quotient_real_denominator, soft_complex_pow_1_5,
};
use crate::quad::{DEFAULT_EPSABS, DEFAULT_EPSREL, DEFAULT_LIMIT, QuadOpts, quad};
use crate::special::{bessel_k1, bessel_kn};

/// Higgs vacuum expectation value in MeV —
/// `_c_scalar_mediator_cross_sections.pyx:9`.
pub const VH: f64 = 246.22795e3;
/// Fine-structure constant — `:10`.
pub const ALPHA_EM: f64 = 1.0 / 137.04;
/// Electron mass in MeV — `:11`.
pub const ME: f64 = 0.510998928;
/// Muon mass in MeV — `:12`.
pub const MMU: f64 = 105.6583715;
/// Neutral pion mass in MeV — `:13`.
pub const MPI0: f64 = 134.9766;
/// Charged pion mass in MeV — `:14`.
pub const MPI: f64 = 139.57018;
/// Chiral-condensate parameter `B₀` in MeV — `:15`.
pub const B0: f64 = 2654.082197477761;
/// Up-quark mass in MeV — `:16`.
pub const MUQ: f64 = 2.3;
/// Down-quark mass in MeV — `:17`.
pub const MDQ: f64 = 4.8;

/// `π`, under the `.pyx`'s spelling of it.
const PI: f64 = std::f64::consts::PI;
/// `π³`, as clang folds `pow(M_PI, 3.0)`.
///
/// A literal because `powf` is not `const`, and deliberately **not** a
/// product of `PI`s, which need not be the same double.
/// `pi_cubed_matches_libm` re-derives it at run time.
const PI_3: f64 = 31.006276680299816;
/// `ln 4`, from the `np.log(4)` at `:283`.
///
/// The one place hazma's compiled layer calls back into NumPy from inside
/// a `cdef` function. It is a constant either way; naming it is Task
/// 5.2's exit criterion, and `log_constants_match_libm` pins the value.
const LN_4: f64 = 1.3862943611198906;
/// `ln 16`, from the two `log(16)` calls at `:283-284`, which clang folds
/// at compile time. See [`LN_4`].
const LN_16: f64 = 2.772588722239781;

/// The quadrature settings `thermal_cross_section` inherits from
/// `scipy.integrate.quad`'s defaults — the `.pyx` passes neither
/// `epsabs` nor `epsrel` (`:1411-1414`).
const THERMAL_EPSABS: f64 = DEFAULT_EPSABS;
/// See [`THERMAL_EPSABS`].
const THERMAL_EPSREL: f64 = DEFAULT_EPSREL;

/// `pow(x, 2.0)`, which is `x · x`.
///
/// Spelled as a function rather than inline so the emitted expressions
/// read like the `.pyx`'s `x ** 2`, and so a reader can see at a glance
/// that a squaring is **not** a fusable multiply in clang's eyes — see
/// the module docs, rule 1. The value is the same either way: `pow`'s
/// exponent-2 case is the correctly rounded product, and so is `x * x`.
#[inline]
fn sq(x: f64) -> f64 {
    x * x
}

/// `σ(x x̄ → S* → f f̄)` in MeV⁻², for a fermion of mass `mf`.
///
/// Zero below `e_cm = max(2 m_f, 2 m_x)`. The only kernel in this file
/// that Cython compiled in complex arithmetic — see the module docs,
/// rule 3 — so it returns a `Result` and its `**1.5` goes through
/// [`soft_complex_pow_1_5`] and its division through
/// [`complex_quotient_real_denominator`].
///
/// # Errors
///
/// [`NonRealResult`] if `__divdc3`'s imaginary part comes back non-zero,
/// which needs a vanishing denominator — `e_cm = 0` (below threshold and
/// unreachable) or a zero-width mediator exactly on resonance. The
/// parity corpus records no raise for this entry point, unlike its two
/// vector siblings.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xx_to_s_to_ff(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    width_s: f64,
    mf: f64,
) -> Result<f64, NonRealResult> {
    if e_cm < 2.0 * mf || e_cm < 2.0 * mx {
        return Ok(0.0);
    }

    let numerator = sq(gsff)
        * sq(gsxx)
        * sq(mf)
        * soft_complex_pow_1_5((-4.0_f64).mul_add(sq(mf), sq(e_cm)))
        * (-4.0_f64).mul_add(sq(mx), sq(e_cm)).sqrt();

    let denominator =
        16.0 * PI * sq(e_cm) * sq(VH) * sq(ms).mul_add(sq(width_s), sq(sq(ms) - sq(e_cm)));

    complex_quotient_real_denominator(numerator, denominator)
}

/// `σ(x x̄ → S* → γγ)` in MeV⁻².
///
/// Zero below `e_cm = 2 m_x`; the photons are massless, so this is the
/// one annihilation channel with a single threshold.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xx_to_s_to_gg(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
) -> f64 {
    if e_cm < 2.0 * mx {
        return 0.0;
    }

    sq(ALPHA_EM)
        * sq(gsFF)
        * sq(gsxx)
        * e_cm.powf(3.0)
        * (-4.0_f64).mul_add(sq(mx), sq(e_cm)).sqrt()
        / (128.0 * sq(lam) * PI_3 * sq(ms).mul_add(sq(width_s), sq(sq(ms) - sq(e_cm))))
}

/// `σ(x x̄ → S* → π⁰π⁰)` in MeV⁻².
///
/// Zero below `e_cm = max(2 m_π⁰, 2 m_x)`. Identical to
/// [`sigma_xx_to_s_to_pipi`] but for the pion mass and a factor 2 in the
/// denominator — identical particles in the final state.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xx_to_s_to_pi0pi0(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> f64 {
    if e_cm < 2.0 * MPI0 || e_cm < 2.0 * mx {
        return 0.0;
    }

    sq(gsxx)
        * ((-4.0_f64).mul_add(sq(MPI0), sq(e_cm)) * (-4.0_f64).mul_add(sq(mx), sq(e_cm))).sqrt()
        * sq(
            (162.0 * gsGG * lam.powf(3.0) * (2.0_f64).mul_add(sq(MPI0), -sq(e_cm))).mul_add(
                sq(VH),
                B0 * (MDQ + MUQ)
                    * (9.0_f64).mul_add(lam, 4.0 * gsGG * vs)
                    * (2.0 * gsGG * VH)
                        .mul_add(vs, (-3.0 * lam).mul_add(VH, 3.0 * gsff * lam * vs))
                    * (2.0 * gsGG * VH).mul_add(
                        (9.0_f64).mul_add(lam, -(4.0 * gsGG * vs)),
                        9.0 * gsff * lam * (3.0_f64).mul_add(lam, 4.0 * gsGG * vs),
                    ),
            ),
        )
        / (419904.0
            * lam.powf(6.0)
            * PI
            * sq(e_cm)
            * VH.powf(4.0)
            * sq((9.0_f64).mul_add(lam, 4.0 * gsGG * vs))
            * sq(ms).mul_add(sq(width_s), sq(sq(ms) - sq(e_cm))))
}

/// `σ(x x̄ → S* → π⁺π⁻)` in MeV⁻².
///
/// Zero below `e_cm = max(2 m_π, 2 m_x)`. See
/// [`sigma_xx_to_s_to_pi0pi0`].
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xx_to_s_to_pipi(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> f64 {
    if e_cm < 2.0 * MPI || e_cm < 2.0 * mx {
        return 0.0;
    }

    sq(gsxx)
        * ((-4.0_f64).mul_add(sq(MPI), sq(e_cm)) * (-4.0_f64).mul_add(sq(mx), sq(e_cm))).sqrt()
        * sq(
            (162.0 * gsGG * lam.powf(3.0) * (2.0_f64).mul_add(sq(MPI), -sq(e_cm))).mul_add(
                sq(VH),
                B0 * (MDQ + MUQ)
                    * (9.0_f64).mul_add(lam, 4.0 * gsGG * vs)
                    * (2.0 * gsGG * VH)
                        .mul_add(vs, (-3.0 * lam).mul_add(VH, 3.0 * gsff * lam * vs))
                    * (2.0 * gsGG * VH).mul_add(
                        (9.0_f64).mul_add(lam, -(4.0 * gsGG * vs)),
                        9.0 * gsff * lam * (3.0_f64).mul_add(lam, 4.0 * gsGG * vs),
                    ),
            ),
        )
        / (209952.0
            * lam.powf(6.0)
            * PI
            * sq(e_cm)
            * VH.powf(4.0)
            * sq((9.0_f64).mul_add(lam, 4.0 * gsGG * vs))
            * sq(ms).mul_add(sq(width_s), sq(sq(ms) - sq(e_cm))))
}

/// `σ(x x̄ → S S)` in MeV⁻², through the t and u channels.
///
/// Zero below `e_cm = max(2 m_s, 2 m_x)`. No s-channel propagator, so
/// none of the mediator's couplings to matter enter — only `gsxx`.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xx_to_ss(e_cm: f64, mx: f64, ms: f64, gsxx: f64) -> f64 {
    if e_cm < 2.0 * ms || e_cm < 2.0 * mx {
        return 0.0;
    }

    gsxx.powf(4.0)
        * (-(((-4.0_f64).mul_add(sq(ms), sq(e_cm)) * (-4.0_f64).mul_add(sq(mx), sq(e_cm))).sqrt()
            * (2.0 * sq(mx)).mul_add(
                (8.0_f64).mul_add(sq(mx), sq(e_cm)),
                (3.0_f64).mul_add(ms.powf(4.0), -(16.0 * sq(ms) * sq(mx))),
            )
            / sq(mx).mul_add(sq(e_cm), (-(4.0 * sq(ms))).mul_add(sq(mx), ms.powf(4.0))))
            + (-(4.0 * sq(ms))).mul_add(
                (4.0_f64).mul_add(sq(mx), sq(e_cm)),
                (16.0 * sq(mx)).mul_add(
                    sq(e_cm),
                    (6.0_f64).mul_add(ms.powf(4.0), -(32.0 * mx.powf(4.0))),
                ) + e_cm.powf(4.0),
            ) * (((-2.0_f64).mul_add(sq(ms), sq(e_cm))
                + ((-4.0_f64).mul_add(sq(ms), sq(e_cm)) * (-4.0_f64).mul_add(sq(mx), sq(e_cm)))
                    .sqrt())
                / ((-2.0_f64).mul_add(sq(ms), sq(e_cm))
                    - ((-4.0_f64).mul_add(sq(ms), sq(e_cm))
                        * (-4.0_f64).mul_add(sq(mx), sq(e_cm)))
                    .sqrt()))
            .ln()
                / (-2.0_f64).mul_add(sq(ms), sq(e_cm)))
        / (32.0 * PI * sq(e_cm) * (-4.0_f64).mul_add(sq(mx), sq(e_cm)))
}

/// `σ(S S → x x̄)` in MeV⁻², the crossing of [`sigma_xx_to_ss`].
///
/// Zero below `e_cm = max(2 m_x, 2 m_s)`. Written in the `.pyx` in terms
/// of the ratios `rs = m_s/e_cm` and `rx = m_x/e_cm` rather than the
/// masses, which is a different rounding from its uncrossed twin and is
/// transcribed as written.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_ss_to_xx(e_cm: f64, mx: f64, ms: f64, gsxx: f64) -> f64 {
    if e_cm < 2.0 * mx || e_cm < 2.0 * ms {
        return 0.0;
    }

    let rs = ms / e_cm;

    let rx = mx / e_cm;

    gsxx.powf(4.0)
        * (-2.0
            * ((-4.0_f64).mul_add(sq(rs), 1.0) * (-4.0_f64).mul_add(sq(rx), 1.0)).sqrt()
            * (16.0_f64).mul_add(
                rx.powf(4.0),
                (3.0_f64).mul_add(rs.powf(4.0), 2.0 * (-8.0_f64).mul_add(sq(rs), 1.0) * sq(rx)),
            )
            / (-(4.0 * sq(rs))).mul_add(sq(rx), rs.powf(4.0) + sq(rx))
            + (-(4.0 * sq(rs))).mul_add(
                (4.0_f64).mul_add(sq(rx), 1.0),
                (-32.0_f64).mul_add(
                    rx.powf(4.0),
                    (16.0_f64).mul_add(sq(rx), (6.0_f64).mul_add(rs.powf(4.0), 1.0)),
                ),
            ) * 2.0
                * (((-2.0_f64).mul_add(sq(rs), 1.0)
                    + ((-4.0_f64).mul_add(sq(rs), 1.0) * (-4.0_f64).mul_add(sq(rx), 1.0)).sqrt())
                .ln()
                    - ((-2.0_f64).mul_add(sq(rs), 1.0)
                        - ((-4.0_f64).mul_add(sq(rs), 1.0) * (-4.0_f64).mul_add(sq(rx), 1.0))
                            .sqrt())
                    .ln())
                / (-2.0_f64).mul_add(sq(rs), 1.0))
        / (8.0 * sq(e_cm) * PI * (-4.0_f64).mul_add(sq(rs), 1.0))
}

/// `σ(x l → x l)` in MeV⁻², summed over lepton charges.
///
/// Zero below `e_cm = m_x + m_l`. The kernel whose tail Cython evaluated
/// through `PyObject` arithmetic because of one `np.log(4)` — module
/// docs, rule 2 — and one of the four this file holds whose `atan`
/// difference cancels catastrophically near `e_cm = 2 m_x`
/// (`docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`).
/// The port reproduces that, under `rules.md` rule 1.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xl_to_xl(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    width_s: f64,
    ml: f64,
) -> f64 {
    if e_cm < mx + ml {
        return 0.0;
    }

    let s = sq(e_cm);

    2.0 * (sq(gsff * ml)
        * sq(gsxx)
        * ((-4.0 * sq(ml))
            .mul_add(
                (-4.0_f64).mul_add(sq(mx), sq(ms)),
                sq(ms) * ((-4.0_f64).mul_add(sq(mx), sq(ms)) - sq(width_s)),
            )
            .mul_add(
                (ms / width_s).atan(),
                (4.0 * sq(ml)).mul_add(
                    (-4.0_f64).mul_add(sq(mx), sq(ms)),
                    sq(ms) * ((4.0_f64).mul_add(sq(mx), -sq(ms)) + sq(width_s)),
                ) * (((-4.0_f64).mul_add(sq(mx), sq(ms)) + s) / (ms * width_s)).atan(),
            )
            + ms * width_s
                * ((4.0_f64).mul_add(sq(mx), -s) + sq(ms) * LN_4
                    - sq(ml) * LN_16
                    - sq(mx) * LN_16
                    + (2.0_f64).mul_add(sq(mx), (2.0_f64).mul_add(sq(ml), -sq(ms)))
                        * (4.0 * sq(ms) * (sq(ms) + sq(width_s))).ln()
                    + (-2.0_f64).mul_add(sq(mx), (-2.0_f64).mul_add(sq(ml), sq(ms)))
                        * sq(ms)
                            .mul_add(
                                (-8.0_f64).mul_add(sq(mx), 2.0 * s) + sq(width_s),
                                ms.powf(4.0) + sq((-4.0_f64).mul_add(sq(mx), s)),
                            )
                            .ln()))
        / (32.0 * ms * PI * sq(e_cm) * (4.0_f64).mul_add(sq(mx), -s) * width_s))
}

/// `σ(x π → x π)` in MeV⁻², summed over pion charges.
///
/// Zero below `e_cm = m_x + m_π`. Ninety lines in the `.pyx`; the four
/// `let` bindings below are common subexpressions the transliterator
/// hoisted, which changes no arithmetic — `coupling_combination` is the
/// factor the source repeats eight times. Cancels near `e_cm = 2 m_x`
/// like [`sigma_xl_to_xl`].
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xpi_to_xpi(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> f64 {
    if e_cm < mx + MPI {
        return 0.0;
    }

    let atan_prefactor = gsff.mul_add(
        (-81.0 * lam.powf(3.0)).mul_add(VH, 48.0 * sq(gsGG) * lam * VH * sq(vs)),
        (27.0 * sq(gsff) * sq(lam) * vs).mul_add(
            (3.0_f64).mul_add(lam, 4.0 * gsGG * vs),
            -(2.0
                * gsGG
                * sq(VH)
                * (8.0 * sq(gsGG)).mul_add(
                    sq(vs),
                    (27.0_f64).mul_add(sq(lam), -(30.0 * gsGG * lam * vs)),
                )),
        ),
    );

    let log_prefactor = sq(atan_prefactor);

    let coupling_combination_sq = (26244.0 * sq(gsGG) * lam.powf(6.0) * VH.powf(4.0)).mul_add(
        sq(ms).mul_add(
            (3.0_f64).mul_add(sq(ms), -(8.0 * sq(mx))) - sq(width_s),
            (4.0_f64).mul_add(
                MPI.powf(4.0),
                -(8.0 * sq(MPI) * (-2.0_f64).mul_add(sq(mx), sq(ms))),
            ),
        ),
        (648.0
            * B0
            * gsGG
            * lam.powf(3.0)
            * (MDQ + MUQ)
            * (2.0_f64).mul_add(sq(mx), sq(MPI) - sq(ms))
            * sq(VH)
            * (9.0_f64).mul_add(lam, 4.0 * gsGG * vs))
        .mul_add(
            atan_prefactor,
            sq(B0) * sq(MDQ + MUQ) * sq((9.0_f64).mul_add(lam, 4.0 * gsGG * vs)) * log_prefactor,
        ),
    );

    let coupling_combination = 2.0
        * (26244.0 * sq(gsGG) * lam.powf(6.0) * VH.powf(4.0)).mul_add(
            (-ms.powf(4.0)).mul_add(
                (4.0_f64).mul_add(sq(mx), 3.0 * sq(width_s)),
                (-(4.0 * sq(MPI) * sq(ms))).mul_add(
                    (-4.0_f64).mul_add(sq(mx), sq(ms)) - sq(width_s),
                    (4.0 * sq(ms) * sq(mx)).mul_add(
                        sq(width_s),
                        (4.0 * MPI.powf(4.0))
                            .mul_add((-4.0_f64).mul_add(sq(mx), sq(ms)), ms.powf(6.0)),
                    ),
                ),
            ),
            (sq(B0)
                * sq(MDQ + MUQ)
                * (-4.0_f64).mul_add(sq(mx), sq(ms))
                * sq((9.0_f64).mul_add(lam, 4.0 * gsGG * vs)))
            .mul_add(
                log_prefactor,
                324.0
                    * B0
                    * gsGG
                    * lam.powf(3.0)
                    * (MDQ + MUQ)
                    * sq(VH)
                    * (9.0_f64).mul_add(lam, 4.0 * gsGG * vs)
                    * atan_prefactor
                    * (2.0 * sq(MPI)).mul_add(
                        (-4.0_f64).mul_add(sq(mx), sq(ms)),
                        sq(ms) * ((4.0_f64).mul_add(sq(mx), -sq(ms)) + sq(width_s)),
                    ),
            ),
        );

    2.0 * (sq(gsxx)
        * (ms * width_s).mul_add(
            coupling_combination_sq.mul_add(
                sq(ms)
                    .mul_add(
                        (-8.0_f64).mul_add(sq(mx), 2.0 * sq(e_cm)) + sq(width_s),
                        ms.powf(4.0) + sq((-4.0_f64).mul_add(sq(mx), sq(e_cm))),
                    )
                    .ln(),
                (-324.0 * gsGG * lam.powf(3.0) * (4.0_f64).mul_add(sq(mx), -sq(e_cm)) * sq(VH))
                    .mul_add(
                        (81.0
                            * gsGG
                            * lam.powf(3.0)
                            * ((4.0_f64)
                                .mul_add(sq(mx), (8.0_f64).mul_add(sq(MPI), -(4.0 * sq(ms))))
                                + sq(e_cm)))
                        .mul_add(
                            sq(VH),
                            2.0 * B0
                                * (MDQ + MUQ)
                                * (9.0_f64).mul_add(lam, 4.0 * gsGG * vs)
                                * atan_prefactor,
                        ),
                        -(coupling_combination_sq * (sq(ms) * (sq(ms) + sq(width_s))).ln()),
                    ),
            ),
            coupling_combination.mul_add(
                (ms / width_s).atan(),
                -(coupling_combination
                    * (((-4.0_f64).mul_add(sq(mx), sq(ms)) + sq(e_cm)) / (ms * width_s)).atan()),
            ),
        )
        / (419904.0
            * lam.powf(6.0)
            * ms
            * PI
            * sq(e_cm)
            * (-4.0_f64).mul_add(sq(mx), sq(e_cm))
            * VH.powf(4.0)
            * sq((9.0_f64).mul_add(lam, 4.0 * gsGG * vs))
            * width_s))
}

/// `σ(x π⁰ → x π⁰)` in MeV⁻².
///
/// Zero below `e_cm = m_x + m_π⁰`. [`sigma_xpi_to_xpi`] with the neutral
/// pion mass and without that kernel's leading factor 2 (no charge sum).
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xpi0_to_xpi0(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> f64 {
    if e_cm < mx + MPI0 {
        return 0.0;
    }

    let atan_prefactor = gsff.mul_add(
        (-81.0 * lam.powf(3.0)).mul_add(VH, 48.0 * sq(gsGG) * lam * VH * sq(vs)),
        (27.0 * sq(gsff) * sq(lam) * vs).mul_add(
            (3.0_f64).mul_add(lam, 4.0 * gsGG * vs),
            -(2.0
                * gsGG
                * sq(VH)
                * (8.0 * sq(gsGG)).mul_add(
                    sq(vs),
                    (27.0_f64).mul_add(sq(lam), -(30.0 * gsGG * lam * vs)),
                )),
        ),
    );

    let log_prefactor = sq(atan_prefactor);

    let coupling_combination_sq = (26244.0 * sq(gsGG) * lam.powf(6.0) * VH.powf(4.0)).mul_add(
        sq(ms).mul_add(
            (3.0_f64).mul_add(sq(ms), -(8.0 * sq(mx))) - sq(width_s),
            (4.0_f64).mul_add(
                MPI0.powf(4.0),
                -(8.0 * sq(MPI0) * (-2.0_f64).mul_add(sq(mx), sq(ms))),
            ),
        ),
        (648.0
            * B0
            * gsGG
            * lam.powf(3.0)
            * (MDQ + MUQ)
            * (2.0_f64).mul_add(sq(mx), sq(MPI0) - sq(ms))
            * sq(VH)
            * (9.0_f64).mul_add(lam, 4.0 * gsGG * vs))
        .mul_add(
            atan_prefactor,
            sq(B0) * sq(MDQ + MUQ) * sq((9.0_f64).mul_add(lam, 4.0 * gsGG * vs)) * log_prefactor,
        ),
    );

    let coupling_combination = 2.0
        * (26244.0 * sq(gsGG) * lam.powf(6.0) * VH.powf(4.0)).mul_add(
            (-ms.powf(4.0)).mul_add(
                (4.0_f64).mul_add(sq(mx), 3.0 * sq(width_s)),
                (-(4.0 * sq(MPI0) * sq(ms))).mul_add(
                    (-4.0_f64).mul_add(sq(mx), sq(ms)) - sq(width_s),
                    (4.0 * sq(ms) * sq(mx)).mul_add(
                        sq(width_s),
                        (4.0 * MPI0.powf(4.0))
                            .mul_add((-4.0_f64).mul_add(sq(mx), sq(ms)), ms.powf(6.0)),
                    ),
                ),
            ),
            (sq(B0)
                * sq(MDQ + MUQ)
                * (-4.0_f64).mul_add(sq(mx), sq(ms))
                * sq((9.0_f64).mul_add(lam, 4.0 * gsGG * vs)))
            .mul_add(
                log_prefactor,
                324.0
                    * B0
                    * gsGG
                    * lam.powf(3.0)
                    * (MDQ + MUQ)
                    * sq(VH)
                    * (9.0_f64).mul_add(lam, 4.0 * gsGG * vs)
                    * atan_prefactor
                    * (2.0 * sq(MPI0)).mul_add(
                        (-4.0_f64).mul_add(sq(mx), sq(ms)),
                        sq(ms) * ((4.0_f64).mul_add(sq(mx), -sq(ms)) + sq(width_s)),
                    ),
            ),
        );

    sq(gsxx)
        * (ms * width_s).mul_add(
            coupling_combination_sq.mul_add(
                sq(ms)
                    .mul_add(
                        (-8.0_f64).mul_add(sq(mx), 2.0 * sq(e_cm)) + sq(width_s),
                        ms.powf(4.0) + sq((-4.0_f64).mul_add(sq(mx), sq(e_cm))),
                    )
                    .ln(),
                (-324.0 * gsGG * lam.powf(3.0) * (4.0_f64).mul_add(sq(mx), -sq(e_cm)) * sq(VH))
                    .mul_add(
                        (81.0
                            * gsGG
                            * lam.powf(3.0)
                            * ((4.0_f64)
                                .mul_add(sq(mx), (8.0_f64).mul_add(sq(MPI0), -(4.0 * sq(ms))))
                                + sq(e_cm)))
                        .mul_add(
                            sq(VH),
                            2.0 * B0
                                * (MDQ + MUQ)
                                * (9.0_f64).mul_add(lam, 4.0 * gsGG * vs)
                                * atan_prefactor,
                        ),
                        -(coupling_combination_sq * (sq(ms) * (sq(ms) + sq(width_s))).ln()),
                    ),
            ),
            coupling_combination.mul_add(
                (ms / width_s).atan(),
                -(coupling_combination
                    * (((-4.0_f64).mul_add(sq(mx), sq(ms)) + sq(e_cm)) / (ms * width_s)).atan()),
            ),
        )
        / (419904.0
            * lam.powf(6.0)
            * ms
            * PI
            * sq(e_cm)
            * (-4.0_f64).mul_add(sq(mx), sq(e_cm))
            * VH.powf(4.0)
            * sq((9.0_f64).mul_add(lam, 4.0 * gsGG * vs))
            * width_s)
}

/// `σ(x γ → x γ)` in MeV⁻².
///
/// Zero below `e_cm = m_x`, **and** exactly at `e_cm = 2 m_x`, where the
/// `.pyx` short-circuits with the comment "for e_cm = 2mx there is
/// complete destructive interference". That guard is transcribed as
/// written; whether it is a physics statement or a workaround for the
/// 0/0 the cancellation produces is the open question in
/// `docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xg_to_xg(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
) -> f64 {
    // "for e_cm = 2mx there is complete destructive interference"
    if e_cm < mx || e_cm == 2.0 * mx {
        return 0.0;
    }

    let s = sq(e_cm);

    sq(ALPHA_EM)
        * sq(gsFF)
        * sq(gsxx)
        * (-width_s).mul_add(
            (sq(ms) * ((3.0_f64).mul_add(sq(ms), -(8.0 * sq(mx))) - sq(width_s))).mul_add(
                sq(ms)
                    .mul_add(
                        (-8.0_f64).mul_add(sq(mx), 2.0 * s) + sq(width_s),
                        ms.powf(4.0) + sq((-4.0_f64).mul_add(sq(mx), s)),
                    )
                    .ln(),
                ((4.0_f64).mul_add(sq(ms), -(4.0 * sq(mx))) - s).mul_add(
                    (4.0_f64).mul_add(sq(mx), -s),
                    sq(ms)
                        * ((-3.0_f64).mul_add(sq(ms), 8.0 * sq(mx)) + sq(width_s))
                        * (sq(ms) * (sq(ms) + sq(width_s))).ln(),
                ),
            ),
            (-2.0
                * (-ms.powf(3.0)).mul_add(
                    (4.0_f64).mul_add(sq(mx), 3.0 * sq(width_s)),
                    (4.0 * ms * sq(mx)).mul_add(sq(width_s), ms.powf(5.0)),
                ))
            .mul_add(
                (ms / width_s).atan(),
                2.0 * (-ms.powf(3.0)).mul_add(
                    (4.0_f64).mul_add(sq(mx), 3.0 * sq(width_s)),
                    (4.0 * ms * sq(mx)).mul_add(sq(width_s), ms.powf(5.0)),
                ) * (((-4.0_f64).mul_add(sq(mx), sq(ms)) + s) / (ms * width_s)).atan(),
            ),
        )
        / (128.0 * sq(lam) * PI_3 * (4.0_f64).mul_add(sq(mx), -s) * s * width_s)
}

/// `σ(x S → x S)` in MeV⁻².
///
/// Zero below `e_cm = m_x + m_s`. Carries its own closed form at exactly
/// `e_cm = 2 m_x`, where the general expression is 0/0 — the one place in
/// this file where the `.pyx` supplies the limit rather than the
/// singularity. The corpus samples both branches: `2 m_x` is a grid
/// anchor.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn sigma_xs_to_xs(e_cm: f64, mx: f64, ms: f64, gsxx: f64) -> f64 {
    if e_cm < mx + ms {
        return 0.0;
    }

    if e_cm == 2.0 * mx {
        return gsxx.powf(4.0)
            * sq((9.0_f64).mul_add(
                mx.powf(4.0),
                (-(10.0 * sq(ms))).mul_add(sq(mx), ms.powf(4.0)),
            ))
            / (36.0
                * mx.powf(4.0)
                * sq((2.0_f64).mul_add(sq(ms), -(3.0 * sq(mx))))
                * PI
                * sq(e_cm));
    }

    let s = sq(e_cm);

    gsxx.powf(4.0)
        * ((10.0_f64).mul_add(
            sq(s),
            (50.0 * sq(mx)).mul_add(
                s,
                (-(8.0 * sq(ms))).mul_add(
                    s,
                    (4.0_f64).mul_add(
                        mx.powf(4.0),
                        (4.0_f64).mul_add(ms.powf(4.0), -(24.0 * sq(ms) * sq(mx))),
                    ) + 4.0 * sq((-4.0_f64).mul_add(sq(mx), sq(ms))) * sq(sq(mx) - s)
                        / ((2.0_f64).mul_add(sq(ms), -(3.0 * sq(mx)))
                            * (4.0_f64).mul_add(sq(mx), -s))
                        - 4.0 * sq((-4.0_f64).mul_add(sq(mx), sq(ms))) * sq(sq(mx) - s)
                            / (((2.0_f64).mul_add(sq(ms), sq(mx)) - s)
                                * (4.0_f64).mul_add(sq(mx), -s)),
                ),
            ),
        ) + 4.0
            * (sq(mx) - s)
            * ((18.0 * sq(mx)).mul_add(
                s,
                (15.0_f64).mul_add(
                    mx.powf(4.0),
                    (-2.0_f64).mul_add(ms.powf(4.0), -(16.0 * sq(ms) * sq(mx))),
                ),
            ) - sq(s))
            * ((2.0_f64).mul_add(sq(ms), -(3.0 * sq(mx)))
                / ((2.0_f64).mul_add(sq(ms), sq(mx)) - s))
                .abs()
                .ln()
            / (4.0_f64).mul_add(sq(mx), -s))
        / (64.0 * PI * sq(e_cm) * sq(sq(mx) - s))
}

/// The sum of all six annihilation channels, `σ_all(e_cm)` in MeV⁻².
///
/// The Cython exported this as a public `def` that nothing imported, so
/// Task 5.2 drops it from the public surface and keeps it here as the
/// [`thermal_cross_section`] integrand's own helper — its only real
/// consumer. Summation order is the `.pyx`'s (`:225-243`):
/// `e + μ + γγ + π⁰π⁰ + π⁺π⁻ + SS`.
///
/// # Errors
///
/// As [`sigma_xx_to_s_to_ff`], the one summand that can fail.
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xx_to_all(
    e_cm: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> Result<f64, NonRealResult> {
    let sig_e = sigma_xx_to_s_to_ff(e_cm, mx, ms, gsxx, gsff, width_s, ME)?;
    let sig_mu = sigma_xx_to_s_to_ff(e_cm, mx, ms, gsxx, gsff, width_s, MMU)?;
    let sig_g = sigma_xx_to_s_to_gg(e_cm, mx, ms, gsxx, gsFF, lam, width_s);
    let sig_pi0 = sigma_xx_to_s_to_pi0pi0(e_cm, mx, ms, gsxx, gsff, gsGG, lam, width_s, vs);
    let sig_pi = sigma_xx_to_s_to_pipi(e_cm, mx, ms, gsxx, gsff, gsGG, lam, width_s, vs);
    let sig_s = sigma_xx_to_ss(e_cm, mx, ms, gsxx);

    Ok(sig_e + sig_mu + sig_g + sig_pi0 + sig_pi + sig_s)
}

/// The thermally averaged `⟨σv⟩` in MeV⁻², at `x = m_x / T`.
///
/// Above `x = 300` this returns exactly `0.0` — "if x is really large,
/// we will get divide by zero errors" (`:1400-1402`). The **vector**
/// model clips `x` to 300 and keeps returning the value there instead,
/// and the corpus pins both (`test/parity/cases.py`'s `_thermal_blocks`
/// says so at length); unifying them would move published numbers.
///
/// The upper limit is `max(50/x, 100)`, where the vector's floor is 150.
/// The break points are the endpoint, the mediator resonance `z = m_s/m_x`
/// and the `SS` threshold `z = 2 m_s/m_x`.
///
/// Neither `epsabs` nor `epsrel` is passed, so this inherits scipy's
/// defaults, which the integrand's ~1e-27 scale satisfies on the first
/// Kronrod pass — the quadrature does not converge, and that is
/// reproduced rather than fixed
/// (`docs/followups/todo/thermal-cross-section-quadrature-never-converges.md`).
///
/// # Errors
///
/// As [`sigma_xx_to_s_to_ff`], if the integrand's `σ_all` fails.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn thermal_cross_section(
    x: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> Result<f64, NonRealResult> {
    // "If x is really large, we will get divide by zero errors."
    if x > 300.0 {
        return Ok(0.0);
    }

    let two_k2 = 2.0 * bessel_kn(2, x);
    let prefactor = x / sq(two_k2);

    // `max(50.0 / x, 100.0)`, in Python's evaluation order.
    let floor = 50.0 / x;
    let upper = if 100.0 > floor { 100.0 } else { floor };

    // "points at which integrand may have trouble are: 1. endpoint;
    // 2. when ss final state is accessible => z = 2 ms / mx;
    // 3. when we hit mediator resonance => z = ms / mx"
    let ratio = ms / mx;
    let points = [2.0, ratio, 2.0 * ratio];

    let mut nonreal = false;
    let mut integrand = |z: f64| {
        match sigma_xx_to_all(mx * z, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs) {
            Ok(sigma) => ((sigma * sq(z)) * (sq(z) - 4.0)) * bessel_k1(x * z),
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
        // statement about the options, never about the integrand.
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
        ALPHA_EM, B0, LN_4, LN_16, MDQ, ME, MMU, MPI, MPI0, MUQ, PI_3, VH, sigma_ss_to_xx,
        sigma_xg_to_xg, sigma_xl_to_xl, sigma_xpi_to_xpi, sigma_xpi0_to_xpi0, sigma_xs_to_xs,
        sigma_xx_to_all, sigma_xx_to_s_to_ff, sigma_xx_to_s_to_gg, sigma_xx_to_s_to_pi0pi0,
        sigma_xx_to_s_to_pipi, sigma_xx_to_ss, sq, thermal_cross_section,
    };

    /// A representative model point: the parity corpus's `open_resonance`
    /// `HiggsPortal(mx=100, ms=300, gsxx=1, stheta=1e-1)`, with the derived
    /// couplings rounded to values a reader can check by eye. Nothing here
    /// depends on them being that model's exact numbers — the corpus is
    /// where the port is compared against the Cython.
    const MX: f64 = 100.0;
    const MS: f64 = 300.0;
    const GSXX: f64 = 1.0;
    const GSFF: f64 = 0.1;
    const GSGG: f64 = 0.1;
    const GSFF_PHOTON: f64 = 0.1;
    const LAM: f64 = 1.0e5;
    const WIDTH_S: f64 = 2.5;
    const VS: f64 = 1.0;

    // -- The constants clang folded ---------------------------------------

    /// `PI_3` is `pow(M_PI, 3.0)`'s answer, not a product of `PI`s.
    ///
    /// The reason it is a literal: `powf` is not `const`. The reason it is
    /// not spelled as a product is measured rather than assumed — on this
    /// platform the left-associated `PI*PI*PI` is a different double.
    #[test]
    fn pi_cubed_matches_libm() {
        assert_eq!(PI_3, std::f64::consts::PI.powf(3.0));
    }

    /// `LN_4` and `LN_16` are libm's `log`, from the two spellings the
    /// `.pyx` used: a NumPy call and a C call clang folded.
    ///
    /// `ln 16` is deliberately **not** derived as `2 * LN_4`: the two need
    /// not be the same double, and it is the folded `log(16.0)` the
    /// Cython computed.
    #[test]
    fn log_constants_match_libm() {
        assert_eq!(LN_4, 4.0_f64.ln());
        assert_eq!(LN_16, 16.0_f64.ln());
    }

    /// The nine module constants are the `.pyx`'s own, transcribed
    /// verbatim (`rules.md` rule 4).
    ///
    /// `ALPHA_EM` is the one worth stating: `1/137.04` is a third value
    /// beside `crate::constants::pdg`'s `1/137.035999084` and `legacy`'s
    /// `1/137`, and consolidating the tables is a declared follow-up, not
    /// a side effect of this port.
    #[test]
    fn the_module_constants_are_the_pyx_values() {
        assert_eq!(VH, 246.22795e3);
        assert_eq!(ALPHA_EM, 1.0 / 137.04);
        assert_eq!(ME, 0.510998928);
        assert_eq!(MMU, 105.6583715);
        assert_eq!(MPI0, 134.9766);
        assert_eq!(MPI, 139.57018);
        assert_eq!(B0, 2654.082197477761);
        assert_eq!(MUQ, 2.3);
        assert_eq!(MDQ, 4.8);
        assert_ne!(ALPHA_EM, crate::constants::pdg::ALPHA_EM);
    }

    /// `sq` is `pow(x, 2.0)`, which is the correctly rounded product.
    #[test]
    fn squaring_matches_libm_pow() {
        for x in [1e-300, 1e-8, 0.5, 1.0, 3.7, 1e8, 1e150] {
            assert_eq!(sq(x), x.powf(2.0));
        }
    }

    // -- Thresholds -------------------------------------------------------

    /// Every kernel returns exactly `0.0` below its own threshold, and
    /// something non-zero just above it.
    ///
    /// `0.0` exactly, not "small": the corpus compares the sub-threshold
    /// region with `atol = 0`, so a port that returned 1e-300 there would
    /// fail (`test/parity/tolerances.py`, "`atol` is 0.0 everywhere").
    #[test]
    fn each_channel_opens_at_its_own_threshold() {
        let below = |e: f64| e * (1.0 - 1e-12);
        let above = |e: f64| e * (1.0 + 1e-12);

        // xx -> S* -> f fbar: max(2 mf, 2 mx), and with mx = 100 the two
        // leptons land on opposite sides of that max -- 2 me is far below
        // 2 mx while 2 mmu (211.3 MeV) is above it. Both are checked, so
        // the test would fail if the kernel used only one of the two
        // thresholds.
        let ff = |e: f64, mf: f64| sigma_xx_to_s_to_ff(e, MX, MS, GSXX, GSFF, WIDTH_S, mf).unwrap();
        assert_eq!(ff(below(2.0 * MX), ME), 0.0);
        assert!(ff(above(2.0 * MX), ME) > 0.0);
        assert_eq!(ff(below(2.0 * MMU), MMU), 0.0);
        assert!(ff(above(2.0 * MMU), MMU) > 0.0);
        // ... and the muon channel is still shut between the two.
        assert_eq!(ff(above(2.0 * MX), MMU), 0.0);

        let gg = |e: f64| sigma_xx_to_s_to_gg(e, MX, MS, GSXX, GSFF_PHOTON, LAM, WIDTH_S);
        assert_eq!(gg(below(2.0 * MX)), 0.0);
        assert!(gg(above(2.0 * MX)) > 0.0);

        // Both pion thresholds sit above 2 mx here (269.95 and 279.14
        // MeV against 200), so each channel is still shut where the dark
        // matter one opens.
        let pi0 = |e: f64| sigma_xx_to_s_to_pi0pi0(e, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS);
        assert_eq!(pi0(above(2.0 * MX)), 0.0);
        assert_eq!(pi0(below(2.0 * MPI0)), 0.0);
        assert!(pi0(above(2.0 * MPI0)) > 0.0);

        let pipi = |e: f64| sigma_xx_to_s_to_pipi(e, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS);
        assert_eq!(pipi(above(2.0 * MX)), 0.0);
        assert_eq!(pipi(below(2.0 * MPI)), 0.0);
        assert!(pipi(above(2.0 * MPI)) > 0.0);

        // xx -> SS and its crossing: max(2 ms, 2 mx), and ms > mx here.
        let ss = |e: f64| sigma_xx_to_ss(e, MX, MS, GSXX);
        assert_eq!(ss(below(2.0 * MS)), 0.0);
        assert!(ss(above(2.0 * MS)) > 0.0);
        let sx = |e: f64| sigma_ss_to_xx(e, MX, MS, GSXX);
        assert_eq!(sx(below(2.0 * MS)), 0.0);
        assert!(sx(above(2.0 * MS)) > 0.0);

        // Elastic scattering: mx + m_target.
        let xl = |e: f64| sigma_xl_to_xl(e, MX, MS, GSXX, GSFF, WIDTH_S, MMU);
        assert_eq!(xl(below(MX + MMU)), 0.0);
        assert_ne!(xl(above(MX + MMU)), 0.0);

        let xpi = |e: f64| sigma_xpi_to_xpi(e, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS);
        assert_eq!(xpi(below(MX + MPI)), 0.0);
        assert_ne!(xpi(above(MX + MPI)), 0.0);

        let xpi0 = |e: f64| sigma_xpi0_to_xpi0(e, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS);
        assert_eq!(xpi0(below(MX + MPI0)), 0.0);
        assert_ne!(xpi0(above(MX + MPI0)), 0.0);

        // The photon is massless, so this one opens at `mx` alone.
        let xg = |e: f64| sigma_xg_to_xg(e, MX, MS, GSXX, GSFF_PHOTON, LAM, WIDTH_S);
        assert_eq!(xg(below(MX)), 0.0);
        assert_ne!(xg(above(MX)), 0.0);

        let xs = |e: f64| sigma_xs_to_xs(e, MX, MS, GSXX);
        assert_eq!(xs(below(MX + MS)), 0.0);
        assert_ne!(xs(above(MX + MS)), 0.0);
    }

    /// `sigma_xg_to_xg` is exactly `0.0` at `e_cm = 2 mx`, and non-zero on
    /// either side of it.
    ///
    /// The `.pyx` short-circuits there — "complete destructive
    /// interference" — and the guard is `==`, so it catches exactly one
    /// double. Whether that is physics or a workaround for the 0/0 the
    /// `atan` cancellation produces is
    /// `docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`'s
    /// question; reproducing it is `rules.md` rule 1's answer.
    #[test]
    fn the_photon_elastic_cross_section_is_zero_exactly_at_two_mx() {
        let xg = |e: f64| sigma_xg_to_xg(e, MX, MS, GSXX, GSFF_PHOTON, LAM, WIDTH_S);
        assert_eq!(xg(2.0 * MX), 0.0);
        assert_ne!(xg(2.0 * MX * (1.0 - 1e-13)), 0.0);
        assert_ne!(xg(2.0 * MX * (1.0 + 1e-13)), 0.0);
    }

    /// `sigma_xs_to_xs`'s `e_cm = 2 mx` branch is the finite limit of the
    /// general expression, not a hole in it.
    ///
    /// Approaching `2 mx` from either side, the general form converges to
    /// the special-cased value **linearly**: measured on this platform
    /// the relative gap is 1.96e-5, 1.96e-6 and 1.96e-7 at relative
    /// offsets of 1e-5, 1e-6 and 1e-7, so the assertion below is that
    /// gap < 3x offset. Convergence stops there — at offsets of 1e-9 and
    /// below the gap *grows* again, to 4e-7 at 1e-10 and 1.9e-6 at
    /// 1e-11, because the general expression's `atan` difference has by
    /// then cancelled away its significant bits
    /// (`docs/followups/todo/scalar-elastic-cross-sections-cancel-in-atan-difference.md`).
    /// The convergent decades are what pin the limit; the floor is
    /// recorded here rather than asserted, since it is the defect this
    /// port reproduces rather than behavior to lock in.
    #[test]
    fn the_mediator_elastic_limit_is_the_general_expressions_limit() {
        // `mx + ms` must be below `2 mx`, so this needs `ms < mx`: the
        // corpus's `closed_resonance` point (mx = 300, ms = 200).
        let (mx, ms) = (300.0, 200.0);
        let at = sigma_xs_to_xs(2.0 * mx, mx, ms, GSXX);
        assert!(at > 0.0);
        for offset in [-1e-5, 1e-5, -1e-6, 1e-6, -1e-7, 1e-7] {
            let near = sigma_xs_to_xs(2.0 * mx * (1.0 + offset), mx, ms, GSXX);
            let gap = (near / at - 1.0).abs();
            assert!(gap < 3.0 * offset.abs(), "at {offset}: gap {gap}");
        }
    }

    // -- Statements the Cython never made ---------------------------------

    /// `sigma_xx_to_all` is the sum of the six annihilation channels.
    ///
    /// Not a tautology of the implementation: it is asserted against the
    /// six kernels called independently, so a summand dropped from the
    /// private helper — the failure mode that would silently bias every
    /// thermal average — fails here.
    #[test]
    fn the_total_is_the_sum_of_the_open_channels() {
        let e_cm = 700.0;
        let total = sigma_xx_to_all(
            e_cm,
            MX,
            MS,
            GSXX,
            GSFF,
            GSGG,
            GSFF_PHOTON,
            LAM,
            WIDTH_S,
            VS,
        )
        .unwrap();
        let parts = sigma_xx_to_s_to_ff(e_cm, MX, MS, GSXX, GSFF, WIDTH_S, ME).unwrap()
            + sigma_xx_to_s_to_ff(e_cm, MX, MS, GSXX, GSFF, WIDTH_S, MMU).unwrap()
            + sigma_xx_to_s_to_gg(e_cm, MX, MS, GSXX, GSFF_PHOTON, LAM, WIDTH_S)
            + sigma_xx_to_s_to_pi0pi0(e_cm, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS)
            + sigma_xx_to_s_to_pipi(e_cm, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS)
            + sigma_xx_to_ss(e_cm, MX, MS, GSXX);
        assert_eq!(total.to_bits(), parts.to_bits());
        assert!(total > 0.0);
    }

    /// The `f f̄` channel's high-energy limit is `σ s → gsff² gsxx² mf²/
    /// (16 π vh²)`.
    ///
    /// Far above every mass and far off resonance the two square roots
    /// and the propagator all go to their asymptotic forms and the whole
    /// expression collapses to a constant times `1/s`. Derived from the
    /// closed form rather than measured from it — an arithmetic slip in
    /// the prefactor moves this and nothing in the corpus would say which
    /// factor moved.
    #[test]
    fn the_fermion_channel_has_the_expected_high_energy_limit() {
        let expected = sq(GSFF) * sq(GSXX) * sq(MMU) / (16.0 * std::f64::consts::PI * sq(VH));
        let e_cm = 1e9;
        let got = sigma_xx_to_s_to_ff(e_cm, MX, MS, GSXX, GSFF, WIDTH_S, MMU).unwrap()
            * sq(e_cm)
            * sq(e_cm)
            / sq(e_cm);
        assert!(
            (got / expected - 1.0).abs() < 1e-8,
            "sigma*s -> {got}, expected {expected}"
        );
    }

    /// The `π⁰π⁰` and `π⁺π⁻` channels differ by exactly the factor 2 for
    /// identical final-state particles, once the pion masses agree.
    ///
    /// Evaluated with `MPI0` substituted for `MPI` is not something the
    /// signature allows, so the check runs the other way: at an energy far
    /// above both thresholds the phase-space factors agree to the mass
    /// splitting, and the ratio is 2 to that accuracy. That is enough to
    /// catch the denominators being swapped (419904 vs 209952), which is
    /// the transcription error this pair invites.
    #[test]
    fn the_two_pion_channels_differ_by_the_identical_particle_factor() {
        let e_cm = 1e6;
        let neutral = sigma_xx_to_s_to_pi0pi0(e_cm, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS);
        let charged = sigma_xx_to_s_to_pipi(e_cm, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS);
        assert!((charged / neutral - 2.0).abs() < 1e-6);
    }

    /// The elastic pion cross sections differ by the same charge-sum
    /// factor 2 their `.pyx` expressions carry.
    ///
    /// `__sigma_xpi_to_xpi` opens with a literal `2.0 *` that
    /// `__sigma_xpi0_to_xpi0` does not — the charge sum. Far above both
    /// thresholds the mass difference washes out and the ratio is that
    /// factor, which is what pins the leading 2 as transcribed rather
    /// than invented.
    #[test]
    fn the_two_elastic_pion_channels_differ_by_the_charge_sum() {
        let e_cm = 1e7;
        let charged = sigma_xpi_to_xpi(e_cm, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS);
        let neutral = sigma_xpi0_to_xpi0(e_cm, MX, MS, GSXX, GSFF, GSGG, LAM, WIDTH_S, VS);
        assert!(
            (charged / neutral - 2.0).abs() < 1e-6,
            "{charged} vs {neutral}"
        );
    }

    /// On resonance the `f f̄` channel peaks, and the peak scales as
    /// `1/width²`.
    ///
    /// The Breit–Wigner denominator is `(ms² − s)² + ms² Γ²`, which at
    /// `s = ms²` is `ms² Γ²`. Halving the width therefore quadruples the
    /// cross section there, and nowhere else.
    #[test]
    fn the_propagator_peaks_on_resonance() {
        let at = |w: f64| sigma_xx_to_s_to_ff(MS, MX, MS, GSXX, GSFF, w, MMU).unwrap();
        assert!((at(WIDTH_S / 2.0) / at(WIDTH_S) - 4.0).abs() < 1e-9);
        let off = |w: f64| sigma_xx_to_s_to_ff(4.0 * MS, MX, MS, GSXX, GSFF, w, MMU).unwrap();
        assert!((off(WIDTH_S / 2.0) / off(WIDTH_S) - 1.0).abs() < 1e-6);
    }

    // -- The thermal average ----------------------------------------------

    /// Above `x = 300` the scalar model returns exactly `0.0`, where the
    /// vector model saturates instead.
    ///
    /// The two disagree, deliberately: `test/parity/cases.py`'s
    /// `_thermal_blocks` pins both, and unifying them would move
    /// published numbers. The boundary is `x > 300`, so `x = 300` itself
    /// still integrates.
    #[test]
    fn the_thermal_average_cuts_off_above_three_hundred() {
        let at = |x: f64| {
            thermal_cross_section(x, MX, MS, GSXX, GSFF, GSGG, GSFF_PHOTON, LAM, WIDTH_S, VS)
                .unwrap()
        };
        assert!(at(300.0) > 0.0);
        for x in [300.000_000_1, 301.0, 1e3, 1e6] {
            assert_eq!(at(x), 0.0, "x = {x}");
        }
    }

    /// The thermal average is positive and falls with `x` across the
    /// freeze-out region.
    ///
    /// A Boltzmann weight `K₁(x z)` on an integrand that opens at `z = 2`
    /// suppresses the whole integral as the temperature drops, and the
    /// `x/(2K₂(x))²` prefactor does not undo it. Monotonicity over four
    /// decades is a statement the closed forms cannot make on their own.
    #[test]
    fn the_thermal_average_falls_with_inverse_temperature() {
        let at = |x: f64| {
            thermal_cross_section(x, MX, MS, GSXX, GSFF, GSGG, GSFF_PHOTON, LAM, WIDTH_S, VS)
                .unwrap()
        };
        let mut previous = f64::INFINITY;
        for x in [0.5, 1.0, 5.0, 20.0, 100.0, 250.0] {
            let value = at(x);
            assert!(value > 0.0, "x = {x} gave {value}");
            assert!(value < previous, "x = {x}: {value} not below {previous}");
            previous = value;
        }
    }

    /// The upper limit switches from the constant floor to `50/x` below
    /// `x = 0.5`, and the scalar floor is 100 where the vector's is 150.
    ///
    /// Checked through the integral rather than by reading the constant:
    /// at `x = 0.5` the two expressions coincide, so the value is
    /// continuous there, while an integrand that is still non-negligible
    /// at `z = 100` would make the *choice* of floor visible.
    #[test]
    fn the_upper_limit_is_continuous_where_the_branch_switches() {
        let at = |x: f64| {
            thermal_cross_section(x, MX, MS, GSXX, GSFF, GSGG, GSFF_PHOTON, LAM, WIDTH_S, VS)
                .unwrap()
        };
        let left = at(0.5 * (1.0 - 1e-9));
        let right = at(0.5 * (1.0 + 1e-9));
        assert!((left / right - 1.0).abs() < 1e-6, "{left} vs {right}");
    }
}
