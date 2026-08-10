//! The QUADPACK subset hazma's compiled layer integrates with.
//!
//! A PyO3-free translation of netlib QUADPACK
//! (`projects/cython-to-rust/rules.md`, Rust conventions rule 3 — plain
//! `fn`s taking a closure, no PyO3 types, so `cargo test` needs no GIL).
//! [`crate::quad_probe`] is the Python-visible half.
//!
//! # Sources and licensing
//!
//! Upstream: **QUADPACK**, Piessens, de Doncker-Kapenga, Überhuber and
//! Kahaner (Springer, 1983), as published on netlib
//! (<https://www.netlib.org/quadpack/>) — `dqk15.f`, `dqk21.f`,
//! `dqelg.f`, `dqpsrt.f`, `dqagse.f`, `dqagpe.f`, retrieved 2026-08-10.
//! QUADPACK is **public domain**; it is the same Fortran scipy vendors
//! (translated to C there since 1.12). Nothing here derives from GSL's
//! GPL-3 reimplementation, which is what
//! `projects/cython-to-rust/adrs/ADR-0002-license-clean-numerics.md`
//! requires (`rules.md` rule 5 / Licensing 1).
//!
//! The translation is deliberately literal: same variable names, same
//! branch order, same magic constants, and **1-based indexing preserved**
//! by giving every array a dead element 0. Idiomatic Rust would be easier
//! to read and much harder to check against the Fortran, and the point of
//! this module is that a reader can put the two side by side. Fortran
//! `go to`s become labelled blocks or `break`s in the same order; each
//! carries the original statement label in a comment.
//!
//! # What calls these, and with what
//!
//! Every live integral is over a **finite** interval, so `qagi` and the
//! transformed-infinite machinery are out of scope. All of the call sites
//! below reach QUADPACK through `scipy.integrate.quad`, so [`quad`] — not
//! [`qagse`] or [`qagpe`] — is the function a ported kernel should call.
//!
//! | Cython call site | Interval | Settings |
//! | --- | --- | --- |
//! | `hazma/spectra/_photon/_pion.pyx:123` | cos θ ∈ [−1, 1] | `points=[-1, 1]`, `epsabs=1e-10`, `epsrel=1e-5` |
//! | `hazma/spectra/_photon/_rho.pyx:52`, `:123` | boosted energy | `epsabs=1e-10`, `epsrel=1e-5` |
//! | `hazma/spectra/_positron/_pion.pyx:58` | boosted energy | `epsabs=1e-10`, `epsrel=1e-4` |
//! | `hazma/spectra/_neutrino/_pion.pyx:124`, `:127` | boosted energy | scipy defaults |
//! | `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx:1411` | z ∈ [2, max(50/x, 100)] | `points=[2, ms/mx, 2·ms/mx]` |
//! | `hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx:656` | z ∈ [2, max(50/x, 150)] | `points=[2, mv/mx, 2·mv/mx]` |
//! | `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:184`, `scalar_mediator_positron_spec.pyx:209` | cos θ ∈ [−1, 1] | `points=[-1, 1]`, `epsabs=1e-10`, `epsrel=1e-5` |
//! | `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:219`, `vector_mediator_positron_spec.pyx:210` | cos θ ∈ [−1, 1] | `points=[-1, 1]`, `epsabs=1e-10`, `epsrel=1e-5` |
//!
//! # Only `qk21` is on the live path
//!
//! `scipy.integrate.quad` on a finite interval runs `qagse` without
//! `points` and `qagpe` with them, and **both** evaluate with the 21-point
//! Gauss–Kronrod rule. [`qk15`] is reachable from no live call site; it is
//! ported because this task's exit criteria name it, because QUADPACK's
//! own reference problems exercise it, and because a second, independent
//! rule is what lets a test cross-check [`qk21`] on the same integrand
//! rather than only against itself.

// The Gauss–Kronrod tables are transcribed verbatim from the Fortran
// `data` statements, which give 31 significant digits. `f64` keeps 17, and
// re-rounding them by hand to silence the lint is exactly the edit that
// would put a wrong digit in a quadrature rule. Same allow, and the same
// reason, as `crate::constants`.
#![allow(clippy::excessive_precision)]
// The next three are the price of the literal translation this module is
// built on (see the module docs). Each of clippy's rewrites is correct in
// isolation and each one costs a line of the Fortran correspondence: an
// iterator adaptor in place of `do 120 k = 1,last`, `enumerate()` in place
// of a hand-carried `indx`, `levcur < levmax` in place of QUADPACK's
// `levcur+1 .le. levmax`. The 1-based indexing that makes the
// correspondence readable is also what makes clippy want the rewrites, so
// taking them here would only move the risk.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::int_plus_one)]

/// `d1mach(4)` — the largest relative spacing.
const EPMACH: f64 = f64::EPSILON;

/// `d1mach(1)` — the smallest positive magnitude.
const UFLOW: f64 = f64::MIN_POSITIVE;

/// `d1mach(2)` — the largest positive magnitude.
const OFLOW: f64 = f64::MAX;

/// scipy's default subdivision limit, and hazma's at every call site.
pub const DEFAULT_LIMIT: usize = 50;

/// scipy's default absolute tolerance (`quad`'s `epsabs`).
pub const DEFAULT_EPSABS: f64 = 1.49e-8;

/// scipy's default relative tolerance (`quad`'s `epsrel`).
pub const DEFAULT_EPSREL: f64 = 1.49e-8;

/// The outcome flag QUADPACK returns in `ier`, in scipy's numbering.
///
/// scipy raises `ValueError` only for [`Ier::InvalidInput`]; every other
/// non-`Ok` value is an `IntegrationWarning` beside a usable result, and
/// hazma's call sites take `quad(...)[0]` regardless. [`quad`] mirrors
/// that split exactly: `Err` for invalid input, `Ok` carrying the flag for
/// everything else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ier {
    /// `ier = 0` — normal, requested accuracy achieved.
    Ok,
    /// `ier = 1` — the subdivision limit was reached.
    MaxSubdivisions,
    /// `ier = 2` — roundoff prevents the requested tolerance.
    Roundoff,
    /// `ier = 3` — extremely bad integrand behaviour somewhere in range.
    BadIntegrand,
    /// `ier = 4` — the extrapolation table does not converge.
    NoConvergence,
    /// `ier = 5` — the integral is probably divergent or slowly
    /// convergent.
    Divergent,
    /// `ier = 6` — invalid input. Never returned inside [`QuadOutcome`];
    /// [`quad`] turns it into [`QuadError`].
    InvalidInput,
}

impl Ier {
    /// The raw QUADPACK code, so a test can compare against scipy's
    /// `full_output` dictionary without a translation table.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Ier::Ok => 0,
            Ier::MaxSubdivisions => 1,
            Ier::Roundoff => 2,
            Ier::BadIntegrand => 3,
            Ier::NoConvergence => 4,
            Ier::Divergent => 5,
            Ier::InvalidInput => 6,
        }
    }

    fn from_code(code: i32) -> Ier {
        match code {
            0 => Ier::Ok,
            1 => Ier::MaxSubdivisions,
            2 => Ier::Roundoff,
            3 => Ier::BadIntegrand,
            4 => Ier::NoConvergence,
            5 => Ier::Divergent,
            _ => Ier::InvalidInput,
        }
    }
}

/// Everything `dqagse`/`dqagpe` return.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadOutcome {
    /// Approximation to the integral.
    pub value: f64,
    /// Estimate of the modulus of the absolute error.
    pub abserr: f64,
    /// Number of integrand evaluations.
    pub neval: usize,
    /// Number of subintervals produced.
    pub last: usize,
    /// The termination flag; anything but [`Ier::Ok`] is scipy's
    /// `IntegrationWarning` case.
    pub ier: Ier,
}

/// The invalid-input cases, i.e. everything for which scipy raises
/// `ValueError` instead of warning.
///
/// Each variant names the condition QUADPACK checks rather than repeating
/// scipy's message text, because scipy reconstructs those messages in
/// Python from a decision tree over the *unfiltered* arguments — its
/// "Number of break points (N)" counts the caller's list while QUADPACK
/// counted the filtered one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadError {
    /// `epsabs <= 0` together with an unachievable `epsrel`.
    ///
    /// QUADPACK's test is `epsrel < max(50·epmach, 0.5e-28)`.
    ToleranceUnachievable,
    /// `limit < 1` (`qagse`), or `limit <= npts` (`qagpe`), where `npts`
    /// is the count of breakpoints **after** scipy's filtering.
    LimitTooSmall {
        /// The `limit` that was passed.
        limit: usize,
        /// Interior breakpoints left after filtering; `0` for `qagse`.
        npts: usize,
    },
    /// A breakpoint outside `[a, b]` reached `qagpe`.
    ///
    /// Unreachable through [`quad`], which filters exactly as scipy does;
    /// it exists because [`qagpe`] is public and faithful.
    BreakpointOutsideInterval,
}

impl std::fmt::Display for QuadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuadError::ToleranceUnachievable => write!(
                f,
                "if 'epsabs' <= 0, 'epsrel' must exceed both 5e-29 and 50 * machine epsilon"
            ),
            QuadError::LimitTooSmall { limit, npts } => write!(
                f,
                "'limit' ({limit}) must exceed the number of interior break points ({npts}) \
                 and leave at least one subinterval"
            ),
            QuadError::BreakpointOutsideInterval => {
                write!(f, "all break points must lie within the integration limits")
            }
        }
    }
}

impl std::error::Error for QuadError {}

/// Everything `dqk15`/`dqk21` return for one interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KronrodOutcome {
    /// The Kronrod approximation to ∫f.
    pub result: f64,
    /// Estimate of the modulus of the absolute error.
    pub abserr: f64,
    /// Approximation to ∫|f|.
    pub resabs: f64,
    /// Approximation to ∫|f − ∫f/(b−a)|.
    pub resasc: f64,
}

// ---------------------------------------------------------------------
// dqk15 / dqk21 — the Gauss–Kronrod rules
// ---------------------------------------------------------------------

/// Abscissae of the 15-point Kronrod rule, `xgk` in `dqk15.f`.
///
/// Even indices (1-based `xgk(2)`, `xgk(4)`, …) are the 7-point Gauss
/// abscissae; odd ones are the points optimally added to them. Only the
/// non-negative half is stored, the rule being symmetric.
const XGK15: [f64; 8] = [
    0.991455371120812639206854697526329,
    0.949107912342758524526189684047851,
    0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,
    0.586087235467691130294144838258730,
    0.405845151377397166906606412076961,
    0.207784955007898467600689403773245,
    0.000000000000000000000000000000000,
];

/// Weights of the 15-point Kronrod rule, `wgk` in `dqk15.f`.
const WGK15: [f64; 8] = [
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
];

/// Weights of the 7-point Gauss rule, `wg` in `dqk15.f`.
///
/// Seven points is odd, so `wg(4)` is the centre weight — which is why
/// [`qk15`] seeds `resg` with `fc * wg(4)` where [`qk21`] seeds it with
/// zero.
const WG15: [f64; 4] = [
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327,
];

/// Abscissae of the 21-point Kronrod rule, `xgk` in `dqk21.f`.
const XGK21: [f64; 11] = [
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.000000000000000000000000000000000,
];

/// Weights of the 21-point Kronrod rule, `wgk` in `dqk21.f`.
const WGK21: [f64; 11] = [
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
];

/// Weights of the 10-point Gauss rule, `wg` in `dqk21.f`.
const WG21: [f64; 5] = [
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
];

/// The shared tail of `dqk15`/`dqk21`: scale to `[a, b]` and turn the
/// Kronrod−Gauss difference into QUADPACK's error estimate.
///
/// `resk`, `resg`, `resabs` and `resasc` arrive on the reference interval;
/// `hlgth` is the half-length of `[a, b]`.
fn kronrod_error(resk: f64, resg: f64, resabs: f64, resasc: f64, hlgth: f64) -> KronrodOutcome {
    let dhlgth = hlgth.abs();
    let result = resk * hlgth;
    let resabs = resabs * dhlgth;
    let resasc = resasc * dhlgth;
    let mut abserr = ((resk - resg) * hlgth).abs();
    if resasc != 0.0 && abserr != 0.0 {
        abserr = resasc * (1.0_f64).min((200.0 * abserr / resasc).powf(1.5));
    }
    if resabs > UFLOW / (50.0 * EPMACH) {
        abserr = ((EPMACH * 50.0) * resabs).max(abserr);
    }
    KronrodOutcome {
        result,
        abserr,
        resabs,
        resasc,
    }
}

/// 15-point Gauss–Kronrod rule on `[a, b]` — `dqk15`.
///
/// Not reachable from any hazma call site (see the module docs); ported
/// for the exit criteria and as an independent check on [`qk21`].
pub fn qk15<F>(f: &mut F, a: f64, b: f64) -> KronrodOutcome
where
    F: FnMut(f64) -> f64,
{
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);

    // 1-based, to match `fv1(j)` / `fv2(j)` in the Fortran.
    let mut fv1 = [0.0_f64; 8];
    let mut fv2 = [0.0_f64; 8];

    let fc = f(centr);
    let mut resg = fc * WG15[3];
    let mut resk = fc * WGK15[7];
    let mut resabs = resk.abs();

    // do 10 j = 1,3 — the Gauss abscissae, at even Kronrod indices.
    for j in 1..=3 {
        let jtw = 2 * j;
        let absc = hlgth * XGK15[jtw - 1];
        let fval1 = f(centr - absc);
        let fval2 = f(centr + absc);
        fv1[jtw - 1] = fval1;
        fv2[jtw - 1] = fval2;
        let fsum = fval1 + fval2;
        resg += WG15[j - 1] * fsum;
        resk += WGK15[jtw - 1] * fsum;
        resabs += WGK15[jtw - 1] * (fval1.abs() + fval2.abs());
    }

    // do 15 j = 1,4 — the added Kronrod abscissae, at odd indices.
    for j in 1..=4 {
        let jtwm1 = 2 * j - 1;
        let absc = hlgth * XGK15[jtwm1 - 1];
        let fval1 = f(centr - absc);
        let fval2 = f(centr + absc);
        fv1[jtwm1 - 1] = fval1;
        fv2[jtwm1 - 1] = fval2;
        let fsum = fval1 + fval2;
        resk += WGK15[jtwm1 - 1] * fsum;
        resabs += WGK15[jtwm1 - 1] * (fval1.abs() + fval2.abs());
    }

    let reskh = resk * 0.5;
    let mut resasc = WGK15[7] * (fc - reskh).abs();
    for j in 1..=7 {
        resasc += WGK15[j - 1] * ((fv1[j - 1] - reskh).abs() + (fv2[j - 1] - reskh).abs());
    }

    kronrod_error(resk, resg, resabs, resasc, hlgth)
}

/// 21-point Gauss–Kronrod rule on `[a, b]` — `dqk21`.
///
/// The rule every live hazma integral runs on: both `qagse` and `qagpe`
/// evaluate with it and nothing else.
pub fn qk21<F>(f: &mut F, a: f64, b: f64) -> KronrodOutcome
where
    F: FnMut(f64) -> f64,
{
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);

    let mut fv1 = [0.0_f64; 11];
    let mut fv2 = [0.0_f64; 11];

    // Ten Gauss points is even, so there is no centre Gauss weight and
    // `resg` starts at zero — the one structural difference from `qk15`.
    let mut resg = 0.0_f64;
    let fc = f(centr);
    let mut resk = WGK21[10] * fc;
    let mut resabs = resk.abs();

    // do 10 j = 1,5
    for j in 1..=5 {
        let jtw = 2 * j;
        let absc = hlgth * XGK21[jtw - 1];
        let fval1 = f(centr - absc);
        let fval2 = f(centr + absc);
        fv1[jtw - 1] = fval1;
        fv2[jtw - 1] = fval2;
        let fsum = fval1 + fval2;
        resg += WG21[j - 1] * fsum;
        resk += WGK21[jtw - 1] * fsum;
        resabs += WGK21[jtw - 1] * (fval1.abs() + fval2.abs());
    }

    // do 15 j = 1,5
    for j in 1..=5 {
        let jtwm1 = 2 * j - 1;
        let absc = hlgth * XGK21[jtwm1 - 1];
        let fval1 = f(centr - absc);
        let fval2 = f(centr + absc);
        fv1[jtwm1 - 1] = fval1;
        fv2[jtwm1 - 1] = fval2;
        let fsum = fval1 + fval2;
        resk += WGK21[jtwm1 - 1] * fsum;
        resabs += WGK21[jtwm1 - 1] * (fval1.abs() + fval2.abs());
    }

    let reskh = resk * 0.5;
    let mut resasc = WGK21[10] * (fc - reskh).abs();
    for j in 1..=10 {
        resasc += WGK21[j - 1] * ((fv1[j - 1] - reskh).abs() + (fv2[j - 1] - reskh).abs());
    }

    kronrod_error(resk, resg, resabs, resasc, hlgth)
}

// ---------------------------------------------------------------------
// dqelg — Wynn's epsilon algorithm
// ---------------------------------------------------------------------

/// Wynn's ε-algorithm on the condensed table — `dqelg`.
///
/// `n` and `nres` are **in/out**, exactly as in the Fortran: the routine
/// truncates the table by assigning to `n`, and the caller's `numrl2` must
/// see that. `epstab` is 1-based with 52 usable slots (index `0` unused),
/// `res3la` 1-based with 3.
///
/// Returns `(result, abserr)`.
fn qelg(n: &mut usize, epstab: &mut [f64], res3la: &mut [f64], nres: &mut usize) -> (f64, f64) {
    *nres += 1;
    let mut abserr = OFLOW;
    let mut result = epstab[*n];
    if *n < 3 {
        // 100
        return (result, abserr.max(5.0 * EPMACH * result.abs()));
    }
    let limexp = 50_usize;
    epstab[*n + 2] = epstab[*n];
    let newelm = (*n - 1) / 2;
    epstab[*n] = OFLOW;
    let num = *n;
    let mut k1 = *n;

    // The Fortran's `do 40` either runs to completion (falling into the
    // shift at 50), jumps out to 50 having truncated `n`, or jumps out to
    // 100 on detected convergence.
    let mut converged = false;
    for i in 1..=newelm {
        let k2 = k1 - 1;
        let k3 = k1 - 2;
        let mut res = epstab[k1 + 2];
        let e0 = epstab[k3];
        let e1 = epstab[k2];
        let e2 = res;
        let e1abs = e1.abs();
        let delta2 = e2 - e1;
        let err2 = delta2.abs();
        let tol2 = e2.abs().max(e1abs) * EPMACH;
        let delta3 = e1 - e0;
        let err3 = delta3.abs();
        let tol3 = e1abs.max(e0.abs()) * EPMACH;
        if !(err2 > tol2 || err3 > tol3) {
            // e0, e1, e2 agree to machine accuracy: convergence.
            result = res;
            abserr = err2 + err3;
            converged = true;
            break;
        }
        // 10
        let e3 = epstab[k1];
        epstab[k1] = e1;
        let delta1 = e1 - e3;
        let err1 = delta1.abs();
        let tol1 = e1abs.max(e3.abs()) * EPMACH;

        // 20 — two elements are very close, or the table is behaving
        // irregularly: drop part of it by shortening `n`.
        let mut truncate = err1 <= tol1 || err2 <= tol2 || err3 <= tol3;
        let mut ss = 0.0;
        if !truncate {
            ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3;
            let epsinf = (ss * e1).abs();
            truncate = epsinf <= 0.1e-3;
        }
        if truncate {
            *n = i + i - 1;
            break;
        }

        // 30 — compute a new element and possibly improve `result`.
        res = e1 + 1.0 / ss;
        epstab[k1] = res;
        k1 -= 2;
        let error = err2 + (res - e2).abs() + err3;
        if error <= abserr {
            abserr = error;
            result = res;
        }
    }
    if converged {
        // 100
        return (result, abserr.max(5.0 * EPMACH * result.abs()));
    }

    // 50 — shift the table.
    if *n == limexp {
        *n = 2 * (limexp / 2) - 1;
    }
    let mut ib = 1_usize;
    if (num / 2) * 2 == num {
        ib = 2;
    }
    let ie = newelm + 1;
    for _i in 1..=ie {
        let ib2 = ib + 2;
        epstab[ib] = epstab[ib2];
        ib = ib2;
    }
    if num != *n {
        let mut indx = num - *n + 1;
        for i in 1..=*n {
            epstab[i] = epstab[indx];
            indx += 1;
        }
    }
    // 80
    if *nres < 4 {
        res3la[*nres] = result;
        abserr = OFLOW;
    } else {
        // 90 — error estimate from the last three results.
        abserr =
            (result - res3la[3]).abs() + (result - res3la[2]).abs() + (result - res3la[1]).abs();
        res3la[1] = res3la[2];
        res3la[2] = res3la[3];
        res3la[3] = result;
    }
    // 100
    (result, abserr.max(5.0 * EPMACH * result.abs()))
}

// ---------------------------------------------------------------------
// dqpsrt — maintain the descending ordering of the error estimates
// ---------------------------------------------------------------------

/// Keep `iord` ordered by descending `elist` and pick the next interval
/// to bisect — `dqpsrt`.
///
/// `maxerr`, `ermax` and `nrmax` are in/out. All arrays are 1-based.
#[allow(
    clippy::too_many_arguments,
    reason = "one parameter per Fortran argument; regrouping them would \
              break the line-by-line correspondence this module trades on"
)]
fn qpsrt(
    limit: usize,
    last: usize,
    maxerr: &mut usize,
    ermax: &mut f64,
    elist: &[f64],
    iord: &mut [usize],
    nrmax: &mut usize,
) {
    if last <= 2 {
        iord[1] = 1;
        iord[2] = 2;
        // 90
        *maxerr = iord[*nrmax];
        *ermax = elist[*maxerr];
        return;
    }

    // 10 — this part runs only when subdivision *increased* the error.
    let errmax = elist[*maxerr];
    if *nrmax != 1 {
        let ido = *nrmax - 1;
        for _i in 1..=ido {
            let isucc = iord[*nrmax - 1];
            if errmax <= elist[isucc] {
                break;
            }
            iord[*nrmax] = isucc;
            *nrmax -= 1;
        }
    }

    // 30 — how much of the list still has to stay ordered.
    let mut jupbn = last;
    if last > limit / 2 + 2 {
        jupbn = limit + 3 - last;
    }
    let errmin = elist[last];

    let jbnd = jupbn - 1;
    let ibeg = *nrmax + 1;

    // 40 — insert errmax, traversing the list top-down.
    let mut inserted_at: Option<usize> = None;
    if ibeg <= jbnd {
        for i in ibeg..=jbnd {
            let isucc = iord[i];
            if errmax >= elist[isucc] {
                inserted_at = Some(i);
                break;
            }
            iord[i - 1] = isucc;
        }
    }

    let Some(i) = inserted_at else {
        // 50 — errmax belongs at the bottom of the maintained list.
        iord[jbnd] = *maxerr;
        iord[jupbn] = last;
        // 90
        *maxerr = iord[*nrmax];
        *ermax = elist[*maxerr];
        return;
    };

    // 60 — insert errmin, traversing the list bottom-up.
    iord[i - 1] = *maxerr;
    let mut k = jbnd;
    let mut placed = false;
    for _j in i..=jbnd {
        let isucc = iord[k];
        if errmin < elist[isucc] {
            // 80
            iord[k + 1] = last;
            placed = true;
            break;
        }
        iord[k + 1] = isucc;
        k -= 1;
    }
    if !placed {
        iord[i] = last;
    }

    // 90
    *maxerr = iord[*nrmax];
    *ermax = elist[*maxerr];
}

// ---------------------------------------------------------------------
// dqagse — adaptive, with ε-extrapolation, no breakpoints
// ---------------------------------------------------------------------

/// Adaptive quadrature with ε-extrapolation on a finite interval —
/// `dqagse`, the routine behind `scipy.integrate.quad(..., points=None)`.
///
/// # Errors
///
/// [`QuadError::ToleranceUnachievable`] when `epsabs <= 0` and `epsrel` is
/// below QUADPACK's floor, and [`QuadError::LimitTooSmall`] when
/// `limit < 1`.
pub fn qagse<F>(
    f: &mut F,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Result<QuadOutcome, QuadError>
where
    F: FnMut(f64) -> f64,
{
    // The Fortran declares `alist(limit)` and indexes from 1, so a
    // `limit`-length list needs `limit + 1` slots here. `limit == 0` is
    // rejected below, matching scipy's "there must be at least one
    // subinterval"; QUADPACK itself expresses that as `limit.lt.1`
    // upstream in dqags.
    if limit < 1 {
        return Err(QuadError::LimitTooSmall { limit, npts: 0 });
    }
    if epsabs <= 0.0 && epsrel < (50.0 * EPMACH).max(0.5e-28) {
        return Err(QuadError::ToleranceUnachievable);
    }

    let mut alist = vec![0.0_f64; limit + 1];
    let mut blist = vec![0.0_f64; limit + 1];
    let mut rlist = vec![0.0_f64; limit + 1];
    let mut elist = vec![0.0_f64; limit + 1];
    let mut iord = vec![0_usize; limit + 2];
    let mut rlist2 = [0.0_f64; 53];
    let mut res3la = [0.0_f64; 4];

    let mut ier = 0_i32;
    let mut ierro = 0_i32;
    alist[1] = a;
    blist[1] = b;

    // First approximation. `defabs` takes dqk21's `resabs`, and the
    // Fortran's local `resabs` takes its `resasc` — the names cross over
    // at the call site, which is worth reading twice.
    let first = qk21(f, a, b);
    let mut result = first.result;
    let mut abserr = first.abserr;
    let defabs = first.resabs;
    let resabs_as_asc = first.resasc;

    let dres = result.abs();
    let mut errbnd = epsabs.max(epsrel * dres);
    let mut last = 1_usize;
    rlist[1] = result;
    elist[1] = abserr;
    iord[1] = 1;
    if abserr <= 100.0 * EPMACH * defabs && abserr > errbnd {
        ier = 2;
    }
    if limit == 1 {
        ier = 1;
    }
    if ier != 0 || (abserr <= errbnd && abserr != resabs_as_asc) || abserr == 0.0 {
        // 140
        return Ok(QuadOutcome {
            value: result,
            abserr,
            neval: 42 * last - 21,
            last,
            ier: Ier::from_code(ier),
        });
    }

    // Initialization.
    rlist2[1] = result;
    let mut errmax = abserr;
    let mut maxerr = 1_usize;
    let mut area = result;
    let mut errsum = abserr;
    abserr = OFLOW;
    let mut nrmax = 1_usize;
    let mut nres = 0_usize;
    let mut numrl2 = 2_usize;
    let mut ktmin = 0_i32;
    let mut extrap = false;
    let mut noext = false;
    let mut iroff1 = 0_i32;
    let mut iroff2 = 0_i32;
    let mut iroff3 = 0_i32;
    let mut ksgn: i32 = -1;
    if dres >= (1.0 - 50.0 * EPMACH) * defabs {
        ksgn = 1;
    }

    let mut small = 0.0_f64;
    let mut erlarg = 0.0_f64;
    let mut ertest = 0.0_f64;
    let mut correc = 0.0_f64;

    // Which Fortran label the main loop left through. Falling off the end
    // of the `do 90` also lands on 100, so that is the default.
    let mut goto_115 = false;

    'mainloop: for last_index in 2..=limit {
        last = last_index;

        // Bisect the subinterval with the nrmax-th largest error.
        let a1 = alist[maxerr];
        let b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        let a2 = b1;
        let b2 = blist[maxerr];
        let erlast = errmax;
        let left = qk21(f, a1, b1);
        let right = qk21(f, a2, b2);
        let (area1, error1, defab1) = (left.result, left.abserr, left.resasc);
        let (area2, error2, defab2) = (right.result, right.abserr, right.resasc);

        let area12 = area1 + area2;
        let erro12 = error1 + error2;
        errsum += erro12 - errmax;
        area += area12 - rlist[maxerr];
        if !(defab1 == error1 || defab2 == error2) {
            if !((rlist[maxerr] - area12).abs() > 0.1e-4 * area12.abs() || erro12 < 0.99 * errmax) {
                if extrap {
                    iroff2 += 1;
                } else {
                    iroff1 += 1;
                }
            }
            // 10
            if last > 10 && erro12 > errmax {
                iroff3 += 1;
            }
        }
        // 15
        rlist[maxerr] = area1;
        rlist[last] = area2;
        errbnd = epsabs.max(epsrel * area.abs());

        if iroff1 + iroff2 >= 10 || iroff3 >= 20 {
            ier = 2;
        }
        if iroff2 >= 5 {
            ierro = 3;
        }
        if last == limit {
            ier = 1;
        }
        if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * (a2.abs() + 1000.0 * UFLOW) {
            ier = 4;
        }

        // Append the newly-created intervals.
        if error2 > error1 {
            // 20
            alist[maxerr] = a2;
            alist[last] = a1;
            blist[last] = b1;
            rlist[maxerr] = area2;
            rlist[last] = area1;
            elist[maxerr] = error2;
            elist[last] = error1;
        } else {
            alist[last] = a2;
            blist[maxerr] = b1;
            blist[last] = b2;
            elist[maxerr] = error1;
            elist[last] = error2;
        }

        // 30
        qpsrt(
            limit,
            last,
            &mut maxerr,
            &mut errmax,
            &elist,
            &mut iord,
            &mut nrmax,
        );
        if errsum <= errbnd {
            goto_115 = true;
            break 'mainloop;
        }
        if ier != 0 {
            break 'mainloop;
        }
        if last == 2 {
            // 80
            small = (b - a).abs() * 0.375;
            erlarg = errsum;
            ertest = errbnd;
            rlist2[2] = area;
            continue 'mainloop;
        }
        if noext {
            continue 'mainloop;
        }
        erlarg -= erlast;
        if (b1 - a1).abs() > small {
            erlarg += erro12;
        }
        if !extrap {
            // Is the interval to be bisected next the smallest one?
            if (blist[maxerr] - alist[maxerr]).abs() > small {
                continue 'mainloop;
            }
            extrap = true;
            nrmax = 2;
        }
        // 40
        if !(ierro == 3 || erlarg <= ertest) {
            // The smallest interval has the largest error: shrink erlarg
            // over the larger intervals before bisecting.
            let id = nrmax;
            let mut jupbnd = last;
            if last > 2 + limit / 2 {
                jupbnd = limit + 3 - last;
            }
            let mut back_to_90 = false;
            for _k in id..=jupbnd {
                maxerr = iord[nrmax];
                errmax = elist[maxerr];
                if (blist[maxerr] - alist[maxerr]).abs() > small {
                    back_to_90 = true;
                    break;
                }
                nrmax += 1;
            }
            if back_to_90 {
                continue 'mainloop;
            }
        }

        // 60 — extrapolate.
        numrl2 += 1;
        rlist2[numrl2] = area;
        let (reseps, abseps) = qelg(&mut numrl2, &mut rlist2, &mut res3la, &mut nres);
        ktmin += 1;
        if ktmin > 5 && abserr < 0.1e-2 * errsum {
            ier = 5;
        }
        if abseps < abserr {
            ktmin = 0;
            abserr = abseps;
            result = reseps;
            correc = erlarg;
            ertest = epsabs.max(epsrel * reseps.abs());
            if abserr <= ertest {
                break 'mainloop;
            }
        }

        // 70 — prepare bisection of the smallest interval.
        if numrl2 == 1 {
            noext = true;
        }
        if ier == 5 {
            break 'mainloop;
        }
        maxerr = iord[1];
        errmax = elist[maxerr];
        nrmax = 1;
        extrap = false;
        small *= 0.5;
        erlarg = errsum;
    }

    // Set the final result and error estimate — Fortran labels 100..140,
    // nested so that every `go to` is a forward `break`.
    'l130: {
        'l115: {
            'l110: {
                'l105: {
                    if goto_115 {
                        break 'l115;
                    }
                    // 100
                    if abserr == OFLOW {
                        break 'l115;
                    }
                    if ier + ierro == 0 {
                        break 'l110;
                    }
                    if ierro == 3 {
                        abserr += correc;
                    }
                    if ier == 0 {
                        ier = 3;
                    }
                    if result != 0.0 && area != 0.0 {
                        break 'l105;
                    }
                    if abserr > errsum {
                        break 'l115;
                    }
                    if area == 0.0 {
                        break 'l130;
                    }
                    break 'l110;
                }
                // 105
                if abserr / result.abs() > errsum / area.abs() {
                    break 'l115;
                }
            }
            // 110 — test on divergence.
            if ksgn == -1 && result.abs().max(area.abs()) <= defabs * 0.1e-1 {
                break 'l130;
            }
            if 0.1e-1 > (result / area) || (result / area) > 0.1e3 || errsum > area.abs() {
                ier = 6;
            }
            break 'l130;
        }
        // 115 — compute the global integral sum.
        result = 0.0;
        for k in 1..=last {
            result += rlist[k];
        }
        abserr = errsum;
    }
    // 130
    if ier > 2 {
        ier -= 1;
    }
    // 140
    Ok(QuadOutcome {
        value: result,
        abserr,
        neval: 42 * last - 21,
        last,
        ier: Ier::from_code(ier),
    })
}

// ---------------------------------------------------------------------
// dqagpe — adaptive, with ε-extrapolation and user breakpoints
// ---------------------------------------------------------------------

/// Adaptive quadrature with ε-extrapolation and user-supplied
/// breakpoints — `dqagpe`, the routine behind
/// `scipy.integrate.quad(..., points=...)`.
///
/// `points` holds the `npts` interior breakpoints only; the Fortran's
/// `npts2 = npts + 2` and its two scratch slots are an artefact of
/// Fortran's fixed-size arrays and are not part of this signature.
/// Unsorted input is fine (the routine sorts, as the Fortran does), but
/// **a breakpoint outside `[a, b]` is an error here**, not something to
/// ignore — [`quad`] is the layer that filters, because that is where
/// scipy filters.
///
/// `npts == 0` is a normal case, not a degenerate one, and it is where
/// five of hazma's twelve call sites land: `points=[-1, 1]` on `[-1, 1]`
/// leaves nothing after scipy's filtering, and `points is None` is what
/// selects `qagse` — not "no break points survived".
///
/// It does not quite reduce to [`qagse`] either. `qagpe` decides an
/// interval is the smallest by subdivision `level`, `qagse` by comparing
/// its length against `small`, and `qagpe` therefore starts extrapolating
/// one bisection earlier. Measured (Task 3.3, scipy 1.18.0): across 3,776
/// random (integrand, tolerance, limit) combinations the two returned
/// identical values, `neval` and `last` on every run that converged, and
/// differed on 45 — all of them runs that exhausted `limit`. On
/// `|x − 1/3|^(−9/10)·cos(50x)` over `[0, 1]` at `limit = 10` the gap is
/// 11%, which is what `test/test_core_quad.py` uses to tell them apart.
///
/// # Errors
///
/// [`QuadError::ToleranceUnachievable`], [`QuadError::LimitTooSmall`] when
/// `limit <= npts`, and [`QuadError::BreakpointOutsideInterval`].
pub fn qagpe<F>(
    f: &mut F,
    a: f64,
    b: f64,
    points: &[f64],
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Result<QuadOutcome, QuadError>
where
    F: FnMut(f64) -> f64,
{
    let npts = points.len();
    let npts2 = npts + 2;
    if limit <= npts {
        return Err(QuadError::LimitTooSmall { limit, npts });
    }
    if epsabs <= 0.0 && epsrel < (50.0 * EPMACH).max(0.5e-28) {
        return Err(QuadError::ToleranceUnachievable);
    }

    let mut alist = vec![0.0_f64; limit + 1];
    let mut blist = vec![0.0_f64; limit + 1];
    let mut rlist = vec![0.0_f64; limit + 1];
    let mut elist = vec![0.0_f64; limit + 1];
    let mut iord = vec![0_usize; limit + 2];
    let mut level = vec![0_i32; limit + 1];
    let mut ndin = vec![0_i32; npts2 + 1];
    let mut pts = vec![0.0_f64; npts2 + 1];
    let mut rlist2 = [0.0_f64; 53];
    let mut res3la = [0.0_f64; 4];

    let mut ier = 0_i32;
    let mut ierro = 0_i32;
    alist[1] = a;
    blist[1] = b;

    let sign = if a > b { -1.0_f64 } else { 1.0_f64 };
    pts[1] = a.min(b);
    for (i, &p) in points.iter().enumerate() {
        pts[i + 2] = p;
    }
    pts[npts + 2] = a.max(b);
    let nint = npts + 1;
    let mut a1 = pts[1];
    if npts != 0 {
        let nintp1 = nint + 1;
        // The Fortran's `do 20` is a literal selection sort over
        // pts(1..nintp1); reproduced rather than replaced by `sort`, since
        // it is what decides the tie order among equal breakpoints.
        for i in 1..=nint {
            for j in (i + 1)..=nintp1 {
                if pts[i] > pts[j] {
                    pts.swap(i, j);
                }
            }
        }
        if pts[1] != a.min(b) || pts[nintp1] != a.max(b) {
            return Err(QuadError::BreakpointOutsideInterval);
        }
    }

    // 40 — first integral and error approximations, one per subinterval.
    let mut result = 0.0_f64;
    let mut abserr = 0.0_f64;
    let mut resabs = 0.0_f64;
    for i in 1..=nint {
        let b1 = pts[i + 1];
        let piece = qk21(f, a1, b1);
        abserr += piece.abserr;
        result += piece.result;
        ndin[i] = 0;
        if piece.abserr == piece.resasc && piece.abserr != 0.0 {
            ndin[i] = 1;
        }
        resabs += piece.resabs;
        level[i] = 0;
        elist[i] = piece.abserr;
        alist[i] = a1;
        blist[i] = b1;
        rlist[i] = piece.result;
        iord[i] = i;
        a1 = b1;
    }
    let mut errsum = 0.0_f64;
    for i in 1..=nint {
        if ndin[i] == 1 {
            elist[i] = abserr;
        }
        errsum += elist[i];
    }

    // Test on accuracy.
    let mut last = nint;
    let mut neval = 21 * nint;
    let dres = result.abs();
    let mut errbnd = epsabs.max(epsrel * dres);
    if abserr <= 0.1e3 * EPMACH * resabs && abserr > errbnd {
        ier = 2;
    }
    if nint != 1 {
        // 70 — put the npts largest errors at the head of iord.
        for i in 1..=npts {
            let jlow = i + 1;
            let mut ind1 = iord[i];
            let mut k = 0_usize;
            for j in jlow..=nint {
                let ind2 = iord[j];
                if elist[ind1] <= elist[ind2] {
                    ind1 = ind2;
                    k = j;
                }
            }
            if ind1 != iord[i] {
                iord[k] = iord[i];
                iord[i] = ind1;
            }
        }
        if limit < npts2 {
            ier = 1;
        }
    }
    // 80
    if ier != 0 || abserr <= errbnd {
        // 210
        if ier > 2 {
            ier -= 1;
        }
        return Ok(QuadOutcome {
            value: result * sign,
            abserr,
            neval,
            last,
            ier: Ier::from_code(ier),
        });
    }

    // Initialization.
    rlist2[1] = result;
    let mut maxerr = iord[1];
    let mut errmax = elist[maxerr];
    let mut area = result;
    let mut nrmax = 1_usize;
    let mut nres = 0_usize;
    let mut numrl2 = 1_usize;
    let mut ktmin = 0_i32;
    let mut extrap = false;
    let mut noext = false;
    let mut erlarg = errsum;
    let mut ertest = errbnd;
    let mut levmax = 1_i32;
    let mut iroff1 = 0_i32;
    let mut iroff2 = 0_i32;
    let mut iroff3 = 0_i32;
    let mut correc = 0.0_f64;
    abserr = OFLOW;
    let mut ksgn: i32 = -1;
    if dres >= (1.0 - 50.0 * EPMACH) * resabs {
        ksgn = 1;
    }

    let mut goto_190 = false;

    'mainloop: for last_index in npts2..=limit {
        last = last_index;

        let levcur = level[maxerr] + 1;
        let a1 = alist[maxerr];
        let b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        let a2 = b1;
        let b2 = blist[maxerr];
        let erlast = errmax;
        let left = qk21(f, a1, b1);
        let right = qk21(f, a2, b2);
        let (area1, error1, defab1) = (left.result, left.abserr, left.resasc);
        let (area2, error2, defab2) = (right.result, right.abserr, right.resasc);

        neval += 42;
        let area12 = area1 + area2;
        let erro12 = error1 + error2;
        errsum += erro12 - errmax;
        area += area12 - rlist[maxerr];
        if !(defab1 == error1 || defab2 == error2) {
            if !((rlist[maxerr] - area12).abs() > 0.1e-4 * area12.abs() || erro12 < 0.99 * errmax) {
                if extrap {
                    iroff2 += 1;
                } else {
                    iroff1 += 1;
                }
            }
            // 90
            if last > 10 && erro12 > errmax {
                iroff3 += 1;
            }
        }
        // 95
        level[maxerr] = levcur;
        level[last] = levcur;
        rlist[maxerr] = area1;
        rlist[last] = area2;
        errbnd = epsabs.max(epsrel * area.abs());

        if iroff1 + iroff2 >= 10 || iroff3 >= 20 {
            ier = 2;
        }
        if iroff2 >= 5 {
            ierro = 3;
        }
        if last == limit {
            ier = 1;
        }
        if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * (a2.abs() + 1000.0 * UFLOW) {
            ier = 4;
        }

        if error2 > error1 {
            // 100
            alist[maxerr] = a2;
            alist[last] = a1;
            blist[last] = b1;
            rlist[maxerr] = area2;
            rlist[last] = area1;
            elist[maxerr] = error2;
            elist[last] = error1;
        } else {
            alist[last] = a2;
            blist[maxerr] = b1;
            blist[last] = b2;
            elist[maxerr] = error1;
            elist[last] = error2;
        }

        // 110
        qpsrt(
            limit,
            last,
            &mut maxerr,
            &mut errmax,
            &elist,
            &mut iord,
            &mut nrmax,
        );
        if errsum <= errbnd {
            goto_190 = true;
            break 'mainloop;
        }
        if ier != 0 {
            break 'mainloop;
        }
        if noext {
            continue 'mainloop;
        }
        erlarg -= erlast;
        if levcur + 1 <= levmax {
            erlarg += erro12;
        }
        if !extrap {
            // Is the interval to be bisected next the smallest one?
            if level[maxerr] + 1 <= levmax {
                continue 'mainloop;
            }
            extrap = true;
            nrmax = 2;
        }
        // 120
        if !(ierro == 3 || erlarg <= ertest) {
            let id = nrmax;
            let mut jupbnd = last;
            if last > 2 + limit / 2 {
                jupbnd = limit + 3 - last;
            }
            let mut back_to_160 = false;
            for _k in id..=jupbnd {
                maxerr = iord[nrmax];
                errmax = elist[maxerr];
                if level[maxerr] + 1 <= levmax {
                    back_to_160 = true;
                    break;
                }
                nrmax += 1;
            }
            if back_to_160 {
                continue 'mainloop;
            }
        }

        // 140 — extrapolate. Unlike `qagse`, `qagpe` skips straight to
        // 155 until the table holds three elements.
        numrl2 += 1;
        rlist2[numrl2] = area;
        if numrl2 > 2 {
            let (reseps, abseps) = qelg(&mut numrl2, &mut rlist2, &mut res3la, &mut nres);
            ktmin += 1;
            if ktmin > 5 && abserr < 0.1e-2 * errsum {
                ier = 5;
            }
            if abseps < abserr {
                ktmin = 0;
                abserr = abseps;
                result = reseps;
                correc = erlarg;
                ertest = epsabs.max(epsrel * reseps.abs());
                if abserr < ertest {
                    break 'mainloop;
                }
            }
            // 150
            if numrl2 == 1 {
                noext = true;
            }
            if ier >= 5 {
                break 'mainloop;
            }
        }
        // 155 — prepare bisection of the smallest interval.
        maxerr = iord[1];
        errmax = elist[maxerr];
        nrmax = 1;
        extrap = false;
        levmax += 1;
        erlarg = errsum;
    }

    // Set the final result — Fortran labels 170..210.
    'l210: {
        'l190: {
            'l180: {
                'l175: {
                    if goto_190 {
                        break 'l190;
                    }
                    // 170
                    if abserr == OFLOW {
                        break 'l190;
                    }
                    if ier + ierro == 0 {
                        break 'l180;
                    }
                    if ierro == 3 {
                        abserr += correc;
                    }
                    if ier == 0 {
                        ier = 3;
                    }
                    if result != 0.0 && area != 0.0 {
                        break 'l175;
                    }
                    if abserr > errsum {
                        break 'l190;
                    }
                    if area == 0.0 {
                        break 'l210;
                    }
                    break 'l180;
                }
                // 175
                if abserr / result.abs() > errsum / area.abs() {
                    break 'l190;
                }
            }
            // 180 — test on divergence.
            if ksgn == -1 && result.abs().max(area.abs()) <= resabs * 0.1e-1 {
                break 'l210;
            }
            if 0.1e-1 > (result / area) || (result / area) > 0.1e3 || errsum > area.abs() {
                ier = 6;
            }
            break 'l210;
        }
        // 190 — compute the global integral sum.
        result = 0.0;
        for k in 1..=last {
            result += rlist[k];
        }
        abserr = errsum;
    }
    // 210
    if ier > 2 {
        ier -= 1;
    }
    Ok(QuadOutcome {
        value: result * sign,
        abserr,
        neval,
        last,
        ier: Ier::from_code(ier),
    })
}

// ---------------------------------------------------------------------
// The scipy-shaped driver
// ---------------------------------------------------------------------

/// Settings for [`quad`], defaulting exactly as `scipy.integrate.quad`
/// does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadOpts<'a> {
    /// Absolute error tolerance.
    pub epsabs: f64,
    /// Relative error tolerance.
    pub epsrel: f64,
    /// Upper bound on the number of subintervals.
    pub limit: usize,
    /// Break points, as passed to `quad(points=...)` — unsorted,
    /// duplicated, endpoint-coincident and out-of-interval entries are all
    /// accepted and handled as scipy handles them. `None` selects
    /// [`qagse`]; `Some` selects [`qagpe`], **even if nothing survives the
    /// filtering**.
    pub points: Option<&'a [f64]>,
}

impl Default for QuadOpts<'_> {
    fn default() -> Self {
        QuadOpts {
            epsabs: DEFAULT_EPSABS,
            epsrel: DEFAULT_EPSREL,
            limit: DEFAULT_LIMIT,
            points: None,
        }
    }
}

/// Filter break points the way `scipy.integrate._quadpack_py._quad` does.
///
/// scipy's three lines are `np.unique(points)`, then `[a < p]`, then
/// `[p < b]` — so the contract is: **sort ascending, drop duplicates, and
/// keep only strictly interior points.** Endpoint-coincident points, points
/// outside `[a, b]` and `NaN`s all vanish (every comparison against `NaN`
/// is false), and `-0.0`/`0.0` collapse to one entry because they compare
/// equal.
///
/// Pinned empirically against scipy 1.18.0 rather than read off the
/// QUADPACK documentation, as this task's exit criteria require — and it
/// matters, because both live degeneracies land in the discard branch:
/// `points=[-1, 1]` on `[-1, 1]` leaves **nothing**, and a heavy mediator
/// pushes `m/mx` and `2m/mx` past the upper bound of the thermal integral.
/// `test/test_core_quad.py` re-derives every clause from scipy.
///
/// `a` and `b` must already be ordered (`a <= b`), which is what [`quad`]
/// guarantees by flipping first — as scipy does.
fn filter_points(points: &[f64], a: f64, b: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = points.to_vec();
    // `np.unique` sorts with NaNs last; `total_cmp` orders them the same
    // way and, unlike `partial_cmp`, is a total order so the sort is
    // well-defined. -0.0 sorts before 0.0 under `total_cmp`, which is the
    // order `np.unique` produces too, and the dedup below then keeps -0.0.
    sorted.sort_by(f64::total_cmp);
    sorted.dedup_by(|x, y| *x == *y);
    sorted.retain(|&p| a < p && p < b);
    sorted
}

/// `scipy.integrate.quad` for a finite interval, closure in place of the
/// Python callable.
///
/// This — not [`qagse`] or [`qagpe`] — is what a ported kernel calls, so
/// that the argument preprocessing the `.pyx` inherited from scipy is
/// inherited once, here, instead of at every call site. It reproduces
/// scipy's whole finite-interval path: order the limits and negate the
/// result if they were reversed, filter `points` (see [`filter_points`]),
/// then dispatch to `qagpe` when `points` is `Some` and `qagse` when it is
/// `None`.
///
/// Like scipy, an abnormal termination that still produced a usable answer
/// is **not** an error: `ier` rides along in the returned
/// [`QuadOutcome`] where scipy raises an `IntegrationWarning`. hazma's call
/// sites all take the value and ignore the warning.
///
/// # Errors
///
/// Only the cases where scipy raises `ValueError`; see [`QuadError`].
pub fn quad<F>(f: &mut F, a: f64, b: f64, opts: &QuadOpts<'_>) -> Result<QuadOutcome, QuadError>
where
    F: FnMut(f64) -> f64,
{
    let flip = b < a;
    let (lo, hi) = if flip { (b, a) } else { (a, b) };

    let mut outcome = match opts.points {
        None => qagse(f, lo, hi, opts.epsabs, opts.epsrel, opts.limit)?,
        Some(points) => {
            let interior = filter_points(points, lo, hi);
            qagpe(f, lo, hi, &interior, opts.epsabs, opts.epsrel, opts.limit)?
        }
    };
    if flip {
        outcome.value = -outcome.value;
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Half of QUADPACK's own reference tolerance discussion in one
    /// number: the rules are exact to a few ulp of the integrand's scale,
    /// so anything beyond this on a *smooth* problem is a translation bug
    /// rather than rounding.
    const SMOOTH_RTOL: f64 = 1e-14;

    fn assert_close(got: f64, want: f64, rtol: f64, what: &str) {
        let err = (got - want).abs() / want.abs();
        assert!(
            err <= rtol,
            "{what}: got {got:e}, want {want:e}, relative error {err:e} > {rtol:e}"
        );
    }

    // -- the rules themselves ------------------------------------------

    /// A Gauss–Kronrod rule is pinned by its degree of exactness, and
    /// nothing else about the tables is checked here on purpose: a wrong
    /// digit in `XGK`/`WGK` breaks exactness at some degree, while a
    /// spot-check against one integral could be passed by a rule that is
    /// merely close.
    ///
    /// The 15-point Kronrod rule is exact through degree 22 (`3n + 1` for
    /// the odd `n = 7` Gauss rule it extends), the 21-point through degree
    /// 31 (`3n + 1` for `n = 10`).
    enum Rule {
        K15,
        K21,
    }

    fn assert_exact_through_degree(rule: &Rule, degree: i32, name: &str) {
        for k in 0..=degree {
            // ∫_{-1}^{1} x^k dx = 2/(k+1) for even k, 0 for odd k.
            let mut integrand = move |x: f64| x.powi(k);
            let got = match rule {
                Rule::K15 => qk15(&mut integrand, -1.0, 1.0),
                Rule::K21 => qk21(&mut integrand, -1.0, 1.0),
            }
            .result;
            if k % 2 == 0 {
                let want = 2.0 / f64::from(k + 1);
                assert_close(got, want, 1e-14, &format!("{name} on x^{k}"));
            } else {
                assert!(
                    got.abs() < 1e-15,
                    "{name} on x^{k}: got {got:e}, want 0 by symmetry"
                );
            }
        }
    }

    #[test]
    fn qk15_is_exact_through_degree_22() {
        assert_exact_through_degree(&Rule::K15, 22, "qk15");
    }

    #[test]
    fn qk21_is_exact_through_degree_31() {
        assert_exact_through_degree(&Rule::K21, 31, "qk21");
    }

    #[test]
    fn the_rules_are_not_exact_one_degree_further() {
        // The complement of the two tests above: without this, a rule
        // built from *more* points than intended (or from the wrong
        // family) would pass them. x^24 and x^32 are the first even
        // degrees past each rule's guarantee.
        //
        // The thresholds sit an order of magnitude below the measured
        // errors (5.7e-9 for qk15 on x^24, 4.4e-12 for qk21 on x^32) and
        // several orders above the ~5e-17 rounding floor these same rules
        // show at their last exact degree, so the assertion is about the
        // jump across the boundary rather than about either number.
        let mut p24 = |x: f64| x.powi(24);
        let err15 = (qk15(&mut p24, -1.0, 1.0).result - 2.0 / 25.0).abs();
        assert!(err15 > 1e-10, "qk15 should not integrate x^24 exactly");

        let mut p32 = |x: f64| x.powi(32);
        let err21 = (qk21(&mut p32, -1.0, 1.0).result - 2.0 / 33.0).abs();
        assert!(err21 > 1e-13, "qk21 should not integrate x^32 exactly");
    }

    #[test]
    fn the_rules_report_the_integral_of_the_absolute_value() {
        // `resabs` is the rule applied to |f|, which is what makes it
        // differ from `result` for a sign-changing integrand. It is not
        // ∫|f|: |x| is not a polynomial, so the 21-point rule misses the
        // true value 1 by 3.7e-3 here. Pinning it against the rule run on
        // |x| states the actual invariant, exactly.
        let mut f = |x: f64| x;
        let out = qk21(&mut f, -1.0, 1.0);
        assert!(out.result.abs() < 1e-16, "∫x over [-1,1] is zero");

        let mut abs_f = |x: f64| x.abs();
        assert_eq!(out.resabs, qk21(&mut abs_f, -1.0, 1.0).result);
        // …and that is a genuinely different number from the truth.
        assert!((out.resabs - 1.0).abs() > 1e-3);
    }

    #[test]
    fn the_rules_agree_on_a_smooth_integrand() {
        // Two independent rules, one integrand: this catches a table
        // error that happened to preserve the degree of exactness of one
        // of them. ∫_0^1 exp(x) dx = e − 1.
        let want = std::f64::consts::E - 1.0;
        let mut f = |x: f64| x.exp();
        assert_close(qk15(&mut f, 0.0, 1.0).result, want, SMOOTH_RTOL, "qk15 exp");
        assert_close(qk21(&mut f, 0.0, 1.0).result, want, SMOOTH_RTOL, "qk21 exp");
    }

    #[test]
    fn a_reversed_interval_negates_the_rule() {
        let mut f = |x: f64| x.exp();
        let forward = qk21(&mut f, 0.0, 1.0);
        let backward = qk21(&mut f, 1.0, 0.0);
        assert_close(
            backward.result,
            -forward.result,
            SMOOTH_RTOL,
            "reversed qk21",
        );
        // resabs and resasc use |hlgth|, so they do not flip sign.
        assert_close(backward.resabs, forward.resabs, SMOOTH_RTOL, "resabs");
    }

    // -- qelg ----------------------------------------------------------

    #[test]
    fn qelg_accelerates_a_geometric_sequence() {
        // The partial sums of Σ (1/2)^k converge to 2 linearly. Wynn's
        // ε-algorithm is exact on a single geometric tail, so three
        // elements are enough to land on the limit — which is a much
        // sharper statement than "it got closer".
        let mut epstab = [0.0_f64; 53];
        let mut res3la = [0.0_f64; 4];
        let mut nres = 0_usize;
        let partial = [1.0, 1.5, 1.75];
        let mut result = f64::NAN;
        for (i, &s) in partial.iter().enumerate() {
            let mut n = i + 1;
            epstab[n] = s;
            let (r, _e) = qelg(&mut n, &mut epstab, &mut res3la, &mut nres);
            result = r;
        }
        assert_close(result, 2.0, 1e-15, "qelg on a geometric sequence");
    }

    #[test]
    fn qelg_returns_the_last_element_below_three_entries() {
        let mut epstab = [0.0_f64; 53];
        let mut res3la = [0.0_f64; 4];
        let mut nres = 0_usize;
        epstab[1] = 3.25;
        let mut n = 1_usize;
        let (result, abserr) = qelg(&mut n, &mut epstab, &mut res3la, &mut nres);
        assert_eq!(result, 3.25);
        // abserr starts at oflow and is only floored by 5·epmach·|result|.
        assert_eq!(abserr, OFLOW);
        assert_eq!(nres, 1);
    }

    // -- qags / qagp ---------------------------------------------------

    /// QUADPACK's own first reference problem: ∫_0^1 x^α · ln(1/x) dx,
    /// exactly 1/(α+1)², an endpoint-singular integrand that only the
    /// extrapolation handles well. (Piessens et al., §1.2.)
    #[test]
    fn qags_solves_the_quadpack_log_singularity() {
        for alpha in [-0.9_f64, -0.5, 0.0, 1.0, 2.0] {
            let mut f = |x: f64| {
                if x <= 0.0 {
                    0.0
                } else {
                    x.powf(alpha) * (1.0 / x).ln()
                }
            };
            let out = qagse(&mut f, 0.0, 1.0, 1e-10, 1e-10, DEFAULT_LIMIT).unwrap();
            let want = 1.0 / (alpha + 1.0).powi(2);
            assert_eq!(out.ier, Ier::Ok, "alpha = {alpha}");
            assert_close(out.value, want, 1e-10, &format!("log singularity {alpha}"));
        }
    }

    /// The interior algebraic singularity QUADPACK ships `qagp` for:
    /// ∫_0^1 |x − 1/3|^(−1/2) dx = 2(√(1/3) + √(2/3)).
    #[test]
    fn qagp_solves_an_interior_algebraic_singularity() {
        let c = 1.0 / 3.0;
        let mut f = |x: f64| {
            let d = (x - c).abs();
            if d == 0.0 { 0.0 } else { d.powf(-0.5) }
        };
        let want = 2.0 * (c.sqrt() + (1.0 - c).sqrt());
        let out = qagpe(&mut f, 0.0, 1.0, &[c], 1e-10, 1e-10, DEFAULT_LIMIT).unwrap();
        assert_eq!(out.ier, Ier::Ok);
        assert_close(out.value, want, 1e-9, "interior singularity with a point");
    }

    #[test]
    fn qagp_with_no_break_points_still_integrates() {
        // The live `points=[-1, 1]` shape after scipy's filtering: zero
        // interior break points, so `nint == 1` and `qagpe` runs its
        // single-interval path.
        let mut f = |x: f64| x * x;
        let out = qagpe(&mut f, -1.0, 1.0, &[], 1e-10, 1e-5, DEFAULT_LIMIT).unwrap();
        assert_eq!(out.ier, Ier::Ok);
        assert_close(out.value, 2.0 / 3.0, SMOOTH_RTOL, "x^2 with no points");
        assert_eq!(out.last, 1);
        assert_eq!(out.neval, 21);
    }

    #[test]
    fn qagp_sorts_unsorted_break_points() {
        let mut f = |x: f64| (x - 1.0).abs().powf(-0.5) + (x - 2.0).abs().powf(-0.5);
        let sorted = qagpe(&mut f, 0.0, 3.0, &[1.0, 2.0], 1e-10, 1e-10, 50).unwrap();
        let mut g = |x: f64| (x - 1.0).abs().powf(-0.5) + (x - 2.0).abs().powf(-0.5);
        let unsorted = qagpe(&mut g, 0.0, 3.0, &[2.0, 1.0], 1e-10, 1e-10, 50).unwrap();
        assert_eq!(sorted.value, unsorted.value);
        assert_eq!(sorted.neval, unsorted.neval);
    }

    #[test]
    fn a_reversed_interval_negates_the_integral() {
        let mut f = |x: f64| x.exp();
        let forward = quad(&mut f, 0.0, 1.0, &QuadOpts::default()).unwrap();
        let mut g = |x: f64| x.exp();
        let backward = quad(&mut g, 1.0, 0.0, &QuadOpts::default()).unwrap();
        assert_eq!(backward.value, -forward.value);

        // And with break points, where `qagpe`'s own `sign` handling is
        // never exercised through `quad` because the flip happens first.
        let pts = [0.5_f64];
        let opts = QuadOpts {
            points: Some(&pts),
            ..QuadOpts::default()
        };
        let mut h = |x: f64| x.exp();
        let fwd = quad(&mut h, 0.0, 1.0, &opts).unwrap();
        let mut k = |x: f64| x.exp();
        let bwd = quad(&mut k, 1.0, 0.0, &opts).unwrap();
        assert_eq!(bwd.value, -fwd.value);
    }

    #[test]
    fn qagpe_handles_a_reversed_interval_itself() {
        // `quad` never lets this happen, but `qagpe` is public and the
        // Fortran's `sign` branch is real code.
        let mut f = |x: f64| x.exp();
        let out = qagpe(&mut f, 1.0, 0.0, &[], 1e-10, 1e-10, 50).unwrap();
        assert_close(
            out.value,
            -(std::f64::consts::E - 1.0),
            1e-13,
            "reversed qagpe",
        );
    }

    // -- the scipy-shaped preprocessing --------------------------------

    #[test]
    fn filter_points_matches_scipys_three_lines() {
        // sort + dedup + strictly interior.
        assert_eq!(filter_points(&[2.0, 1.0, 2.0], 0.0, 3.0), vec![1.0, 2.0]);
        // endpoint-coincident points are dropped: the live
        // `points=[-1, 1]` case.
        assert_eq!(filter_points(&[-1.0, 1.0], -1.0, 1.0), Vec::<f64>::new());
        // out-of-interval points are dropped: the live heavy-mediator
        // thermal case.
        assert_eq!(filter_points(&[-5.0, 1.0, 9.0], 0.0, 3.0), vec![1.0]);
        // NaN loses every comparison, so it never survives the interior
        // test.
        assert_eq!(filter_points(&[f64::NAN, 1.0], 0.0, 3.0), vec![1.0]);
        // Signed zeros compare equal and collapse to the one that sorts
        // first.
        assert_eq!(filter_points(&[0.0, -0.0], -1.0, 1.0), vec![-0.0]);
        assert!(filter_points(&[0.0, -0.0], -1.0, 1.0)[0].is_sign_negative());
    }

    #[test]
    fn points_some_but_empty_still_selects_qagpe() {
        // The distinction that makes the live `points=[-1, 1]` calls run
        // `qagpe` rather than `qagse`, since scipy dispatches on
        // `points is None` before it filters. On this integrand the two
        // routines happen to agree; the observable difference is which
        // code path runs, so this test pins the neval bookkeeping that
        // differs between them.
        let empty: [f64; 0] = [];
        let opts = QuadOpts {
            epsabs: 1e-10,
            epsrel: 1e-5,
            points: Some(&empty),
            ..QuadOpts::default()
        };
        let mut f = |x: f64| x * x;
        let with_points = quad(&mut f, -1.0, 1.0, &opts).unwrap();
        let mut g = |x: f64| x * x;
        let without = quad(
            &mut g,
            -1.0,
            1.0,
            &QuadOpts {
                epsabs: 1e-10,
                epsrel: 1e-5,
                ..QuadOpts::default()
            },
        )
        .unwrap();
        assert_eq!(with_points.value, without.value);
        // qagpe counts 21·nint; qagse counts 42·last − 21.
        assert_eq!(with_points.neval, 21);
        assert_eq!(without.neval, 21);
    }

    // -- error and abnormal-termination behavior -----------------------

    #[test]
    fn invalid_tolerances_are_an_error_not_a_panic() {
        let mut f = |x: f64| x;
        assert_eq!(
            quad(
                &mut f,
                0.0,
                1.0,
                &QuadOpts {
                    epsabs: 0.0,
                    epsrel: 0.0,
                    ..QuadOpts::default()
                }
            ),
            Err(QuadError::ToleranceUnachievable)
        );
    }

    #[test]
    fn a_limit_below_one_is_an_error() {
        let mut f = |x: f64| x;
        assert_eq!(
            quad(
                &mut f,
                0.0,
                1.0,
                &QuadOpts {
                    limit: 0,
                    ..QuadOpts::default()
                }
            ),
            Err(QuadError::LimitTooSmall { limit: 0, npts: 0 })
        );
    }

    #[test]
    fn too_many_break_points_for_the_limit_is_an_error() {
        let pts: Vec<f64> = (1..=5).map(f64::from).collect();
        let mut f = |x: f64| x;
        assert_eq!(
            quad(
                &mut f,
                0.0,
                6.0,
                &QuadOpts {
                    limit: 5,
                    points: Some(&pts),
                    ..QuadOpts::default()
                }
            ),
            Err(QuadError::LimitTooSmall { limit: 5, npts: 5 })
        );
    }

    #[test]
    fn a_break_point_outside_the_interval_reaches_qagpe_as_an_error() {
        let mut f = |x: f64| x;
        assert_eq!(
            qagpe(&mut f, 0.0, 1.0, &[7.0], 1e-8, 1e-8, 50),
            Err(QuadError::BreakpointOutsideInterval)
        );
        // …and cannot reach it through `quad`, which filters first.
        let pts = [7.0_f64];
        let mut g = |x: f64| x;
        let out = quad(
            &mut g,
            0.0,
            1.0,
            &QuadOpts {
                points: Some(&pts),
                ..QuadOpts::default()
            },
        )
        .unwrap();
        assert_close(out.value, 0.5, SMOOTH_RTOL, "filtered break point");
    }

    #[test]
    fn hitting_the_subdivision_limit_reports_it_and_still_returns() {
        // A genuinely divergent integral: ∫_0^1 1/x dx. scipy warns and
        // hands back the partial sum, and so must this.
        let mut f = |x: f64| if x == 0.0 { 0.0 } else { 1.0 / x };
        let out = qagse(&mut f, 0.0, 1.0, 1e-10, 1e-10, 10).unwrap();
        assert_ne!(out.ier, Ier::Ok);
        assert!(out.value.is_finite());
        assert!(out.last <= 10, "last = {} exceeded the limit", out.last);
    }

    #[test]
    fn a_nan_integrand_does_not_panic() {
        let mut f = |_x: f64| f64::NAN;
        let out = qagse(&mut f, 0.0, 1.0, 1e-10, 1e-10, DEFAULT_LIMIT).unwrap();
        assert!(out.value.is_nan() || out.value == 0.0);
    }

    #[test]
    fn a_zero_width_interval_integrates_to_zero() {
        let mut f = |x: f64| x.exp();
        let out = quad(&mut f, 1.5, 1.5, &QuadOpts::default()).unwrap();
        assert_eq!(out.value, 0.0);
        assert_eq!(out.ier, Ier::Ok);
    }

    #[test]
    fn ier_codes_round_trip() {
        for code in 0..=6 {
            assert_eq!(Ier::from_code(code).code(), code);
        }
    }
}
