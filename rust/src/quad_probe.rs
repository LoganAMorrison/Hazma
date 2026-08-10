//! `hazma._core.quad` — the Python-visible half of [`crate::quad`].
//!
//! Registration only, like the per-domain submodules. This is a **test
//! surface, not physics**: the kernels Phases 04–06 port call
//! [`crate::quad::quad`] directly in Rust with a Rust closure, and nothing
//! under `hazma/` imports this module. It exists because Task 3.3's exit
//! criteria are stated against `scipy.integrate.quad` — an oracle that
//! lives in Python — so `test/test_core_quad.py` needs a way to put the
//! *same* integrand through both. Same role `lib.rs`'s `roundtrip` plays
//! for the dispatch contract, and `special_probe` for the cephes shim.
//!
//! Taking a Python callable is the point rather than a convenience. A
//! probe that exposed a fixed menu of Rust integrands would compare a Rust
//! integrand against a Python one and attribute the difference to the
//! quadrature; with a callback, the integrand is byte-identical on both
//! sides and every remaining difference is the algorithm. It is also what
//! the Cython does today — `scipy.integrate.quad` re-enters Python once
//! per node there too.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;

use crate::quad::{self, QuadOpts};

/// Integrate `f` over `[a, b]`, mirroring `scipy.integrate.quad`.
///
/// Returns `(value, abserr, neval, last, ier)`. `ier` is QUADPACK's raw
/// termination flag in scipy's numbering, returned rather than warned
/// about so a test can assert on it; scipy's own `full_output` dictionary
/// reports the same number.
///
/// Raises `ValueError` for exactly the inputs scipy raises `ValueError`
/// for. An exception from `f` propagates unchanged: it is recorded on the
/// first occurrence, the integrand short-circuits to `NaN` for the rest of
/// the run so no further Python code executes, and the original error is
/// re-raised once QUADPACK returns.
#[pyfunction]
#[pyo3(name = "quad")]
#[pyo3(signature = (f, a, b, epsabs=quad::DEFAULT_EPSABS, epsrel=quad::DEFAULT_EPSREL, limit=quad::DEFAULT_LIMIT, points=None))]
#[pyo3(text_signature = "(f, a, b, epsabs=1.49e-8, epsrel=1.49e-8, limit=50, points=None)")]
#[allow(
    clippy::too_many_arguments,
    reason = "one parameter per `scipy.integrate.quad` keyword this port supports"
)]
fn quad_py(
    f: &Bound<'_, PyAny>,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
    points: Option<Vec<f64>>,
) -> PyResult<(f64, f64, usize, usize, i32)> {
    let failure: RefCell<Option<PyErr>> = RefCell::new(None);

    let mut integrand = |x: f64| -> f64 {
        if failure.borrow().is_some() {
            return f64::NAN;
        }
        match f.call1((x,)).and_then(|v| v.extract::<f64>()) {
            Ok(value) => value,
            Err(err) => {
                *failure.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };

    let opts = QuadOpts {
        epsabs,
        epsrel,
        limit,
        points: points.as_deref(),
    };
    let outcome = quad::quad(&mut integrand, a, b, &opts);

    if let Some(err) = failure.into_inner() {
        return Err(err);
    }
    let outcome = outcome.map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        outcome.value,
        outcome.abserr,
        outcome.neval,
        outcome.last,
        outcome.ier.code(),
    ))
}

/// Apply the 15-point Gauss–Kronrod rule once, without subdivision.
///
/// Returns `(result, abserr, resabs, resasc)`. No scipy entry point
/// exposes a bare rule, so the oracle for this one is a reference value
/// computed in the test rather than scipy.
#[pyfunction]
#[pyo3(text_signature = "(f, a, b)")]
fn qk15(f: &Bound<'_, PyAny>, a: f64, b: f64) -> PyResult<(f64, f64, f64, f64)> {
    apply_rule(f, a, b, Rule::K15)
}

/// Apply the 21-point Gauss–Kronrod rule once, without subdivision.
///
/// Returns `(result, abserr, resabs, resasc)`. This is the rule both
/// `qags` and `qagp` run on, so a single-interval `quad` result and this
/// must agree bit for bit.
#[pyfunction]
#[pyo3(text_signature = "(f, a, b)")]
fn qk21(f: &Bound<'_, PyAny>, a: f64, b: f64) -> PyResult<(f64, f64, f64, f64)> {
    apply_rule(f, a, b, Rule::K21)
}

/// Which Gauss–Kronrod rule [`apply_rule`] should run.
///
/// A two-variant enum rather than a function pointer: `qk15` and `qk21`
/// are generic over the integrand, and naming them as `fn` pointers over
/// `&mut dyn FnMut` needs a higher-ranked lifetime the monomorphized item
/// cannot supply.
enum Rule {
    K15,
    K21,
}

/// Shared body of [`qk15`] and [`qk21`]: wrap the Python callable, run the
/// rule, and re-raise any exception it produced.
fn apply_rule(f: &Bound<'_, PyAny>, a: f64, b: f64, rule: Rule) -> PyResult<(f64, f64, f64, f64)> {
    let failure: RefCell<Option<PyErr>> = RefCell::new(None);
    let mut integrand = |x: f64| -> f64 {
        if failure.borrow().is_some() {
            return f64::NAN;
        }
        match f.call1((x,)).and_then(|v| v.extract::<f64>()) {
            Ok(value) => value,
            Err(err) => {
                *failure.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let out = match rule {
        Rule::K15 => quad::qk15(&mut integrand, a, b),
        Rule::K21 => quad::qk21(&mut integrand, a, b),
    };
    if let Some(err) = failure.into_inner() {
        return Err(err);
    }
    Ok((out.result, out.abserr, out.resabs, out.resasc))
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(quad_py, module)?)?;
    module.add_function(wrap_pyfunction!(qk15, module)?)?;
    module.add_function(wrap_pyfunction!(qk21, module)?)?;
    Ok(())
}
