//! `hazma._core.special` — the Python-visible half of [`crate::special`].
//!
//! Registration only, like the per-domain submodules. These three
//! functions are a **test surface, not physics**: the kernels Phases
//! 04–06 port call `crate::special` directly in Rust, and nothing under
//! `hazma/` imports this module. It exists because Task 3.2's exit
//! criteria are stated against `scipy.special` — an oracle that lives in
//! Python — so `test/test_core_special.py` needs a way to call the Rust
//! side on the same grid. Same role `lib.rs`'s `roundtrip` plays for the
//! dispatch contract.
//!
//! Each wrapper goes through [`crate::dispatch::map_unary`], so all
//! three accept a scalar or a 1-D `float64` array and follow the same
//! contract as every ported entry point. That is not free array support
//! for its own sake: sweeping thousands of points against scipy through
//! a Python-level loop is exactly the shape of test that gets quietly
//! trimmed to a dozen points later.

use pyo3::prelude::*;

use crate::dispatch::map_unary;
use crate::special;

/// Spence's integral in scipy's convention — `Li₂(1 − x)`.
///
/// See [`special::spence`]; the convention, not the values, is the thing
/// this binding exists to pin.
#[pyfunction]
#[pyo3(text_signature = "(x)")]
fn spence(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_unary(x, "Spence arguments", special::spence)
}

/// Modified Bessel function of the second kind, order one — `K₁(x)`.
#[pyfunction]
#[pyo3(text_signature = "(x)")]
fn bessel_k1(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_unary(x, "Bessel arguments", special::bessel_k1)
}

/// Modified Bessel function of the second kind, integer order —
/// `Kₙ(x)`.
///
/// `n` is a scalar order and `x` the mapped argument, matching
/// `scipy.special.kn(n, x)`.
#[pyfunction]
#[pyo3(text_signature = "(n, x)")]
fn bessel_kn(n: i32, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_unary(x, "Bessel arguments", |value| special::bessel_kn(n, value))
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(spence, module)?)?;
    module.add_function(wrap_pyfunction!(bessel_k1, module)?)?;
    module.add_function(wrap_pyfunction!(bessel_kn, module)?)?;
    Ok(())
}
