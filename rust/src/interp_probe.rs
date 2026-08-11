//! `hazma._core.interp` — the Python-visible half of [`crate::interp`].
//!
//! Registration only, like the per-domain submodules. This is a **test
//! surface, not physics**: the kernels Phase 04 ports call
//! [`crate::interp::interp`] directly in Rust, and nothing under `hazma/`
//! imports this module. It exists because Task 3.4's exit criterion is
//! stated against `np.interp` — an oracle that lives in Python — so
//! `test/test_core_interp.py` needs a way to put the same abscissae
//! through both. Same role `special_probe` plays for the cephes shim and
//! `quad_probe` for the QUADPACK port, and it joins them in
//! `test/parity/cases.py`'s `_CORE_TEST_ONLY_MODULES` rather than
//! widening that exemption.
//!
//! The abscissa goes through [`crate::dispatch::map_unary`], so a sweep
//! runs as one array call rather than a Python loop over tens of
//! thousands of points — the kind of test that otherwise gets quietly
//! trimmed later.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dispatch::map_unary;
use crate::interp;

/// Linear interpolation on an ascending grid — `numpy.interp(x, xp, fp)`.
///
/// `x` follows the usual dispatch contract (scalar or 1-D `float64`
/// array); `xp` and `fp` must both be 1-D `float64` arrays.
///
/// # Errors
///
/// Raises `ValueError`, with NumPy's own wording, for an empty grid or
/// for `xp` and `fp` of different lengths. Doing it here rather than in
/// the kernel keeps [`crate::interp::interp`]'s asserts unreachable from
/// Python.
#[pyfunction]
#[pyo3(name = "interp")]
#[pyo3(text_signature = "(x, xp, fp)")]
fn interp_py(
    x: &Bound<'_, PyAny>,
    xp: PyReadonlyArray1<'_, f64>,
    fp: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyAny>> {
    // `to_vec`, not `as_slice`: the live tables are rows of a transposed
    // `np.loadtxt` result, so they are strided views rather than
    // contiguous buffers and `as_slice` refuses them. Copying is the
    // right answer for a test surface — a Phase 04 kernel owns its table
    // in Rust and never sees a NumPy stride at all.
    let xp = xp.as_array().to_vec();
    let fp = fp.as_array().to_vec();
    if xp.is_empty() {
        return Err(PyValueError::new_err("array of sample points is empty"));
    }
    if xp.len() != fp.len() {
        return Err(PyValueError::new_err(
            "fp and xp are not of the same length.",
        ));
    }
    map_unary(x, "Interpolation abscissae", |value| {
        interp::interp(value, &xp, &fp)
    })
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(interp_py, module)?)?;
    Ok(())
}
