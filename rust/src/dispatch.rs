//! The PyO3 boundary: one shape for every public entry point.
//!
//! `projects/cython-to-rust/references/numerics-replacements.md`
//! ("Entry-point dispatch contract") fixes the contract the Cython layer
//! already exposes and the port must keep:
//!
//! * a Python `float`, a NumPy scalar, or a 0-d array returns a Python
//!   `float`;
//! * a 1-D `float64` array returns a fresh 1-D `float64` array of the
//!   same length;
//! * anything else raises `ValueError`.
//!
//! Implemented once here so no kernel re-derives it. Phases 03–06 call
//! [`map_unary`] rather than touching PyO3 in a kernel module.

use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyFloat;

/// Apply `kernel` elementwise to `obj` under the dispatch contract.
///
/// `quantity` names the argument in every error message — pass the same
/// wording the Cython twin used (e.g. `"Photon energies"`), so the port
/// does not silently reword a user-visible exception.
///
/// # Errors
///
/// Returns `ValueError` if `obj` is an array of more than one dimension,
/// an array whose dtype is not `float64`, or anything that is neither a
/// real scalar nor a NumPy array.
pub fn map_unary<F>(obj: &Bound<'_, PyAny>, quantity: &str, kernel: F) -> PyResult<Py<PyAny>>
where
    F: Fn(f64) -> f64,
{
    let py = obj.py();

    // Plain floats first, and not only for speed. `cast::<PyUntypedArray>`
    // reaches for NumPy's array-API capsule, and the `numpy` crate
    // *panics* rather than raising if NumPy cannot be imported — so
    // without this arm `roundtrip(1.5)` would abort in an interpreter
    // that has no NumPy. hazma depends on NumPy at runtime, so that is a
    // latent trap rather than a live bug, but a scalar has no business
    // touching NumPy at all. `np.float64` is a subclass of `float` and
    // takes this path too; `np.float32` is not, and falls through.
    if let Ok(value) = obj.cast::<PyFloat>() {
        return Ok(PyFloat::new(py, kernel(value.value())).into_any().unbind());
    }

    // Then arrays. A NumPy scalar that is not a `float` subclass is not an
    // ndarray either, and falls through to the extract arm at the end.
    if let Ok(array) = obj.cast::<PyUntypedArray>() {
        let ndim = array.ndim();
        if ndim > 1 {
            return Err(PyValueError::new_err(format!(
                "{quantity} must be 0 or 1-dimensional."
            )));
        }
        // Extracting the read-only view is also the dtype check: a
        // non-float64 array fails here. Mapping the failure to
        // ValueError (PyO3 would raise TypeError) keeps one exception
        // type across the whole contract.
        let readonly: PyReadonlyArrayDyn<'_, f64> = obj.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "{quantity} must be a float64 array; got dtype {}.",
                array.dtype()
            ))
        })?;
        let view = readonly.as_array();

        if ndim == 0 {
            let value = *view
                .first()
                .expect("a 0-dimensional array always holds exactly one element");
            return Ok(PyFloat::new(py, kernel(value)).into_any().unbind());
        }

        // A fresh Vec, never a view onto the caller's buffer: the result
        // must not alias the input, and `view` may be non-contiguous.
        let mapped: Vec<f64> = view.iter().map(|&x| kernel(x)).collect();
        return Ok(mapped.into_pyarray(py).into_any().unbind());
    }

    if let Ok(value) = obj.extract::<f64>() {
        return Ok(PyFloat::new(py, kernel(value)).into_any().unbind());
    }

    Err(PyValueError::new_err(format!(
        "{quantity} must be a float or a NumPy array."
    )))
}
