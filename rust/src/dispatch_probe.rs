//! `hazma._core.dispatch` — the Python-visible half of [`crate::dispatch`].
//!
//! Registration only, like the per-domain submodules. This is a **test
//! surface, not physics**: Phases 04–06 call [`crate::dispatch`]'s three
//! helpers from their own `#[pyfunction]`s, and nothing under `hazma/`
//! imports this module. It exists because the contract's error text is a
//! *user-visible* part of the public API and the oracle for it is the
//! `.pyx` sources, which live in Python — so `test/test_core_dispatch.py`
//! needs to render every message the tree contains and compare bytes.
//! Same role `special_probe`, `quad_probe`, `interp_probe` and
//! `boost_probe` play for their halves, and it joins them in
//! `test/parity/cases.py`'s `_CORE_TEST_ONLY_MODULES` rather than
//! widening that exemption.
//!
//! Every probe here takes the `quantity` wording as an argument, which
//! the top-level `roundtrip` (Phase 02's scaffold probe, wording fixed to
//! `"Input values"`) cannot. That is the whole reason this module exists
//! rather than two more functions beside it: a test that checks
//! `"Photon energies must be 0 or 1-dimensional."` byte for byte has to be
//! able to ask for that wording.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::dispatch::{map_flavors, map_unary, require_vector};
use crate::kernels;

/// [`crate::dispatch::map_unary`] over the identity, with caller-chosen
/// error wording.
#[pyfunction]
#[pyo3(name = "roundtrip")]
#[pyo3(text_signature = "(x, quantity)")]
fn roundtrip_py(x: &Bound<'_, PyAny>, quantity: &str) -> PyResult<Py<PyAny>> {
    map_unary(x, quantity, kernels::roundtrip)
}

/// [`crate::dispatch::map_flavors`] over
/// [`crate::kernels::roundtrip_flavors`], with caller-chosen error wording.
#[pyfunction]
#[pyo3(name = "roundtrip_flavors")]
#[pyo3(text_signature = "(x, quantity)")]
fn roundtrip_flavors_py(x: &Bound<'_, PyAny>, quantity: &str) -> PyResult<Py<PyAny>> {
    map_flavors(x, quantity, kernels::roundtrip_flavors)
}

/// [`crate::dispatch::require_vector`], returned as a fresh array.
///
/// Handing the extracted values back is what makes the extraction
/// checkable: length, order and bit patterns all have to survive the trip
/// through `Vec<f64>`, and the result must not alias the caller's buffer.
#[pyfunction]
#[pyo3(name = "roundtrip_vector")]
#[pyo3(text_signature = "(values, quantity)")]
fn roundtrip_vector_py<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    quantity: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(require_vector(values, quantity)?.into_pyarray(py))
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(roundtrip_py, module)?)?;
    module.add_function(wrap_pyfunction!(roundtrip_flavors_py, module)?)?;
    module.add_function(wrap_pyfunction!(roundtrip_vector_py, module)?)?;
    Ok(())
}
