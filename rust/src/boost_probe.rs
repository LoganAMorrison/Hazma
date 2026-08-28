//! `hazma._core.boost` — the Python-visible half of [`crate::boost`].
//!
//! Registration only, like the per-domain submodules. This is a **test
//! surface, not physics**: the kernels Phase 04 ports call
//! [`crate::boost`] directly in Rust, and nothing under `hazma/` imports
//! this module. It exists because Task 3.4's oracle lives in Python: it
//! was the Cython twin itself, reached through
//! `hazma._utils.boost.__pyx_capi__`, and since Phase 06 Task 6.4 deleted
//! that extension it is the pair of reference implementations
//! `test/test_core_boost.py` carries. Either way that module needs a way
//! to put the same arguments through both. Same role `special_probe`
//! plays for the cephes shim and
//! `quad_probe` for the QUADPACK port, and it joins them in
//! `test/parity/cases.py`'s `_CORE_TEST_ONLY_MODULES` rather than
//! widening that exemption.
//!
//! Each wrapper maps over the argument its Cython call sites sweep — the
//! parent energy for the boost parameters, the product energy for the
//! line, the lab-frame energy for the tabulated integral — through
//! [`crate::dispatch::map_unary`], so a sweep is one array call rather
//! than a Python loop.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::boost;
use crate::dispatch::map_unary;

/// The velocity of a particle — `β = sqrt(1 − (m/E)²)`, in units of `c`.
///
/// `energy` is the mapped argument and `mass` a scalar, matching how the
/// spectra sweep a parent energy at fixed mass.
#[pyfunction]
#[pyo3(text_signature = "(energy, mass)")]
fn boost_beta(energy: &Bound<'_, PyAny>, mass: f64) -> PyResult<Py<PyAny>> {
    map_unary(energy, "Parent energies", |value| {
        boost::boost_beta(value, mass)
    })
}

/// The Lorentz factor of a particle — `γ = E / m`.
#[pyfunction]
#[pyo3(text_signature = "(energy, mass)")]
fn boost_gamma(energy: &Bound<'_, PyAny>, mass: f64) -> PyResult<Py<PyAny>> {
    map_unary(energy, "Parent energies", |value| {
        boost::boost_gamma(value, mass)
    })
}

/// Boost a rest-frame line `δ(E − e0)` into the lab frame, in MeV⁻¹.
///
/// `e` — the product's lab-frame energy — is the mapped argument, which
/// is the argument every call site sweeps. The parameter order matches
/// the Cython's `boost_delta_function(e0, e, m, beta)`.
#[pyfunction]
#[pyo3(text_signature = "(e0, e, m, beta)")]
fn boost_delta_function(e0: f64, e: &Bound<'_, PyAny>, m: f64, beta: f64) -> PyResult<Py<PyAny>> {
    map_unary(e, "Product energies", |value| {
        boost::boost_delta_function(e0, value, m, beta)
    })
}

/// Boost a tabulated rest-frame spectrum into the lab frame, in MeV⁻¹.
///
/// `photon_energy` is the mapped argument; `x` and `y` are the table's
/// 1-D `float64` columns.
///
/// # Errors
///
/// Raises `ValueError` for the guards the Cython states as `assert`s —
/// `beta` outside `(0, 1)` and columns of different lengths — plus the
/// empty table the Cython leaves undefined
/// (`projects/cython-to-rust/rules.md` rule 9).
#[pyfunction]
#[pyo3(text_signature = "(photon_energy, beta, x, y)")]
fn boost_integrate_linear_interp(
    photon_energy: &Bound<'_, PyAny>,
    beta: f64,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyAny>> {
    // `to_vec`, not `as_slice`: the live tables are rows of a transposed
    // `np.loadtxt` result, so they are strided views rather than
    // contiguous buffers and `as_slice` refuses them. Copying is the
    // right answer for a test surface — a Phase 04 kernel owns its table
    // in Rust and never sees a NumPy stride at all.
    let x = x.as_array().to_vec();
    let y = y.as_array().to_vec();
    // Validate once here rather than per element: the guards depend only
    // on `beta` and the table, so a swept array either fails everywhere
    // or nowhere, and mapping the error out of the kernel would mean
    // deciding what value a failed element takes.
    if let Err(err) = boost::boost_integrate_linear_interp(1.0, beta, &x, &y) {
        return Err(PyValueError::new_err(err.to_string()));
    }
    map_unary(photon_energy, "Photon energies", |value| {
        boost::boost_integrate_linear_interp(value, beta, &x, &y)
            .expect("the guards depend only on beta and the table, both just checked")
    })
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(boost_beta, module)?)?;
    module.add_function(wrap_pyfunction!(boost_gamma, module)?)?;
    module.add_function(wrap_pyfunction!(boost_delta_function, module)?)?;
    module.add_function(wrap_pyfunction!(boost_integrate_linear_interp, module)?)?;
    Ok(())
}
