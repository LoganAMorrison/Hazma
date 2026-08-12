//! The PyO3 boundary: one shape for every public entry point.
//!
//! `projects/cython-to-rust/references/numerics-replacements.md`
//! ("Entry-point dispatch contract") fixes the contract the Cython layer
//! exposes and the port keeps:
//!
//! * a Python `float`, a NumPy scalar, or a 0-d array of any numeric
//!   dtype returns a Python `float`;
//! * a 1-D `float64` array, or any sequence that converts to one, returns
//!   a fresh 1-D `float64` array of the same length;
//! * a higher-rank array, a 1-D array that is not `float64`, or a 0-d
//!   array that is not numeric raises `ValueError`;
//! * anything that is neither a real number nor a sequence raises
//!   `TypeError`.
//!
//! [`map_flavors`] is the same contract with the neutrino return shape —
//! a 3-tuple for a scalar, a `(3, N)` array for a grid — and
//! [`require_vector`] is the third live shape, an argument that must be a
//! 1-D array and is never a scalar (`partial_widths`).
//!
//! Implemented once here so no kernel re-derives it. Phases 03–06 call
//! these three rather than touching PyO3 in a kernel module (`rules.md`
//! rule 8).
//!
//! # Relationship to the Cython these replace
//!
//! Task 3.5 measured all 43 surviving top-level `def`s and found four
//! dispatch shapes, not the one the reference described. The rule the
//! port follows, stated once:
//!
//! **Every exception the Cython raises *explicitly* keeps its type; only
//! its `assert`s change type** (`rules.md` rule 9 — today they vanish
//! under `python -O`). So a rank error becomes `ValueError` carrying the
//! assert's own message text verbatim, a dtype error stays `ValueError`,
//! a non-number stays `TypeError`, and `partial_widths`' explicit
//! `raise ValueError` keeps both its type and its wording.
//!
//! Three deliberate widenings come with that, each recorded in the task
//! note and none of which can break a call that works today:
//!
//! * a 0-d array takes the scalar path instead of raising, which is what
//!   the 18 cross-section entry points already do (`.item()`);
//! * a list or tuple is accepted, which the 17 `hasattr`-dispatching entry
//!   points already do (`np.array(...)`) and the cross sections do not;
//! * a rank error names the argument rather than the buffer.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};

/// The number of neutrino flavors a spectrum kernel returns, in the
/// Cython's row order: electron, muon, tau.
const N_FLAVORS: usize = 3;

/// An entry point's dispatched argument: the scalar it was, or a borrowed
/// `float64` view of the grid it was.
enum Argument<'py> {
    Scalar(f64),
    Vector(PyReadonlyArray1<'py, f64>),
}

/// `"<quantity> must be 0 or 1-dimensional."` — byte-identical to the
/// `assert` message of every spectra entry point.
fn dimension_error(quantity: &str) -> PyErr {
    PyValueError::new_err(format!("{quantity} must be 0 or 1-dimensional."))
}

/// `ValueError` naming the dtype that was passed.
///
/// The Cython raises `ValueError` here too, but with Cython's own buffer
/// wording — which is not one string to match: the spectra extensions say
/// `expected 'double'` and the mediator ones `expected 'float64_t'` for
/// the same rejection, and both name C types rather than the dtype. The
/// port states the dtype instead, and keeps the exception type.
fn dtype_error(array: &Bound<'_, PyUntypedArray>, quantity: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{quantity} must be a float64 array; got dtype {}.",
        array.dtype()
    ))
}

/// `TypeError` for an argument that is neither a real number nor a
/// sequence — the type CPython raises today when such a value reaches a
/// `cdef double` parameter.
fn type_error(quantity: &str) -> PyErr {
    PyTypeError::new_err(format!("{quantity} must be a float or a NumPy array."))
}

/// `numpy.asarray(obj)`, for the sequence branch only.
///
/// Mirrors the `np.array(...)` the spectra entry points call before their
/// memoryview cast, which is what makes `dnde_photon([10.0, 20.0], 200.0)`
/// work today. `asarray` rather than `array` because this is only ever
/// reached for something that is *not* already an ndarray, so there is no
/// copy to avoid and no aliasing to create.
fn as_numpy_array<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    obj.py().import("numpy")?.call_method1("asarray", (obj,))
}

/// Whether an array's dtype is one a `float64` can hold: boolean,
/// signed or unsigned integer, or floating point.
///
/// Asked of the *dtype* rather than of `float(array)`, because PyO3's
/// `extract::<f64>` goes through `PyNumber_Float` and NumPy's 0-d
/// `__float__` forwards to the element — so a 0-d `<U4` array holding
/// `"15.0"` extracts as `15.0`. The dtype's `kind` is the only thing that
/// distinguishes "a number NumPy stored narrowly" from "a string that
/// happens to parse".
fn has_numeric_dtype(array: &Bound<'_, PyUntypedArray>) -> PyResult<bool> {
    let kind: String = array.dtype().getattr("kind")?.extract()?;
    Ok(matches!(kind.as_str(), "b" | "i" | "u" | "f"))
}

/// Classify an ndarray by rank, and take its `float64` view.
///
/// A 0-d array is the scalar it holds, whatever its numeric dtype — the
/// cross-section entry points' `.item()`, generalised. The `float64` rule
/// therefore binds 1-D arrays only, which is also what makes
/// `f(np.array(4))` behave like `f(4)` rather than like `f(np.array([4]))`.
fn classify_array<'py>(
    array: &Bound<'py, PyUntypedArray>,
    quantity: &str,
) -> PyResult<Argument<'py>> {
    match array.ndim() {
        0 if !has_numeric_dtype(array)? => Err(dtype_error(array, quantity)),
        0 => array
            .as_any()
            .extract::<f64>()
            .map(Argument::Scalar)
            .map_err(|_| dtype_error(array, quantity)),
        1 => array
            .as_any()
            .extract::<PyReadonlyArray1<'py, f64>>()
            .map(Argument::Vector)
            .map_err(|_| dtype_error(array, quantity)),
        _ => Err(dimension_error(quantity)),
    }
}

/// Apply the dispatch contract to `obj`, without evaluating anything.
///
/// # Errors
///
/// `ValueError` for a rank or dtype violation, `TypeError` for a value
/// that is neither a real number nor a sequence.
fn classify<'py>(obj: &Bound<'py, PyAny>, quantity: &str) -> PyResult<Argument<'py>> {
    // Plain floats first, and not only for speed. `cast::<PyUntypedArray>`
    // reaches for NumPy's array-API capsule, and the `numpy` crate
    // *panics* rather than raising if NumPy cannot be imported — so
    // without this arm `roundtrip(1.5)` would abort in an interpreter
    // that has no NumPy. hazma depends on NumPy at runtime, so that is a
    // latent trap rather than a live bug, but a scalar has no business
    // touching NumPy at all. `np.float64` is a subclass of `float` and
    // takes this path too; `np.float32` is not, and falls through.
    if let Ok(value) = obj.cast::<PyFloat>() {
        return Ok(Argument::Scalar(value.value()));
    }

    // Then arrays, by rank. A NumPy scalar that is not a `float` subclass
    // is not an ndarray either, and falls through to the extract arm.
    if let Ok(array) = obj.cast::<PyUntypedArray>() {
        return classify_array(array, quantity);
    }

    // Then sequences, under the Cython's own `hasattr(x, '__len__')`
    // predicate, and before the scalar fallback below because that is the
    // order the Cython tests them in. Swapping the two arms is *not*
    // observable from Python — a mutation that swaps them leaves all 118
    // tests in `test/test_core_dispatch.py` green, because the only
    // objects with both a `__len__` and a working `__float__` are 0-d
    // ndarrays, which the arm above already took. The order is fidelity
    // to the Cython, not a guard; the guard against a string parsing as a
    // number is `has_numeric_dtype`.
    //
    // Testing `__len__` rather than converting everything is what keeps
    // `None` and a bare object a `TypeError` — `np.asarray` would turn
    // either into a 0-d object array and report a dtype instead of a type.
    if obj.hasattr("__len__")? {
        let converted = as_numpy_array(obj)?;
        let array = converted
            .cast::<PyUntypedArray>()
            .map_err(|_| type_error(quantity))?;
        return classify_array(array, quantity);
    }

    // `np.float32`, `np.int64`, `np.bool_`, and Python `int`/`bool`: all
    // scalars the Cython accepts for an energy argument today, none of
    // which defines `__len__`.
    if let Ok(value) = obj.extract::<f64>() {
        return Ok(Argument::Scalar(value));
    }

    Err(type_error(quantity))
}

/// Apply `kernel` elementwise to `obj` under the dispatch contract.
///
/// `quantity` names the argument in every error message — pass the same
/// wording the Cython twin used (e.g. `"Photon energies"`), so the port
/// does not silently reword a user-visible exception.
///
/// # Errors
///
/// `ValueError` if `obj` is an array of more than one dimension or a 1-D
/// array whose dtype is not `float64`; `TypeError` if it is neither a
/// real number nor a sequence.
pub fn map_unary<F>(obj: &Bound<'_, PyAny>, quantity: &str, kernel: F) -> PyResult<Py<PyAny>>
where
    F: Fn(f64) -> f64,
{
    let py = obj.py();
    match classify(obj, quantity)? {
        Argument::Scalar(value) => Ok(PyFloat::new(py, kernel(value)).into_any().unbind()),
        Argument::Vector(values) => {
            // A fresh Vec, never a view onto the caller's buffer: the
            // result must not alias the input, and the view may be
            // non-contiguous.
            let mapped: Vec<f64> = values.as_array().iter().map(|&x| kernel(x)).collect();
            Ok(mapped.into_pyarray(py).into_any().unbind())
        }
    }
}

/// [`map_unary`] for a kernel returning one value per neutrino flavor.
///
/// The only non-uniform return shape in the whole public surface: a
/// scalar argument gives a 3-tuple of `float`, a grid of `N` energies
/// gives a `(3, N)` `float64` array whose rows are electron, muon and tau
/// — the layout `hazma/spectra/_neutrino/*.pyx` builds today.
///
/// `kernel` is called **once per energy**, not once per (energy, flavor):
/// the Cython computes the three flavors together from one shared
/// kinematic evaluation, and calling three times would triple the work
/// and could not be assumed to give the same rounding.
///
/// # Errors
///
/// As [`map_unary`].
pub fn map_flavors<F>(obj: &Bound<'_, PyAny>, quantity: &str, kernel: F) -> PyResult<Py<PyAny>>
where
    F: Fn(f64) -> [f64; N_FLAVORS],
{
    let py = obj.py();
    match classify(obj, quantity)? {
        Argument::Scalar(value) => Ok(PyTuple::new(py, kernel(value))?.into_any().unbind()),
        Argument::Vector(values) => {
            let view = values.as_array();
            let flavors: Vec<[f64; N_FLAVORS]> = view.iter().map(|&x| kernel(x)).collect();
            let mut rows = Vec::with_capacity(N_FLAVORS * flavors.len());
            for flavor in 0..N_FLAVORS {
                rows.extend(flavors.iter().map(|triple| triple[flavor]));
            }
            let array = Array2::from_shape_vec((N_FLAVORS, flavors.len()), rows)
                .expect("N_FLAVORS x n built from exactly N_FLAVORS * n values");
            Ok(array.into_pyarray(py).into_any().unbind())
        }
    }
}

/// Extract an argument that must be a 1-D `float64` array, never a scalar.
///
/// The third live shape: `partial_widths` in the mediator decay spectra,
/// which is the only public argument the Cython rejects a scalar for
/// explicitly rather than by falling through a dispatch branch. Both
/// messages are that call site's own, verbatim.
///
/// # Errors
///
/// `ValueError` if `obj` has no `__len__` (`"<quantity> must be a list or
/// array."`), if it is not 1-D (`"<quantity> must be 1-dimensional."`), or
/// if its dtype is not `float64`.
pub fn require_vector(obj: &Bound<'_, PyAny>, quantity: &str) -> PyResult<Vec<f64>> {
    if !obj.hasattr("__len__")? {
        return Err(PyValueError::new_err(format!(
            "{quantity} must be a list or array."
        )));
    }

    // `converted` must outlive the borrow taken from it, so it is bound
    // here rather than inside the match arm.
    let converted;
    let array = match obj.cast::<PyUntypedArray>() {
        Ok(array) => array,
        Err(_) => {
            converted = as_numpy_array(obj)?;
            converted
                .cast::<PyUntypedArray>()
                .map_err(|_| type_error(quantity))?
        }
    };

    if array.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "{quantity} must be 1-dimensional."
        )));
    }
    let values: PyReadonlyArray1<'_, f64> = array
        .as_any()
        .extract()
        .map_err(|_| dtype_error(array, quantity))?;
    Ok(values.as_array().to_vec())
}
