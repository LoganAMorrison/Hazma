//! `hazma._core.mediator_tables` — the Python-visible half of
//! [`crate::kernels::mediator_tables`].
//!
//! Registration only, and a **test surface, not physics**: Tasks 6.2 and
//! 6.3 call [`crate::kernels::mediator_tables`] directly in Rust and
//! nothing under `hazma/` imports this module. It joins
//! `test/parity/cases.py`'s `_CORE_TEST_ONLY_MODULES` beside
//! `special` / `quad` / `interp` / `boost` / `dispatch`.
//!
//! It exists for the same reason [`crate::interp_probe`] does — the
//! oracle lives in Python. Two claims Task 6.1 makes cannot be checked
//! from `cargo test` alone:
//!
//! * the grid is `numpy.logspace(start, log10(m/2), 500)`. `cargo` can
//!   pin the *algorithm* — unfused `i * step + start`, the last point
//!   substituted from `stop` — but not the agreement with NumPy, because
//!   that rests on Rust's `log10`/`powf` and NumPy's `power` loop
//!   reaching the same libm. They do on the capturing platform;
//!   hard-coding its bits into a `cargo` test would turn a Linux CI job
//!   red for a platform difference rather than a defect
//!   (`projects/cython-to-rust/learnings/phase-04-spectra-kernels.md`
//!   §4). Comparing live against `numpy.logspace` re-derives the claim
//!   wherever the suite runs.
//! * the tables hold the Phase 04 kernels themselves. Exposing the
//!   built columns lets `test/test_core_mediator_tables.py` assert them
//!   against `hazma._core.photon` / `hazma._core.positron` — the same
//!   kernels through the public entry points — rather than against a
//!   re-derivation.

use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dispatch::map_unary;
use crate::kernels::mediator_tables::{
    self, BelowGrid, PhotonMode, PositronMode, RestFrameTable, ScalarPhotonModes,
};

/// One table set as Python sees it: `(energies, charged_pion, muon)`.
type TableColumns<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// `numpy.logspace(start, stop, num)`, from Rust.
///
/// `start` and `stop` are base-10 exponents; the result is a fresh 1-D
/// `float64` array of `num` points.
///
/// # Errors
///
/// `ValueError` if `num < 2`. NumPy accepts 0 and 1; the kernel does not,
/// because every hazma call site passes 500 and reproducing NumPy's
/// degenerate cases would be untested code.
#[pyfunction]
#[pyo3(name = "logspace")]
#[pyo3(text_signature = "(start, stop, num)")]
fn logspace_py<'py>(
    py: Python<'py>,
    start: f64,
    stop: f64,
    num: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if num < 2 {
        return Err(PyValueError::new_err("num must be at least 2"));
    }
    Ok(mediator_tables::logspace(start, stop, num).to_pyarray(py))
}

/// The decay modules' photon tables for a mediator of mass `mass` MeV.
///
/// Returns `(energies, charged_pion, muon)` — the shared abscissae in
/// MeV and the two `dN/dE` columns in MeV⁻¹.
#[pyfunction]
#[pyo3(name = "photon_tables")]
#[pyo3(text_signature = "(mass)")]
fn photon_tables_py(py: Python<'_>, mass: f64) -> TableColumns<'_> {
    let tables = mediator_tables::photon_tables(mass);
    (
        tables.charged_pion.energies().to_pyarray(py),
        tables.charged_pion.values().to_pyarray(py),
        tables.muon.values().to_pyarray(py),
    )
}

/// The positron modules' tables for a mediator of mass `mass` MeV.
///
/// Returns `(energies, charged_pion, muon)`, units as for
/// [`photon_tables_py`].
#[pyfunction]
#[pyo3(name = "positron_tables")]
#[pyo3(text_signature = "(mass)")]
fn positron_tables_py(py: Python<'_>, mass: f64) -> TableColumns<'_> {
    let tables = mediator_tables::positron_tables(mass);
    (
        tables.charged_pion.energies().to_pyarray(py),
        tables.charged_pion.values().to_pyarray(py),
        tables.muon.values().to_pyarray(py),
    )
}

/// The `1/E`-tail lookup the decay modules use, over a table built from
/// `kernel_values` on `energies`.
///
/// Exposed so `test/test_core_mediator_tables.py` can put the same
/// abscissae through this and through `numpy.interp` plus the Cython's
/// `if eng_gam < 10**-1` guard, rather than inferring the branch from a
/// spectrum. `energies` and `values` are the columns
/// [`photon_tables_py`] returns.
///
/// # Errors
///
/// Whatever the dispatch contract raises for `x`.
#[pyfunction]
#[pyo3(name = "lookup")]
#[pyo3(text_signature = "(x, energies, values, inverse_energy_tail)")]
fn lookup_py(
    x: &Bound<'_, PyAny>,
    energies: numpy::PyReadonlyArray1<'_, f64>,
    values: numpy::PyReadonlyArray1<'_, f64>,
    inverse_energy_tail: bool,
) -> PyResult<Py<PyAny>> {
    let below = if inverse_energy_tail {
        BelowGrid::InverseEnergy
    } else {
        BelowGrid::Clamp
    };
    let table = RestFrameTable::from_columns(
        energies.as_array().to_vec(),
        values.as_array().to_vec(),
        below,
    )
    .map_err(PyValueError::new_err)?;
    map_unary(x, "Energies", |energy| table.lookup(energy))
}

/// The vector decay module's `mode` string, resolved.
///
/// Returns the variant's name, or `None` for a string the Cython would
/// answer with `0.0`.
#[pyfunction]
#[pyo3(name = "photon_mode")]
#[pyo3(text_signature = "(mode)")]
fn photon_mode_py(mode: &str) -> Option<(&'static str, bool)> {
    PhotonMode::parse(mode).map(|parsed| (photon_mode_name(parsed), parsed.has_line()))
}

/// The stable name `photon_mode` reports for a variant.
fn photon_mode_name(mode: PhotonMode) -> &'static str {
    match mode {
        PhotonMode::Total => "Total",
        PhotonMode::ElectronFsr => "ElectronFsr",
        PhotonMode::ChargedPionFsr => "ChargedPionFsr",
        PhotonMode::ChargedPionDecay => "ChargedPionDecay",
        PhotonMode::NeutralPionLine => "NeutralPionLine",
        PhotonMode::MuonFsr => "MuonFsr",
        PhotonMode::MuonDecay => "MuonDecay",
    }
}

/// The positron modules' `fs` string, resolved.
#[pyfunction]
#[pyo3(name = "positron_mode")]
#[pyo3(text_signature = "(fs)")]
fn positron_mode_py(fs: &str) -> Option<&'static str> {
    PositronMode::parse(fs).map(|parsed| match parsed {
        PositronMode::Total => "Total",
        PositronMode::ElectronLine => "ElectronLine",
        PositronMode::MuonDecay => "MuonDecay",
        PositronMode::ChargedPionDecay => "ChargedPionDecay",
    })
}

/// The scalar decay module's mode *list*, folded to its bitflag.
///
/// The integer this returns is the one
/// `scalar_mediator_decay_spectrum.pyx:253-266` builds, so a test can
/// compare the two directly.
#[pyfunction]
#[pyo3(name = "scalar_photon_mode_bits")]
#[pyo3(text_signature = "(modes)")]
fn scalar_photon_mode_bits_py(modes: Vec<String>) -> u32 {
    ScalarPhotonModes::from_names(modes).bits()
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(logspace_py, module)?)?;
    module.add_function(wrap_pyfunction!(photon_tables_py, module)?)?;
    module.add_function(wrap_pyfunction!(positron_tables_py, module)?)?;
    module.add_function(wrap_pyfunction!(lookup_py, module)?)?;
    module.add_function(wrap_pyfunction!(photon_mode_py, module)?)?;
    module.add_function(wrap_pyfunction!(positron_mode_py, module)?)?;
    module.add_function(wrap_pyfunction!(scalar_photon_mode_bits_py, module)?)?;
    Ok(())
}
