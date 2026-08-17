//! `hazma._core.photon` — photon spectrum kernels.
//!
//! Registration only: the math is in [`crate::kernels`] and the argument
//! and error handling in [`crate::dispatch`], so nothing here computes or
//! classifies anything (`projects/cython-to-rust/rules.md`, Rust
//! conventions rules 2–3).
//!
//! Like [`crate::positron`], and unlike the `special` / `quad` / `interp`
//! / `boost` / `dispatch` probes, this submodule is **not** a test
//! surface: every function it registers is what
//! `hazma/spectra/_photon/__init__.py` calls, so
//! `test/parity/cases.py`'s `rust_core_kernels()` counts them.
//!
//! The seven tabulated spectra landed here in Task 4.2 and the radiative
//! muon spectrum in Task 4.3. Tasks 4.4–4.5 add the pion and rho kernels
//! beside them.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dispatch::map_unary;
use crate::kernels::photon_muon;
use crate::kernels::photon_tables::{self, Spectrum};

/// Every tabulated entry point, over one implementation.
///
/// `photon_energies` is the mapped argument — a float, a NumPy scalar, a
/// 0-d numeric array, or a 1-D `float64` array (or a sequence that
/// converts to one). `parent_energy` is the decaying meson's total energy
/// in MeV.
///
/// The quantity wording is `"Photon energies"`, which is the wording all
/// five `.pyx` files used in the `assert` this replaces — see
/// [`crate::dispatch`]'s module docs for why an `assert`'s message
/// survives while its exception type does not.
///
/// # Errors
///
/// `ValueError` if `parent_energy` implies a boost velocity outside
/// `(0, 1)`, which in practice means a `NaN` — the Cython raises a bare
/// `AssertionError` there. Plus whatever the dispatch contract raises for
/// `photon_energies`.
fn dnde_tabulated(
    photon_energies: &Bound<'_, PyAny>,
    parent_energy: f64,
    spectrum: &Spectrum,
) -> PyResult<Py<PyAny>> {
    // Resolved once, before any element is evaluated: the branch and its
    // guard depend on the parent energy alone, so a swept grid either
    // fails everywhere or nowhere and there is no question of what value
    // a failed element takes.
    let branch = photon_tables::branch(parent_energy, spectrum.mass)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    map_unary(photon_energies, "Photon energies", |energy| {
        photon_tables::dnde(energy, branch, spectrum)
    })
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from radiative muon decay.
///
/// `photon_energies` is the mapped argument — a float, a NumPy scalar, a
/// 0-d numeric array, or a 1-D `float64` array (or a sequence that
/// converts to one). `muon_energy` is the muon's total energy in MeV.
///
/// The quantity wording is `"Photon energies"`, which is the wording
/// `hazma/spectra/_photon/_muon.pyx` used in the `assert` this replaces.
/// Unlike the tabulated entry points above there is no parent-energy
/// guard to resolve first: the kernel's own `emu < m_mu` short circuit
/// returns zero, as the Cython does.
///
/// The advertised signature is not positional-only: the Cython entry
/// point was a `def` and accepted both arguments by keyword.
#[pyfunction]
#[pyo3(text_signature = "(photon_energies, muon_energy)")]
fn dnde_photon_muon(photon_energies: &Bound<'_, PyAny>, muon_energy: f64) -> PyResult<Py<PyAny>> {
    map_unary(photon_energies, "Photon energies", |energy| {
        photon_muon::dnde_photon_muon(energy, muon_energy)
    })
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from charged-kaon decay.
#[pyfunction]
#[pyo3(text_signature = "(photon_energies, kaon_energy)")]
fn dnde_photon_charged_kaon(
    photon_energies: &Bound<'_, PyAny>,
    kaon_energy: f64,
) -> PyResult<Py<PyAny>> {
    dnde_tabulated(photon_energies, kaon_energy, &photon_tables::CHARGED_KAON)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from long-kaon decay.
#[pyfunction]
#[pyo3(text_signature = "(photon_energies, kaon_energy)")]
fn dnde_photon_long_kaon(
    photon_energies: &Bound<'_, PyAny>,
    kaon_energy: f64,
) -> PyResult<Py<PyAny>> {
    dnde_tabulated(photon_energies, kaon_energy, &photon_tables::LONG_KAON)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from short-kaon decay.
#[pyfunction]
#[pyo3(text_signature = "(photon_energies, kaon_energy)")]
fn dnde_photon_short_kaon(
    photon_energies: &Bound<'_, PyAny>,
    kaon_energy: f64,
) -> PyResult<Py<PyAny>> {
    dnde_tabulated(photon_energies, kaon_energy, &photon_tables::SHORT_KAON)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from η decay.
#[pyfunction]
#[pyo3(text_signature = "(photon_energies, eta_energy)")]
fn dnde_photon_eta(photon_energies: &Bound<'_, PyAny>, eta_energy: f64) -> PyResult<Py<PyAny>> {
    dnde_tabulated(photon_energies, eta_energy, &photon_tables::ETA)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from η′ decay.
#[pyfunction]
#[pyo3(text_signature = "(photon_energies, eta_prime_energy)")]
fn dnde_photon_eta_prime(
    photon_energies: &Bound<'_, PyAny>,
    eta_prime_energy: f64,
) -> PyResult<Py<PyAny>> {
    dnde_tabulated(photon_energies, eta_prime_energy, &photon_tables::ETA_PRIME)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from ω decay.
#[pyfunction]
#[pyo3(text_signature = "(photon_energies, omega_energy)")]
fn dnde_photon_omega(photon_energies: &Bound<'_, PyAny>, omega_energy: f64) -> PyResult<Py<PyAny>> {
    dnde_tabulated(photon_energies, omega_energy, &photon_tables::OMEGA)
}

/// The photon spectrum `dN/dE` in MeV⁻¹ from φ decay.
#[pyfunction]
#[pyo3(text_signature = "(photon_energies, phi_energy)")]
fn dnde_photon_phi(photon_energies: &Bound<'_, PyAny>, phi_energy: f64) -> PyResult<Py<PyAny>> {
    dnde_tabulated(photon_energies, phi_energy, &photon_tables::PHI)
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dnde_photon_muon, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_photon_charged_kaon, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_photon_long_kaon, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_photon_short_kaon, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_photon_eta, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_photon_eta_prime, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_photon_omega, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_photon_phi, module)?)?;
    Ok(())
}
