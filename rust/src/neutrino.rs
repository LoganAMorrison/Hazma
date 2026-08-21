//! `hazma._core.neutrino` — neutrino spectrum kernels.
//!
//! Registration only: the math is in [`crate::kernels`] and the argument
//! and error handling in [`crate::dispatch`], so nothing here computes
//! or classifies anything (`projects/cython-to-rust/rules.md`, Rust
//! conventions rules 2-3).
//!
//! Both entry points landed in Phase 04 Task 4.6, and they are the one
//! non-uniform return shape in hazma's public surface: a 3-tuple of
//! `float` for a scalar argument, a `(3, N)` `float64` array whose rows
//! are electron, muon and tau for a grid. [`crate::dispatch::map_flavors`]
//! owns that shape; these two functions only choose the kernel and the
//! quantity wording.
//!
//! Unlike the `special` / `quad` / `interp` / `boost` / `dispatch`
//! probes, this submodule is **not** a test surface: every function it
//! registers is what `hazma/spectra/_neutrino/__init__.py` calls, so
//! `test/parity/cases.py`'s `rust_core_kernels()` counts them.

use pyo3::prelude::*;

use crate::dispatch::map_flavors;
use crate::kernels::{neutrino_muon, neutrino_pion};

/// The neutrino spectra `dN/dE` in MeV⁻¹ from the decay of a muon.
///
/// `neutrino_energies` is the mapped argument — a float, a NumPy scalar,
/// a 0-d numeric array, or a 1-D `float64` array (or a sequence that
/// converts to one). `muon_energy` is the muon's total energy in MeV.
///
/// The quantity wording is `"Neutrino energies"`. The `assert` this
/// replaces says `"Photon energies must be 0 or 1-dimensional."` — a
/// copy-paste defect in `hazma/spectra/_neutrino/_muon.pyx:205` that its
/// `_pion.pyx` sibling does not share, and the one place in the port
/// where the message is deliberately **not** the twin's. Task 3.5 decided
/// it (`rules.md` rule 9's roster), and
/// `test/test_core_dispatch.py::TestCythonMessageParity` pins the roster
/// so the divergence stays declared rather than accidental.
///
/// The advertised signature is not positional-only: the Cython entry
/// point was a `def` and accepted both arguments by keyword.
#[pyfunction]
#[pyo3(text_signature = "(neutrino_energies, muon_energy)")]
fn dnde_neutrino_muon(
    neutrino_energies: &Bound<'_, PyAny>,
    muon_energy: f64,
) -> PyResult<Py<PyAny>> {
    map_flavors(neutrino_energies, "Neutrino energies", |energy| {
        neutrino_muon::dnde_neutrino_muon(energy, muon_energy).to_array()
    })
}

/// The neutrino spectra `dN/dE` in MeV⁻¹ from the decay of a charged
/// pion.
///
/// `neutrino_energies` is the mapped argument, as above; `pion_energy` is
/// the pion's total energy in MeV.
///
/// The quantity wording is `"Neutrino energies"`, which
/// `hazma/spectra/_neutrino/_pion.pyx` spells correctly.
#[pyfunction]
#[pyo3(text_signature = "(neutrino_energies, pion_energy)")]
fn dnde_neutrino_charged_pion(
    neutrino_energies: &Bound<'_, PyAny>,
    pion_energy: f64,
) -> PyResult<Py<PyAny>> {
    map_flavors(neutrino_energies, "Neutrino energies", |energy| {
        neutrino_pion::dnde_neutrino_charged_pion(energy, pion_energy).to_array()
    })
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dnde_neutrino_muon, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_neutrino_charged_pion, module)?)?;
    Ok(())
}
