//! `hazma._core.positron` — positron spectrum kernels.
//!
//! Registration only: the math is in [`crate::kernels`] and the argument
//! and error handling in [`crate::dispatch`], so nothing here computes
//! or classifies anything (`projects/cython-to-rust/rules.md`, Rust
//! conventions rules 2–3).
//!
//! Unlike the `special` / `quad` / `interp` / `boost` / `dispatch`
//! probes, this submodule is **not** a test surface: every function it
//! registers is what `hazma/spectra/_positron/__init__.py` calls, so
//! `test/parity/cases.py`'s `rust_core_kernels()` counts them and the
//! parity corpus leaves bit-equality mode accordingly. That is the
//! intended reading — a kernel is served here.
//!
//! Both entry points landed here: `dnde_positron_muon` in Task 4.1 and
//! `dnde_positron_charged_pion` in Task 4.6, which is every
//! `hazma.spectra.dnde_positron_*` spectrum the compiled layer serves —
//! the other eight are pure-Python table lookups in the wrapper.

use pyo3::prelude::*;

use crate::dispatch::map_unary;
use crate::kernels::{positron_muon, positron_pion};

/// The positron spectrum `dN/dE` in MeV⁻¹ from the decay of a muon.
///
/// `positron_energies` is the mapped argument — a float, a NumPy scalar,
/// a 0-d numeric array, or a 1-D `float64` array (or a sequence that
/// converts to one). `muon_energy` is the muon's total energy in MeV.
///
/// The quantity wording is `"Positron energies"`, which is the wording
/// `hazma/spectra/_positron/_muon.pyx` used in the `assert` this
/// replaces — see `dispatch`'s module docs for why an `assert`'s message
/// survives while its exception type does not.
///
/// The advertised signature is not positional-only: the Cython entry
/// point was a `def` and accepted both arguments by keyword.
#[pyfunction]
#[pyo3(text_signature = "(positron_energies, muon_energy)")]
fn dnde_positron_muon(
    positron_energies: &Bound<'_, PyAny>,
    muon_energy: f64,
) -> PyResult<Py<PyAny>> {
    map_unary(positron_energies, "Positron energies", |energy| {
        positron_muon::dnde_positron_muon(energy, muon_energy)
    })
}

/// The positron spectrum `dN/dE` in MeV⁻¹ from the decay of a charged
/// pion.
///
/// `positron_energies` is the mapped argument — a float, a NumPy scalar,
/// a 0-d numeric array, or a 1-D `float64` array (or a sequence that
/// converts to one). `pion_energy` is the pion's total energy in MeV.
///
/// The quantity wording is `"Positron energies"`, which is the wording
/// `hazma/spectra/_positron/_pion.pyx` used in the `assert` this
/// replaces — the same string its `_muon.pyx` sibling used.
///
/// The advertised signature is not positional-only: the Cython entry
/// point was a `def` and accepted both arguments by keyword.
#[pyfunction]
#[pyo3(text_signature = "(positron_energies, pion_energy)")]
fn dnde_positron_charged_pion(
    positron_energies: &Bound<'_, PyAny>,
    pion_energy: f64,
) -> PyResult<Py<PyAny>> {
    map_unary(positron_energies, "Positron energies", |energy| {
        positron_pion::dnde_positron_charged_pion(energy, pion_energy)
    })
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dnde_positron_muon, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_positron_charged_pion, module)?)?;
    Ok(())
}
