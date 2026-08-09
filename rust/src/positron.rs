//! `hazma._core.positron` — positron spectrum kernels.
//!
//! Empty scaffold. Phase 04 fills it with the `dnde_positron_*` kernels
//! currently in `hazma/spectra/_positron/*.pyx`.

use pyo3::prelude::*;

/// Populate the submodule. Nothing to register yet.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = module;
    Ok(())
}
