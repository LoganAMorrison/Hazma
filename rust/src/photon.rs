//! `hazma._core.photon` — photon spectrum kernels.
//!
//! Empty scaffold. Phase 04 fills it with the `dnde_photon_*` kernels
//! currently in `hazma/spectra/_photon/*.pyx`.

use pyo3::prelude::*;

/// Populate the submodule. Nothing to register yet.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = module;
    Ok(())
}
