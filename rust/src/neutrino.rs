//! `hazma._core.neutrino` — neutrino spectrum kernels.
//!
//! Empty scaffold. Phase 04 fills it with the `dnde_neutrino_*` kernels
//! currently in `hazma/spectra/_neutrino/*.pyx`. These are the one
//! non-uniform return shape in the dispatch contract: a 3-tuple for
//! scalar input, a `(3, N)` array for array input.

use pyo3::prelude::*;

/// Populate the submodule. Nothing to register yet.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = module;
    Ok(())
}
