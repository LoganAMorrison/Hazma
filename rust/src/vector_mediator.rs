//! `hazma._core.vector_mediator` — vector-mediator kernels.
//!
//! Empty scaffold. Phase 05 fills it with the cross sections and thermal
//! ⟨σv⟩ from `hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx`;
//! Phase 06 adds the spectrum modules.

use pyo3::prelude::*;

/// Populate the submodule. Nothing to register yet.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = module;
    Ok(())
}
