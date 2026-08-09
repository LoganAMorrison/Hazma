//! `hazma._core.scalar_mediator` — scalar-mediator kernels.
//!
//! Empty scaffold. Phase 05 fills it with the cross sections and thermal
//! ⟨σv⟩ from `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx`;
//! Phase 06 adds the spectrum modules.

use pyo3::prelude::*;

/// Populate the submodule. Nothing to register yet.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = module;
    Ok(())
}
