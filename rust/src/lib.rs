//! `hazma._core` — hazma's compiled numerical layer.
//!
//! One `cdylib`, five per-domain submodules, built against CPython's
//! limited API (`abi3-py310`). This is the final import path from the
//! day the crate exists: Phases 03–06 of the cython-to-rust project fill
//! the submodules and repoint hazma's pure-Python wrappers at them one
//! kernel at a time, and no public Python import path changes when they
//! do (`projects/cython-to-rust/rules.md`, Rust conventions rule 2).
//!
//! Layering inside the crate mirrors that rule: [`kernels`] is plain
//! GIL-free Rust with no PyO3 types at all, and everything PyO3 touches
//! sits above it — [`dispatch`] owns argument conversion, array glue and
//! error mapping, this module owns registration, and the per-domain
//! submodules are registration only. So "the PyO3 layer lives
//! separately" (rules.md rule 8) is about keeping it out of [`kernels`],
//! not about confining it to one file.

mod dispatch;
mod kernels;
mod neutrino;
mod photon;
mod positron;
mod scalar_mediator;
mod vector_mediator;

use pyo3::prelude::*;

/// Return the input, having passed it through Rust.
///
/// The scaffold's plumbing probe: it exercises the whole
/// [`dispatch::map_unary`] contract — scalar in / `float` out, 1-D array
/// in / fresh 1-D array out, `ValueError` on everything else — without
/// any physics to get wrong. Phase 02 Task 2.3's test module is written
/// against it and is the template later kernel swaps copy.
#[pyfunction]
#[pyo3(text_signature = "(x, /)")]
fn roundtrip(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    dispatch::map_unary(x, "Input values", kernels::roundtrip)
}

/// Create a submodule, attach it to `parent`, and register it in
/// `sys.modules`.
///
/// Attaching it as an attribute alone makes `from hazma._core import
/// photon` work but leaves `from hazma._core.photon import
/// dnde_photon_muon` an ImportError — the import system never learns the
/// child exists. Registering the fully-qualified name in `sys.modules`
/// makes both forms work, so a Phase 04 wrapper can use whichever reads
/// better.
///
/// The child is *created* under its fully-qualified name so its
/// `__name__` matches the `sys.modules` key; it is then attached under
/// the bare `name`, which is what `hazma._core.photon` must resolve to.
fn add_submodule(
    parent: &Bound<'_, PyModule>,
    name: &str,
    register: fn(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = parent.py();
    let parent_name: String = parent.name()?.extract()?;
    let qualified = format!("{parent_name}.{name}");

    let child = PyModule::new(py, &qualified)?;
    register(&child)?;
    parent.add(name, &child)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(&qualified, &child)?;
    Ok(())
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__doc__", "Compiled numerical kernels for hazma.")?;
    module.add_function(wrap_pyfunction!(roundtrip, module)?)?;

    add_submodule(module, "photon", photon::register)?;
    add_submodule(module, "positron", positron::register)?;
    add_submodule(module, "neutrino", neutrino::register)?;
    add_submodule(module, "scalar_mediator", scalar_mediator::register)?;
    add_submodule(module, "vector_mediator", vector_mediator::register)?;

    Ok(())
}
