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
//!
//! [`constants`], [`special`], [`quad`], [`interp`] and [`boost`] sit
//! below `kernels` in that stack and are `pub` while their neighbours are
//! private — no kernel reads any of them yet, and a private module of
//! unread items is a wall of `dead_code`. Publishing them is also honest:
//! the constant tables, the cephes shim, the QUADPACK port and the
//! interpolation/boost foundation are the crate's most reusable surface,
//! and Phases 03–06 consume them from every kernel module.
//!
//! The six `*_probe` modules are the exception to "registration only
//! means per-domain": they register `special`, `quad`, `interp`,
//! `boost`, `dispatch` and `mediator_tables` submodules that expose the
//! foundation modules to Python purely so `test/test_core_special.py`,
//! `test/test_core_quad.py`, `test/test_core_interp.py`,
//! `test/test_core_boost.py`, `test/test_core_dispatch.py`,
//! `test/test_core_mediator_tables.py` and `test/test_core_photon_tables.py`
//! can compare them against their oracles — scipy, NumPy, a Python
//! reference implementation, and the Phase 04 kernels' own entry points.
//! Two of those oracles used to be the Cython itself, reached through
//! `__pyx_capi__` or read out of the `.pyx` sources; Phase 06 Task 6.4
//! deleted the last of them, and what replaced each is recorded in the
//! module that used it. No hazma module imports any of these probes.
//!
//! Because nothing but the test suite reaches them, they compile only
//! under the `test-probes` feature, which `rust/Cargo.toml` leaves out
//! of `default`. A development install and the crate's own gates turn it
//! on; the wheel and the sdist do not, so a released `hazma._core`
//! carries the five per-domain submodules and nothing else. The test
//! modules above import the probes at module scope rather than through
//! `importorskip`, so a build without the feature fails collection
//! instead of quietly skipping — `test/conftest.py` turns that failure
//! into the install command that fixes it.

pub mod boost;
#[cfg(feature = "test-probes")]
mod boost_probe;
pub mod constants;
mod dispatch;
#[cfg(feature = "test-probes")]
mod dispatch_probe;
pub mod interp;
#[cfg(feature = "test-probes")]
mod interp_probe;
// Without `test-probes`, a handful of kernel items lose their only
// non-test reader: `roundtrip_flavors` served `dispatch_probe` alone, and
// four of `mediator_tables`'s accessors served `mediator_tables_probe`.
// That is the feature working, not dead code, so the lint is relaxed for
// exactly the build that compiles the probes out. Every gate this repo
// runs builds them in, so `dead_code` still covers this module where it
// is enforced.
#[cfg_attr(not(feature = "test-probes"), allow(dead_code))]
mod kernels;
#[cfg(feature = "test-probes")]
mod mediator_tables_probe;
mod neutrino;
mod photon;
mod positron;
pub mod quad;
#[cfg(feature = "test-probes")]
mod quad_probe;
mod scalar_mediator;
pub mod special;
#[cfg(feature = "test-probes")]
mod special_probe;
mod vector_mediator;

use pyo3::prelude::*;

/// Return the input, having passed it through Rust.
///
/// The scaffold's plumbing probe: it exercises the whole
/// [`dispatch::map_unary`] contract — scalar in / `float` out, 1-D array
/// in / fresh 1-D array out, `ValueError` on everything else — without
/// any physics to get wrong. Phase 02 Task 2.3's test module is written
/// against it and is the template later kernel swaps copy.
///
/// The advertised signature is `(x)`, not `(x, /)`: `text_signature` is a
/// claim PyO3 does not enforce, `roundtrip(x=1.5)` works, and the Cython
/// entry points this crate replaces are `def` functions that accept their
/// arguments by keyword. A positional-only claim would misdescribe this
/// function and, copied into a Phase 04 wrapper, narrow the public API.
#[pyfunction]
#[pyo3(text_signature = "(x)")]
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

    #[cfg(feature = "test-probes")]
    {
        add_submodule(module, "special", special_probe::register)?;
        add_submodule(module, "quad", quad_probe::register)?;
        add_submodule(module, "interp", interp_probe::register)?;
        add_submodule(module, "boost", boost_probe::register)?;
        add_submodule(module, "dispatch", dispatch_probe::register)?;
        add_submodule(module, "mediator_tables", mediator_tables_probe::register)?;
    }

    Ok(())
}
