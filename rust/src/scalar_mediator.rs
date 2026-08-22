//! `hazma._core.scalar_mediator` — scalar-mediator kernels.
//!
//! Registration only: the math is in [`crate::kernels::scalar_xs`] and
//! the argument and error handling in [`crate::dispatch`], so nothing
//! here computes or classifies anything
//! (`projects/cython-to-rust/rules.md`, Rust conventions rules 2–3).
//!
//! Twelve entry points, all landed in cython-to-rust Task 5.2 — every
//! consumed public `def` of
//! `hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx`, which
//! that task deleted. The thirteenth, `sigma_xx_to_all`, is deliberately
//! **not** here: nothing in hazma imported it, so the plan drops it
//! rather than porting it (`phase-05-mediator-cross-sections.md`). It
//! survives as a private helper of the thermal integrand, which is the
//! only caller it ever had.
//!
//! # Arguments
//!
//! `e_cms` is the mapped argument on all twelve — a float, a NumPy
//! scalar, a 0-d numeric array, or a 1-D `float64` array (or a sequence
//! that converts to one); `thermal_cross_section` takes a scalar `x`
//! instead, because the Cython declared it `double`. Every other
//! argument is a scalar, in MeV or dimensionless: `mx` and `ms` are the
//! dark-matter and mediator masses; `gsxx`, `gsff`, `gsGG` and `gsFF`
//! the mediator's couplings to dark matter, fermions, gluons and
//! photons; `lam` the cut-off scale of the `SGG` and `SFF`
//! interactions; `width_s` the mediator's full decay width in MeV; `vs`
//! the mediator VEV; and, where present, `mf` the final-state fermion
//! mass in MeV.
//!
//! Every one is accepted by keyword, and every channel accepts the full
//! ten (or eleven) whether its kernel reads them or not: the Cython
//! entry points were `def`s with that signature and the wrapper still
//! passes all of them positionally, so dropping or renaming one would
//! narrow the public API.
//!
//! Phase 06 adds the spectrum modules alongside these.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::dispatch::{map_unary, map_unary_try};
use crate::kernels::scalar_xs;
use crate::kernels::soft_complex::NonRealResult;

/// The wording every `NonRealResult` reaches Python with.
///
/// Not the Cython's, for the reason [`crate::vector_mediator`] gives at
/// length: two thirds of `__Pyx_SoftComplexToDouble`'s message is advice
/// about a Cython compiler directive, which is wrong advice in a tree
/// with no Cython in it. The **type** is what the port owes.
const NON_REAL_MESSAGE: &str = "Cross section is complex at this center-of-mass energy: the \
     kinematic factor raised to the power 3/2 reached a vanishing \
     denominator.";

/// Map the kernels' failure onto the exception the Cython raised.
fn non_real(_: NonRealResult) -> PyErr {
    PyTypeError::new_err(NON_REAL_MESSAGE)
}

/// The name every dispatched argument goes by in an error message.
///
/// The Cython cross sections had no `assert` to borrow a wording from —
/// they dispatch on `hasattr(e_cms, '__len__')` and let the buffer cast
/// raise — so this is the port's own, matching the argument's spelling in
/// the `.pyx` signature.
const ENERGIES: &str = "Center of mass energies";
/// Cross section for `x x̄ → S* → f f̄` in MeV⁻².
///
/// See the module docs for the arguments.
///
/// # Errors
///
/// `TypeError` if the Cython's complex `**` operator produced a
/// non-real result. See [`NON_REAL_MESSAGE`].
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs, mf)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xx_to_s_to_ff(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
    mf: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsGG, gsFF, lam, vs);
    map_unary_try(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xx_to_s_to_ff(e_cm, mx, ms, gsxx, gsff, width_s, mf).map_err(non_real)
    })
}

/// Cross section for `x x̄ → S* → γγ` in MeV⁻².
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xx_to_s_to_gg(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsff, gsGG, vs);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xx_to_s_to_gg(e_cm, mx, ms, gsxx, gsFF, lam, width_s)
    })
}

/// Cross section for `x x̄ → S* → π⁰π⁰` in MeV⁻².
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xx_to_s_to_pi0pi0(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsFF,);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xx_to_s_to_pi0pi0(e_cm, mx, ms, gsxx, gsff, gsGG, lam, width_s, vs)
    })
}

/// Cross section for `x x̄ → S* → π⁺π⁻` in MeV⁻².
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xx_to_s_to_pipi(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsFF,);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xx_to_s_to_pipi(e_cm, mx, ms, gsxx, gsff, gsGG, lam, width_s, vs)
    })
}

/// Cross section for `x x̄ → S S` in MeV⁻², through the t and u channels.
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xx_to_ss(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsff, gsGG, gsFF, lam, width_s, vs);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xx_to_ss(e_cm, mx, ms, gsxx)
    })
}

/// Cross section for `S S → x x̄` in MeV⁻².
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_ss_to_xx(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsff, gsGG, gsFF, lam, width_s, vs);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_ss_to_xx(e_cm, mx, ms, gsxx)
    })
}

/// Elastic cross section for `x l → x l` in MeV⁻², summed over charges.
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs, mf)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xl_to_xl(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
    mf: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsGG, gsFF, lam, vs);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xl_to_xl(e_cm, mx, ms, gsxx, gsff, width_s, mf)
    })
}

/// Elastic cross section for `x π → x π` in MeV⁻², summed over charges.
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xpi_to_xpi(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsFF,);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xpi_to_xpi(e_cm, mx, ms, gsxx, gsff, gsGG, lam, width_s, vs)
    })
}

/// Elastic cross section for `x π⁰ → x π⁰` in MeV⁻².
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xpi0_to_xpi0(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsFF,);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xpi0_to_xpi0(e_cm, mx, ms, gsxx, gsff, gsGG, lam, width_s, vs)
    })
}

/// Elastic cross section for `x γ → x γ` in MeV⁻².
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xg_to_xg(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsff, gsGG, vs);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xg_to_xg(e_cm, mx, ms, gsxx, gsFF, lam, width_s)
    })
}

/// Elastic cross section for `x S → x S` in MeV⁻².
///
/// See the module docs for the arguments.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn sigma_xs_to_xs(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/scalar_mediator/
    // _scalar_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gsff, gsGG, gsFF, lam, width_s, vs);
    map_unary(e_cms, ENERGIES, |e_cm| {
        scalar_xs::sigma_xs_to_xs(e_cm, mx, ms, gsxx)
    })
}

/// Thermally averaged `⟨σv⟩` in MeV⁻², at `x = mx / T`.
///
/// `x` is a scalar only — the Cython declared it `double`, so this entry
/// point never had the array dispatch the eleven above do.
///
/// # Errors
///
/// As [`sigma_xx_to_s_to_ff`]. Unreachable in practice — see
/// [`crate::kernels::scalar_xs::thermal_cross_section`].
#[pyfunction]
#[pyo3(text_signature = "(x, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)")]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn thermal_cross_section(
    x: f64,
    mx: f64,
    ms: f64,
    gsxx: f64,
    gsff: f64,
    gsGG: f64,
    gsFF: f64,
    lam: f64,
    width_s: f64,
    vs: f64,
) -> PyResult<f64> {
    scalar_xs::thermal_cross_section(x, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)
        .map_err(non_real)
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sigma_xx_to_s_to_ff, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_s_to_gg, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_s_to_pi0pi0, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_s_to_pipi, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_ss, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_ss_to_xx, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xl_to_xl, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xpi_to_xpi, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xpi0_to_xpi0, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xg_to_xg, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xs_to_xs, module)?)?;
    module.add_function(wrap_pyfunction!(thermal_cross_section, module)?)?;
    Ok(())
}
