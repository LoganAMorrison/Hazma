//! `hazma._core.vector_mediator` — vector-mediator kernels.
//!
//! Registration only: the math is in [`crate::kernels::vector_xs`] and
//! the argument and error handling in [`crate::dispatch`], so nothing
//! here computes or classifies anything
//! (`projects/cython-to-rust/rules.md`, Rust conventions rules 2–3).
//!
//! Six entry points, all landed in cython-to-rust Task 5.1 —
//! every consumed public `def` of
//! `hazma/vector_mediator/_c_vector_mediator_cross_sections.pyx`, which
//! that task deleted. The seventh, `sigma_xx_to_all`, is deliberately
//! **not** here: nothing in hazma imported it, so the plan drops it
//! rather than porting it (`phase-05-mediator-cross-sections.md`). It
//! survives as a private helper of the thermal integrand, which is the
//! only caller it ever had.
//!
//! Phase 06 adds the spectrum modules alongside these.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::dispatch::{map_unary, map_unary_try};
use crate::kernels::vector_xs::{self, NonRealResult};

/// The wording every `NonRealResult` reaches Python with.
///
/// Not the Cython's. `__Pyx_SoftComplexToDouble` says "Cannot convert
/// 'complex' with non-zero imaginary component to 'double' (this most
/// likely comes from the '**' operator; use 'cython.cpow(True)' to return
/// 'nan' instead of a complex number)", and two thirds of that is advice
/// about a Cython compiler directive — advice that is wrong in a tree
/// with no Cython in it and worse than useless after Phase 07. The
/// **type** is what the port owes (`crate::dispatch`'s module docs, and
/// what `test/parity/generate.py` records), so `TypeError` it stays,
/// with a message that names the same cause and the argument that
/// produced it.
const NON_REAL_MESSAGE: &str = "Cross section is complex at this center-of-mass energy: the \
     kinematic factor raised to the power 3/2 divides by a vanishing \
     denominator, which happens at e_cm = 2 * mx exactly.";

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

/// Cross section for `x x̄ → V* → f f̄` in MeV⁻².
///
/// `e_cms` is the mapped argument — a float, a NumPy scalar, a 0-d
/// numeric array, or a 1-D `float64` array (or a sequence that converts
/// to one). Every other argument is a scalar in MeV or dimensionless:
/// `mx` and `mv` are the dark-matter and mediator masses, `gvxx` and
/// `gvll` the mediator's couplings to dark matter and to the final-state
/// lepton, `width_v` the mediator's full decay width in MeV, and `ml` the
/// lepton mass in MeV.
///
/// The advertised signature is not positional-only: the Cython entry
/// point was a `def` and accepted every argument by keyword.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, mv, gvxx, gvll, width_v, ml)")]
#[allow(clippy::too_many_arguments)]
fn sigma_xx_to_v_to_ff(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvll: f64,
    width_v: f64,
    ml: f64,
) -> PyResult<Py<PyAny>> {
    map_unary(e_cms, ENERGIES, |e_cm| {
        vector_xs::sigma_xx_to_v_to_ff(e_cm, mx, mv, gvxx, gvll, width_v, ml)
    })
}

/// Cross section for `x x̄ → V* → π⁺π⁻` in MeV⁻².
///
/// `e_cms` is the mapped argument; see [`sigma_xx_to_v_to_ff`]. `gvuu`,
/// `gvdd`, `gvss`, `gvee` and `gvmumu` are the mediator's couplings to
/// up, down and strange quarks, electrons and muons — the last three are
/// unused by this channel and are kept because the Cython signature had
/// them and the wrapper passes them positionally.
///
/// # Errors
///
/// `TypeError` at `e_cm = 2 mx`, where the Cython's complex `**` operator
/// produced a non-real result. See [`NON_REAL_MESSAGE`].
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)")]
#[allow(clippy::too_many_arguments)]
fn sigma_xx_to_v_to_pipi(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    gvss: f64,
    gvee: f64,
    gvmumu: f64,
    width_v: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/vector_mediator/
    // _vector_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gvss, gvee, gvmumu);
    map_unary_try(e_cms, ENERGIES, |e_cm| {
        vector_xs::sigma_xx_to_v_to_pipi(e_cm, mx, mv, gvxx, gvuu, gvdd, width_v).map_err(non_real)
    })
}

/// Cross section for `x x̄ → V* → π⁰ γ` in MeV⁻².
///
/// Arguments as [`sigma_xx_to_v_to_pipi`].
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)")]
#[allow(clippy::too_many_arguments)]
fn sigma_xx_to_v_to_pi0g(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    gvss: f64,
    gvee: f64,
    gvmumu: f64,
    width_v: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/vector_mediator/
    // _vector_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gvss, gvee, gvmumu);
    map_unary(e_cms, ENERGIES, |e_cm| {
        vector_xs::sigma_xx_to_v_to_pi0g(e_cm, mx, mv, gvxx, gvuu, gvdd, width_v)
    })
}

/// Cross section for `x x̄ → V* → π⁰ V` in MeV⁻².
///
/// Arguments as [`sigma_xx_to_v_to_pipi`].
///
/// # Errors
///
/// As [`sigma_xx_to_v_to_pipi`].
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)")]
#[allow(clippy::too_many_arguments)]
fn sigma_xx_to_v_to_pi0v(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    gvss: f64,
    gvee: f64,
    gvmumu: f64,
    width_v: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/vector_mediator/
    // _vector_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gvss, gvee, gvmumu);
    map_unary_try(e_cms, ENERGIES, |e_cm| {
        vector_xs::sigma_xx_to_v_to_pi0v(e_cm, mx, mv, gvxx, gvuu, gvdd, width_v).map_err(non_real)
    })
}

/// Cross section for `x x̄ → V V` in MeV⁻², through the t and u channels.
///
/// Arguments as [`sigma_xx_to_v_to_pipi`]; this channel uses none of the
/// quark or lepton couplings, only `gvxx`.
#[pyfunction]
#[pyo3(text_signature = "(e_cms, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)")]
#[allow(clippy::too_many_arguments)]
fn sigma_xx_to_vv(
    e_cms: &Bound<'_, PyAny>,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    gvss: f64,
    gvee: f64,
    gvmumu: f64,
    width_v: f64,
) -> PyResult<Py<PyAny>> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/vector_mediator/
    // _vector_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = (gvuu, gvdd, gvss, gvee, gvmumu, width_v);
    map_unary(e_cms, ENERGIES, |e_cm| {
        vector_xs::sigma_xx_to_vv(e_cm, mx, mv, gvxx)
    })
}

/// Thermally averaged `⟨σv⟩` in MeV⁻², at `x = mx / T`.
///
/// `x` is a scalar only — the Cython declared it `double`, so this entry
/// point never had the array dispatch the five above do, and PyO3 raises
/// the same `TypeError` for a non-number that CPython raised there.
///
/// # Errors
///
/// As [`sigma_xx_to_v_to_pipi`], if the integrand reaches the threshold.
/// Unreachable in practice — see
/// [`crate::kernels::vector_xs::thermal_cross_section`].
#[pyfunction]
#[pyo3(text_signature = "(x, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)")]
#[allow(clippy::too_many_arguments)]
fn thermal_cross_section(
    x: f64,
    mx: f64,
    mv: f64,
    gvxx: f64,
    gvuu: f64,
    gvdd: f64,
    gvss: f64,
    gvee: f64,
    gvmumu: f64,
    width_v: f64,
) -> PyResult<f64> {
    // Declared, accepted by keyword, and unused by this channel: the
    // Cython signature had them and `hazma/vector_mediator/
    // _vector_mediator_cross_sections.py` still passes all ten, so
    // dropping or renaming one would narrow the public API.
    let _ = gvss;
    vector_xs::thermal_cross_section(x, mx, mv, gvxx, gvuu, gvdd, gvee, gvmumu, width_v)
        .map_err(non_real)
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sigma_xx_to_v_to_ff, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_v_to_pipi, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_v_to_pi0g, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_v_to_pi0v, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_vv, module)?)?;
    module.add_function(wrap_pyfunction!(thermal_cross_section, module)?)?;
    Ok(())
}
