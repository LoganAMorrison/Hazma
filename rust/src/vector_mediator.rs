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
//! # The decay spectrum
//!
//! [`dnde_decay_v`] and [`dnde_decay_v_pt`] are the seventh and eighth
//! entry points, landed in Task 6.2 — the whole public surface of
//! `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx`, which
//! that task deleted. Their math is in
//! [`crate::kernels::vector_decay_photon`]. They are a *pair* rather
//! than one dispatching function because the `.pyx` was: `dnde_decay_v`
//! declared `np.ndarray[double] eng_gam` and `dnde_decay_v_pt` declared
//! `double eng_gam`, and
//! `hazma/vector_mediator/_vector_mediator_spectra.py:99-102` still
//! chooses between them on `hasattr(e_gams, "__len__")`.
//!
//! Phase 06 Task 6.3 adds the positron spectrum alongside them.

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;

use crate::dispatch::{map_unary, map_unary_try, require_vector};
use crate::kernels::mediator_decay_positron;
use crate::kernels::mediator_tables::{PartialWidths, PhotonMode, PositronMode, SpectrumError};
use crate::kernels::soft_complex::NonRealResult;
use crate::kernels::vector_decay_photon;
use crate::kernels::vector_xs;
use numpy::IntoPyArray;

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

/// The wording the port gives `dnde_decay_v`'s energy argument.
///
/// The `.pyx` had no `assert` here to borrow from — it declared
/// `np.ndarray[double] eng_gam` and let Cython's own argument check and
/// buffer cast raise — so this is the spelling the argument has in the
/// `.pyx` signature, matching what the scalar twin's `assert` called the
/// same quantity.
const PHOTON_ENERGIES: &str = "Photon energies";

/// The wording the port gives `partial_widths`, as in the scalar twin.
const PARTIAL_WIDTHS: &str = "Partial widths";

/// Cython's own out-of-bounds wording, verbatim — see
/// [`crate::scalar_mediator`]'s copy for how it was measured.
const OUT_OF_BOUNDS_MESSAGE: &str = "Out of bounds on buffer access (axis 0)";

/// The wording a complex FSR coefficient reaches Python with.
///
/// The charged pion's, here, where the scalar twin's is the lepton's:
/// the two `.pyx` put the `1.5` exponent on different factors.
const NON_REAL_SPECTRUM_MESSAGE: &str = "Photon spectrum is complex at this mediator mass: the final-state \
     radiation coefficient raised to the power 3/2 divides by a vanishing \
     denominator, which happens at mv = 2 * m_pi exactly.";

/// Map a kernel failure onto the exception the Cython raised.
fn spectrum_error(error: SpectrumError) -> PyErr {
    match error {
        SpectrumError::OutOfBounds => PyIndexError::new_err(OUT_OF_BOUNDS_MESSAGE),
        SpectrumError::NonReal => PyTypeError::new_err(NON_REAL_SPECTRUM_MESSAGE),
    }
}

/// Photon `dN/dE` in MeV⁻¹ from the decay of a boosted vector mediator,
/// over a grid of energies.
///
/// # Parameters
///
/// * `eng_gam` — lab-frame photon energies in MeV, as a 1-D `float64`
///   array. Never a scalar: use [`dnde_decay_v_pt`] for that, as the
///   `.pyx` required and as the Python wrapper still does.
/// * `eng_v` — the mediator's total energy, MeV.
/// * `mv` — the mediator's mass, MeV.
/// * `pws` — the four normalised partial widths, in the order
///   `[e e, mu mu, pi0 g, pi pi]`.
/// * `mode` — the channel: `"total"`, `"e e g"`, `"pi pi g"`,
///   `"pi pi"`, `"pi0 g"`, `"mu mu g"` or `"mu mu"`. Anything else —
///   including `None` — gives `0.0` at every energy, which is what the
///   `.pyx` gives (see
///   [`crate::kernels::mediator_tables`]'s docs, and the follow-up
///   `docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`).
///
/// # Returns
///
/// A fresh 1-D `float64` array of `dN/dE` in MeV⁻¹.
///
/// # Errors
///
/// `ValueError` if either array argument is not 1-D `float64`, or has no
/// `__len__` at all; `IndexError` for a `pws` shorter than four;
/// `TypeError` where the charged-pion FSR coefficient comes back
/// complex.
///
/// Two divergences from the Cython, both on paths no working call takes.
/// A scalar `eng_gam` raised `TypeError` there and raises `ValueError`
/// here (`"Photon energies must be a list or array."`), and a `list` was
/// refused there and is accepted here — the same widening
/// [`crate::dispatch`] declares for every other entry point.
#[pyfunction]
#[pyo3(text_signature = "(eng_gam, eng_v, mv, pws, mode)")]
fn dnde_decay_v(
    py: Python<'_>,
    eng_gam: &Bound<'_, PyAny>,
    eng_v: f64,
    mv: f64,
    pws: &Bound<'_, PyAny>,
    mode: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let energies = require_vector(eng_gam, PHOTON_ENERGIES)?;
    let widths = require_vector(pws, PARTIAL_WIDTHS)?;
    let selected = mode.and_then(PhotonMode::parse);
    let tables = vector_decay_photon::tables_for(mv);
    let widths = PartialWidths::new(&widths);

    let spectrum = energies
        .iter()
        .map(|&energy| {
            vector_decay_photon::spectrum_point(energy, eng_v, mv, widths, selected, &tables)
                .map_err(spectrum_error)
        })
        .collect::<PyResult<Vec<f64>>>()?;
    Ok(spectrum.into_pyarray(py).into_any().unbind())
}

/// Photon `dN/dE` in MeV⁻¹ from the decay of a boosted vector mediator,
/// at one energy.
///
/// The scalar-argument twin of [`dnde_decay_v`]; arguments and errors are
/// that function's, except that `eng_gam` is a single energy in MeV and
/// PyO3 raises the same `TypeError` for a non-number that CPython raised
/// at the `.pyx`'s `double eng_gam`.
///
/// # Errors
///
/// As [`dnde_decay_v`], minus the `eng_gam` array cases.
#[pyfunction]
#[pyo3(text_signature = "(eng_gam, eng_v, mv, pws, mode)")]
fn dnde_decay_v_pt(
    eng_gam: f64,
    eng_v: f64,
    mv: f64,
    pws: &Bound<'_, PyAny>,
    mode: Option<&str>,
) -> PyResult<f64> {
    let widths = require_vector(pws, PARTIAL_WIDTHS)?;
    let selected = mode.and_then(PhotonMode::parse);
    let tables = vector_decay_photon::tables_for(mv);

    vector_decay_photon::spectrum_point(
        eng_gam,
        eng_v,
        mv,
        PartialWidths::new(&widths),
        selected,
        &tables,
    )
    .map_err(spectrum_error)
}

/// The wording the port gives the positron module's energy argument.
///
/// The `.pyx` did not check it — `np.ndarray[double] eng_ps` let the
/// buffer cast raise — so this is the spelling the argument has in its
/// signature, as with [`PHOTON_ENERGIES`] above.
const POSITRON_ENERGIES: &str = "Positron energies";

/// Positron `dN/dE` in MeV⁻¹ from the decay of a boosted vector
/// mediator, over a grid of energies.
///
/// `hazma/vector_mediator/vector_mediator_positron_spec.pyx` called this
/// `dnde_decay_v`, which is the name [`dnde_decay_v`] above already has
/// from the *photon* module Task 6.2 ported. Both `.pyx` exported that
/// spelling from different extensions; one PyO3 submodule cannot, so the
/// positron pair is spelled out here and
/// `hazma/vector_mediator/_vector_mediator_positron_spectra.py` imports
/// it under the Cython name. [`crate::scalar_mediator`]'s twin follows
/// the same spelling even though nothing there collides.
///
/// # Parameters
///
/// * `eng_ps` — lab-frame positron energies in MeV, as a 1-D `float64`
///   array. Never a scalar: use [`dnde_positron_decay_v_pt`] for that, as
///   the `.pyx` required and as the Python wrapper still does.
/// * `eng_v` — the mediator's total energy, MeV.
/// * `mv` — the mediator's mass, MeV.
/// * `pws` — the three normalised partial widths, in the order
///   `[e e, mu mu, pi pi]` — **not** the four-element `[e e, mu mu,
///   pi0 g, pi pi]` the photon pair takes.
/// * `fs` — the channel: `"total"`, `"e e"`, `"mu mu"` or `"pi pi"`.
///   Anything else — including `None` — gives `0.0` at every energy,
///   which is what the `.pyx` gives.
///
/// # Returns
///
/// A fresh 1-D `float64` array of `dN/dE` in MeV⁻¹.
///
/// # Errors
///
/// As [`crate::scalar_mediator`]'s twin: `ValueError` for a rank or
/// dtype violation in either array argument, `IndexError` for a `pws`
/// shorter than three once an index it does not have is read.
#[pyfunction]
#[pyo3(text_signature = "(eng_ps, eng_v, mv, pws, fs)")]
fn dnde_positron_decay_v(
    py: Python<'_>,
    eng_ps: &Bound<'_, PyAny>,
    eng_v: f64,
    mv: f64,
    pws: &Bound<'_, PyAny>,
    fs: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let energies = require_vector(eng_ps, POSITRON_ENERGIES)?;
    let widths = require_vector(pws, PARTIAL_WIDTHS)?;
    let selected = fs.and_then(PositronMode::parse);
    let tables = mediator_decay_positron::tables_for(mv);
    let widths = PartialWidths::new(&widths);

    let spectrum = energies
        .iter()
        .map(|&energy| {
            mediator_decay_positron::spectrum_point(energy, eng_v, mv, widths, selected, &tables)
                .map_err(spectrum_error)
        })
        .collect::<PyResult<Vec<f64>>>()?;
    Ok(spectrum.into_pyarray(py).into_any().unbind())
}

/// Positron `dN/dE` in MeV⁻¹ from the decay of a boosted vector
/// mediator, at one energy.
///
/// The scalar-argument twin of [`dnde_positron_decay_v`], and
/// `dnde_decay_v_pt` in the `.pyx` that exported it.
///
/// # Errors
///
/// As [`dnde_positron_decay_v`], minus the `eng_ps` array cases.
#[pyfunction]
#[pyo3(text_signature = "(eng_p, eng_v, mv, pws, fs)")]
fn dnde_positron_decay_v_pt(
    eng_p: f64,
    eng_v: f64,
    mv: f64,
    pws: &Bound<'_, PyAny>,
    fs: Option<&str>,
) -> PyResult<f64> {
    let widths = require_vector(pws, PARTIAL_WIDTHS)?;
    let selected = fs.and_then(PositronMode::parse);
    let tables = mediator_decay_positron::tables_for(mv);

    mediator_decay_positron::spectrum_point(
        eng_p,
        eng_v,
        mv,
        PartialWidths::new(&widths),
        selected,
        &tables,
    )
    .map_err(spectrum_error)
}

/// Populate the submodule.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sigma_xx_to_v_to_ff, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_v_to_pipi, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_v_to_pi0g, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_v_to_pi0v, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_xx_to_vv, module)?)?;
    module.add_function(wrap_pyfunction!(thermal_cross_section, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_decay_v, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_decay_v_pt, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_positron_decay_v, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_positron_decay_v_pt, module)?)?;
    Ok(())
}
