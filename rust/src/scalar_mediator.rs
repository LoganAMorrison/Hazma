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
//! # The decay spectrum
//!
//! [`scalar_mediator_decay_spectrum`] is the thirteenth entry point,
//! landed in Task 6.2 — the whole public surface of
//! `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx`, which
//! that task deleted. Its math is in
//! [`crate::kernels::scalar_decay_photon`]. It is the only function here
//! that takes an array argument other than the mapped one
//! (`partial_widths`) and the only one with a container argument
//! (`modes`), and both are handled below rather than in a kernel because
//! both are Python-object questions (`rules.md` rule 8).
//!
//! # The positron spectrum
//!
//! [`dnde_positron_decay_s`] and [`dnde_positron_decay_s_pt`] are the
//! fourteenth and fifteenth, landed in Task 6.3 — the whole public
//! surface of `hazma/scalar_mediator/scalar_mediator_positron_spec.pyx`,
//! which that task deleted. Both were `dnde_decay_s`/`dnde_decay_s_pt`
//! in the Cython, and
//! `hazma/scalar_mediator/_scalar_mediator_positron_spectra.py` still
//! imports them under those names; they are spelled out here because the
//! vector twin's Cython names collide with the photon pair Task 6.2
//! registered in [`crate::vector_mediator`], and the two models are
//! easier to read named alike than named differently for a reason that
//! only applies to one of them.
//!
//! Their math is in [`crate::kernels::mediator_decay_positron`], which is
//! one module serving both models because the two `.pyx` were the same
//! text.

use numpy::IntoPyArray;
use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;

use crate::dispatch::{map_unary, map_unary_try, require_vector};
use crate::kernels::mediator_decay_positron;
use crate::kernels::mediator_tables::{
    PartialWidths, PositronMode, ScalarPhotonModes, SpectrumError,
};
use crate::kernels::scalar_decay_photon;
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

/// The wording the `.pyx`'s rank `assert` gave the mapped argument.
///
/// `scalar_mediator_decay_spectrum.pyx:270` said `"Photon energies must
/// be 0 or 1-dimensional."`, and the port reproduces the message and
/// promotes the `AssertionError` to a `ValueError` (`rules.md` rule 9).
const PHOTON_ENERGIES: &str = "Photon energies";

/// The wording the `.pyx`'s two `partial_widths` checks gave it.
///
/// Both were that call site's own text — the `raise ValueError("Partial
/// widths must be a list or array.")` at `:249` and the `assert ...,
/// "Partial widths must be 1-dimensional."` at `:251` —
/// and [`require_vector`] emits both verbatim.
const PARTIAL_WIDTHS: &str = "Partial widths";

/// Cython's own out-of-bounds wording, verbatim.
///
/// `@cython.boundscheck(True)` on the integrand and the entry point
/// means a short `partial_widths` raised
/// `IndexError: Out of bounds on buffer access (axis 0)`, measured
/// against the shipped 2.1.0 extension rather than read off the
/// generated C. The axis is the only one a 1-D buffer has.
const OUT_OF_BOUNDS_MESSAGE: &str = "Out of bounds on buffer access (axis 0)";

/// The wording a complex FSR coefficient reaches Python with.
///
/// Not the Cython's, for the reason [`crate::vector_mediator`]'s
/// `NON_REAL_MESSAGE` gives: `__Pyx_SoftComplexToDouble`'s text is two
/// thirds advice about a Cython compiler directive. The **type** is what
/// the port owes, so `TypeError` it stays.
const NON_REAL_SPECTRUM_MESSAGE: &str = "Photon spectrum is complex at this mediator mass: the final-state \
     radiation coefficient raised to the power 3/2 divides by a vanishing \
     denominator, which happens at ms = 2 * m_lepton exactly.";

/// Map a kernel failure onto the exception the Cython raised.
fn spectrum_error(error: SpectrumError) -> PyErr {
    match error {
        SpectrumError::OutOfBounds => PyIndexError::new_err(OUT_OF_BOUNDS_MESSAGE),
        SpectrumError::NonReal => PyTypeError::new_err(NON_REAL_SPECTRUM_MESSAGE),
    }
}

/// Photon `dN/dE` in MeV⁻¹ from the decay of a boosted scalar mediator.
///
/// # Parameters
///
/// * `photon_energies` — the mapped argument: lab-frame photon energies
///   in MeV, as a float, a NumPy scalar, a 0-d numeric array, or a 1-D
///   `float64` array (or a sequence that converts to one).
/// * `sm_energy` — the mediator's total energy, MeV.
/// * `sm_mass` — the mediator's mass, MeV.
/// * `partial_widths` — the five normalised partial widths, in the order
///   `[e e, mu mu, pi0 pi0, pi pi, g g]`, as a 1-D `float64` array or a
///   sequence that converts to one.
/// * `modes` — any container of mode names; membership is decided with
///   Python's `in`, as the `.pyx` decided it, so a `str` works and sets
///   every bit whose name it contains. Omitted means all seven.
///
/// # Returns
///
/// `dN/dE` in MeV⁻¹: a float for a scalar argument, a fresh 1-D
/// `float64` array for a grid.
///
/// # Errors
///
/// `ValueError` for a rank or dtype violation in either array argument,
/// `TypeError` for a `photon_energies` that is neither a real number nor
/// a sequence, whatever `modes.__contains__` raises, `IndexError` for a
/// `partial_widths` too short for the channels asked for, and `TypeError`
/// where an FSR coefficient comes back complex.
#[pyfunction]
#[pyo3(
    text_signature = "(photon_energies, sm_energy, sm_mass, partial_widths, \
                      modes=['pi pi', 'mu mu', 'pi0 pi0', 'g g', 'e e g', \
                      'pi pi g', 'mu mu g'])"
)]
#[pyo3(signature = (photon_energies, sm_energy, sm_mass, partial_widths, modes=None))]
fn scalar_mediator_decay_spectrum(
    photon_energies: &Bound<'_, PyAny>,
    sm_energy: f64,
    sm_mass: f64,
    partial_widths: &Bound<'_, PyAny>,
    modes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let widths = require_vector(partial_widths, PARTIAL_WIDTHS)?;
    let selected = scalar_photon_modes(modes)?;
    // Built once per call. The `.pyx` rebuilt a 500-point,
    // quadrature-backed table on *every* call because its cache was
    // never populated; the memo in
    // `crate::kernels::mediator_tables` returns the same numbers from
    // the same inputs, so this is performance only (`rules.md` rules 3
    // and 12).
    let tables = scalar_decay_photon::tables_for(sm_mass);
    let pws = PartialWidths::new(&widths);

    map_unary_try(photon_energies, PHOTON_ENERGIES, |eng_gam| {
        scalar_decay_photon::spectrum_point(eng_gam, sm_energy, sm_mass, pws, selected, &tables)
            .map_err(spectrum_error)
    })
}

/// Fold `modes` into a bit set with Python's `in`, one name at a time.
///
/// The `.pyx` wrote `if "pi pi" in modes: bitflag += BITFLAG_PP` seven
/// times (`:253-266`), so membership is `__contains__` and not a list
/// comparison — `modes="pi pi g"` sets the `"pi pi"` and `"pi pi g"`
/// bits today by substring, and a set or a tuple works as well as a
/// list. Testing each name once also means a repeated entry cannot
/// double a flag into its neighbour's bit.
///
/// `None` means the argument was omitted, and the `.pyx`'s default is
/// every mode. Passing `modes=None` explicitly raised `TypeError` there
/// (`"pi pi" in None`) and takes the default here — a divergence no
/// working call can notice, since no working call passes it.
fn scalar_photon_modes(modes: Option<&Bound<'_, PyAny>>) -> PyResult<ScalarPhotonModes> {
    let Some(modes) = modes else {
        return Ok(ScalarPhotonModes::from_names(ScalarPhotonModes::NAMES));
    };
    let mut bits = 0;
    for name in ScalarPhotonModes::NAMES {
        if modes.contains(name)? {
            bits |= ScalarPhotonModes::bit_for(name)
                .expect("every NAMES entry has a bit by construction");
        }
    }
    Ok(ScalarPhotonModes::from_bits(bits))
}

/// The wording the port gives the positron modules' energy argument.
///
/// Neither `.pyx` checked it — `np.ndarray[double] eng_ps` let the
/// buffer cast raise — so this is the spelling the argument has in the
/// `.pyx` signature, as in [`crate::vector_mediator`]'s photon copy.
const POSITRON_ENERGIES: &str = "Positron energies";

/// Positron `dN/dE` in MeV⁻¹ from the decay of a boosted scalar
/// mediator, over a grid of energies.
///
/// # Parameters
///
/// * `eng_ps` — lab-frame positron energies in MeV, as a 1-D `float64`
///   array. Never a scalar: use [`dnde_positron_decay_s_pt`] for that, as
///   the `.pyx` required and as the Python wrapper still does.
/// * `eng_s` — the mediator's total energy, MeV.
/// * `ms` — the mediator's mass, MeV.
/// * `pws` — the three normalised partial widths, in the order
///   `[e e, mu mu, pi pi]`.
/// * `fs` — the channel: `"total"`, `"e e"`, `"mu mu"` or `"pi pi"`.
///   Anything else — including `None` — gives `0.0` at every energy,
///   which is what the `.pyx` gives (see
///   `docs/followups/todo/mediator-spectra-accept-unknown-mode-strings.md`).
///
/// # Returns
///
/// A fresh 1-D `float64` array of `dN/dE` in MeV⁻¹.
///
/// # Errors
///
/// `ValueError` if either array argument is not 1-D `float64`, or has no
/// `__len__` at all; `IndexError` for a `pws` shorter than three, once
/// an index it does not have is actually read.
///
/// The same two divergences from the Cython [`crate::vector_mediator`]'s
/// photon pair declares, and on the same paths no working call takes: a
/// scalar `eng_ps` raised `TypeError` there and raises `ValueError` here,
/// and a `list` was refused there and is accepted here.
#[pyfunction]
#[pyo3(text_signature = "(eng_ps, eng_s, ms, pws, fs)")]
fn dnde_positron_decay_s(
    py: Python<'_>,
    eng_ps: &Bound<'_, PyAny>,
    eng_s: f64,
    ms: f64,
    pws: &Bound<'_, PyAny>,
    fs: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let energies = require_vector(eng_ps, POSITRON_ENERGIES)?;
    let widths = require_vector(pws, PARTIAL_WIDTHS)?;
    let selected = fs.and_then(PositronMode::parse);
    let tables = mediator_decay_positron::tables_for(ms);
    let widths = PartialWidths::new(&widths);

    let spectrum = energies
        .iter()
        .map(|&energy| {
            mediator_decay_positron::spectrum_point(energy, eng_s, ms, widths, selected, &tables)
                .map_err(spectrum_error)
        })
        .collect::<PyResult<Vec<f64>>>()?;
    Ok(spectrum.into_pyarray(py).into_any().unbind())
}

/// Positron `dN/dE` in MeV⁻¹ from the decay of a boosted scalar
/// mediator, at one energy.
///
/// The scalar-argument twin of [`dnde_positron_decay_s`]; arguments and
/// errors are that function's, except that `eng_p` is a single energy in
/// MeV and PyO3 raises the same `TypeError` for a non-number that CPython
/// raised at the `.pyx`'s `double eng_p`.
///
/// # Errors
///
/// As [`dnde_positron_decay_s`], minus the `eng_ps` array cases.
#[pyfunction]
#[pyo3(text_signature = "(eng_p, eng_s, ms, pws, fs)")]
fn dnde_positron_decay_s_pt(
    eng_p: f64,
    eng_s: f64,
    ms: f64,
    pws: &Bound<'_, PyAny>,
    fs: Option<&str>,
) -> PyResult<f64> {
    let widths = require_vector(pws, PARTIAL_WIDTHS)?;
    let selected = fs.and_then(PositronMode::parse);
    let tables = mediator_decay_positron::tables_for(ms);

    mediator_decay_positron::spectrum_point(
        eng_p,
        eng_s,
        ms,
        PartialWidths::new(&widths),
        selected,
        &tables,
    )
    .map_err(spectrum_error)
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
    module.add_function(wrap_pyfunction!(scalar_mediator_decay_spectrum, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_positron_decay_s, module)?)?;
    module.add_function(wrap_pyfunction!(dnde_positron_decay_s_pt, module)?)?;
    Ok(())
}
