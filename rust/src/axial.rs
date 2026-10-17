//! `hazma._core.axial` — axial current kernels.
//!
//! Registration only: the math is in [`crate::kernels::axial_tensors`] and
//! [`crate::kernels::axial_three_pi`], and the batch layout in
//! [`crate::dispatch::components`]. The callers are the private wrappers in
//! `hazma/form_factors/axial/`, which validate every argument, broadcast it
//! and reshape it to `(components, N)`, so each entry point here takes a
//! flat batch and returns one value, or one `(components, N)` array, per
//! sample. The one error raised here, [`tensor_projections`]'s, is a
//! property of the numbers rather than of the arguments, and its messages
//! are the public ones.

use num_complex::Complex64;
use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dispatch::{component_major, components};
use crate::kernels::axial_tensors::{self, ProjectionError};
use crate::kernels::axial_three_pi::{Channel, Momenta, ThreePion, angular_rule};

type Projections<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

fn projections(py: Python<'_>, pairs: impl Iterator<Item = (f64, f64)>) -> Projections<'_> {
    let (transverse, longitudinal): (Vec<f64>, Vec<f64>) = pairs.unzip();
    (transverse.into_pyarray(py), longitudinal.into_pyarray(py))
}

/// Return `H^μ H^ν*` for a `(4, N)` complex batch as `(16, N)`, row-major
/// in `(μ, ν)`.
#[pyfunction]
fn hadronic_tensor<'py>(
    py: Python<'py>,
    current: PyReadonlyArray2<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let currents = components::<_, 4>(&current, "current")?;
    Ok(component_major(
        py,
        currents.samples().map(|h| {
            let tensor = axial_tensors::hadronic_tensor(&h);
            std::array::from_fn::<_, 16, _>(|k| tensor[k / 4][k % 4])
        }),
    ))
}

/// Return the `(T, L)` projections of a `(4, N)` complex current batch
/// against a `(4, N)` real momentum batch.
#[pyfunction]
fn current_projections<'py>(
    py: Python<'py>,
    current: PyReadonlyArray2<'py, Complex64>,
    q: PyReadonlyArray2<'py, f64>,
) -> PyResult<Projections<'py>> {
    let currents = components::<_, 4>(&current, "current")?;
    let momenta = components::<_, 4>(&q, "q")?;
    Ok(projections(
        py,
        currents
            .samples()
            .zip(momenta.samples())
            .map(|(h, q)| axial_tensors::current_projections(&h, &q)),
    ))
}

/// Return the `(T, L)` projections of a `(16, N)` Hermitian tensor batch,
/// row-major in `(μ, ν)`, against a `(4, N)` momentum batch.
///
/// # Errors
///
/// `ValueError` if any sample overflows, has a materially negative
/// projected density, or is ill-conditioned for its `q` at `rtol`. A batch
/// failing in several ways reports the overflow first, then the negative
/// density.
#[pyfunction]
fn tensor_projections<'py>(
    py: Python<'py>,
    tensor: PyReadonlyArray2<'py, Complex64>,
    q: PyReadonlyArray2<'py, f64>,
    rtol: f64,
) -> PyResult<Projections<'py>> {
    let tensors = components::<_, 16>(&tensor, "tensor")?;
    let momenta = components::<_, 4>(&q, "q")?;
    let results: Vec<Result<(f64, f64), ProjectionError>> = tensors
        .samples()
        .zip(momenta.samples())
        .map(|(w, q)| {
            let rows: [[Complex64; 4]; 4] =
                std::array::from_fn(|mu| std::array::from_fn(|nu| w[4 * mu + nu]));
            axial_tensors::tensor_projections(&rows, &q, rtol)
        })
        .collect();
    if let Some(error) = results.iter().filter_map(|r| r.err()).max() {
        return Err(PyValueError::new_err(match error {
            ProjectionError::Overflow => "The tensor calculation overflowed; rescale the inputs.",
            ProjectionError::NegativeDensity => "tensor has a negative projected density.",
            ProjectionError::IllConditioned => {
                "tensor projection is ill-conditioned for this q: its roundoff bound \
                 exceeds rtol times the rest-frame norm. Project the current with \
                 current_projections or supply the tensor in the q rest frame."
            }
        }));
    }
    Ok(projections(py, results.into_iter().flatten()))
}

/// Integrate `(M, K)` transverse and longitudinal densities against
/// `(M, K)` weights over their last axis.
///
/// # Errors
///
/// `ValueError` if the three arrays differ in shape or are not
/// C-contiguous.
#[pyfunction]
fn integrate_spectral_functions<'py>(
    py: Python<'py>,
    transverse: PyReadonlyArray2<'py, f64>,
    longitudinal: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, f64>,
) -> PyResult<Projections<'py>> {
    let shape = weights.shape();
    if transverse.shape() != shape || longitudinal.shape() != shape {
        return Err(PyValueError::new_err(
            "densities and weights must share one shape.",
        ));
    }
    let (rows, width) = (shape[0], shape[1]);
    let (transverse, longitudinal, measure) = (
        contiguous(&transverse)?,
        contiguous(&longitudinal)?,
        contiguous(&weights)?,
    );
    Ok(projections(
        py,
        (0..rows).map(|i| {
            let row = i * width..(i + 1) * width;
            (
                axial_tensors::integrate(&transverse[row.clone()], &measure[row.clone()]),
                axial_tensors::integrate(&longitudinal[row.clone()], &measure[row]),
            )
        }),
    ))
}

fn contiguous<'a>(array: &'a PyReadonlyArray2<'_, f64>) -> PyResult<&'a [f64]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err("densities and weights must be C-contiguous."))
}

fn three_pi(pion_mass: f64, decay_constant: f64, neutral: bool) -> ThreePion {
    ThreePion {
        channel: if neutral {
            Channel::Neutral
        } else {
            Channel::Charged
        },
        pion_mass,
        decay_constant,
    }
}

/// Split the 12 components of one `(4, 3)` row-major event into rows.
fn event(components: [f64; 12]) -> Momenta {
    std::array::from_fn(|mu| std::array::from_fn(|i| components[3 * mu + i]))
}

/// Return the LO three-pion current of a `(12, N)` momentum batch, row-major
/// in `(μ, particle)`, as `(4, N)`, dimensionless.
#[pyfunction]
fn three_pi_current<'py>(
    py: Python<'py>,
    momenta: PyReadonlyArray2<'py, f64>,
    g3: f64,
    pion_mass: f64,
    decay_constant: f64,
    neutral: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let events = components::<_, 12>(&momenta, "momenta")?;
    let model = three_pi(pion_mass, decay_constant, neutral);
    Ok(component_major(
        py,
        events.samples().map(|p| model.current(&event(p), g3)),
    ))
}

/// Return the spin-averaged weight of a `(12, N)` momentum batch,
/// dimensionless.
#[pyfunction]
fn three_pi_differential_weight<'py>(
    py: Python<'py>,
    momenta: PyReadonlyArray2<'py, f64>,
    g3: f64,
    pion_mass: f64,
    decay_constant: f64,
    neutral: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let events = components::<_, 12>(&momenta, "momenta")?;
    let model = three_pi(pion_mass, decay_constant, neutral);
    let weights: Vec<f64> = events
        .samples()
        .map(|p| model.differential_weight(&event(p), g3))
        .collect();
    Ok(weights.into_pyarray(py))
}

/// Return `(A_T, A_L)` in MeV² at each energy of a 1-D batch (MeV), with
/// a `quadrature_order`-point Gauss–Legendre rule.
#[pyfunction]
fn three_pi_spectral_functions<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<'py, f64>,
    g3: f64,
    pion_mass: f64,
    decay_constant: f64,
    neutral: bool,
    quadrature_order: usize,
) -> PyResult<Projections<'py>> {
    let model = three_pi(pion_mass, decay_constant, neutral);
    let rule = angular_rule(quadrature_order);
    Ok(projections(
        py,
        energies
            .as_array()
            .iter()
            .map(|&energy| model.spectral_functions(energy, g3, &rule)),
    ))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hadronic_tensor, module)?)?;
    module.add_function(wrap_pyfunction!(current_projections, module)?)?;
    module.add_function(wrap_pyfunction!(tensor_projections, module)?)?;
    module.add_function(wrap_pyfunction!(integrate_spectral_functions, module)?)?;
    module.add_function(wrap_pyfunction!(three_pi_current, module)?)?;
    module.add_function(wrap_pyfunction!(three_pi_differential_weight, module)?)?;
    module.add_function(wrap_pyfunction!(three_pi_spectral_functions, module)?)?;
    Ok(())
}
