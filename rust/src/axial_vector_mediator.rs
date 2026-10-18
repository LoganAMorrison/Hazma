//! `hazma._core.axial_vector_mediator` — rates of the axial-vector
//! mediator model.
//!
//! Registration only: the math is in [`crate::kernels::axial_rates`]. The
//! caller is `hazma/axial_vector_mediator/_model.py`, which validates every
//! argument and flattens it, so each entry point takes flat batches of
//! equal length and returns one value per sample.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kernels::axial_rates::{self, DarkSector, FermionPair};

/// `(m, g_V^χ, g_A^χ, M, Γ)`, masses and width in MeV.
type Dark = (f64, f64, f64, f64, f64);

fn same_length(lengths: &[usize]) -> PyResult<()> {
    if lengths.windows(2).all(|pair| pair[0] == pair[1]) {
        Ok(())
    } else {
        Err(PyValueError::new_err("batches must have equal lengths."))
    }
}

/// Return the tree-level `(A_T, A_L)` of a fermion pair in MeV² at each
/// energy.
#[pyfunction]
fn fermion_pair_densities<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<'py, f64>,
    mass: f64,
    vector: f64,
    axial: f64,
    colors: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let fermion = FermionPair {
        mass,
        vector,
        axial,
        colors,
    };
    let (transverse, longitudinal): (Vec<f64>, Vec<f64>) = energies
        .as_array()
        .iter()
        .map(|&q| fermion.densities(q))
        .unzip();
    (transverse.into_pyarray(py), longitudinal.into_pyarray(py))
}

/// Return `σ v_Møller` in MeV⁻² from per-energy densities in MeV².
///
/// # Errors
///
/// `ValueError` if the three batches differ in length.
#[pyfunction]
fn sigma_v<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<'py, f64>,
    transverse: PyReadonlyArray1<'py, f64>,
    longitudinal: PyReadonlyArray1<'py, f64>,
    dark: Dark,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (dm_mass, dm_vector, dm_axial, mediator_mass, mediator_width) = dark;
    let sector = DarkSector {
        dm_mass,
        dm_vector,
        dm_axial,
        mediator_mass,
        mediator_width,
    };
    let (energies, transverse, longitudinal) = (
        energies.as_array(),
        transverse.as_array(),
        longitudinal.as_array(),
    );
    same_length(&[energies.len(), transverse.len(), longitudinal.len()])?;
    let values: Vec<f64> = energies
        .iter()
        .zip(transverse.iter().zip(longitudinal.iter()))
        .map(|(&q, (&t, &l))| sector.sigma_v(q, t, l))
        .collect();
    Ok(values.into_pyarray(py))
}

/// Return `σ = σ v_Møller / v_Møller` in MeV⁻² at each energy.
///
/// # Errors
///
/// `ValueError` if the two batches differ in length.
#[pyfunction]
fn cross_section<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<'py, f64>,
    sigma_v: PyReadonlyArray1<'py, f64>,
    dm_mass: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (energies, sigma_v) = (energies.as_array(), sigma_v.as_array());
    same_length(&[energies.len(), sigma_v.len()])?;
    let values: Vec<f64> = energies
        .iter()
        .zip(sigma_v.iter())
        .map(|(&q, &sv)| axial_rates::cross_section(q, sv, dm_mass))
        .collect();
    Ok(values.into_pyarray(py))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fermion_pair_densities, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_v, module)?)?;
    module.add_function(wrap_pyfunction!(cross_section, module)?)?;
    Ok(())
}
