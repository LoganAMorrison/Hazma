//! `hazma._core.axial_vector_mediator` — rates and rest-frame yields of
//! the axial-vector mediator model.
//!
//! Registration only: the math is in [`crate::kernels::axial_rates`] and
//! [`crate::kernels::axial_yields`]. The caller is
//! `hazma/axial_vector_mediator/_model.py`, which validates every argument
//! and flattens it, so each entry point takes flat batches of equal length
//! and returns one value per sample, or per bin between consecutive edges.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dispatch::component_major;
use crate::kernels::axial_rates::{self, DarkSector, FermionPair};
use crate::kernels::axial_three_pi::Channel as PionChannel;
use crate::kernels::axial_yields::{Channel, Probe, ThreePionParents};

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

/// Return the helicity-weighted transverse density of a fermion pair in
/// MeV² at each energy.
#[pyfunction]
fn fermion_pair_helicity_density<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<'py, f64>,
    mass: f64,
    vector: f64,
    axial: f64,
    colors: f64,
) -> Bound<'py, PyArray1<f64>> {
    let fermion = FermionPair {
        mass,
        vector,
        axial,
        colors,
    };
    let values: Vec<f64> = energies
        .as_array()
        .iter()
        .map(|&q| fermion.helicity_density(q))
        .collect();
    values.into_pyarray(py)
}

/// Evaluate `channel`'s `product` yields at `points`: densities at each
/// energy, or counts between consecutive edges when `binned`. The rows are
/// one for `"photon"`, `[e⁺, e⁻]` for `"charged_lepton"` and the flavors
/// `[e, μ, τ]` for `"neutrino"`.
fn channel_yields<'py>(
    py: Python<'py>,
    channel: &Channel,
    product: &str,
    points: &PyReadonlyArray1<'py, f64>,
    binned: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let points = points.as_array().to_vec();
    let probes: Vec<Probe> = if binned {
        points
            .windows(2)
            .map(|edges| Probe::Bin(edges[0], edges[1]))
            .collect()
    } else {
        points
            .iter()
            .map(|&energy| Probe::Density(energy))
            .collect()
    };
    let probes = probes.iter().copied();
    match product {
        "photon" => Ok(component_major(
            py,
            probes.map(|probe| [channel.photons(probe)]),
        )),
        "charged_lepton" => Ok(component_major(
            py,
            probes.map(|probe| channel.charged_leptons(probe)),
        )),
        "neutrino" => Ok(component_major(
            py,
            probes.map(|probe| channel.neutrinos(probe)),
        )),
        _ => Err(PyValueError::new_err(format!(
            "unknown product {product:?}."
        ))),
    }
}

/// Return the `product` yields of `μ⁺μ⁻` with each muon at `muon_energy`
/// in MeV and effective `polarization`.
///
/// # Errors
///
/// `ValueError` for an unknown `product`.
#[pyfunction]
fn muon_pair_yields<'py>(
    py: Python<'py>,
    product: &str,
    points: PyReadonlyArray1<'py, f64>,
    binned: bool,
    muon_energy: f64,
    polarization: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let channel = Channel::MuonPair {
        muon_energy,
        polarization,
    };
    channel_yields(py, &channel, product, &points, binned)
}

/// Return the `product` yields of a three-pion state, with `parents`
/// `(neutral, invariant energy in MeV, production pion mass in MeV,
/// transverse fraction of σv)`.
///
/// # Errors
///
/// `ValueError` for an unknown `product`.
#[pyfunction]
fn three_pion_yields<'py>(
    py: Python<'py>,
    product: &str,
    points: PyReadonlyArray1<'py, f64>,
    binned: bool,
    parents: (bool, f64, f64, f64),
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (neutral, energy, pion_mass, transverse_fraction) = parents;
    let pion_channel = if neutral {
        PionChannel::Neutral
    } else {
        PionChannel::Charged
    };
    let channel = Channel::ThreePion(ThreePionParents::new(
        pion_channel,
        energy,
        pion_mass,
        transverse_fraction,
    ));
    channel_yields(py, &channel, product, &points, binned)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fermion_pair_densities, module)?)?;
    module.add_function(wrap_pyfunction!(fermion_pair_helicity_density, module)?)?;
    module.add_function(wrap_pyfunction!(muon_pair_yields, module)?)?;
    module.add_function(wrap_pyfunction!(three_pion_yields, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_v, module)?)?;
    module.add_function(wrap_pyfunction!(cross_section, module)?)?;
    Ok(())
}
