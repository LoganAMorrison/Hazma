//! `hazma._core.axial_vector_mediator` — rates, rest-frame yields,
//! polarized decay cascades, resolved radiative photons and the CMB
//! deposition of the cascades of the axial-vector mediator model.
//!
//! Registration only: the math is in [`crate::kernels::axial_rates`],
//! [`crate::kernels::axial_yields`],
//! [`crate::kernels::polarized_cascade`],
//! [`crate::kernels::axial_radiation`] and
//! [`crate::kernels::deposition`]. The caller is
//! `hazma/axial_vector_mediator/_model.py`, which validates every argument
//! and flattens it, so each entry point takes flat batches of equal length
//! and returns one value per sample, or per bin between consecutive edges.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dispatch::component_major;
use crate::kernels::axial_radiation::{
    PairRadiation, RestPhotons, ThreePionRadiation, lab_photons, rest_photons,
};
use crate::kernels::axial_rates::{self, DarkSector, FermionPair};
use crate::kernels::axial_three_pi::Channel as PionChannel;
use crate::kernels::axial_yields::{Channel, Probe, ThreePionParents, Yields};
use crate::kernels::deposition::{Efficiency, deposited_energy};
use crate::kernels::polarized_cascade::{Boost, Cascade, Decay, Helicities, PairDecay};

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

/// Return the probes of `points`: a density at each energy, or a count
/// between consecutive edges when `binned`.
fn probes(points: &PyReadonlyArray1<'_, f64>, binned: bool) -> Vec<Probe> {
    let points = points.as_array().to_vec();
    if binned {
        points
            .windows(2)
            .map(|edges| Probe::Bin(edges[0], edges[1]))
            .collect()
    } else {
        points
            .iter()
            .map(|&energy| Probe::Density(energy))
            .collect()
    }
}

/// Evaluate `channel`'s `product` yields at `points`: densities at each
/// energy, or counts between consecutive edges when `binned`. The rows are
/// one for `"photon"`, `[e⁺, e⁻]` for `"charged_lepton"` and the flavors
/// `[e, μ, τ]` for `"neutrino"`.
fn channel_yields<'py>(
    py: Python<'py>,
    channel: &impl Yields,
    product: &str,
    points: &PyReadonlyArray1<'py, f64>,
    binned: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let probes = probes(points, binned);
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

/// Return a mediator's boost from `(mass, lab energy)` in MeV and its
/// helicity fractions from `(F₊, F₀, F₋)`.
fn mediator_state(mediator: (f64, f64), helicities: (f64, f64, f64)) -> (Boost, Helicities) {
    let (plus, zero, minus) = helicities;
    (
        Boost::new(mediator.0, mediator.1),
        Helicities { plus, zero, minus },
    )
}

/// Return the `product` yields of a moving mediator's decay into the
/// fermion pair of `lepton` (`"e"`, `"mu"`, `"ve"`, `"vm"` or `"vt"`),
/// with `fermion` its `(mass in MeV, g_V, g_A)`, `mediator` the mediator's
/// `(mass, lab energy)` in MeV and `helicities` its `(F₊, F₀, F₋)`.
///
/// # Errors
///
/// `ValueError` for an unknown `lepton` or `product`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pair_cascade_yields<'py>(
    py: Python<'py>,
    product: &str,
    points: PyReadonlyArray1<'py, f64>,
    binned: bool,
    lepton: &str,
    fermion: (f64, f64, f64),
    mediator: (f64, f64),
    helicities: (f64, f64, f64),
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cascade = pair_cascade(lepton, fermion, mediator, helicities)?;
    channel_yields(py, &cascade, product, &points, binned)
}

/// Return the `product` yields of a moving mediator's decay into
/// `π⁺π⁻π⁰` through the LO transverse current, with every pion produced at
/// `pion_mass` in MeV, and `mediator` and `helicities` as in
/// `pair_cascade_yields`.
///
/// # Errors
///
/// `ValueError` for an unknown `product`.
#[pyfunction]
fn three_pion_cascade_yields<'py>(
    py: Python<'py>,
    product: &str,
    points: PyReadonlyArray1<'py, f64>,
    binned: bool,
    pion_mass: f64,
    mediator: (f64, f64),
    helicities: (f64, f64, f64),
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cascade = three_pion_cascade(pion_mass, mediator, helicities);
    channel_yields(py, &cascade, product, &points, binned)
}

/// Return the decay of a moving mediator into the fermion pair of
/// `lepton`, with the arguments of `pair_cascade_yields`.
fn pair_cascade(
    lepton: &str,
    fermion: (f64, f64, f64),
    mediator: (f64, f64),
    helicities: (f64, f64, f64),
) -> PyResult<Cascade> {
    let (mass, vector, axial) = fermion;
    let pair = PairDecay {
        mediator_mass: mediator.0,
        mass,
        vector,
        axial,
    };
    let decay = match lepton {
        "e" => Decay::ElectronPair(pair),
        "mu" => Decay::MuonPair(pair),
        "ve" => Decay::NeutrinoPair(pair, 0),
        "vm" => Decay::NeutrinoPair(pair, 1),
        "vt" => Decay::NeutrinoPair(pair, 2),
        _ => {
            return Err(PyValueError::new_err(format!("unknown lepton {lepton:?}.")));
        }
    };
    let (boost, helicities) = mediator_state(mediator, helicities);
    Ok(Cascade {
        decay,
        boost,
        helicities,
    })
}

/// Return the decay of a moving mediator into `π⁺π⁻π⁰`, with the arguments
/// of `three_pion_cascade_yields`.
fn three_pion_cascade(
    pion_mass: f64,
    mediator: (f64, f64),
    helicities: (f64, f64, f64),
) -> Cascade {
    let parents = ThreePionParents::new(PionChannel::Charged, mediator.0, pion_mass, 1.0);
    let (boost, helicities) = mediator_state(mediator, helicities);
    Cascade {
        decay: Decay::ThreePion(parents),
        boost,
        helicities,
    }
}

/// Return the energy in MeV that a mediator decay's `product`
/// (`"photon"` or `"charged_lepton"`, both charges summed) deposits, per
/// decay: `∫ E dN/dE f(E − rest_mass) dE` from `lower` to `upper`, the
/// mediator's lab energy, with `f` linear in the `efficiency` table of
/// `(kinetic energies in MeV, fractions)`.
///
/// # Errors
///
/// `ValueError` for an unknown `product` or a table whose two arrays differ
/// in length or are empty.
fn cascade_deposition(
    cascade: &Cascade,
    product: &str,
    efficiency: &(PyReadonlyArray1<'_, f64>, PyReadonlyArray1<'_, f64>),
    rest_mass: f64,
    (lower, upper): (f64, f64),
) -> PyResult<f64> {
    let (kinetic_energies, fractions) = (efficiency.0.as_slice()?, efficiency.1.as_slice()?);
    if kinetic_energies.is_empty() || kinetic_energies.len() != fractions.len() {
        return Err(PyValueError::new_err(
            "the efficiency table needs equal, nonempty arrays.",
        ));
    }
    let table = Efficiency {
        kinetic_energies,
        fractions,
    };
    let edges = cascade.box_edges();
    match product {
        "photon" => Ok(deposited_energy(
            |energy| cascade.photons(Probe::Density(energy)),
            lower,
            upper,
            rest_mass,
            &table,
            &edges,
        )),
        "charged_lepton" => Ok(deposited_energy(
            |energy| cascade.charged_leptons(Probe::Density(energy)).iter().sum(),
            lower,
            upper,
            rest_mass,
            &table,
            &edges,
        )),
        _ => Err(PyValueError::new_err(format!(
            "unknown deposited product {product:?}."
        ))),
    }
}

/// Return the energy in MeV that `product` of a moving mediator's decay into
/// the fermion pair of `lepton` deposits, as `cascade_deposition` folds it,
/// with the decay's arguments as in `pair_cascade_yields`.
///
/// # Errors
///
/// As `pair_cascade_yields` and `cascade_deposition`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pair_cascade_deposition(
    product: &str,
    lepton: &str,
    fermion: (f64, f64, f64),
    mediator: (f64, f64),
    helicities: (f64, f64, f64),
    efficiency: (PyReadonlyArray1<'_, f64>, PyReadonlyArray1<'_, f64>),
    rest_mass: f64,
    lower: f64,
) -> PyResult<f64> {
    let cascade = pair_cascade(lepton, fermion, mediator, helicities)?;
    cascade_deposition(
        &cascade,
        product,
        &efficiency,
        rest_mass,
        (lower, mediator.1),
    )
}

/// Return the energy in MeV that `product` of a moving mediator's decay into
/// `π⁺π⁻π⁰` deposits, as `cascade_deposition` folds it, with the decay's
/// arguments as in `three_pion_cascade_yields`.
///
/// # Errors
///
/// As `cascade_deposition`.
#[pyfunction]
fn three_pion_cascade_deposition(
    product: &str,
    pion_mass: f64,
    mediator: (f64, f64),
    helicities: (f64, f64, f64),
    efficiency: (PyReadonlyArray1<'_, f64>, PyReadonlyArray1<'_, f64>),
    rest_mass: f64,
    lower: f64,
) -> PyResult<f64> {
    let cascade = three_pion_cascade(pion_mass, mediator, helicities);
    cascade_deposition(
        &cascade,
        product,
        &efficiency,
        rest_mass,
        (lower, mediator.1),
    )
}

/// Return the rest-frame resolved photons of `spectrum` at `points`, as
/// `channel_yields` evaluates them.
fn rest_radiation<'py>(
    py: Python<'py>,
    spectrum: &impl RestPhotons,
    points: &PyReadonlyArray1<'py, f64>,
    binned: bool,
) -> Bound<'py, PyArray1<f64>> {
    let values: Vec<f64> = probes(points, binned)
        .into_iter()
        .map(|probe| rest_photons(spectrum, probe))
        .collect();
    values.into_pyarray(py)
}

/// Return the lab resolved photons of `spectrum` from a mediator with
/// `boost` and `helicities`, at `points`.
fn lab_radiation<'py>(
    py: Python<'py>,
    spectrum: &impl RestPhotons,
    (boost, helicities): (Boost, Helicities),
    points: &PyReadonlyArray1<'py, f64>,
    binned: bool,
) -> Bound<'py, PyArray1<f64>> {
    let values: Vec<f64> = probes(points, binned)
        .into_iter()
        .map(|probe| lab_photons(spectrum, boost, helicities, probe))
        .collect();
    values.into_pyarray(py)
}

/// Return the resolved `π⁺π⁻π⁰γ` photons per Born annihilation of a source
/// of invariant `energy` in MeV whose rate is `transverse_fraction`
/// transverse, with the π⁰ and π± `masses` in MeV and `emission`
/// `(α, cut in MeV)`.
#[pyfunction]
fn three_pion_radiation<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    binned: bool,
    energy: f64,
    masses: (f64, f64),
    transverse_fraction: f64,
    emission: (f64, f64),
) -> Bound<'py, PyArray1<f64>> {
    let (alpha, cut) = emission;
    let spectrum = ThreePionRadiation::new(
        energy,
        [masses.0, masses.1],
        transverse_fraction,
        alpha,
        cut,
    );
    rest_radiation(py, &spectrum, &points, binned)
}

/// Return the lab resolved `π⁺π⁻π⁰γ` photons per decay of a moving
/// mediator, with `masses` and `emission` as in `three_pion_radiation` and
/// `mediator` and `helicities` as in `pair_cascade_yields`.
#[pyfunction]
fn three_pion_radiation_cascade<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    binned: bool,
    masses: (f64, f64),
    emission: (f64, f64),
    mediator: (f64, f64),
    helicities: (f64, f64, f64),
) -> Bound<'py, PyArray1<f64>> {
    let (alpha, cut) = emission;
    let spectrum = ThreePionRadiation::new(mediator.0, [masses.0, masses.1], 1.0, alpha, cut);
    let state = mediator_state(mediator, helicities);
    lab_radiation(py, &spectrum, state, &points, binned)
}

/// Return the Altarelli–Parisi photons per annihilation into a lepton pair
/// of `mass` in MeV from a source of invariant `energy` in MeV, with
/// `emission` as in `three_pion_radiation`.
#[pyfunction]
fn pair_radiation<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    binned: bool,
    energy: f64,
    mass: f64,
    emission: (f64, f64),
) -> Bound<'py, PyArray1<f64>> {
    let (alpha, cut) = emission;
    let spectrum = PairRadiation {
        energy,
        mass,
        alpha,
        cut,
        tensor: 0.0,
    };
    rest_radiation(py, &spectrum, &points, binned)
}

/// Return the lab Altarelli–Parisi photons per decay of a moving mediator
/// into the lepton pair of `fermion` `(mass in MeV, g_V, g_A)`, each photon
/// along its lepton, with `emission` as in `three_pion_radiation` and
/// `mediator` and `helicities` as in `pair_cascade_yields`.
#[pyfunction]
fn pair_radiation_cascade<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    binned: bool,
    fermion: (f64, f64, f64),
    emission: (f64, f64),
    mediator: (f64, f64),
    helicities: (f64, f64, f64),
) -> Bound<'py, PyArray1<f64>> {
    let ((mass, vector, axial), (alpha, cut)) = (fermion, emission);
    let decay = PairDecay {
        mediator_mass: mediator.0,
        mass,
        vector,
        axial,
    };
    let spectrum = PairRadiation {
        energy: mediator.0,
        mass,
        alpha,
        cut,
        tensor: decay.analyzer().tensor,
    };
    let state = mediator_state(mediator, helicities);
    lab_radiation(py, &spectrum, state, &points, binned)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fermion_pair_densities, module)?)?;
    module.add_function(wrap_pyfunction!(fermion_pair_helicity_density, module)?)?;
    module.add_function(wrap_pyfunction!(muon_pair_yields, module)?)?;
    module.add_function(wrap_pyfunction!(three_pion_yields, module)?)?;
    module.add_function(wrap_pyfunction!(pair_cascade_yields, module)?)?;
    module.add_function(wrap_pyfunction!(three_pion_cascade_yields, module)?)?;
    module.add_function(wrap_pyfunction!(pair_cascade_deposition, module)?)?;
    module.add_function(wrap_pyfunction!(three_pion_cascade_deposition, module)?)?;
    module.add_function(wrap_pyfunction!(three_pion_radiation, module)?)?;
    module.add_function(wrap_pyfunction!(three_pion_radiation_cascade, module)?)?;
    module.add_function(wrap_pyfunction!(pair_radiation, module)?)?;
    module.add_function(wrap_pyfunction!(pair_radiation_cascade, module)?)?;
    module.add_function(wrap_pyfunction!(sigma_v, module)?)?;
    module.add_function(wrap_pyfunction!(cross_section, module)?)?;
    Ok(())
}
