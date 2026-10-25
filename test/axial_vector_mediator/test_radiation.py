"""Resolved prompt photons of `hazma.axial_vector_mediator`."""

from __future__ import annotations

import numpy as np
import pytest
import radiation_oracle as oracle
from scipy import integrate

from hazma import parameters
from hazma._core import axial_vector_mediator as core
from hazma.axial_vector_mediator import AxialVectorMediator, LeptonCouplings
from hazma.form_factors.axial import AxialCurrent
from hazma.spectra.altarelli_parisi import dnde_photon_ap_fermion
from hazma.spectra.boost import dnde_boost

ALPHA = parameters.alpha_em
PION = parameters.neutral_pion_mass
PHYSICAL = (parameters.neutral_pion_mass, parameters.charged_pion_mass)
MEDIATOR = 450.0
LAB = 600.0
#: A centre-of-mass energy inside the three-pion channel's declared range,
#: for dark matter of mass `REST_MX`.
REST = 580.0
REST_MX = 280.0
LEPTONS = LeptonCouplings(
    gvee=0.3,
    gaee=-0.4,
    gvmumu=0.5,
    gamumu=0.7,
    gvtautau=0.0,
    gatautau=0.0,
    gnue=0.1,
    gnumu=0.2,
    gnutau=0.3,
)
#: Monte Carlo agreement is required within this many standard errors per
#: bin, over about ten bins each.
SIGMAS = 4.0


def _model(**overrides: object) -> AxialVectorMediator:
    arguments = dict(
        mx=600.0,
        mv=MEDIATOR,
        gvxx=0.0,
        gaxx=1.0,
        quarks=AxialCurrent(
            1e-3, -1e-3, 0.0, qcd_scheme="MSbar", scale_mev=2000.0, provenance="test"
        ),
        leptons=LEPTONS,
        gacc=0.0,
        theory="chpt",
    )
    arguments.update(overrides)
    return AxialVectorMediator(**arguments)


def _assert_within_errors(
    actual: np.ndarray, expected: np.ndarray, errors: np.ndarray
) -> None:
    np.testing.assert_array_less(np.abs(actual - expected), SIGMAS * errors + 1e-15)


def test_radiative_diagrams_obey_the_ward_identity_and_factorize() -> None:
    p0, plus, minus, k, _ = oracle.events(560.0, PHYSICAL, 2000, seed=5)
    polarization = np.array([0.3, 0.2, -0.5, 0.8])[:, None]
    m = oracle.amplitude((p0, plus, minus, k), polarization, PHYSICAL[0])
    scale = np.sqrt(np.sum(m**2, axis=0)) * k[0]
    # k.M cancels two terms of about the scale's size, to rounding.
    np.testing.assert_array_less(np.abs(oracle.dot(k, m)) / scale, 1e-12)
    q = p0 + plus + minus + k
    born = oracle.dot(polarization, oracle.current(q, p0, q - p0, PHYSICAL[0]))
    # The diagrams differ from (eps.H) J by a term along k, which no physical
    # photon sees, so their polarization sum -M.M is the eikonal one.
    np.testing.assert_allclose(
        -oracle.dot(m, m), born**2 * oracle.photon_sum(plus, minus, k), rtol=1e-10
    )


@pytest.mark.parametrize(
    ("energy", "reference", "error"),
    [(450.0, 9.12e-4, 0.28e-4), (500.0, 3.14e-3, 0.10e-3)],
)
def test_restricted_lo_fractions_reproduce_the_reference(
    energy: float, reference: float, error: float
) -> None:
    # The mediator note's spectra study: physical pion masses, rest-frame
    # photons above 1 MeV, its alpha and its one-sigma Monte Carlo error.
    alpha = 1.0 / 137.035999
    fraction = core.three_pion_radiation(
        np.array([1.0, energy]), True, energy, PHYSICAL, 1.0, (alpha, 1.0)
    )[0]
    assert abs(fraction - reference) <= 2.0 * error
    edges = np.array([1.0, energy])
    expected, errors = oracle.photon_counts(
        energy, PHYSICAL, edges, alpha=alpha, cut=1.0, n=400_000, seed=11
    )
    _assert_within_errors(fraction, expected[0], errors[0])


@pytest.mark.parametrize("transverse_fraction", [1.0, 0.0])
def test_rest_spectrum_matches_the_monte_carlo(transverse_fraction: float) -> None:
    energy, cut = 560.0, 2.0
    edges = np.array([2.0, 4.0, 8.0, 16.0, 30.0, 50.0, 70.0, 90.0, 120.0])
    counts = core.three_pion_radiation(
        edges, True, energy, (PION, PION), transverse_fraction, (ALPHA, cut)
    )
    expected, errors = oracle.photon_counts(
        energy,
        (PION, PION),
        edges,
        alpha=ALPHA,
        cut=cut,
        n=400_000,
        seed=3,
        transverse_fraction=transverse_fraction,
    )
    _assert_within_errors(counts, expected, errors)


@pytest.mark.parametrize(
    "diagonal", [(0.0, 1.0, 0.0), (0.5, 0.0, 0.5)], ids=["longitudinal", "transverse"]
)
def test_polarized_cascade_matches_the_monte_carlo(
    diagonal: tuple[float, float, float],
) -> None:
    edges = np.array([2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 65.0, 80.0])
    density = np.diag(diagonal).astype(complex)
    counts = _model().mediator_decay_resolved_photon_counts(
        edges, LAB, density=density, cut=1.0
    )["pi pi pi0"]
    expected, errors = oracle.photon_counts(
        MEDIATOR,
        (PION, PION),
        edges,
        alpha=ALPHA,
        cut=1.0,
        n=400_000,
        seed=7,
        lab=(LAB / MEDIATOR, diagonal),
    )
    _assert_within_errors(counts, expected, errors)


def test_helicity_changes_the_three_pion_cascade() -> None:
    # The outer edges hold every lab photon: the cut's box starts near
    # 0.45 MeV and the endpoint's ends below 100 MeV.
    edges = np.array([0.1, 40.0, 200.0])
    model = _model()
    zero, transverse = (
        model.mediator_decay_resolved_photon_counts(
            edges, LAB, density=np.diag(d).astype(complex), cut=1.0
        )["pi pi pi0"]
        for d in ((0.0, 1.0, 0.0), (0.5, 0.0, 0.5))
    )
    # The alignment moves the hardest photons by several percent, between
    # bins, and keeps their total.
    assert not np.isclose(zero[1], transverse[1], rtol=1e-2)
    np.testing.assert_allclose(zero.sum(), transverse.sum(), rtol=1e-10)


def test_soft_limit_is_the_eikonal_radiator_over_the_born() -> None:
    # A vector dark-matter coupling gives a purely transverse rate.
    model = _model(mx=REST_MX, gvxx=1.0, gaxx=0.0)
    e_cm = REST
    omega = 1e-8 * e_cm

    def radiator(s: float) -> float:
        beta = np.sqrt(1.0 - 4.0 * PION**2 / s)
        return (1.0 + beta**2) * np.log((1.0 + beta) / (1.0 - beta)) / beta - 2.0

    def weight(s: float) -> float:
        kallen = (s - (e_cm - PION) ** 2) * (s - (e_cm + PION) ** 2)
        return kallen**1.5 * np.sqrt(1.0 - 4.0 * PION**2 / s)

    bounds = (4.0 * PION**2, (e_cm - PION) ** 2)
    mean = (
        integrate.quad(lambda s: weight(s) * radiator(s), *bounds, epsrel=1e-12)[0]
        / integrate.quad(weight, *bounds, epsrel=1e-12)[0]
    )
    photons = model.rest_frame_resolved_photons(omega, e_cm, cut=omega / 2.0)
    # The approach is linear in omega, so 1e-8 leaves well under 1e-5.
    np.testing.assert_allclose(
        omega * photons["pi pi pi0"], ALPHA / np.pi * mean, rtol=1e-5
    )


def test_lepton_pairs_are_altarelli_parisi_alone() -> None:
    model = _model(mx=REST_MX)
    e_cm, cut = REST, 3.0
    energies = np.array([1.0, 2.9, 3.0, 50.0, 200.0, 251.0, 252.0, 280.0])
    photons = model.rest_frame_resolved_photons(energies, e_cm, cut=cut)
    for fs, mass in (
        ("e e", parameters.electron_mass),
        ("mu mu", parameters.muon_mass),
    ):
        # The kinematic endpoint clips the leading-log spectrum's tail.
        endpoint = (e_cm**2 - 4.0 * mass**2) / (2.0 * e_cm)
        inside = (energies >= cut) & (energies < endpoint)
        expected = np.where(
            inside, 2.0 * dnde_photon_ap_fermion(energies, e_cm**2, mass), 0.0
        )
        np.testing.assert_allclose(photons[fs], expected, rtol=1e-12, atol=0.0)
    # The Born photons of a lepton pair hold no prompt radiation, so the
    # resolved photons are its only radiation.
    born = model.rest_frame_yields(energies, e_cm, product="photon")
    np.testing.assert_array_equal(born["e e"], 0.0)


def test_isotropic_lepton_cascade_is_the_boosted_rest_frame() -> None:
    model = _model()
    beta = np.sqrt(1.0 - (MEDIATOR / LAB) ** 2)

    def rest(energy: float) -> float:
        points = np.array([energy])
        emission = (ALPHA, 1.0)
        return core.pair_radiation(
            points, False, MEDIATOR, parameters.electron_mass, emission
        )[0]

    energies = np.array([5.0, 40.0, 150.0, 300.0])
    expected = dnde_boost(rest, energies, beta, vectorized=False)
    photons = model.mediator_decay_resolved_photons(
        energies, LAB, density=np.eye(3) / 3.0, cut=1.0
    )["e e"]
    # dnde_boost's quad meets its tolerance only to about 1e-5 on the
    # kinks the cut puts in the lab density.
    np.testing.assert_allclose(photons, expected, rtol=1e-5)


def test_shapes_keys_and_zeros() -> None:
    model = _model(mx=REST_MX)
    energies = np.array([[5.0, 20.0], [40.0, 80.0]])
    rest = model.rest_frame_resolved_photons(energies, REST, cut=1.0)
    assert set(rest) == set(model.rest_frame_yields(energies, REST, product="photon"))
    assert all(np.shape(value) == energies.shape for value in rest.values())
    np.testing.assert_array_equal(rest["ve ve"], 0.0)
    np.testing.assert_array_equal(rest["g g"], 0.0)
    assert isinstance(
        model.rest_frame_resolved_photons(5.0, REST, cut=1.0)["e e"], float
    )
    lab = model.mediator_decay_resolved_photons(
        energies, LAB, density=np.eye(3) / 3.0, cut=1.0
    )
    assert set(lab) == {"e e", "mu mu", "ve ve", "vm vm", "vt vt", "pi pi pi0"}
    assert all(np.shape(value) == energies.shape for value in lab.values())
    np.testing.assert_array_equal(lab["vm vm"], 0.0)


def test_counts_round_trip_by_bin_width() -> None:
    model = _model(mx=REST_MX)
    width = 1e-3
    for energy in (8.0, 45.0):
        edges = np.array([energy - width / 2.0, energy + width / 2.0])
        rest = model.rest_frame_resolved_photon_counts(edges, REST, cut=1.0)
        density = model.rest_frame_resolved_photons(energy, REST, cut=1.0)
        lab = model.mediator_decay_resolved_photon_counts(
            edges, LAB, density=np.diag([0.2, 0.5, 0.3]).astype(complex), cut=1.0
        )
        lab_density = model.mediator_decay_resolved_photons(
            energy, LAB, density=np.diag([0.2, 0.5, 0.3]).astype(complex), cut=1.0
        )
        for fs in ("e e", "mu mu", "pi pi pi0"):
            np.testing.assert_allclose(rest[fs][0] / width, density[fs], rtol=1e-6)
            np.testing.assert_allclose(lab[fs][0] / width, lab_density[fs], rtol=1e-6)


def test_photons_below_the_cut_vanish() -> None:
    model = _model(mx=REST_MX)
    photons = model.rest_frame_resolved_photons(np.array([0.5, 0.99]), REST, cut=1.0)
    for value in photons.values():
        np.testing.assert_array_equal(value, 0.0)


def test_open_tau_pair_is_refused() -> None:
    taus = LeptonCouplings(
        gvee=0.0,
        gaee=0.0,
        gvmumu=0.0,
        gamumu=0.0,
        gvtautau=0.5,
        gatautau=0.0,
        gnue=0.0,
        gnumu=0.0,
        gnutau=0.0,
    )
    no_quarks = AxialCurrent(
        0.0, 0.0, 0.0, qcd_scheme="MSbar", scale_mev=2000.0, provenance="test"
    )
    # A heavy mediator keeps the unmodeled "v v" closed.
    rest = _model(mx=REST_MX, mv=5000.0, quarks=no_quarks, leptons=taus)
    with pytest.raises(ValueError, match="tau tau"):
        rest.rest_frame_resolved_photons(10.0, 3600.0, cut=1.0)
    moving = _model(mv=4000.0, quarks=no_quarks, leptons=taus)
    with pytest.raises(ValueError, match="tau tau"):
        moving.mediator_decay_resolved_photons(
            10.0, 5000.0, density=np.eye(3) / 3.0, cut=1.0
        )


@pytest.mark.parametrize("cut", [0.0, -1.0, np.inf, np.nan])
def test_cut_must_be_positive_and_finite(cut: float) -> None:
    model = _model(mx=REST_MX)
    with pytest.raises(ValueError, match="cut"):
        model.rest_frame_resolved_photons(10.0, REST, cut=cut)
    with pytest.raises(ValueError, match="cut"):
        model.mediator_decay_resolved_photon_counts(
            np.array([1.0, 2.0]), LAB, density=np.eye(3) / 3.0, cut=cut
        )


def test_photon_energies_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        _model(mx=REST_MX).rest_frame_resolved_photons(
            np.array([0.0, 1.0]), REST, cut=1.0
        )
