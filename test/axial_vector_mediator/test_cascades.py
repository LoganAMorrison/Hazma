"""Polarized decay cascades of `hazma.axial_vector_mediator`."""

from __future__ import annotations

import numpy as np
import pytest
import spinor_oracle as oracle
from scipy import integrate

from hazma import parameters
from hazma._core import axial_vector_mediator as core
from hazma.axial_vector_mediator import AxialVectorMediator, LeptonCouplings
from hazma.form_factors.axial import AxialCurrent
from hazma.spectra.boost import boost_delta_function, dnde_boost

MEDIATOR = 450.0
#: The mediator's lab energy: a pair at threshold for 600 MeV dark matter.
LAB = 600.0
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
ISOTROPIC = np.eye(3) / 3.0
TRANSVERSE = np.diag([0.5, 0.0, 0.5])
#: A density with F₊ ≠ F₋ and coherences, which only its diagonal reaches.
ORIENTED = np.array(
    [[0.6, 0.1 + 0.2j, 0.05], [0.1 - 0.2j, 0.25, 0.0], [0.05, 0.0, 0.15]]
)
PRODUCTS = ("photon", "positron", "electron_positron", "neutrino")


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


def _beta() -> float:
    return np.sqrt(1.0 - (MEDIATOR / LAB) ** 2)


def test_open_decays_and_shapes() -> None:
    model = _model()
    energies = np.array([[20.0, 80.0], [150.0, 250.0]])
    yields = model.mediator_decay_yields(
        energies, LAB, product="photon", density=TRANSVERSE
    )
    assert set(yields) == {"e e", "mu mu", "ve ve", "vm vm", "vt vt", "pi pi pi0"}
    assert all(value.shape == energies.shape for value in yields.values())
    neutrinos = model.mediator_decay_yields(
        energies, LAB, product="neutrino", density=TRANSVERSE
    )
    assert neutrinos["pi pi pi0"].shape == (3, *energies.shape)
    scalar = model.mediator_decay_yields(
        80.0, LAB, product="photon", density=TRANSVERSE
    )
    assert isinstance(scalar["pi pi pi0"], float)
    assert scalar["pi pi pi0"] == yields["pi pi pi0"][0, 1]
    assert isinstance(
        model.mediator_pair_yields(80.0, LAB, product="photon", density=TRANSVERSE),
        float,
    )


@pytest.mark.parametrize(
    ("density", "message"),
    [
        (np.eye(2) / 2.0, "3x3"),
        (np.full((3, 3), "a"), "3x3"),
        (np.diag([np.nan, 0.5, 0.5]), "finite"),
        (np.array([[0.5, 0.1, 0], [0.2, 0.0, 0], [0, 0, 0.5]]), "Hermitian"),
        (np.diag([0.5, 0.5, 0.5]), "unit trace"),
        (np.diag([0.7, -0.1, 0.4]), "positive semidefinite"),
        (np.array([[0.5, 0.4, 0], [0.4, 0.1, 0], [0, 0, 0.4]]), "positive"),
    ],
)
def test_densities_are_validated(density: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _model().mediator_decay_yields(50.0, LAB, product="photon", density=density)


def test_only_the_diagonal_reaches_the_yields() -> None:
    model = _model()
    edges = np.linspace(1.0, 500.0, 11)
    coherent = model.mediator_decay_counts(
        edges, LAB, product="positron", density=ORIENTED
    )
    diagonal = model.mediator_decay_counts(
        edges, LAB, product="positron", density=np.diag(np.diag(ORIENTED).real)
    )
    for fs, counts in coherent.items():
        np.testing.assert_array_equal(counts, diagonal[fs])


def test_inputs_are_validated() -> None:
    model = _model()
    with pytest.raises(ValueError, match="exceed the mediator mass"):
        model.mediator_decay_yields(50.0, MEDIATOR, product="photon", density=ISOTROPIC)
    with pytest.raises(ValueError, match="product"):
        model.mediator_decay_yields(50.0, LAB, product="pion", density=ISOTROPIC)
    with pytest.raises(ValueError, match="positive"):
        model.mediator_decay_yields(0.0, LAB, product="photon", density=ISOTROPIC)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.mediator_decay_counts(
            [5.0, 1.0], LAB, product="neutrino", density=ISOTROPIC
        )
    taus = LeptonCouplings(
        gvee=0.0,
        gaee=0.0,
        gvmumu=0.0,
        gamumu=0.0,
        gvtautau=0.1,
        gatautau=0.0,
        gnue=0.0,
        gnumu=0.0,
        gnutau=0.0,
    )
    no_quarks = AxialCurrent(
        0.0, 0.0, 0.0, qcd_scheme="MSbar", scale_mev=2000.0, provenance="test"
    )
    heavy = _model(mv=4000.0, quarks=no_quarks, leptons=taus)
    with pytest.raises(ValueError, match="tau tau"):
        heavy.mediator_decay_yields(50.0, 5000.0, product="photon", density=ISOTROPIC)


def test_isotropic_electrons_are_the_boosted_line() -> None:
    # Within float rounding of the two box formulas.
    energies = np.array([40.0, 100.0, 250.0, 450.0])
    positrons = _model().mediator_decay_yields(
        energies, LAB, product="positron", density=ISOTROPIC
    )["e e"]
    expected = boost_delta_function(
        energies, MEDIATOR / 2.0, parameters.electron_mass, _beta()
    )
    np.testing.assert_allclose(positrons, expected, rtol=1e-12)


def _rest_polarization(mass: float, vector: float, axial: float) -> float:
    """Return the on-shell fermion helicity, the rates model's A_h/A_T."""
    q = np.array([MEDIATOR])
    helicity = core.fermion_pair_helicity_density(q, mass, vector, axial, 1.0)[0]
    return helicity / core.fermion_pair_densities(q, mass, vector, axial, 1.0)[0][0]


def test_isotropic_muons_are_the_boosted_rest_frame() -> None:
    # An isotropic mediator has isotropic, uniformly polarized muons, so its
    # lab e⁺ are Hazma's boost of the rest-frame pair. quad's 1.5e-8 sets
    # the tolerance, with margin for the kinked boost integrand (1e-6).
    polarization = _rest_polarization(parameters.muon_mass, 0.5, 0.7)

    def rest(energy: float) -> float:
        points = np.array([energy])
        return core.muon_pair_yields(
            "charged_lepton", points, False, MEDIATOR / 2.0, polarization
        )[0][0]

    energies = np.array([5.0, 30.0, 90.0, 180.0, 260.0])
    expected = dnde_boost(
        rest, energies, _beta(), mass=parameters.electron_mass, vectorized=False
    )
    positrons = _model().mediator_decay_yields(
        energies, LAB, product="positron", density=ISOTROPIC
    )["mu mu"]
    np.testing.assert_allclose(positrons, expected, rtol=1e-6)


def test_isotropic_three_pions_are_the_boosted_rest_frame() -> None:
    def rest(energy: float) -> float:
        parents = (False, MEDIATOR, parameters.neutral_pion_mass, 1.0)
        return core.three_pion_yields("photon", np.array([energy]), False, parents)[0][
            0
        ]

    energies = np.array([20.0, 60.0, 110.0, 200.0])
    expected = dnde_boost(rest, energies, _beta(), vectorized=False)
    photons = _model().mediator_decay_yields(
        energies, LAB, product="photon", density=ISOTROPIC
    )["pi pi pi0"]
    # dnde_boost's quad meets its tolerance only to about 1e-5 on the
    # integrand's kinks.
    np.testing.assert_allclose(photons, expected, rtol=1e-5)


def _oracle_lepton_density(
    energy: float, *, antimuon: bool, fractions: np.ndarray
) -> float:
    """Return a muon's lab e± density from explicit spinors and Lorentz boosts.

    Each muon's angular density and rest-frame spin come from
    `spinor_oracle.polarized_vector_decay`; its spin four-vector is boosted
    to the lab, where the lab helicity is its rest-frame spin along its lab
    momentum. Hazma's polarized muon kernel then decays it, with the
    effective polarization of `rest_frame_yields` (minus the helicity of a
    μ⁺).
    """
    mass, vector, axial = parameters.muon_mass, 0.5, 0.7
    gamma, gamma_beta = LAB / MEDIATOR, np.sqrt(LAB**2 - MEDIATOR**2) / MEDIATOR
    muon_energy = MEDIATOR / 2.0
    momentum = np.sqrt(muon_energy**2 - mass**2)

    def decay(helicity: int, cosine: float) -> tuple[float, np.ndarray, np.ndarray]:
        return oracle.polarized_vector_decay(
            MEDIATOR,
            mf=mass,
            gvf=vector,
            gaf=axial,
            helicity=helicity,
            cos_theta=cosine,
        )

    nodes, weights = np.polynomial.legendre.leggauss(8)
    norm = sum(w * decay(1, c)[0] for c, w in zip(nodes, weights, strict=True))

    def integrand(cosine: float) -> float:
        # The fermion runs along n; a μ⁺ at `cosine` partners a μ⁻ at -cosine.
        fermion_cosine = -cosine if antimuon else cosine
        sine = np.sqrt(1.0 - cosine**2)
        direction = np.array([-sine if antimuon else sine, 0.0, cosine])
        weight, spin = 0.0, np.zeros(3)
        for helicity, fraction in zip((1, 0, -1), fractions, strict=True):
            rate, fermion, antifermion = decay(helicity, fermion_cosine)
            weight += fraction * rate / norm
            spin += fraction * rate / norm * (antifermion if antimuon else fermion)
        spin /= weight
        # The spin four-vector of rest-frame spin s for momentum k n.
        parallel = spin @ direction
        four = np.array(
            [
                momentum * parallel / mass,
                *(spin + (muon_energy / mass - 1.0) * parallel * direction),
            ]
        )
        lab_four = four.copy()
        lab_four[0] = gamma * four[0] + gamma_beta * four[3]
        lab_four[3] = gamma_beta * four[0] + gamma * four[3]
        lab_momentum = momentum * direction
        lab_energy = gamma * muon_energy + gamma_beta * lab_momentum[2]
        lab_momentum[2] = gamma_beta * muon_energy + gamma * lab_momentum[2]
        # Back to the muon's rest frame along its lab momentum.
        rest_spin = lab_four[1:] - lab_four[0] / (lab_energy + mass) * lab_momentum
        helicity = rest_spin @ lab_momentum / np.linalg.norm(lab_momentum)
        polarization = -helicity if antimuon else helicity
        points = np.array([energy])
        response = core.muon_pair_yields(
            "charged_lepton", points, False, lab_energy, polarization
        )[0][0]
        return weight * response

    value, _ = integrate.quad(integrand, -1.0, 1.0, epsabs=0.0, epsrel=1e-10, limit=200)
    return value


@pytest.mark.parametrize("energy", [20.0, 90.0, 170.0, 260.0])
def test_muon_cascades_match_explicit_spinors(energy: float) -> None:
    # The oracle's quadrature reaches 1e-10 and the kernel's fold 1e-8.
    fractions = np.diag(ORIENTED).real
    model = _model()
    positron = model.mediator_decay_yields(
        energy, LAB, product="positron", density=ORIENTED
    )["mu mu"]
    both = model.mediator_decay_yields(
        energy, LAB, product="electron_positron", density=ORIENTED
    )["mu mu"]
    expected_positron = _oracle_lepton_density(
        energy, antimuon=True, fractions=fractions
    )
    expected_electron = _oracle_lepton_density(
        energy, antimuon=False, fractions=fractions
    )
    np.testing.assert_allclose(positron, expected_positron, rtol=1e-7)
    np.testing.assert_allclose(both - positron, expected_electron, rtol=1e-7)


def test_pair_yield_is_twice_the_branching_sum() -> None:
    # Every decay gives one e⁺ per decay into it, so a pair carries
    # 2 Σ_f B_f of them over the full range, not Σ_f B_f².
    model = _model()
    widths = model.partial_widths()
    edges = np.array([parameters.electron_mass, 2000.0])
    positrons = model.mediator_pair_counts(
        edges, LAB, product="positron", density=TRANSVERSE
    )
    charged = ("e e", "mu mu", "pi pi pi0")
    expected = 2.0 * sum(widths[fs] for fs in charged) / widths["total"]
    # The fold tolerance (1e-8) and the counts below the first edge (one
    # electron mass, where the spectra vanish) bound the sum's error.
    np.testing.assert_allclose(positrons, [expected], rtol=1e-7)
    neutrinos = model.mediator_pair_counts(
        np.array([0.0, 2000.0]), LAB, product="neutrino", density=TRANSVERSE
    )
    branching = {fs: widths[fs] / widths["total"] for fs in widths}
    expected = 2.0 * np.array(
        [
            2.0 * (branching["ve ve"] + branching["mu mu"] + branching["pi pi pi0"]),
            2.0 * (branching["vm vm"] + branching["mu mu"])
            + 4.0 * branching["pi pi pi0"],
            2.0 * branching["vt vt"],
        ]
    )
    # The three-pion neutrino bins add the pion decay folds' tolerances
    # (1e-8 each) and the counts missing below the first edge.
    np.testing.assert_allclose(neutrinos[:, 0], expected, rtol=1e-6)


def test_width_override_rescales_the_pair_branching_fractions() -> None:
    # Branching fractions divide by the model's total width, which is the
    # override when given, so doubling it halves every pair count.
    default = _model()
    doubled = _model(width=2.0 * default.partial_widths()["total"])
    edges = np.array([parameters.electron_mass, 2000.0])
    counts = [
        model.mediator_pair_counts(edges, LAB, product="positron", density=TRANSVERSE)
        for model in (default, doubled)
    ]
    np.testing.assert_allclose(counts[1] / counts[0], 0.5, rtol=1e-12)


@pytest.mark.parametrize("product", PRODUCTS)
def test_counts_round_trip_by_bin_width(product: str) -> None:
    model = _model()
    for energy in (30.0, 140.0):
        width = 1e-4
        edges = np.array([energy - width / 2.0, energy + width / 2.0])
        counts = model.mediator_decay_counts(
            edges, LAB, product=product, density=ORIENTED
        )
        yields = model.mediator_decay_yields(
            energy, LAB, product=product, density=ORIENTED
        )
        for fs, value in yields.items():
            # A 1e-4 MeV bin averages the density to 1e-5 (its curvature and
            # the fold tolerance); atol covers yields that vanish.
            np.testing.assert_allclose(
                counts[fs][..., 0] / width, value, rtol=1e-5, atol=1e-12
            )
