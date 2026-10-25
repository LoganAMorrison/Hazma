"""Gamma-ray-limit and CMB adapters of `hazma.axial_vector_mediator`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from hazma import parameters
from hazma.axial_vector_mediator import AxialVectorMediator, LeptonCouplings
from hazma.cmb import f_eff_ep, f_eff_g, vx_cmb
from hazma.form_factors.axial import AxialCurrent
from hazma.gamma_ray_parameters import comptel_diffuse
from hazma.single_channel import SingleChannelAnn

NO_QUARKS = AxialCurrent(
    0.0, 0.0, 0.0, qcd_scheme="MSbar", scale_mev=2000.0, provenance="test"
)
ISOVECTOR = AxialCurrent(
    1e-3, -1e-3, 0.0, qcd_scheme="MSbar", scale_mev=2000.0, provenance="test"
)
NO_LEPTONS = LeptonCouplings(
    gvee=0.0,
    gaee=0.0,
    gvmumu=0.0,
    gamumu=0.0,
    gvtautau=0.0,
    gatautau=0.0,
    gnue=0.0,
    gnumu=0.0,
    gnutau=0.0,
)
ELECTRONS = LeptonCouplings(
    gvee=1e-3,
    gaee=0.0,
    gvmumu=0.0,
    gamumu=0.0,
    gvtautau=0.0,
    gatautau=0.0,
    gnue=0.0,
    gnumu=0.0,
    gnutau=0.0,
)
#: L_mu - L_tau below the dimuon threshold: the electrons couple only through
#: a small vector mixing, so nearly every mediator decays into neutrinos.
L_MU_L_TAU = LeptonCouplings(
    gvee=1e-7,
    gaee=0.0,
    gvmumu=3e-5,
    gamumu=0.0,
    gvtautau=3e-5,
    gatautau=0.0,
    gnue=0.0,
    gnumu=3e-5,
    gnutau=3e-5,
)
TRANSVERSE = np.diag([0.5, 0.0, 0.5])
ISOTROPIC = np.eye(3) / 3.0


def _model(**overrides: object) -> AxialVectorMediator:
    arguments = dict(
        mx=100.0,
        mv=75.0,
        gvxx=0.0,
        gaxx=1.0,
        quarks=NO_QUARKS,
        leptons=NO_LEPTONS,
        gacc=0.0,
        theory="chpt",
    )
    arguments.update(overrides)
    return AxialVectorMediator(**arguments)


def _cmb_e_cm(mx: float) -> float:
    """Return the centre-of-mass energy `TheoryCMB.f_eff` uses at its x_kd."""
    return 2.0 * mx * (1.0 + 0.5 * vx_cmb(mx, 1e-4) ** 2)


@pytest.mark.parametrize(
    ("mx", "pinned"),
    # The original gamma-gamma line energies of the mediator note. Its
    # 0.723585 and 0.600670 read the same table linearly in log E; Hazma's
    # table is linear in E.
    [(67.488, 0.7241273), (273.93, 0.6053788)],
)
def test_a_photon_line_deposits_the_table_at_its_energy(
    mx: float, pinned: float
) -> None:
    # Only the charm triangle couples, so the dark matter annihilates into
    # "g g" alone: two photons of e_cm/2 and no continuum.
    model = _model(mx=mx, mv=1000.0, gacc=1e-3)
    assert model.annihilation_branching_fractions(2.0 * mx)["g g"] == 1.0
    energy = _cmb_e_cm(mx) / 2.0
    line = float(f_eff_g(energy)) * energy / mx
    # The line and the fold share every operation but the sum.
    assert model.f_eff_g() == pytest.approx(line, rel=1e-14)
    assert model.f_eff_ep() == 0.0
    assert model.f_eff() == pytest.approx(pinned, abs=5e-8)
    p_ann = 3.5e-31
    assert model.cmb_limit(p_ann=p_ann) == pytest.approx(2.0 * p_ann * mx / line)


def _transverse_box_deposition(
    model: AxialVectorMediator, mediator_energy: float
) -> float:
    """Return the energy in MeV a transverse e⁺e⁻ decay's electrons deposit.

    Each electron's angular density about the flight direction is
    ``[1 - h P₂(cos θ)]/2`` for ``F₀ = 0``, with the tensor power
    ``h = (4r - 1)/(2 + 4r)``, ``r = m_e²/M²``, of a vector coupling.
    """
    mass, mv = parameters.electron_mass, model.mv
    gamma = mediator_energy / mv
    gamma_beta = np.sqrt(gamma**2 - 1.0)
    energy = mv / 2.0
    momentum = np.sqrt(energy**2 - mass**2)
    ratio = (mass / mv) ** 2
    tensor = (4.0 * ratio - 1.0) / (2.0 + 4.0 * ratio)

    def integrand(cosine: float) -> float:
        lab = gamma * energy + gamma_beta * momentum * cosine
        weight = (1.0 - tensor * (3.0 * cosine**2 - 1.0) / 2.0) / 2.0
        fraction = np.interp(lab - mass, f_eff_ep.x, f_eff_ep.y)
        return lab * fraction * weight

    # Break where a lab energy crosses a table node.
    nodes = (f_eff_ep.x + mass - gamma * energy) / (gamma_beta * momentum)
    points = nodes[(nodes > -1.0) & (nodes < 1.0)]
    value = integrate.quad(
        integrand, -1.0, 1.0, points=points, epsabs=0.0, epsrel=1e-13, limit=400
    )[0]
    return 2.0 * value


def test_a_transverse_electron_box_folds_to_its_angular_integral() -> None:
    model = _model(leptons=ELECTRONS)
    e_cm = _cmb_e_cm(model.mx)
    expected = 2.0 * _transverse_box_deposition(model, e_cm / 2.0) / (2.0 * model.mx)
    folded = model.mediator_pair_f_eff(density=TRANSVERSE)
    # Both integrals converge far below this; it is the fold's tolerance.
    assert folded == pytest.approx(expected, rel=1e-8)


def test_an_isotropic_electron_box_is_flat() -> None:
    # With F₀ = 1/3 the box is flat between its ends, whatever the analyzer.
    model = _model(leptons=ELECTRONS)
    mass, mv = parameters.electron_mass, model.mv
    mediator_energy = _cmb_e_cm(model.mx) / 2.0
    gamma = mediator_energy / mv
    spread = np.sqrt(gamma**2 - 1.0) * np.sqrt((mv / 2.0) ** 2 - mass**2)
    low, high = gamma * mv / 2.0 - spread, gamma * mv / 2.0 + spread

    def integrand(energy: float) -> float:
        return energy * np.interp(energy - mass, f_eff_ep.x, f_eff_ep.y)

    nodes = f_eff_ep.x + mass
    box = integrate.quad(
        integrand,
        low,
        high,
        points=nodes[(nodes > low) & (nodes < high)],
        epsabs=0.0,
        epsrel=1e-13,
    )[0] / (high - low)
    # Two mediators, each with two electrons of the box.
    expected = 4.0 * box / (2.0 * model.mx)
    folded = model.mediator_pair_f_eff(density=ISOTROPIC)
    assert folded == pytest.approx(expected, rel=1e-8)


def test_secluded_l_mu_l_tau_conditional_efficiency() -> None:
    # The mediator note's secluded benchmark: m_chi = 100 MeV, M = 75 MeV,
    # a transverse pair at threshold. Only the e⁺e⁻ decays deposit, so the
    # efficiency over B(e e) is the efficiency of a pair that decays into
    # electrons. The note's 0.952771 reads the table linearly in log E.
    model = _model(gaxx=0.00760666, leptons=L_MU_L_TAU)
    widths = model.partial_widths()
    branching = widths["e e"] / widths["total"]
    conditional = model.mediator_pair_f_eff(density=TRANSVERSE) / branching
    e_cm = _cmb_e_cm(model.mx)
    expected = 2.0 * _transverse_box_deposition(model, e_cm / 2.0) / (2.0 * model.mx)
    assert conditional == pytest.approx(expected, rel=1e-8)
    assert conditional == pytest.approx(0.9544662, abs=5e-8)


@pytest.mark.parametrize(
    ("mx", "mv", "pinned"),
    # The model's own fold, which a 16000-point trapezoid over
    # `mediator_pair_yields` reproduces to 1.2e-7. The mediator note's
    # spin-corrected 0.3967 and 0.3705 conserve the pair's energy, where
    # the charged pions here decay at their physical mass and inject 2%
    # more of it; its unpolarized 0.4022 and 0.3750 read the table
    # linearly in log E.
    [(600.0, 450.0, 0.4021864), (900.0, 500.0, 0.3750662)],
)
def test_three_pion_cascade_efficiency(mx: float, mv: float, pinned: float) -> None:
    model = _model(mx=mx, mv=mv, quarks=ISOVECTOR)
    assert model.mediator_pair_f_eff(density=TRANSVERSE) == pytest.approx(
        pinned, abs=5e-8
    )


def test_the_pair_efficiency_is_validated() -> None:
    model = _model(leptons=ELECTRONS)
    with pytest.raises(ValueError, match="closed"):
        _model(mv=150.0, leptons=ELECTRONS).mediator_pair_f_eff(density=TRANSVERSE)
    with pytest.raises(ValueError, match="x_kd"):
        model.mediator_pair_f_eff(density=TRANSVERSE, x_kd=0.0)
    with pytest.raises(ValueError, match="3x3"):
        model.mediator_pair_f_eff(density=np.eye(2) / 2.0)


def test_the_photon_spectra_add_the_resolved_prompt_photons() -> None:
    model = _model(mx=290.0, mv=500.0, gaxx=0.0, gvxx=1.0, leptons=ELECTRONS)
    model.quarks = ISOVECTOR
    e_cm = 590.0
    energies = np.array([[2.0, 20.0], [150.0, 280.0]])
    spectra = model.spectra(energies, e_cm)
    branching = model.annihilation_branching_fractions(e_cm)
    continuum = model.rest_frame_yields(energies, e_cm, product="photon")
    # The resolved density above the cut is the same for any cut below it.
    resolved = model.rest_frame_resolved_photons(energies, e_cm, cut=1.0)
    for fs in ("e e", "pi pi pi0"):
        assert branching[fs] > 0.0
        expected = branching[fs] * (continuum[fs] + resolved[fs])
        np.testing.assert_allclose(spectra[fs], expected, rtol=1e-14)
    scalar = model.total_spectrum(20.0, e_cm)
    assert scalar == pytest.approx(spectra["total"][0, 1], rel=1e-14)


def test_electron_pair_gamma_rays_match_the_single_channel_model() -> None:
    # Both take the Altarelli-Parisi photons of each electron, so the limits
    # agree through the same telescope response.
    model = _model(mx=50.0, mv=200.0, gaxx=0.0, gvxx=1.0, leptons=ELECTRONS)
    single = SingleChannelAnn(50.0, "e e", 1.0)
    energies = np.geomspace(0.1, 49.0, 7)
    np.testing.assert_allclose(
        model.total_spectrum(energies, 100.0),
        single.total_spectrum(energies, 100.0),
        rtol=1e-13,
    )
    assert model.binned_limit(comptel_diffuse) == pytest.approx(
        single.binned_limit(comptel_diffuse), rel=1e-12
    )
