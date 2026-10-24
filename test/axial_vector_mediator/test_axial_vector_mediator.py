"""Rates, widths and declared support of `hazma.axial_vector_mediator`."""

from __future__ import annotations

import numpy as np
import pytest
import spinor_oracle as oracle

from hazma import parameters
from hazma._core import axial_vector_mediator as core
from hazma.axial_vector_mediator import AxialVectorMediator, LeptonCouplings
from hazma.form_factors.axial import (
    AxialCurrent,
    AxialFormFactorGammaGamma,
    AxialFormFactorPiPiPi0,
    InvisibleCut,
    TriangleCouplings,
)
from hazma.relic_density._thermal_functions import thermal_cross_section


def _current(gau: float, gad: float, gas: float) -> AxialCurrent:
    return AxialCurrent(
        gau, gad, gas, qcd_scheme="MSbar", scale_mev=2000.0, provenance="test"
    )


ZERO = _current(0.0, 0.0, 0.0)
ISOVECTOR = _current(1e-3, -1e-3, 0.0)
#: The hypercharge-aligned benchmark direction of the mediator note,
#: `sec:ax-uv`, which has octet and singlet components.
HYPERCHARGE = _current(1e-3, -1e-3, -1e-3)
LEPTONS = LeptonCouplings(
    gvee=0.3,
    gaee=-0.4,
    gvmumu=0.5,
    gamumu=0.7,
    gvtautau=-0.2,
    gatautau=0.6,
    gnue=0.1,
    gnumu=0.2,
    gnutau=0.3,
)
#: ``(mass, g_V, g_A)`` of each fermion pair under LEPTONS.
PAIRS = {
    "e e": (parameters.electron_mass, 0.3, -0.4),
    "mu mu": (parameters.muon_mass, 0.5, 0.7),
    "tau tau": (parameters.tau_mass, -0.2, 0.6),
    "ve ve": (0.0, 0.05, -0.05),
    "vm vm": (0.0, 0.1, -0.1),
    "vt vt": (0.0, 0.15, -0.15),
}


def _model(**overrides: object) -> AxialVectorMediator:
    arguments = dict(
        mx=150.0,
        mv=500.0,
        gvxx=0.4,
        gaxx=0.9,
        quarks=ISOVECTOR,
        leptons=LEPTONS,
        gacc=0.0,
        theory="chpt",
    )
    arguments.update(overrides)
    return AxialVectorMediator(**arguments)


def _pair_densities(s: float, mass: float, gv: float, ga: float) -> tuple[float, float]:
    # ``(A_T, A_L)`` of a fermion pair, transcribed from ``lepton_spectral``
    # of the mediator note's ``ax_xsec.py``.
    beta = np.sqrt(1 - 4 * mass**2 / s)
    transverse = (
        beta / (6 * np.pi) * (gv**2 * (s + 2 * mass**2) + ga**2 * (s - 4 * mass**2))
    )
    return transverse, ga**2 * mass**2 * beta / np.pi


@pytest.mark.parametrize("width", [0.0, 30.0])
@pytest.mark.parametrize("final_state", ["e e", "mu mu", "tau tau", "vm vm"])
def test_fermion_pairs_match_explicit_spinors(final_state: str, width: float) -> None:
    # The note reached 2e-14 (`sec:ax-match`); the oracle's sixteen-term
    # complex sums add a few ulp per energy, so 1e-12.
    mass, gv, ga = PAIRS[final_state]
    mx, mv = (2000.0, 5000.0) if final_state == "tau tau" else (150.0, 500.0)
    model = _model(mx=mx, mv=mv, quarks=ZERO, width=width)
    for e_cm in (2.0 * mx * 1.0001, 2.0 * mx * 1.3, 0.97 * mv, 1.2 * mv):
        expected = oracle.sigma_v_fermion_pair(
            e_cm, mx=mx, gvxx=0.4, gaxx=0.9, mv=mv, width=width, mf=mass, gvf=gv, gaf=ga
        )
        actual = model.annihilation_sigma_v_funcs()[final_state](e_cm)
        assert actual == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("final_state", list(PAIRS))
def test_zero_width_matches_the_unsplit_spin_one_rate(final_state: str) -> None:
    # With the unsplit Proca propagator ``(g - kk/M²)/(s - M²)`` the rate is
    # ``{[g_V²(s+2m²) + g_A²(s-6m²)] A_T + 2m²g_A²[A_T + A_L(1-s/M²)²]}/(s(s-M²)²)``,
    # from the dark-matter trace ``-L^mu_mu`` and ``k.L.k/s`` directly.
    # Roundoff only, so 1e-13.
    mass, gv, ga = PAIRS[final_state]
    mx, mv = (2000.0, 5000.0) if final_state == "tau tau" else (150.0, 500.0)
    model = _model(mx=mx, mv=mv, quarks=ZERO, width=0.0)
    e_cm = 1.15 * max(2.0 * mx, 2.0 * mass)
    s = e_cm**2
    transverse, longitudinal = _pair_densities(s, mass, gv, ga)
    trace = 0.16 * (s + 2 * mx**2) + 0.81 * (s - 6 * mx**2)
    k_l_k = 2 * mx**2 * 0.81
    expected = (
        trace * transverse + k_l_k * (transverse + longitudinal * (1 - s / mv**2) ** 2)
    ) / (s * (s - mv**2) ** 2)
    actual = model.annihilation_sigma_v_funcs()[final_state](e_cm)
    assert actual == pytest.approx(expected, rel=1e-13)


def test_gamma_gamma_is_the_note_line_rate() -> None:
    # ``sigma v = g_A² m² alpha² |A|²/(4 pi³ M⁴)`` (note ``eq:ax-gg``), with
    # ``A`` from the amplitude entry point and independent of the width.
    # Roundoff only, so 1e-12.
    triangles = TriangleCouplings(electron=-0.4, muon=0.7, tau=0.6, charm=2e-3)
    model = _model(quarks=HYPERCHARGE, gacc=2e-3, width=12.0)
    cut = InvisibleCut(dm_mass=150.0, dm_axial=0.9, mediator_mass=500.0)
    e_cm = np.array([300.0, 450.0, 560.0])
    amplitude = AxialFormFactorGammaGamma().amplitude(
        e_cm, HYPERCHARGE, triangles, invisible=cut
    )
    expected = (
        0.81 * 150.0**2 * parameters.alpha_em**2 * np.abs(amplitude) ** 2
    ) / (4 * np.pi**3 * 500.0**4)
    actual = model.annihilation_sigma_v_funcs()["g g"](e_cm)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_partial_widths() -> None:
    # Textbook ``Gamma = M beta [g_V²(1+2r) + g_A² beta²]/(12 pi)`` for the
    # fermion pairs, and the three-pion width of its form factor. 1e-13.
    model = _model()
    widths = model.partial_widths()
    mv = 500.0
    for final_state, (mass, gv, ga) in PAIRS.items():
        r = mass**2 / mv**2
        beta = np.sqrt(max(1 - 4 * r, 0.0))
        expected = mv * beta * (gv**2 * (1 + 2 * r) + ga**2 * beta**2) / (12 * np.pi)
        assert widths[final_state] == pytest.approx(expected, rel=1e-13, abs=0.0)
    r = (150.0 / mv) ** 2
    beta = np.sqrt(1 - 4 * r)
    dm = mv * beta * (0.16 * (1 + 2 * r) + 0.81 * beta**2) / (12 * np.pi)
    assert widths["x x"] == pytest.approx(dm, rel=1e-13)
    three_pi = AxialFormFactorPiPiPi0().width(mv, ISOVECTOR)
    assert widths["pi pi pi0"] == pytest.approx(three_pi, rel=1e-14)
    assert "g g" not in widths
    total = sum(v for k, v in widths.items() if k != "total")
    assert widths["total"] == pytest.approx(total, rel=1e-15)
    assert model.width_v == widths["total"]


def test_width_override_replaces_the_sum() -> None:
    width = 3.5
    assert _model(width=width).width_v == width


@pytest.mark.parametrize("final_state", ["e e", "mu mu", "pi pi pi0", "g g"])
def test_threshold_limits_of_sigma_and_sigma_v_agree(final_state: str) -> None:
    """The rate is continuous at ``2 mx``, and sigma v_Mol reproduces it."""
    model = _model(mx=250.0, mv=560.0, quarks=HYPERCHARGE)
    sigma_v = model.annihilation_sigma_v_funcs()[final_state]
    sigma = model.annihilation_cross_section_funcs()[final_state]
    at_threshold = sigma_v(500.0)
    assert at_threshold > 0.0
    for e_cm in (500.0 * (1 + 1e-12), 500.0 * (1 + 1e-6)):
        v_mol = 2.0 * np.sqrt((e_cm - 500.0) * (e_cm + 500.0)) / e_cm
        assert sigma(e_cm) * v_mol == pytest.approx(sigma_v(e_cm), rel=1e-14)
    # sigma v departs from its threshold value as v², about 1e-11 here.
    assert sigma_v(500.0 * (1 + 1e-12)) == pytest.approx(at_threshold, rel=1e-9)
    assert sigma(500.0) == np.inf
    assert sigma(499.0) == 0.0
    assert sigma_v(499.0) == 0.0


def test_arrays_in_arrays_out() -> None:
    model = _model()
    sigma_v = model.annihilation_sigma_v_funcs()["mu mu"]
    e_cm = np.array([[310.0, 400.0], [450.0, 290.0]])
    values = sigma_v(e_cm)
    assert values.shape == e_cm.shape
    for index in np.ndindex(e_cm.shape):
        scalar = sigma_v(float(e_cm[index]))
        assert isinstance(scalar, float)
        assert values[index] == scalar
    sigma = model.annihilation_cross_section_funcs()["mu mu"](e_cm)
    assert sigma.shape == e_cm.shape


def test_thresholds_and_resonances() -> None:
    model = _model(quarks=HYPERCHARGE)
    thresholds = model.annihilation_thresholds()
    assert set(thresholds) == set(model.list_annihilation_final_states())
    assert thresholds["mu mu"] == 2 * parameters.muon_mass
    assert thresholds["pi0 pi0 pi0"] == 3 * parameters.neutral_pion_mass
    resonances = model.annihilation_resonances()
    assert resonances[0] == (500.0, model.width_v)
    assert [m for m, _ in resonances[1:]] == [
        parameters.neutral_pion_mass,
        parameters.eta_mass,
        parameters.eta_prime_mass,
    ]
    assert _model(quarks=ZERO).annihilation_resonances() == [
        (500.0, _model(quarks=ZERO).width_v)
    ]


def test_rates_are_per_pair_and_lines_carry_no_multiplicity() -> None:
    # The line branching fraction is the channel's share of the rate per
    # pair; the two photons and any Dirac population factor are left to the
    # consumers.
    model = _model(quarks=HYPERCHARGE)
    e_cm = 400.0
    sigmas = model.annihilation_cross_sections(e_cm)
    lines = model.gamma_ray_lines(e_cm)
    assert lines["g g"]["energy"] == e_cm / 2
    assert lines["g g"]["bf"] == pytest.approx(sigmas["g g"] / sigmas["total"])
    assert model.positron_lines(e_cm)["e e"]["energy"] == e_cm / 2


def test_three_pions_read_only_the_isovector_component() -> None:
    # G parity keeps the octet and singlet directions out of three pions,
    # so the rate is the same computation for the isovector card.
    full = _model(quarks=_current(4e-4, -1.6e-3, 7e-4), width=2.0)
    isovector = _model(quarks=_current(1e-3, -1e-3, 0.0), width=2.0)
    e_cm = np.array([420.0, 500.0, 590.0])
    # The two cards' T3 coefficients may differ by an ulp of rounding.
    for final_state in ("pi pi pi0", "pi0 pi0 pi0"):
        np.testing.assert_allclose(
            full.annihilation_sigma_v_funcs()[final_state](e_cm),
            isovector.annihilation_sigma_v_funcs()[final_state](e_cm),
            rtol=1e-14,
        )


def test_rcht_is_declared_but_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="rcht"):
        _model(theory="rcht")
    with pytest.raises(ValueError, match="theory"):
        _model(theory="lattice")


def test_three_pions_past_their_support_raise() -> None:
    model = _model(mx=320.0)
    with pytest.raises(ValueError, match="declared only up to 600"):
        model.annihilation_sigma_v_funcs()["pi pi pi0"](700.0)


def test_width_past_the_three_pion_support_needs_an_override() -> None:
    with pytest.raises(ValueError, match="declared only up to 600"):
        _model(mv=700.0)
    width = 4.0
    assert _model(mv=700.0, width=width).width_v == width
    # Without an isovector coupling, three pions vanish by symmetry.
    assert _model(mv=700.0, quarks=ZERO).partial_widths()["pi pi pi0"] == 0.0


def test_totals_refuse_an_open_mediator_pair() -> None:
    model = _model(mx=150.0, mv=200.0, quarks=ZERO)
    with pytest.raises(ValueError, match="v v"):
        model.annihilation_cross_sections(450.0)
    with pytest.raises(ValueError, match="v v"):
        model.annihilation_sigma_v(np.array([350.0, 450.0]))
    assert model.annihilation_cross_section_funcs()["e e"](450.0) > 0.0
    assert model.annihilation_sigma_v(350.0)["total"] > 0.0


def test_totals_refuse_eta_pi_pi_for_octet_and_singlet_sources() -> None:
    opening = parameters.eta_mass + 2 * parameters.neutral_pion_mass
    model = _model(mx=420.0, mv=1000.0, quarks=HYPERCHARGE, width=5.0)
    with pytest.raises(ValueError, match="eta pi0 pi0"):
        model.annihilation_cross_sections(opening + 30.0)
    with pytest.raises(ValueError, match="eta pi0 pi0"):
        _model(mv=900.0, quarks=HYPERCHARGE)
    assert (
        _model(mx=420.0, mv=1000.0, quarks=ZERO).annihilation_sigma_v(opening + 30.0)[
            "total"
        ]
        > 0.0
    )


def test_rates_reject_points_past_the_mixing_guard() -> None:
    # At the pi0 pole the note's ``g_A = 1e-3``, ``M = 500 MeV`` point has
    # ``epsilon_pi`` of order 2 (`sec:ax-match`).
    model = _model(mx=60.0, quarks=HYPERCHARGE)
    with pytest.raises(ValueError, match="MIXING_GUARD"):
        model.annihilation_sigma_v_funcs()["e e"](parameters.neutral_pion_mass)


#: Edges that hold every e± of a 500 MeV annihilation: no e± is below its
#: rest energy, and the hardest, the "e e" line, is at 250 MeV.
LEPTON_EDGES = np.geomspace(parameters.electron_mass, 400.0, 21)


def test_rest_frame_counts_carry_each_charged_lepton() -> None:
    # One e⁺ per annihilation from the μ⁺ of "mu mu", the π⁺ of "pi pi pi0"
    # and the "e e" line, and CP gives as many e⁻. The three-pion folds are
    # integrated to 1e-8, so 1e-7.
    model = _model()
    positrons = model.rest_frame_counts(LEPTON_EDGES, 500.0, product="positron")
    both = model.rest_frame_counts(LEPTON_EDGES, 500.0, product="electron_positron")
    for fs in ("e e", "mu mu", "pi pi pi0"):
        assert np.sum(positrons[fs]) == pytest.approx(1.0, rel=1e-7)
        np.testing.assert_allclose(both[fs], 2.0 * positrons[fs], rtol=1e-14)
    assert np.sum(positrons["pi0 pi0 pi0"]) == 0.0


@pytest.mark.parametrize("excess", [1e-9, 1e-6, 1e-3])
def test_three_pion_counts_stay_normalized_at_threshold(excess: float) -> None:
    # Just above 3m the Dalitz plot shrinks to a point, where total-energy
    # primitives cancelled and gave 1.0126 e⁺ at 1e-6 MeV and 2.3e5 at
    # 1e-9 MeV. The kinetic-energy form keeps one e⁺ to about 4e-11.
    model = _model(gaxx=0.0)
    q = 3.0 * parameters.neutral_pion_mass + excess
    counts = model.rest_frame_counts(np.array([0.0, 400.0]), q, product="positron")
    assert counts["pi pi pi0"][0] == pytest.approx(1.0, rel=1e-9)


def test_pion_rest_mean_positron_energy_carries_the_muon_spin() -> None:
    # Just above the three-pion threshold the π⁺ decays at rest, so the e⁺
    # yield of "pi pi pi0" is the polarized pion chain's rest spectrum with
    # its two-body e⁺ line. Bins of 0.2 MeV centered on the line hold it
    # exactly, and the midpoint sum of the smooth continuum is second order
    # in the width, near 1e-9 here. The mediator note's closed form with
    # Hazma's masses gives 36.94069258356754 MeV with the spin and
    # 38.430221 MeV without it; the kernel reproduces the first to 1e-11
    # and this binned public path to 1e-8, far inside the 4% the spin moves.
    model = _model(gaxx=0.0)
    e_cm = 3.0 * parameters.neutral_pion_mass + 1e-6
    m_pi, m_e = parameters.charged_pion_mass, parameters.electron_mass
    line = (m_pi**2 + m_e**2) / (2.0 * m_pi)
    width = 0.2
    # The first edge lies between `m_e - 1.5 width` and `m_e - 0.5 width`,
    # below every e⁺ and above zero.
    n_low = int(np.floor((line - m_e) / width)) + 2
    n_high = int(np.ceil(1.0 / width)) + 1
    edges = line + (np.arange(-n_low, n_high + 1) + 0.5) * width
    counts = model.rest_frame_counts(edges, e_cm, product="positron")["pi pi pi0"]
    assert np.sum(counts) == pytest.approx(1.0, rel=1e-9)
    mean = np.sum((edges[:-1] + edges[1:]) / 2.0 * counts)
    assert mean == pytest.approx(36.94069258356754, rel=1e-8)


@pytest.mark.parametrize("width", [0.0, 30.0])
def test_muon_polarization_matches_helicity_spinors(width: float) -> None:
    # The μ⁻ of "mu mu" has the helicity-weighted rate over the rate as its
    # effective polarization, which the e⁺ and neutrino kernels boost. The
    # oracle resolves the μ⁻ spinors by helicity along its momentum and
    # sums the μ⁺ spins, never forming the helicity density; like the rate
    # test, 1e-12. The yields then carry that value: the public e⁺ density
    # is the kernel at the oracle's polarization.
    mass, gv, ga = PAIRS["mu mu"]
    mx, mv = 106.0, 500.0
    model = _model(mx=mx, mv=mv, quarks=ZERO, width=width)
    for e_cm in (2.0 * mx * 1.0001, 300.0, 0.97 * mv, 1.2 * mv):
        expected = oracle.fermion_polarization(
            e_cm, mx=mx, gvxx=0.4, gaxx=0.9, mv=mv, width=width, mf=mass, gvf=gv, gaf=ga
        )
        assert model._muon_polarization(e_cm) == pytest.approx(expected, rel=1e-12)
    energies = np.geomspace(1.0, 250.0, 7)
    expected = oracle.fermion_polarization(
        600.0, mx=mx, gvxx=0.4, gaxx=0.9, mv=mv, width=width, mf=mass, gvf=gv, gaf=ga
    )
    rows = core.muon_pair_yields("charged_lepton", energies, False, 300.0, expected)
    np.testing.assert_allclose(
        model.rest_frame_yields(energies, 600.0, product="positron")["mu mu"],
        rows[0],
        rtol=1e-12,
    )


def test_rest_frame_counts_round_trip_by_bin_width() -> None:
    # A bin of relative width 1e-6 holds the density times its width up to
    # a second-order error near 1e-12; the muon kernels are good to 1e-10.
    model = _model()
    energy, width = 40.0, 40e-6
    edges = np.array([energy - width / 2.0, energy + width / 2.0])
    for product in ("positron", "neutrino", "photon"):
        counts = model.rest_frame_counts(edges, 400.0, product=product)["mu mu"]
        density = model.rest_frame_yields(energy, 400.0, product=product)["mu mu"]
        np.testing.assert_allclose(counts[..., 0] / width, density, rtol=1e-8)


def test_neutrino_yields_are_per_flavor() -> None:
    model = _model()
    scalar = model.rest_frame_yields(10.0, 400.0, product="neutrino")["mu mu"]
    array = model.rest_frame_yields(np.array([10.0, 20.0]), 400.0, product="neutrino")
    assert scalar.shape == (3,)
    assert array["mu mu"].shape == (3, 2)
    np.testing.assert_array_equal(array["mu mu"][:, 0], scalar)
    assert scalar[2] == 0.0
    lines = model.rest_frame_lines(400.0, product="neutrino")
    np.testing.assert_array_equal(lines["vm vm"]["multiplicity"], [0.0, 2.0, 0.0])
    assert lines["vm vm"]["energy"] == pytest.approx(400.0 / 2.0, rel=0.0)


def test_lines_on_the_last_edge_are_counted() -> None:
    model = _model()
    # The two photons of "g g" sit exactly on the last edge, e_cm/2.
    counts = model.rest_frame_counts(np.array([150.0, 200.0]), 400.0, product="photon")
    np.testing.assert_array_equal(counts["g g"], [2.0])


def test_spectra_weight_the_rest_frame_yields() -> None:
    model = _model()
    energies = np.array([10.0, 30.0])
    bfs = model.annihilation_branching_fractions(400.0)
    yields = model.rest_frame_yields(energies, 400.0, product="positron")
    spectra = model.positron_spectra(energies, 400.0)
    np.testing.assert_allclose(spectra["mu mu"], bfs["mu mu"] * yields["mu mu"])
    assert np.all(model.spectra(energies, 400.0)["total"] > 0.0)


def test_tau_yields_are_refused() -> None:
    taus_only = LeptonCouplings(
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
    # A heavy mediator keeps the unmodeled "v v" closed.
    model = _model(mv=5000.0, quarks=ZERO, leptons=taus_only)
    with pytest.raises(ValueError, match="tau decay"):
        model.rest_frame_yields(10.0, 3600.0, product="positron")


def test_rest_frame_inputs_are_validated() -> None:
    model = _model()
    with pytest.raises(ValueError, match="product"):
        model.rest_frame_yields(10.0, 400.0, product="proton")
    with pytest.raises(ValueError, match="increasing"):
        model.rest_frame_counts(np.array([2.0, 1.0]), 400.0, product="positron")
    with pytest.raises(ValueError, match="positive"):
        model.rest_frame_yields(0.0, 400.0, product="photon")


def test_setters_recompute_the_width_or_change_nothing() -> None:
    model = _model()
    mv, before = model.mv, model.width_v
    model.gaxx = 0.5
    assert model.width_v != before
    assert model.width_v == model.partial_widths()["total"]
    width = model.width_v
    with pytest.raises(ValueError, match="declared only up to 600"):
        model.mv = 700.0
    assert model.mv == mv
    assert model.width_v == width
    with pytest.raises(ValueError, match="mx"):
        model.mx = -1.0
    with pytest.raises(TypeError, match="AxialCurrent"):
        model.quarks = (1e-3, -1e-3, 0.0)


ELECTRONS_ONLY = LeptonCouplings(
    gvee=1e-2,
    gaee=0.0,
    gvmumu=0.0,
    gamumu=0.0,
    gvtautau=0.0,
    gatautau=0.0,
    gnue=0.0,
    gnumu=0.0,
    gnutau=0.0,
)


def test_vanishing_channels_neither_open_nor_range_check() -> None:
    model = _model(mx=900.0, mv=5000.0, gaxx=0.0, quarks=ZERO, leptons=ELECTRONS_ONLY)
    assert set(model.annihilation_thresholds()) == {"e e"}
    rates = model.annihilation_sigma_v(1850.0)
    assert rates["g g"] == 0.0
    assert rates["pi pi pi0"] == 0.0
    assert rates["total"] == rates["e e"] > 0.0


def test_thermal_average_runs_and_reaches_the_s_wave() -> None:
    # Pure vector dark matter coupled to electrons alone has an s-wave
    # sigma v = a + b v², so the thermal average departs from its threshold
    # value by O(1/x): -5.0e-5 measured at x = 1e4, hence 2e-4.
    model = _model(
        mx=10.0, mv=5000.0, gvxx=1.0, gaxx=0.0, quarks=ZERO, leptons=ELECTRONS_ONLY
    )
    threshold = model.annihilation_sigma_v_funcs()["e e"](20.0)
    average = thermal_cross_section(1e4, model)
    assert average == pytest.approx(threshold, rel=2e-4)
    assert np.isfinite(thermal_cross_section(20.0, model))


def test_an_exact_zero_width_pole_raises() -> None:
    model = _model(width=0.0)
    with pytest.raises(ValueError, match="exact pole of zero width"):
        model.annihilation_sigma_v_funcs()["e e"](500.0)
    # A vanishing resonant numerator stays zero on the pole: pure axial
    # dark matter at threshold, here with M = 2 mx.
    axial = _model(mx=150.0, mv=300.0, gvxx=0.0, quarks=ZERO, width=0.0)
    assert np.isfinite(axial.annihilation_sigma_v_funcs()["mu mu"](300.0))


def test_mixing_guard_spares_vector_dark_matter() -> None:
    # Only the axial contact feels the meson poles, so pure vector dark
    # matter has a rate at the pi0 pole where the axial model raises.
    model = _model(mx=60.0, gaxx=0.0, quarks=HYPERCHARGE)
    rate = model.annihilation_sigma_v_funcs()["e e"](parameters.neutral_pion_mass)
    assert np.isfinite(rate)
    assert rate > 0.0


def test_branching_fractions_are_finite_at_threshold() -> None:
    model = _model(quarks=HYPERCHARGE)
    at_threshold = model.annihilation_branching_fractions(300.0)
    assert all(np.isfinite(value) for value in at_threshold.values())
    assert sum(at_threshold.values()) == pytest.approx(1.0, rel=1e-14)
    sigmas = model.annihilation_cross_sections(400.0)
    fractions = model.annihilation_branching_fractions(400.0)
    for fs, fraction in fractions.items():
        assert fraction == pytest.approx(sigmas[fs] / sigmas["total"], rel=1e-14)


def test_longitudinal_channels_vanish_without_axial_dark_matter() -> None:
    # "g g" (Landau-Yang) and the LO "pi0 pi0 pi0" (pion pole only) have no
    # transverse density, so with gaxx = 0 they are exact zeros that neither
    # open nor range-check, while "pi pi pi0" keeps its transverse rate.
    model = _model(mx=320.0, mv=5000.0, gaxx=0.0, width=1.0)
    assert {"g g", "pi0 pi0 pi0"}.isdisjoint(model.annihilation_thresholds())
    assert model.annihilation_sigma_v_funcs()["pi0 pi0 pi0"](700.0) == 0.0
    assert "pi pi pi0" in model.annihilation_thresholds()
    with pytest.raises(ValueError, match="declared only up to 600"):
        model.annihilation_sigma_v_funcs()["pi pi pi0"](700.0)


def test_meson_poles_leave_the_resonances_with_the_gamma_gamma_rate() -> None:
    # With gaxx = 0 the "g g" rate vanishes, so its pi0, eta and eta' poles
    # are not resonances and cannot carry the thermal window to the eta'
    # pole, past the three-pion range. The average then departs from the
    # e e s-wave value by O(1/x): 1.2e-4 measured at x = 1e4, hence 5e-4.
    model = _model(gaxx=0.0, quarks=HYPERCHARGE, leptons=ELECTRONS_ONLY)
    assert model.annihilation_resonances() == [(500.0, model.width_v)]
    threshold = model.annihilation_sigma_v_funcs()["e e"](300.0)
    assert thermal_cross_section(1e4, model) == pytest.approx(threshold, rel=5e-4)
