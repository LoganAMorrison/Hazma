"""Coherent longitudinal pi0, eta and eta' poles and the gamma gamma density.

The independent references are the research Berends-Giele implementation of
the mediator note (pinned values), the full-kernel NumPy oracle in
`axial_oracle`, analytic limits, and the note's quoted benchmarks.
"""

from __future__ import annotations

import axial_oracle as oracle
import numpy as np
import pytest

from hazma import parameters
from hazma.form_factors.axial import (
    MIXING_GUARD,
    AxialCurrent,
    AxialFormFactorGammaGamma,
    AxialMesonPoles,
    InvisibleCut,
    TriangleCouplings,
    current_projections,
)


def _current(gau: float, gad: float, gas: float) -> AxialCurrent:
    return AxialCurrent(
        gau, gad, gas, qcd_scheme="MSbar", scale_mev=2000.0, provenance="test"
    )


# The hypercharge-aligned benchmark of the mediator note (sec:ax-uv), with
# g' z_H1 = 2e-3: g_A^u = -g_A^{d,s} = -g_A^l = g' z_H1/2, and charm up-like.
HYPERCHARGE = _current(1e-3, -1e-3, -1e-3)
HYPERCHARGE_TRIANGLES = TriangleCouplings(
    electron=-1e-3, muon=-1e-3, tau=-1e-3, charm=1e-3
)
NO_TRIANGLES = TriangleCouplings(electron=0.0, muon=0.0, tau=0.0, charm=0.0)
ZERO = _current(0.0, 0.0, 0.0)
POLES = AxialMesonPoles()
GAMMA_GAMMA = AxialFormFactorGammaGamma()
MASSES = np.array([POLES.pion_mass, POLES.eta_mass, POLES.eta_prime_mass])
WIDTHS = np.array([POLES.pion_width, POLES.eta_width, POLES.eta_prime_width])
ANGLE = np.deg2rad(POLES.mixing_angle_deg)
# Energies (MeV) on both sides of, and within a width or two of, each pole.
Q_GRID = [1.0, 60.0, 134.9, 135.0, 300.0, 547.8, 547.9, 700.0, 957.7, 1600.0]
SOURCES = [
    HYPERCHARGE,
    _current(0.3, 0.7, -1.1),
    _current(1.0, 1.0, 1.0),
    _current(0.5, -0.5, 0.0),
]


def _flavors(current: AxialCurrent) -> np.ndarray:
    return np.array([current.gau, current.gad, current.gas])


def test_gamma_gamma_density_matches_the_research_berends_giele_evaluation() -> None:
    """Pin A_L against an independent tree-level amplitude assembly.

    `ax_xsec.py` at research commit 3d9c6c7 (mev-gev-pseudo-scalar-axial-
    vector, version3/integrated) builds the Z' -> gamma gamma amplitude by
    Berends-Giele recursion over its derived U(3) ChPT + WZW vertices and adds
    the triangles, then sums photon polarizations numerically. These values
    are its `_gammagamma` output for model "chpt" with alpha_em and the
    lepton and charm masses set to Hazma's. The closed form agrees to at most
    1.4e-15 relative (macOS/arm64); 4e-15 allows platform rounding.
    """
    q = np.array([50.0, 134.0, 300.0, 547.0, 700.0, 957.0, 1200.0, 1600.0])
    expected = np.array(
        [
            7.088522022988635e-10,
            1.8449631892507285e-05,
            3.693905514650761e-08,
            0.0012824985021747432,
            1.271199923649508e-07,
            0.043933205002674856,
            2.3476987223526543e-06,
            2.0755496757798723e-06,
        ]
    )
    result = GAMMA_GAMMA.spectral_functions(q, HYPERCHARGE, HYPERCHARGE_TRIANGLES)
    np.testing.assert_allclose(result.source_longitudinal, expected, rtol=4e-15)
    assert result.units == "MeV^2"


def test_benchmark_mixing_strengths_match_the_note() -> None:
    # The note (sec:ax-inv) quotes c = (0.184, 0.083, -0.101) MeV to three figures.
    np.testing.assert_allclose(
        POLES.source_couplings(HYPERCHARGE), [0.184, 0.083, -0.101], atol=5e-4
    )


@pytest.mark.parametrize("current", SOURCES)
@pytest.mark.parametrize("mediator_mass", [300.0, 2500.0, 1e5])
def test_schur_complement_matches_the_full_kernel_inverse(
    current: AxialCurrent, mediator_mass: float
) -> None:
    """The physical-basis meson resummation equals the full 4x4 inverse.

    The oracle inverts the whole kernel over (a, pi3, eta8, eta0) in the
    generator basis; the kernel sums the diagonal meson block in the physical
    basis in closed form. Both are exact, so they differ by the oracle's
    rounding amplified by the meson block's conditioning, max/min |D_P| up
    to 1.4e5 on this grid (q = 135.0 MeV). The largest measured difference
    is 3.0e-13 relative (macOS/arm64), and 1e-11 allows for that
    conditioning on other platforms.
    The amplitude does not depend on the mediator mass, which only sets K_aa.
    """
    for q in Q_GRID:
        kernel = oracle.longitudinal_kernel(
            q, _flavors(current), MASSES, WIDTHS, ANGLE, 92.2, mediator_mass
        )
        h_eff = oracle.schur_source_amplitude(kernel, oracle.photon_traces() / 92.2)
        # h_eff = -i sqrt(s) a^T D^-1 b, so A_poles = (i sqrt(s)/2) h_eff.
        expected = 0.5j * q * h_eff
        actual = GAMMA_GAMMA.amplitude(q, current, NO_TRIANGLES)
        assert abs(actual - expected) <= 1e-11 * abs(expected), q
        self_energy = POLES.source_self_energy(q, current)
        expected = oracle.source_self_energy(kernel)
        assert abs(self_energy - expected) <= 1e-11 * abs(expected), q


@pytest.mark.parametrize("current", SOURCES)
def test_chiral_meson_poles_cancel_the_axial_contact(current: AxialCurrent) -> None:
    """Ward identity: with massless, stable mesons s c^T D^-1 c = 2 f² Tr(G_A²).

    s/(s - 0) is exactly one, so only the rounding of the trace sum
    separates poles from contact; 1e-14 is a few ulps of the sum.
    """
    chiral = AxialMesonPoles(
        pion_mass=0.0,
        eta_mass=0.0,
        eta_prime_mass=0.0,
        pion_width=0.0,
        eta_width=0.0,
        eta_prime_width=0.0,
    )
    contact = 2 * 92.2**2 * np.sum(_flavors(current) ** 2)
    q = np.array([1e-3, 1.0, 300.0, 1e5])
    np.testing.assert_allclose(
        chiral.source_self_energy(q, current), contact, rtol=1e-14
    )


@pytest.mark.parametrize("current", SOURCES)
def test_massive_poles_leave_the_pcac_remainder(current: AxialCurrent) -> None:
    """Away from the chiral limit the contact minus the poles is PCAC's.

    ``2 f² Tr(G_A²) - s c^T D^-1 c = -sum_P c_P² m_P²/(s - m_P²)`` for stable
    mesons, so the Ward-identity violation scales as m_P². The remainder is
    a difference of two O(f² Tr G_A²) numbers; 1e-12 of the contact covers it.
    """
    stable = AxialMesonPoles(pion_width=0.0, eta_width=0.0, eta_prime_width=0.0)
    c = stable.source_couplings(current)
    contact = 2 * 92.2**2 * np.sum(_flavors(current) ** 2)
    for q in [10.0, 300.0, 700.0, 1500.0]:
        s = q * q
        remainder = -np.sum(c**2 * MASSES**2 / (s - MASSES**2))
        actual = contact - stable.source_self_energy(q, current)
        assert abs(actual - remainder) <= 1e-12 * contact, q


def test_proca_contraction_with_a_longitudinal_current_has_no_mediator_pole() -> None:
    """The unreduced Proca propagator turns k^mu into a contact.

    ``J.P(k).k`` with ``P = -(g - k k/M²)/(s - M²)`` equals ``J.k/M²`` for
    every s != M², including s within 1e-6 of the pole, so the source-
    longitudinal response cannot resonate at the mediator mass. The
    numerical cancellation of (1 - s/M²)/(s - M²) loses ~1e-10 there.
    """
    rng = np.random.default_rng(4)
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    mediator_mass = 700.0
    for scale in [0.3, 1 - 1e-6, 1 + 1e-6, 4.0]:
        rest = np.array([mediator_mass * np.sqrt(scale), 0.0, 0.0, 0.0])
        k = rest + np.array([0.0, *rng.normal(size=3) * 10.0])
        k[0] = np.sqrt(mediator_mass**2 * scale + np.sum(k[1:] ** 2))
        s = k @ metric @ k
        current = rng.normal(size=4) + 1j * rng.normal(size=4)
        propagator = -(metric - np.outer(k, k) / mediator_mass**2) / (
            s - mediator_mass**2
        )
        lowered = metric @ k
        contracted = (metric @ current) @ propagator @ lowered
        contact = (metric @ current) @ k / mediator_mass**2
        assert abs(contracted - contact) <= 1e-9 * abs(contact), scale


def test_amplitude_approaches_the_abj_coefficient_away_from_the_poles() -> None:
    """Far above the poles the light-quark poles sum to the anomaly.

    With the masses lowered 100-fold the support reaches q >> m_P: the
    deviation from N_c sum_q g_A^q Q_q² is the exact pole remainder
    (1/2) sum_P a_P b_P (m_P² - i m_P Gamma_P)/(s - m_P² + i m_P Gamma_P),
    and it falls as 1/s.
    """
    light = AxialFormFactorGammaGamma(
        poles=AxialMesonPoles(
            pion_mass=MASSES[0] / 100,
            eta_mass=MASSES[1] / 100,
            eta_prime_mass=MASSES[2] / 100,
        )
    )
    abj = 3 * np.sum(_flavors(HYPERCHARGE) * np.array([4, 1, 1]) / 9)
    rotation = oracle.physical_rotation(ANGLE)
    a = rotation @ oracle.OCTET_SINGLET @ _flavors(HYPERCHARGE)
    b = rotation @ oracle.photon_traces()
    q = np.array([100.0, 400.0, 1600.0])
    s = q[:, None] ** 2
    masses = MASSES / 100
    remainder = 0.5 * np.sum(
        a
        * b
        * (masses**2 - 1j * masses * WIDTHS)
        / (s - masses**2 + 1j * masses * WIDTHS),
        axis=1,
    )
    deviation = light.amplitude(q, HYPERCHARGE, NO_TRIANGLES) - abj
    np.testing.assert_allclose(deviation, remainder, rtol=1e-9)
    relative = np.abs(deviation / abj)
    assert relative[-1] < 2 * (masses[2] / q[-1]) ** 2
    # From 400 to 1600 MeV the deviation falls 16-fold, up to its own
    # O(m_P²/s) correction, 6e-4 at 400 MeV.
    assert relative[2] / relative[1] == pytest.approx(1 / 16, rel=2e-3)


def test_transverse_gamma_gamma_density_is_exactly_zero() -> None:
    """Landau-Yang: two real photons carry no spin-one transverse weight.

    The returned A_T is identically zero. The explicit current
    ``H^mu = k^mu (k.H)/s`` for summed photon polarizations also projects to
    zero transverse weight up to rounding, and its longitudinal projection,
    with phase space 1/(16 pi), reproduces A_L to 1e-13.
    """
    q = np.array([0.0, 120.0, 547.9, 1600.0])
    result = GAMMA_GAMMA.spectral_functions(q, HYPERCHARGE, HYPERCHARGE_TRIANGLES)
    assert np.all(result.transverse == 0.0)
    assert result.source_longitudinal[0] == 0.0
    for energy in q[1:]:
        amplitude = GAMMA_GAMMA.amplitude(energy, HYPERCHARGE, HYPERCHARGE_TRIANGLES)
        k1 = np.array([energy / 2, 0.3, -0.4, np.sqrt(energy**2 / 4 - 0.25)])
        k2 = np.array([energy / 2, -0.3, 0.4, -np.sqrt(energy**2 / 4 - 0.25)])
        k = k1 + k2
        pols1 = _photon_polarizations(k1)
        pols2 = _photon_polarizations(k2)
        currents = []
        for e1 in pols1:
            for e2 in pols2:
                epsilon = _levi_civita(k1, e1, k2, e2)
                k_dot_h = -2j * parameters.alpha_em / np.pi * amplitude * epsilon
                currents.append(k * k_dot_h / energy**2)
        currents = np.array(currents).T
        momenta = np.repeat(k[:, None], 4, axis=1)
        transverse, longitudinal = current_projections(currents, momenta)
        scale = np.max(np.abs(currents)) ** 2
        assert np.all(np.abs(transverse) <= 1e-14 * scale)
        density = np.sum(longitudinal) / (16 * np.pi)
        expected = GAMMA_GAMMA.spectral_functions(
            energy, HYPERCHARGE, HYPERCHARGE_TRIANGLES
        ).source_longitudinal
        assert density == pytest.approx(expected, rel=1e-13)


def _photon_polarizations(k: np.ndarray) -> list[np.ndarray]:
    direction = k[1:] / np.linalg.norm(k[1:])
    first = np.cross(direction, [1.0, 0.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return [np.concatenate([[0.0], first]), np.concatenate([[0.0], second])]


def _levi_civita(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """``epsilon_{mu nu rho sigma} a^mu b^nu c^rho d^sigma`` with epsilon_0123 = -1."""
    lowered = [np.array([v[0], -v[1], -v[2], -v[3]]) for v in (a, b, c, d)]
    return -np.linalg.det(np.array(lowered))


@pytest.mark.parametrize("pole", range(3))
def test_residue_is_recovered_at_each_pole(pole: int) -> None:
    """``(s - m_P²) A`` tends to ``(1/2) m_P² a_P b_P`` at each stable pole.

    Averaging s' = s(1 +- delta) cancels the regular part to O(delta²) and
    leaves rounding of order ulp/delta, ~1e-10 for delta = 1e-6.
    """
    stable = AxialFormFactorGammaGamma(
        poles=AxialMesonPoles(pion_width=0.0, eta_width=0.0, eta_prime_width=0.0)
    )
    rotation = oracle.physical_rotation(ANGLE)
    a = rotation @ oracle.OCTET_SINGLET @ _flavors(HYPERCHARGE)
    b = rotation @ oracle.photon_traces()
    s = MASSES[pole] ** 2
    delta = 1e-6
    scales = np.array([1 + delta, 1 - delta])
    amplitudes = stable.amplitude(np.sqrt(s * scales), HYPERCHARGE, NO_TRIANGLES)
    residue = np.mean((scales - 1) * s * amplitudes)
    assert residue.real == pytest.approx(0.5 * s * a[pole] * b[pole], rel=1e-8)
    assert abs(residue.imag) <= 1e-12 * abs(residue.real)
    with pytest.raises(ValueError, match="zero width"):
        stable.amplitude(MASSES[pole], HYPERCHARGE, NO_TRIANGLES)


def test_invisible_cut_inserted_once_matches_the_mediator_line_cut() -> None:
    """The meson-block cut equals the a-line pair cut, moved onto the mesons.

    Integrating out ``a`` at first order in its pair self-energy Pi_aa adds
    ``i s Pi_aa c c^T/M⁴`` to the meson block. The oracle computes Pi_aa
    from explicit Dirac traces (with a vector coupling that must drop out)
    and inverts the full kernel; the kernel inserts its own rank-one cut in
    the physical basis. The largest measured difference is 1.7e-12 relative
    (macOS/arm64), within the 1e-11 conditioning allowance of the Schur test.
    """
    dm_mass, dm_axial, mediator_mass = 20.0, 3e3, 2500.0
    cut = InvisibleCut(dm_mass=dm_mass, dm_axial=dm_axial, mediator_mass=mediator_mass)
    c = 92.2 * oracle.OCTET_SINGLET @ _flavors(HYPERCHARGE)
    for q in [30.0, 134.97, 300.0, 547.86, 957.7]:
        pair = oracle.longitudinal_pair_cut(q, dm_mass, 0.7, dm_axial)
        insertion = 1j * q * q * pair * np.outer(c, c) / mediator_mass**4
        kernel = oracle.longitudinal_kernel(
            q,
            _flavors(HYPERCHARGE),
            MASSES,
            WIDTHS,
            ANGLE,
            92.2,
            mediator_mass,
            meson_insertion=insertion,
        )
        h_eff = oracle.schur_source_amplitude(kernel, oracle.photon_traces() / 92.2)
        expected = 0.5j * q * h_eff
        actual = GAMMA_GAMMA.amplitude(q, HYPERCHARGE, NO_TRIANGLES, invisible=cut)
        assert abs(actual - expected) <= 1e-11 * abs(expected), q
    # Below the pair threshold the cut is closed and changes nothing.
    assert GAMMA_GAMMA.amplitude(
        39.0, HYPERCHARGE, NO_TRIANGLES, invisible=cut
    ) == GAMMA_GAMMA.amplitude(39.0, HYPERCHARGE, NO_TRIANGLES)


def test_invisible_widths_reproduce_the_note_benchmarks() -> None:
    """Gamma(P -> chi chibar) against the note's quoted values (sec:ax-inv).

    For the pi0-aligned current G_A = g_q lambda_pi with kappa = g_A^chi g_q/M²
    = 4e-6 GeV^-2 at m_chi = 60 MeV the note quotes Gamma = 1.9263e-14 MeV
    and BR = 2.4696175e-9 against Gamma_SM = 7.8e-6 MeV; for the hypercharge
    benchmark (M = 2.5 GeV, g_A^chi = 1) it quotes BR(pi0) = 3.95e-6. Each
    tolerance is half a unit in the last quoted figure.
    """
    g_q = 1e-3
    aligned = _current(g_q, -g_q, 0.0)
    cut = InvisibleCut(dm_mass=60.0, dm_axial=1.0, mediator_mass=np.sqrt(g_q / 4e-12))
    widths = POLES.invisible_widths(aligned, cut)
    assert widths[0] == pytest.approx(1.9263e-14, abs=5e-19)
    branching = widths[0] / (POLES.pion_width + widths[0])
    assert branching == pytest.approx(2.4696175e-9, abs=5e-17)
    benchmark = InvisibleCut(dm_mass=60.0, dm_axial=1.0, mediator_mass=2500.0)
    pion = POLES.invisible_widths(HYPERCHARGE, benchmark)[0]
    assert pion / (POLES.pion_width + pion) == pytest.approx(3.95e-6, abs=5e-9)
    # The cut adds exactly these widths at each pole of the meson block, so
    # the gamma gamma line shape near the pi0 is the broadened one.
    closed = InvisibleCut(dm_mass=POLES.eta_prime_mass, dm_axial=1.0, mediator_mass=1.0)
    np.testing.assert_array_equal(POLES.invisible_widths(HYPERCHARGE, closed), 0.0)


def test_mixing_expansion_reproduces_the_note_example() -> None:
    """``epsilon_pi`` at the pi0 peak for g_A^u = -g_A^d = 1e-3, M = 500 MeV.

    The note (sec:ax-match) states c_pi² m_pi²/M² ~ 2.4 m_pi Gamma_pi, so
    epsilon_pi = 2.4 there, far above the guard; two figures are quoted.
    """
    isovector = _current(1e-3, -1e-3, 0.0)
    epsilon = POLES.mixing_expansion(MASSES[0], isovector, mediator_mass=500.0)
    assert epsilon.shape == (3,)
    assert epsilon[0] == pytest.approx(2.4, abs=0.05)
    assert epsilon[0] > MIXING_GUARD
    c = POLES.source_couplings(isovector)
    q = np.array([[50.0, 700.0], [300.0, 1500.0]])
    s = q**2
    expected = np.abs(
        c[:, None, None] ** 2
        * s
        / (
            500.0**2
            * (
                s
                - MASSES[:, None, None] ** 2
                + 1j * MASSES[:, None, None] * WIDTHS[:, None, None]
            )
        )
    )
    np.testing.assert_allclose(
        POLES.mixing_expansion(q, isovector, mediator_mass=500.0), expected, rtol=1e-14
    )


def test_scalar_and_array_energies_broadcast() -> None:
    q = np.array([[50.0, 300.0, 700.0], [134.0, 547.0, 1600.0]])
    amplitude = GAMMA_GAMMA.amplitude(q, HYPERCHARGE, HYPERCHARGE_TRIANGLES)
    assert amplitude.shape == q.shape
    scalar = GAMMA_GAMMA.amplitude(300.0, HYPERCHARGE, HYPERCHARGE_TRIANGLES)
    assert isinstance(scalar, complex)
    assert scalar == amplitude[0, 1]
    result = GAMMA_GAMMA.spectral_functions(q, HYPERCHARGE, HYPERCHARGE_TRIANGLES)
    assert result.source_longitudinal.shape == q.shape
    assert isinstance(POLES.source_self_energy(300.0, HYPERCHARGE), complex)
    assert POLES.source_self_energy(q, HYPERCHARGE).shape == q.shape


@pytest.mark.parametrize(
    ("q", "triangles", "weight", "mass"),
    [
        (
            1.0,
            TriangleCouplings(electron=0.0, muon=0.0, tau=0.0, charm=1.0),
            4 / 3,
            parameters.charm_quark_mass,
        ),
        (
            0.6,
            TriangleCouplings(electron=0.0, muon=1.0, tau=0.0, charm=0.0),
            1.0,
            parameters.muon_mass,
        ),
    ],
)
def test_heavy_triangle_decouples_below_its_pair_threshold(
    q: float, triangles: TriangleCouplings, weight: float, mass: float
) -> None:
    """``1 - tau f(tau) = -(x/3 + 8x²/45) + O(x³)`` with ``x = s/(4 m²)``.

    With no light-quark source the amplitude is the triangle alone. This
    two-term reference is good to (12/35) x², 2.2e-11 for the muon here, so
    1e-10 bounds it.
    """
    x = q * q / (4 * mass**2)
    amplitude = GAMMA_GAMMA.amplitude(q, ZERO, triangles)
    expected = -weight * (x / 3 + 8 * x**2 / 45)
    assert amplitude.real == pytest.approx(expected, rel=1e-10)
    assert amplitude.imag == 0.0


def test_light_triangle_carries_the_full_anomaly() -> None:
    """Far above threshold the electron triangle is within tau log² of one.

    It matches the oracle's textbook f(tau), whose 1 - r cancellation costs
    at most ~1e-9 of the logarithm here; 1e-13 of the factor covers it.
    """
    electron = TriangleCouplings(electron=1.0, muon=0.0, tau=0.0, charm=0.0)
    amplitude = GAMMA_GAMMA.amplitude(1600.0, ZERO, electron)
    tau = 4 * parameters.electron_mass**2 / 1600.0**2
    assert abs(amplitude - 1.0) < tau * (np.log(4 / tau) ** 2 + np.pi**2) / 4
    expected = oracle.triangle_factor(1600.0, parameters.electron_mass)
    assert amplitude == pytest.approx(expected, rel=1e-13)


# ``1 - tau f(tau)`` for each triangle fermion at Hazma's masses, from mpmath at
# 40 digits with the textbook f(tau), rounded to the nearest double. The energies
# (MeV) cover the series regime far below threshold, both sides of the series
# switch at tau = 4 (q = m), both sides of the pair threshold, and the 1600 MeV
# ceiling. mpmath is a regeneration-only dependency (pyproject.toml).
TRIANGLE_REFERENCE = {
    "electron": [
        (1e-06, (-3.1913816610769865e-13 + 0j)),
        (0.001, (-3.1913832906630654e-07 + 0j)),
        (0.505888956639, (-0.09439001265884912 + 0j)),
        (0.516108935561, (-0.09889330539284989 + 0j)),
        (0.97089799759, (-0.7402772463059175 + 0j)),
        (1.0, (-0.9402320184636139 + 0j)),
        (1.07309778681, (-1.1480485242180012 - 0.8973834936781886j)),
        (2.0439957844, (0.8167442505000742 - 1.0343363135165171j)),
        (1600.0, (1.0000254271070486 - 1.0317149948837103e-05j)),
    ],
    "muon": [
        (1e-06, (-7.464674999280613e-18 + 0j)),
        (0.001, (-7.464674999369768e-12 + 0j)),
        (1.0, (-7.464764154760664e-06 + 0j)),
        (104.601790755, (-0.09439001265884911 + 0j)),
        (106.714958245, (-0.09889330539284989 + 0j)),
        (200.75091155, (-0.7402772463059181 + 0j)),
        (221.88258645, (-1.1480485242180016 - 0.897383493678188j)),
        (422.633498, (0.8167442505000742 - 1.0343363135165171j)),
        (1600.0, (1.0853643359288592 - 0.1486801117617154j)),
    ],
    "tau": [
        (1e-06, (-2.6394432717293116e-20 + 0j)),
        (0.001, (-2.6394432717294234e-14 + 0j)),
        (1.0, (-2.63944338319589e-08 + 0j)),
        (1600.0, (-0.0759891165726088 + 0j)),
    ],
    "charm": [
        (1e-06, (-5.166677000020666e-20 + 0j)),
        (0.001, (-5.166677000021094e-14 + 0j)),
        (1.0, (-5.166677427133529e-08 + 0j)),
        (1257.3, (-0.0943900126588491 + 0j)),
        (1282.7, (-0.09889330539284988 + 0j)),
        (1600.0, (-0.17030156243779934 + 0j)),
    ],
}


@pytest.mark.parametrize("name", list(TRIANGLE_REFERENCE))
def test_triangle_keeps_its_relative_precision_across_the_support(name: str) -> None:
    """Each triangle against 40-digit mpmath from 1e-6 MeV to the 1600 MeV ceiling.

    Far below threshold the closed form 1 - tau arcsin² cancels like
    ulp*tau, which before the series left the charm triangle 29% wrong at
    1e-4 MeV and exactly zero at 1e-6 MeV. The worst error measured on a
    dense grid is 3.2e-15 relative (macOS/arm64), where the closed form
    takes over at tau = 4; 1e-14 allows libm differences.
    """
    couplings = {"electron": 0.0, "muon": 0.0, "tau": 0.0, "charm": 0.0}
    couplings[name] = 1.0
    weight = 4 / 3 if name == "charm" else 1.0
    reference = np.array(TRIANGLE_REFERENCE[name])
    q, expected = reference[:, 0].real, reference[:, 1]
    amplitude = GAMMA_GAMMA.amplitude(q, ZERO, TriangleCouplings(**couplings))
    np.testing.assert_allclose(amplitude / weight, expected, rtol=1e-14, atol=0)


def test_a_meson_the_source_does_not_couple_to_drops_out_at_its_pole() -> None:
    """An isoscalar source has c_pi = 0, so a stable pi0 pole contributes nothing.

    At the exact zero-width pion pole the amplitude and self-energy are
    finite and continuous with q(1 + 1e-9), where only O(1e-9) changes in
    the eta and eta' terms separate them; a coupled stable pole still raises.
    """
    isoscalar = _current(1e-3, 1e-3, 0.0)
    stable_poles = AxialMesonPoles(pion_width=0.0, eta_width=0.0, eta_prime_width=0.0)
    stable = AxialFormFactorGammaGamma(poles=stable_poles)
    q = np.array([MASSES[0], MASSES[0] * (1 + 1e-9)])
    amplitude = stable.amplitude(q, isoscalar, NO_TRIANGLES)
    assert amplitude[0] == pytest.approx(amplitude[1], rel=1e-8)
    self_energy = stable_poles.source_self_energy(q, isoscalar)
    assert self_energy[0] == pytest.approx(self_energy[1], rel=1e-8)
    assert (
        stable_poles.mixing_expansion(MASSES[0], isoscalar, mediator_mass=500.0)[0]
        == 0.0
    )
    with pytest.raises(ValueError, match="zero width"):
        stable.amplitude(MASSES[0], HYPERCHARGE, NO_TRIANGLES)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="declared support"):
        GAMMA_GAMMA.amplitude(1600.1, HYPERCHARGE, NO_TRIANGLES)
    with pytest.raises(ValueError, match="declared support"):
        GAMMA_GAMMA.spectral_functions(-1.0, HYPERCHARGE, NO_TRIANGLES)
    with pytest.raises(ValueError, match="real and finite"):
        POLES.source_self_energy(np.nan, HYPERCHARGE)
    with pytest.raises(TypeError, match="AxialCurrent"):
        POLES.source_couplings((1e-3, -1e-3, 0.0))
    with pytest.raises(TypeError, match="TriangleCouplings"):
        GAMMA_GAMMA.amplitude(300.0, HYPERCHARGE, (0.0, 0.0, 0.0, 0.0))
    with pytest.raises(TypeError, match="InvisibleCut"):
        POLES.invisible_widths(HYPERCHARGE, (60.0, 1.0, 2500.0))
    with pytest.raises(ValueError, match="1600 MeV"):
        AxialFormFactorGammaGamma(q_max=2000.0)
    with pytest.raises(ValueError, match="mediator_mass"):
        InvisibleCut(dm_mass=1.0, dm_axial=1.0, mediator_mass=0.0)
    with pytest.raises(ValueError, match="dm_mass"):
        InvisibleCut(dm_mass=-1.0, dm_axial=1.0, mediator_mass=1.0)
    with pytest.raises(ValueError, match="pion_width"):
        AxialMesonPoles(pion_width=-1.0)
    with pytest.raises(ValueError, match="mediator_mass"):
        POLES.mixing_expansion(300.0, HYPERCHARGE, mediator_mass=-1.0)
