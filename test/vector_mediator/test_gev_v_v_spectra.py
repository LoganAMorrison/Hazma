"""Photon, positron and neutrino spectra of ``VectorMediatorGeV``'s ``v v`` channel.

Each mediator carries ``e_cm / 2`` and decays isotropically in its own
rest frame. A boost moves particles in energy without creating or
destroying any, and an isotropic source's mean lab energy is the Lorentz factor times
its mean rest-frame energy. The tests pin both statements, which owe
nothing to the boost integrator.

Like every channel of ``VectorMediatorGeV``, these spectra count particles
only: a muon pair yields one positron, one electron neutrino and one muon
neutrino.

Every spectrum is integrated on a log grid that also carries the two edges
of the boosted ``V → f f̄`` line, where the spectrum is discontinuous; with
the edges off the grid the trapezoid misses by up to 2e-3.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import pytest
from scipy.integrate import trapezoid

from hazma.parameters import charged_pion_mass
from hazma.parameters import electron_mass as me
from hazma.parameters import eta_mass, neutral_pion_mass
from hazma.vector_mediator import VectorMediatorGeV
from hazma.vector_mediator._gev import neutrino, positron, spectra

MX = 5e3  # MeV
MV = 1e3  # MeV

# The follow-up's measurement point, gamma = 5.05, and a stronger boost.
CMES = [10.1e3, 40e3]  # MeV

# Masses of the particles each spectrum counts, in MeV.
MASSES = {"positron": me, "e": 0.0, "mu": 0.0, "tau": 0.0}

# The rest-frame spectrum is tabulated on `V_V_REST_FRAME_POINTS` energies.
# Its linear interpolant misses the exact particle number by at most 1.8e-5
# (muon neutrinos; 3.2e-6 for positrons), and the hadronic channel yields by
# 4.4e-6. The boost of the interpolant is exact, so the residual is the same
# at both boosts. The budget is a few times the largest.
NUMBER_RTOL = 1e-4

# The tabulation error cancels in the ratio of mean energies at two boosts,
# measured to 7.7e-9.
ENERGY_RTOL = 1e-7

# Particle number and total energy, keyed by spectrum and center-of-mass energy.
Moments = dict[tuple[str, float], tuple[float, float]]

# The options `dnde_positron_v_v` passes to the n-body channels that accept
# them, by default.
N_BODY_OPTIONS = {"npts": 1 << 15, "nbins": 30}


def _leptophilic() -> VectorMediatorGeV:
    """A mediator whose decays are all leptonic, so counts are exact."""
    return VectorMediatorGeV(
        mx=MX,
        mv=MV,
        gvxx=1.0,
        gvuu=0.0,
        gvdd=0.0,
        gvss=0.0,
        gvee=1.0,
        gvmumu=1.0,
        gvveve=1.0,
        gvvmvm=1.0,
        gvvtvt=1.0,
    )


def _universal() -> VectorMediatorGeV:
    """The follow-up's model: every coupling 1, hadronic decays open."""
    return VectorMediatorGeV(MX, MV, *([1.0] * 9))


def _spectrum_fn(model: VectorMediatorGeV, kind: str) -> Callable[..., np.ndarray]:
    if kind == "positron":
        return model.positron_spectrum_funcs()["v v"]
    return model.neutrino_spectrum_funcs(kind)["v v"]


def _lab_grid(cme: float, mass: float) -> np.ndarray:
    """Lab energies up to the kinematic endpoint, with the line's edges."""
    gamma = 0.5 * cme / MV
    beta = np.sqrt(1.0 - gamma**-2)
    e0 = 0.5 * MV
    p0 = np.sqrt(e0**2 - mass**2)
    edges = [gamma * (e0 - beta * p0), gamma * (e0 + beta * p0)]
    lo = me * (1.0 + 1e-9) if mass > 0.0 else 1e-6
    es = np.geomspace(lo, gamma * MV, 20_001)
    around = [x * (1.0 + s) for x in edges for s in (-1e-12, 1e-12)]
    return np.unique(np.concatenate([es, around]))


def _moments(model: VectorMediatorGeV, kind: str, cme: float) -> tuple[float, float]:
    """Particle number and total energy (MeV) per annihilation."""
    es = _lab_grid(cme, MASSES[kind])
    dnde = _spectrum_fn(model, kind)(es, cme)
    return trapezoid(dnde, es), trapezoid(es * dnde, es)


@pytest.fixture(scope="module")
def leptophilic() -> VectorMediatorGeV:
    return _leptophilic()


@pytest.fixture(scope="module")
def leptophilic_moments(
    leptophilic: VectorMediatorGeV,
) -> Moments:
    return {
        (kind, cme): _moments(leptophilic, kind, cme) for kind in MASSES for cme in CMES
    }


def _leptophilic_count(model: VectorMediatorGeV, kind: str) -> float:
    """Particles per annihilation: two mediators, each decaying at rest."""
    pws = model.partial_widths()
    width = sum(pws.values())
    bf = {key: val / width for key, val in pws.items()}
    per_decay = {
        "positron": bf["e e"] + bf["mu mu"],
        "e": bf["ve ve"] + bf["mu mu"],
        "mu": bf["vm vm"] + bf["mu mu"],
        "tau": bf["vt vt"],
    }
    return 2 * per_decay[kind]


@pytest.mark.parametrize("cme", CMES)
@pytest.mark.parametrize("kind", list(MASSES))
def test_v_v_carries_two_mediators_worth_of_particles(
    leptophilic: VectorMediatorGeV,
    leptophilic_moments: Moments,
    kind: str,
    cme: float,
) -> None:
    """The lab spectrum integrates to twice one mediator's decay yield."""
    number, _ = leptophilic_moments[(kind, cme)]
    assert number == pytest.approx(
        _leptophilic_count(leptophilic, kind), rel=NUMBER_RTOL
    )


@pytest.mark.parametrize("kind", list(MASSES))
def test_v_v_mean_energy_scales_with_gamma(
    leptophilic_moments: Moments, kind: str
) -> None:
    """The mean lab energy per particle is proportional to the boost."""
    lo, hi = CMES
    n_lo, e_lo = leptophilic_moments[(kind, lo)]
    n_hi, e_hi = leptophilic_moments[(kind, hi)]
    assert (e_hi / n_hi) / (e_lo / n_lo) == pytest.approx(hi / lo, rel=ENERGY_RTOL)


def test_hadronic_v_v_positrons_match_the_channel_yields() -> None:
    """With hadronic decays open, the yield is the branching-weighted sum.

    The reference reads each channel through the model's dispatch table at
    a center-of-mass energy equal to the mediator mass, with the n-body
    options the `v v` sum uses, so a channel the sum drops or miswires shows
    up as a missing yield. Both sides read one model instance, because the
    n-body partial widths are phase-space integrals that need not agree
    between instances.
    """
    model = _universal()
    pws = model.partial_widths()
    width = sum(pws.values())
    rest_es = np.geomspace(me * (1.0 + 1e-9), 0.5 * MV, 20_001)
    channel_fns = model._positron_spectrum_funcs()
    per_decay = pws["e e"] / width  # the e+ e- line
    for channel, pw in pws.items():
        if channel in ("e e", "v v") or pw == 0.0:
            continue
        fn = channel_fns[channel]
        params = inspect.signature(fn).parameters
        options = {k: v for k, v in N_BODY_OPTIONS.items() if k in params}
        dnde = fn(rest_es, MV, **options)
        per_decay += pw / width * trapezoid(dnde, rest_es)

    number, _ = _moments(model, "positron", CMES[0])
    assert number == pytest.approx(2 * per_decay, rel=NUMBER_RTOL)


@pytest.mark.parametrize("kind", list(MASSES))
def test_v_v_vanishes_below_threshold(
    leptophilic: VectorMediatorGeV, kind: str
) -> None:
    """Below `e_cm = 2 m_V` the channel is closed."""
    es = np.geomspace(1.0, 1e3, 10)
    cme = 1.9 * MV
    if kind == "positron":
        dnde = positron.dnde_positron_v_v(leptophilic, es, cme)
    else:
        dnde = neutrino.dnde_neutrino_v_v(leptophilic, es, cme, kind)
    np.testing.assert_array_equal(dnde, 0.0)


# PDG's `BR(pi -> mu nu)`, as `hazma._core` carries it in
# `rust/src/constants.rs`. The rest goes to `pi -> e nu`.
BR_PI_TO_MU_NUMU = 0.9998770

# A charged pion yields one electron or positron and one electron-flavored
# neutrino whichever way it decays, and two muon-flavored neutrinos
# through `pi -> mu nu`.
PER_PION = {
    "positron": 1.0,
    "e": 1.0,
    "mu": 2.0 * BR_PI_TO_MU_NUMU,
    "tau": 0.0,
}


@pytest.mark.parametrize("kind", list(MASSES))
def test_v_v_keeps_the_pion_lines_at_the_pion_pair_threshold(kind: str) -> None:
    """Pions from `V -> pi pi` just above threshold keep their prompt lines.

    At `m_V = 2 m_pi (1 + 1e-6)` each pion's lines boost into boxes 0.3%
    wide, narrower than a step of the rest-frame grid's log spacing. With
    only quark couplings `V -> pi pi` and `V -> pi0 gamma` are open, and
    the second yields neither leptons nor neutrinos, so each count is
    exact. The rest-frame grid's residual is 2.5e-5, inside `NUMBER_RTOL`;
    without the box edges on that grid the muon-flavored count lost half
    its value.
    """
    mv = 2.0 * charged_pion_mass * (1.0 + 1e-6)
    model = VectorMediatorGeV(MX, mv, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pws = model.partial_widths()
    bf_pi_pi = pws["pi pi"] / sum(pws.values())

    cme = CMES[0]
    lo = me * (1.0 + 1e-9) if kind == "positron" else 1e-6
    es = np.geomspace(lo, 0.5 * cme, 20_001)
    number = trapezoid(_spectrum_fn(model, kind)(es, cme), es)

    # Two mediators. Half a pion pair's leptons are particles, one pion's worth.
    expected = 2.0 * bf_pi_pi * PER_PION[kind]
    assert number == pytest.approx(expected, rel=NUMBER_RTOL, abs=1e-300)


@pytest.mark.parametrize("kind", list(MASSES))
def test_v_v_accepts_integer_and_scalar_energies(kind: str) -> None:
    """Integer and scalar energies give the floating-point array's values."""
    model = _leptophilic()
    fn = _spectrum_fn(model, kind)
    cme = CMES[0]
    expected = fn(np.array([100.0, 200.0]), cme)

    assert np.all(expected > 0.0)
    np.testing.assert_array_equal(fn(np.array([100, 200]), cme), expected)
    scalar = fn(100.0, cme)
    assert isinstance(scalar, float)
    assert scalar == expected[0]


def test_v_v_positrons_vanish_below_the_electron_pair_threshold() -> None:
    """A mediator lighter than `2 m_e` decays only into neutrinos."""
    model = VectorMediatorGeV(MX, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0)
    es = np.array([1.0, 2.0])
    cme = 10.1e3
    np.testing.assert_array_equal(positron.dnde_positron_v_v(model, es, cme), 0.0)
    assert np.all(neutrino.dnde_neutrino_v_v(model, es, cme, "e") > 0.0)


@pytest.mark.parametrize("kind", list(MASSES))
def test_v_v_vanishes_for_a_stable_mediator(kind: str) -> None:
    """With every Standard Model coupling zero the mediator never decays."""
    model = VectorMediatorGeV(MX, MV, 1.0, *([0.0] * 8))
    np.testing.assert_array_equal(
        _spectrum_fn(model, kind)(np.array([1.0, 2.0]), CMES[0]), 0.0
    )


def _photon_moments(
    model: VectorMediatorGeV, cme: float, line_masses: list[float]
) -> tuple[float, float]:
    """Photon number and total energy (MeV) per annihilation.

    The grid carries both edges of each boosted `V -> M gamma` line, where
    the spectrum is discontinuous.
    """
    gamma = 0.5 * cme / model.mv
    beta = np.sqrt(1.0 - gamma**-2)
    edges = []
    for m in line_masses:
        e0 = 0.5 * (model.mv - m**2 / model.mv)
        edges += [gamma * e0 * (1.0 - beta), gamma * e0 * (1.0 + beta)]
    around = [x * (1.0 + s) for x in edges for s in (-1e-12, 1e-12)]
    es = np.unique(np.concatenate([np.geomspace(1e-6, 0.5 * cme, 20_001), around]))
    dnde = spectra.dnde_photon_v_v(model, es, cme)
    return trapezoid(dnde, es), trapezoid(es * dnde, es)


# PDG's `BR(pi0 -> gamma gamma)`, as `hazma._core` carries it in
# `rust/src/constants.rs`. The neutral pion's photon spectrum counts only
# this mode.
BR_PI0_TO_A_A = 0.98823

# `dnde_photon_v_v` tabulates its rest-frame continuum on the requested
# energies, so the edges of each pion and eta decay box fall between grid
# points. On these grids that costs at most 4.2e-4 of the photon energy
# (2.0e-4 of the number). Placing the `pi0 gamma` line at the charged pion
# mass lowers the `pi0 gamma` energy by 1.5%, and lines left unweighted by
# their branching fractions raise the hadronic energy ninefold. The budget
# is a few times the largest residual.
PHOTON_RTOL = 1e-3


def test_v_v_photons_from_pi0_gamma_match_the_decay_yield() -> None:
    """A mediator that only decays to `pi0 gamma` yields its photon and the pion's.

    At `m_V = 200` MeV with only quark couplings, `V -> pi0 gamma` is the one
    open decay. Each decay gives the monochromatic photon at
    `(m_V - m_pi0^2 / m_V) / 2` and the pion's two photons, which carry the
    pion's energy, so the photon number and energy pin both the line's
    weight and its position.
    """
    mv = 200.0
    model = VectorMediatorGeV(MX, mv, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pws = model.partial_widths()
    assert pws["pi0 gamma"] == sum(pws.values())

    cme = CMES[0]
    number, energy = _photon_moments(model, cme, [neutral_pion_mass])
    e_line = 0.5 * (mv - neutral_pion_mass**2 / mv)
    per_decay_energy = e_line + BR_PI0_TO_A_A * (mv - e_line)
    # Two mediators, each carrying a Lorentz factor of `cme / (2 m_V)`.
    assert number == pytest.approx(2 * (1 + 2 * BR_PI0_TO_A_A), rel=PHOTON_RTOL)
    assert energy == pytest.approx(cme / mv * per_decay_energy, rel=PHOTON_RTOL)


def test_hadronic_v_v_photon_energy_matches_the_channel_yields() -> None:
    """With hadronic decays open, the energy is the branching-weighted sum.

    Photon number diverges with the final-state radiation's infrared tail,
    but the energy it carries does not, and an isotropic source's lab
    energy is the Lorentz factor times its rest-frame energy. The reference
    integrates each channel's continuum at rest through the model's dispatch
    table and adds one photon per `V -> pi0 gamma` and `V -> eta gamma`
    decay at its line energy. The reference integrates the same per-channel
    functions the implementation sums, so it pins the boost, the `2 BR`
    weighting and the line placement but not the channel continua
    themselves.
    """
    model = _universal()
    pws = model.partial_widths()
    width = sum(pws.values())
    lines = {"pi0 gamma": neutral_pion_mass, "eta gamma": eta_mass}
    rest_es = np.geomspace(1e-6, 0.5 * MV, 20_001)
    channel_fns = model._spectrum_funcs()
    per_decay = 0.0
    for channel, pw in pws.items():
        if pw == 0.0:
            continue
        fn = channel_fns[channel]
        params = inspect.signature(fn).parameters
        options = {k: v for k, v in N_BODY_OPTIONS.items() if k in params}
        dnde = fn(rest_es, MV, **options)
        per_decay += pw / width * trapezoid(rest_es * dnde, rest_es)
        if channel in lines:
            per_decay += pw / width * 0.5 * (MV - lines[channel] ** 2 / MV)

    cme = CMES[0]
    _, energy = _photon_moments(model, cme, list(lines.values()))
    assert energy == pytest.approx(0.5 * cme / MV * 2 * per_decay, rel=PHOTON_RTOL)


def test_v_v_photons_vanish_for_a_stable_mediator() -> None:
    """With every Standard Model coupling zero the mediator never decays."""
    model = VectorMediatorGeV(MX, MV, 1.0, *([0.0] * 8))
    np.testing.assert_array_equal(
        spectra.dnde_photon_v_v(model, np.array([1.0, 2.0]), CMES[0]), 0.0
    )
