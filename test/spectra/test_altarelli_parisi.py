r"""Endpoints of the Altarelli-Parisi FSR spectra.

A radiator of mass m recoiling against a partner of mass m_Y in a system of
invariant mass sqrt(s) can emit a photon up to x = 2E/sqrt(s) = 1 - (m + m_Y)²/s.
The AP bracket ln(s(1 - x)/m²) - 1 is negative past x = 1 - e m²/s, so the
spectrum ends at whichever of the two comes first. For an equal pair that is
the kinematic endpoint, where the bracket is still ln 4 - 1, so the spectrum
ends in a step.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from hazma import parameters, spectra
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import muon_mass as mmu
from hazma.parameters import neutral_pion_mass as mpi0

# Relative offset of the probe energies from the endpoint, as a fraction of
# 1 - x_end, so the probes straddle the endpoint by far more than rounding
# but sit close enough that the bracket differs from its endpoint value by
# about 1e-9.
STEP = 1e-9


def _fermion_splitting(x: float) -> float:
    return (1.0 + (1.0 - x) ** 2) / x


def _scalar_splitting(x: float) -> float:
    return 2.0 * (1.0 - x) / x


def _straddle(x_end: float, sqrts: float) -> np.ndarray:
    """Photon energies just below and just above `x_end`, in MeV."""
    dx = STEP * (1.0 - x_end)
    return np.array([x_end - dx, x_end + dx]) * sqrts / 2.0


CASES = [
    pytest.param(spectra.dnde_photon_ap_fermion, _fermion_splitting, mmu, 300.0),
    pytest.param(spectra.dnde_photon_ap_scalar, _scalar_splitting, mpi, 500.0),
]


@pytest.mark.parametrize(("dnde", "split", "mass", "sqrts"), CASES)
def test_an_equal_pair_ends_in_a_step_at_its_endpoint(
    dnde: Callable[..., np.ndarray],
    split: Callable[[float], float],
    mass: float,
    sqrts: float,
) -> None:
    s = sqrts**2
    x_end = 1.0 - 4.0 * mass**2 / s
    below, above = dnde(_straddle(x_end, sqrts), s, mass)

    # At the endpoint s(1 - x)/m² = 4. The probe sits 1e-9 of 1 - x_end
    # inside it, which moves the bracket by 1e-9 / (ln 4 - 1) relative.
    pre = parameters.alpha_em / (2.0 * np.pi) * 2.0 / sqrts
    step = pre * split(x_end) * (np.log(4.0) - 1.0)
    assert below == pytest.approx(step, rel=1e-8)
    assert above == 0.0


@pytest.mark.parametrize(("dnde", "split", "mass", "sqrts"), CASES)
def test_the_default_partner_is_the_radiator_itself(
    dnde: Callable[..., np.ndarray],
    split: Callable[[float], float],
    mass: float,
    sqrts: float,
) -> None:
    es = np.linspace(1.0, sqrts / 2.0, 200)
    s = sqrts**2
    np.testing.assert_array_equal(
        dnde(es, s, mass), dnde(es, s, mass, partner_mass=mass)
    )


@pytest.mark.parametrize(("dnde", "split", "mass", "sqrts"), CASES)
def test_a_heavier_partner_moves_the_endpoint_inward(
    dnde: Callable[..., np.ndarray],
    split: Callable[[float], float],
    mass: float,
    sqrts: float,
) -> None:
    partner = 2.0 * mass
    s = sqrts**2
    x_end = 1.0 - (mass + partner) ** 2 / s
    below, above = dnde(_straddle(x_end, sqrts), s, mass, partner_mass=partner)

    pre = parameters.alpha_em / (2.0 * np.pi) * 2.0 / sqrts
    step = pre * split(x_end) * (np.log(9.0) - 1.0)
    assert below == pytest.approx(step, rel=1e-8)
    assert above == 0.0


@pytest.mark.parametrize(("dnde", "split", "mass", "sqrts"), CASES)
def test_a_massless_partner_ends_where_the_bracket_vanishes(
    dnde: Callable[..., np.ndarray],
    split: Callable[[float], float],
    mass: float,
    sqrts: float,
) -> None:
    # The kinematic endpoint 1 - m²/s lies past 1 - e m²/s, where the
    # bracket reaches zero, so the spectrum falls continuously to zero.
    s = sqrts**2
    x_end = 1.0 - np.e * mass**2 / s
    below, above = dnde(_straddle(x_end, sqrts), s, mass, partner_mass=0.0)

    peak = dnde(np.array([0.1 * sqrts]), s, mass, partner_mass=0.0)[0]
    assert 0.0 < below < 1e-8 * peak
    assert above == 0.0


def _fsr(es: np.ndarray, cme: float, final_states: Sequence[str]) -> np.ndarray:
    with_fsr = spectra.dnde_photon(es, cme, final_states, include_fsr=True)
    without = spectra.dnde_photon(es, cme, final_states, include_fsr=False)
    return with_fsr - without


def test_two_body_fsr_ends_at_the_pair_endpoint() -> None:
    # In mu pi both legs recoil against the other, so both end at
    # x = 1 - (m_mu + m_pi)²/s. A radiator paired with itself would carry
    # the muon on to 1 - 4m_mu²/s, 18 MeV further out.
    cme = 400.0
    e_end = (cme**2 - (mmu + mpi) ** 2) / (2.0 * cme)
    below, above = _fsr(e_end * np.array([1.0 - 1e-6, 1.0 + 1e-6]), cme, ["mu", "pi"])
    assert below > 0.0
    assert above == 0.0


def test_three_body_fsr_pairs_each_radiator_with_its_partner() -> None:
    # In mu mu pi0 the muons radiate in the mu mu pair, whose invariant mass
    # reaches cme - m_pi0, and in the mu pi0 pairs, whose invariant mass
    # reaches cme - m_mu. The latter ends last, at 123.7 MeV. Pairing the
    # muon with itself there would end at 140.5 MeV instead.
    cme = 500.0
    s_mp = (cme - mmu) ** 2
    e_end = (s_mp - (mmu + mpi0) ** 2) / (2.0 * np.sqrt(s_mp))
    e_self = (s_mp - 4.0 * mmu**2) / (2.0 * np.sqrt(s_mp))
    es = np.linspace(e_end, e_self, 20)
    np.testing.assert_array_equal(_fsr(es, cme, ["mu", "mu", "pi0"]), 0.0)
    assert _fsr(np.array([0.9 * e_end]), cme, ["mu", "mu", "pi0"])[0] > 0.0
