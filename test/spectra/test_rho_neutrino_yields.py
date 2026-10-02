"""Pin each rho's neutrino yield to the charged pion's.

The rho tables hold only the two-pion modes: `rho+- -> pi+- pi0` carries one
charged pion and `rho0 -> pi+ pi-` carries two, while the `pi0` decays to
photons alone. Integrated over energy, the charged rho must therefore give
one charged pion's neutrinos of each flavor and the neutral rho twice that. A
particle count is Lorentz invariant, so the parent energies need not match.

The `rho0` entry of the N-body dispatch must route to the neutral rho's
positron and neutrino spectra rather than to zero.
"""

from collections.abc import Callable

import numpy as np
import pytest

from hazma import spectra
from hazma.parameters import standard_model_masses as sm_masses

MRHO = sm_masses["rho"]
MPI = sm_masses["pi"]
MPI0 = sm_masses["pi0"]

# The rho tables reproduce the pion's yield to 9.55e-4 (a shortfall) for the
# muon flavor and 1.0e-4 (an excess) for the electron flavor; the residual is
# the tables' interpolation error, so 2e-3 holds with margin and still fails a
# factor of two.
YIELD_RTOL = 2e-3


def _neutrino_count(
    dnde: Callable[[np.ndarray, float], np.ndarray], parent_energy: float
) -> np.ndarray:
    """Return the (e, mu, tau) neutrino counts of `dnde` at `parent_energy`.

    The spectrum vanishes below 1e-3 MeV to well under the tolerance and
    above the parent energy by kinematics, so the trapezoid runs over that
    range on a log grid fine enough to resolve the muon-decay edge.
    """
    energies = np.geomspace(1e-3, parent_energy, 200_001)
    return np.trapezoid(dnde(energies, parent_energy), energies, axis=-1)


@pytest.mark.parametrize("gamma", [1.5, 5.0])
def test_charged_rho_neutrino_yield_is_one_charged_pion(gamma: float) -> None:
    pion = _neutrino_count(spectra.dnde_neutrino_charged_pion, 1.5 * MPI)
    rho = _neutrino_count(spectra.dnde_neutrino_charged_rho, gamma * MRHO)
    np.testing.assert_allclose(rho, pion, rtol=YIELD_RTOL, atol=1e-12)


@pytest.mark.parametrize("gamma", [1.5, 5.0])
def test_neutral_rho_neutrino_yield_is_two_charged_pions(gamma: float) -> None:
    pion = _neutrino_count(spectra.dnde_neutrino_charged_pion, 1.5 * MPI)
    rho = _neutrino_count(spectra.dnde_neutrino_neutral_rho, gamma * MRHO)
    np.testing.assert_allclose(rho, 2.0 * pion, rtol=YIELD_RTOL, atol=1e-12)


def test_nbody_rho0_carries_the_neutral_rho_spectra() -> None:
    # `rho0 pi0` at 1.5 times threshold. The `pi0` adds no positrons or
    # neutrinos, so each N-body spectrum is the neutral rho's at its two-body
    # energy, and both functions evaluate the same tables.
    cme = 1.5 * (MRHO + MPI0)
    e_rho = (cme**2 + MRHO**2 - MPI0**2) / (2.0 * cme)
    energies = np.geomspace(1.0, cme, 25)

    np.testing.assert_allclose(
        spectra.dnde_positron(energies, cme, ("rho0", "pi0")),
        spectra.dnde_positron_neutral_rho(energies, e_rho),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        spectra.dnde_neutrino(energies, cme, ("rho0", "pi0")),
        spectra.dnde_neutrino_neutral_rho(energies, e_rho),
        rtol=1e-12,
        atol=0.0,
    )
