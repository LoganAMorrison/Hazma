"""Pin `RHNeutrino`'s `l rho` and `v rho` channels to the boosted rho spectra.

`N -> l rho` is a two-body decay, so the rho carries the fixed energy
`E_rho = (m_N^2 + m_rho^2 - m_l^2) / (2 m_N)`. With `l = e` the electron adds
no positrons or neutrinos of its own, so the channel's positron and neutrino
spectra are the charged rho's, boosted to `E_rho`, and its photon spectrum is
the charged rho's plus the electron's Altarelli-Parisi FSR at `s = m_N^2`.
The channel is not self-conjugate, so `RHNeutrino` doubles the photon and
neutrino spectra to count both `l- rho+` and `l+ rho-`; it counts positrons
once.

Each expectation is assembled from public `hazma.spectra` functions, so the
tests hold the channel to the charged rho rather than to whatever
`hazma.spectra.dnde_photon` returns for the `(e, rho)` final state.

`N -> v rho0` is the neutral counterpart. The neutrino is massless, so the
rho0 carries `E_rho = (m_N^2 + m_rho^2) / (2 m_N)`, and the channel's positron
and neutrino spectra are the neutral rho's, boosted to `E_rho`. The channel is
self-conjugate, so `RHNeutrino` applies no doubling to any product.
"""

import numpy as np
import pytest

from hazma import rh_neutrino, spectra
from hazma.parameters import standard_model_masses as sm_masses

ME = sm_masses["e"]
MRHO = sm_masses["rho"]

# Heavy-neutrino masses in MeV, taking the rho from a Lorentz factor of 1.03
# to one of 3.3.
MASSES = [1000.0, 2000.0, 5000.0]

# Both sides evaluate the same kernels at the same energies, so they agree to
# floating-point rounding; 1e-12 leaves room only for summation order.
RTOL = 1e-12


def _setup(mx: float) -> tuple[rh_neutrino.RHNeutrino, float, np.ndarray]:
    model = rh_neutrino.RHNeutrino(mx, 1e-3, "e")
    e_rho = (mx**2 + MRHO**2 - ME**2) / (2.0 * mx)
    # Spans the continuum and runs past the channel's endpoint, where both
    # sides must be zero.
    energies = np.geomspace(1e-3, 0.6, 50) * mx
    return model, e_rho, energies


@pytest.mark.parametrize("mx", MASSES)
def test_l_rho_photon_is_boosted_charged_rho_plus_electron_fsr(mx: float) -> None:
    model, e_rho, es = _setup(mx)
    expected = 2.0 * (
        spectra.dnde_photon_charged_rho(es, e_rho)
        + spectra.dnde_photon_ap_fermion(es, mx**2, mass=ME, charge=-1.0)
    )
    actual = model._spectrum_funcs()["e rho"](es)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("mx", MASSES)
def test_l_rho_positron_is_boosted_charged_rho(mx: float) -> None:
    model, e_rho, es = _setup(mx)
    expected = spectra.dnde_positron_charged_rho(es, e_rho)
    actual = model._positron_spectrum_funcs()["e rho"](es)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)
    assert np.any(actual > 0.0)


@pytest.mark.parametrize("mx", MASSES)
def test_l_rho_neutrino_is_boosted_charged_rho(mx: float) -> None:
    model, e_rho, es = _setup(mx)
    expected = 2.0 * spectra.dnde_neutrino_charged_rho(es, e_rho)
    actual = model._neutrino_spectrum_funcs()["e rho"](es)
    assert np.shape(actual) == (3, es.size)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)
    assert np.any(actual > 0.0)


def _setup_v_rho(mx: float) -> tuple[rh_neutrino.RHNeutrino, float, np.ndarray]:
    model = rh_neutrino.RHNeutrino(mx, 1e-3, "e")
    e_rho = (mx**2 + MRHO**2) / (2.0 * mx)
    energies = np.geomspace(1e-3, 0.6, 50) * mx
    return model, e_rho, energies


@pytest.mark.parametrize("mx", MASSES)
def test_v_rho_positron_is_boosted_neutral_rho(mx: float) -> None:
    model, e_rho, es = _setup_v_rho(mx)
    expected = spectra.dnde_positron_neutral_rho(es, e_rho)
    actual = model._positron_spectrum_funcs()["ve rho"](es)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)
    assert np.any(actual > 0.0)


@pytest.mark.parametrize("mx", MASSES)
def test_v_rho_neutrino_is_boosted_neutral_rho(mx: float) -> None:
    model, e_rho, es = _setup_v_rho(mx)
    expected = spectra.dnde_neutrino_neutral_rho(es, e_rho)
    actual = model._neutrino_spectrum_funcs()["ve rho"](es)
    assert np.shape(actual) == (3, es.size)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)
    assert np.any(actual > 0.0)
