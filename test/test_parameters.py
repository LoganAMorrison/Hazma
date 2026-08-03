"""Tests for the detector energy-resolution convolution."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from hazma.parameters import convolved_spectrum_fn, spec_res_fn


def scalar_only_energy_res(energy):
    """Resolution callback that only accepts scalars, per the documented API.

    Passing an array raises ``ValueError: truth value of an array is
    ambiguous``, so anything consuming this must evaluate it element-wise.
    """
    return 0.1 if energy < 10.0 else 0.2


def constant_energy_res(energy):
    """Resolution callback returning a single value regardless of input shape."""
    return 0.05


def vectorized_energy_res(energy):
    """Resolution callback that handles arrays natively."""
    return np.where(np.asarray(energy) < 10.0, 0.1, 0.2)


def gaussian_line_source(energies):
    """A narrow source spectrum centered at 5 MeV, normalized to one photon."""
    width = 0.5
    energies = np.asarray(energies, dtype=float)
    return np.exp(-((energies - 5.0) ** 2) / (2 * width**2)) / np.sqrt(
        2 * np.pi * width**2
    )


ENERGY_RES_CALLBACKS = [
    scalar_only_energy_res,
    constant_energy_res,
    vectorized_energy_res,
]


@pytest.mark.parametrize("energy_res", ENERGY_RES_CALLBACKS)
def test_convolution_accepts_scalar_and_vector_callbacks(energy_res):
    """`energy_res` is documented as `float -> float` and must stay supported."""
    dnde = convolved_spectrum_fn(
        1e-1, 1e2, energy_res, spec_fn=gaussian_line_source, n_pts=200
    )

    energies = np.geomspace(1e-1, 1e2, 300)
    assert np.all(np.isfinite(dnde(energies)))


@pytest.mark.parametrize("energy_res", ENERGY_RES_CALLBACKS)
def test_line_only_convolution_accepts_scalar_callbacks(energy_res):
    """A line-only convolution must not require a vectorized callback."""
    dnde = convolved_spectrum_fn(
        1e-1, 1e2, energy_res, lines={"g g": {"bf": 1.0, "energy": 5.0}}, n_pts=200
    )

    energies = np.geomspace(1e-1, 1e2, 300)
    assert np.all(np.isfinite(dnde(energies)))


def test_scalar_callback_matches_vectorized_equivalent():
    """Element-wise fallback must give the same answer as a native array call."""
    energies = np.geomspace(1e-1, 1e2, 300)

    scalar = convolved_spectrum_fn(
        1e-1, 1e2, scalar_only_energy_res, spec_fn=gaussian_line_source, n_pts=200
    )
    vector = convolved_spectrum_fn(
        1e-1, 1e2, vectorized_energy_res, spec_fn=gaussian_line_source, n_pts=200
    )

    assert np.allclose(scalar(energies), vector(energies), rtol=1e-12, atol=0.0)


def test_response_width_is_set_by_the_true_energy():
    """The response must not smear photons past where the source spectrum ends.

    The width of the response is `sigma = E * (Delta E / E)` evaluated at the
    *true* energy. Using the reconstructed energy instead lets a sharp feature
    leak to arbitrarily high energies wherever the resolution is poor: a
    Gaussian centered at 100 MeV with a 20% resolution has `sigma = 20` MeV, and
    so reaches back down to a source that cuts off at 6 MeV.
    """
    dnde = convolved_spectrum_fn(
        1e-1, 1e3, scalar_only_energy_res, spec_fn=gaussian_line_source, n_pts=200
    )

    # The source is dead by 8 MeV (6 sigma), and the fine resolution there
    # cannot push photons out to 50 MeV and beyond.
    assert np.all(dnde(np.geomspace(50.0, 1e3, 50)) < 1e-20)


def test_convolution_conserves_photon_number():
    """Smearing redistributes photons in energy; it must not create or lose them."""
    energies = np.geomspace(1e-1, 1e2, 2000)

    dnde = convolved_spectrum_fn(
        1e-1, 1e2, vectorized_energy_res, spec_fn=gaussian_line_source, n_pts=500
    )

    assert trapezoid(dnde(energies), energies) == pytest.approx(1.0, rel=1e-3)


def test_spec_res_fn_arguments_are_not_interchangeable():
    """The width comes from the true energy, so the response is asymmetric."""
    # 4 MeV true -> 10% width; 20 MeV true -> 20% width. Swapping the arguments
    # changes which width is used and therefore the value.
    forward = spec_res_fn(20.0, 4.0, scalar_only_energy_res)
    backward = spec_res_fn(4.0, 20.0, scalar_only_energy_res)

    assert not np.isclose(forward, backward)


def test_spec_res_fn_accepts_an_array_of_true_energies():
    """The convolution integral evaluates the response over a grid of true energies."""
    true_energies = np.array([4.0, 5.0, 20.0])

    response = spec_res_fn(5.0, true_energies, scalar_only_energy_res)

    assert response.shape == true_energies.shape
    assert np.all(np.isfinite(response))
    # The response peaks where the true energy matches the reconstructed one.
    assert np.argmax(response) == 1


def test_zero_width_response_carries_no_photons():
    """A vanishing resolution yields a zero response, matching historical behavior."""
    assert spec_res_fn(5.0, 5.0, lambda energy: 0.0) == 0.0
