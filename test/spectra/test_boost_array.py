"""`hazma.spectra.dnde_boost_array` with a separate rest-frame grid."""

from __future__ import annotations

import numpy as np

from hazma.spectra import boost_delta_function, dnde_boost_array


def _triangle(es: np.ndarray) -> np.ndarray:
    """A piecewise-linear rest-frame spectrum on [10, 50] MeV, peaking at 30."""
    return np.clip(1.0 - np.abs(es - 30.0) / 20.0, 0.0, None)


def test_rest_energies_equal_to_energies_change_nothing() -> None:
    es = np.linspace(0.5, 100.0, 400)
    dnde = _triangle(es)
    np.testing.assert_array_equal(
        dnde_boost_array(dnde, es, 0.6, mass=0.3, rest_energies=es),
        dnde_boost_array(dnde, es, 0.6, mass=0.3),
    )


def test_at_rest_the_spectrum_is_interpolated_onto_energies() -> None:
    rest_es = np.linspace(0.0, 60.0, 61)
    es = np.array([5.0, 20.5, 30.0, 45.25, 70.0])
    boosted = dnde_boost_array(_triangle(rest_es), es, 0.0, rest_energies=rest_es)
    np.testing.assert_allclose(boosted, _triangle(es), rtol=0.0, atol=1e-15)


def test_a_narrow_rest_frame_peak_boosts_like_a_line() -> None:
    """A narrow rest-frame peak boosts like a weighted delta function.

    Four rest-frame points describe a trapezoid of width `4 w` around `e0`,
    and its boost converges to `boost_delta_function` times its area as `w`
    shrinks. The lab energies sit well inside the boosted box, away from
    its edges, where the two agree to second order in `w / e0`.
    """
    e0, w, beta = 30.0, 1e-4, 0.8
    rest_es = np.array([e0 - 2 * w, e0 - w, e0 + w, e0 + 2 * w])
    rest_dnde = np.array([0.0, 1.0, 1.0, 0.0])
    area = 3.0 * w
    es = np.linspace(15.0, 80.0, 7)
    boosted = dnde_boost_array(rest_dnde, es, beta, rest_energies=rest_es)
    # (w / e0)^2 is 1e-11; measured 8.8e-12.
    np.testing.assert_allclose(
        boosted, area * boost_delta_function(es, e0, 0.0, beta), rtol=1e-9
    )


def test_integer_and_scalar_energies_boost_like_floats() -> None:
    """The boosted values do not inherit an integer dtype or need an array."""
    rest_es = np.linspace(0.0, 60.0, 61)
    dnde = _triangle(rest_es)
    expected = dnde_boost_array(
        dnde, np.array([20.0, 40.0]), 0.6, rest_energies=rest_es
    )

    assert np.all(expected > 0.0)
    np.testing.assert_array_equal(
        dnde_boost_array(dnde, np.array([20, 40]), 0.6, rest_energies=rest_es),
        expected,
    )
    assert dnde_boost_array(dnde, 20.0, 0.6, rest_energies=rest_es) == expected[0]
    np.testing.assert_array_equal(
        boost_delta_function(np.array([20, 40]), 30.0, 0.0, 0.6),
        boost_delta_function(np.array([20.0, 40.0]), 30.0, 0.0, 0.6),
    )
