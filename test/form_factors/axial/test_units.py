"""Dimension-aware conversion independent of rate or charge conventions."""

from collections.abc import Callable

import numpy as np
import pytest

from hazma.form_factors.axial._units import gev_to_mev, mev_to_gev


@pytest.mark.parametrize(
    ("power", "expected"),
    [(1, 2000), (-1, 0.002), (-2, 0.000002), (0, 2), (2, 2000000)],
)
def test_pinned_energy_rate_density_and_amplitude_dimensions(
    power: int, expected: float
) -> None:
    # Powers of the exact GeV/MeV ratio need only float multiplication rounding.
    assert gev_to_mev(2, power) == pytest.approx(expected, rel=2e-15, abs=0)
    assert mev_to_gev(expected, power) == pytest.approx(2, rel=2e-15, abs=0)
    assert isinstance(gev_to_mev(2, power), float)


def test_unit_measure_conversion_preserves_spectrum_integral() -> None:
    energies_gev = np.array([0.1, 0.2, 0.4])
    density_gev = np.array([2.0, 3.0, 1.0])
    energies_mev = gev_to_mev(energies_gev, 1)
    density_mev = gev_to_mev(density_gev, -1)
    # Independent trapezoidal measure, without any particle-count factor.
    original = np.sum(np.diff(energies_gev) * (density_gev[1:] + density_gev[:-1]) / 2)
    converted = np.sum(np.diff(energies_mev) * (density_mev[1:] + density_mev[:-1]) / 2)
    assert original == pytest.approx(0.65, rel=2e-15, abs=0)
    assert converted == pytest.approx(original, rel=2e-15, abs=0)


def test_amplitude_phase_space_and_cross_section_dimensions() -> None:
    # For a dimensionless three-body invariant amplitude, dPhi3 has E^2,
    # width=dPhi3/(2M) has E, and a 1/s^2 rate prefactor yields E^-2.
    amplitude_squared = gev_to_mev(3.0, 0)
    phase_space = gev_to_mev(0.2, 2)
    mass = gev_to_mev(0.5, 1)
    rate_prefactor = gev_to_mev(4.0, -4)
    assert amplitude_squared * phase_space / (2 * mass) == pytest.approx(
        gev_to_mev(0.6, 1), rel=2e-15, abs=0
    )
    assert amplitude_squared * phase_space * rate_prefactor == pytest.approx(
        gev_to_mev(2.4, -2), rel=2e-15, abs=0
    )


@pytest.mark.parametrize("power", [-2, -1, 0, 1, 2, 4])
def test_array_roundtrip_preserves_shape_and_promotes_integer_dtype(power: int) -> None:
    values = np.array([[0, -1, 2], [3, 4, 5]], dtype=int)
    converted = gev_to_mev(values, power)
    assert converted.shape == values.shape
    assert converted.dtype == np.float64
    np.testing.assert_allclose(mev_to_gev(converted, power), values, rtol=2e-15, atol=0)
    assert isinstance(gev_to_mev(np.array(2), power), float)


@pytest.mark.parametrize("convert", [gev_to_mev, mev_to_gev])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1j, [1, np.nan], "1", True])
def test_nonfinite_and_nonreal_values(convert: Callable, bad: object) -> None:
    with pytest.raises(ValueError):
        convert(bad, -1)


@pytest.mark.parametrize("bad", [True, 1.0, 0.5, np.nan, np.inf, 1j, "1", [1]])
def test_energy_power_must_be_explicit_integer(bad: object) -> None:
    with pytest.raises(ValueError):
        gev_to_mev(2, bad)


@pytest.mark.parametrize("convert", [gev_to_mev, mev_to_gev])
def test_overflow_and_underflow_are_controlled_domain_errors(convert: Callable) -> None:
    for power in (1000, -1000):
        with pytest.raises(ValueError):
            convert(1, power)
    large_power = 1 if convert is gev_to_mev else -1
    with pytest.raises(ValueError):
        convert(np.finfo(float).max, large_power)
    with pytest.raises(ValueError):
        convert(np.nextafter(0.0, 1.0), -large_power)


def test_signed_zero_and_dimensionless_counts_are_preserved() -> None:
    result = gev_to_mev(np.array([-0.0, 0.0, -2.0, 4.0]), 0)
    np.testing.assert_array_equal(result, [-0.0, 0.0, -2.0, 4.0])
    assert np.signbit(result[0])
