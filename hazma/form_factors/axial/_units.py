"""Explicit natural-unit conversions by the energy power of a quantity."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from hazma.utils import RealOrRealArray

ArrayLike = RealOrRealArray | list[float] | tuple[float, ...]


def _convert(value: ArrayLike, energy_power: int, scale: float) -> RealOrRealArray:
    if isinstance(energy_power, (bool, np.bool_)) or not isinstance(
        energy_power, Integral
    ):
        raise ValueError("energy_power must be an integer.")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("value must contain finite real numbers.")
    values = np.asarray(raw, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("value must contain finite real numbers.")
    try:
        factor = scale ** int(energy_power)
    except OverflowError as exc:
        raise ValueError(
            "Unit conversion factor exceeds the finite float domain."
        ) from exc
    if not np.isfinite(factor) or factor == 0.0:
        raise ValueError("Unit conversion factor exceeds the finite float domain.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = values * factor
    if np.any(~np.isfinite(result)) or np.any((values != 0.0) & (result == 0.0)):
        raise ValueError("Converted value exceeds the nonzero finite float domain.")
    return float(result) if result.ndim == 0 else result


def gev_to_mev(value: ArrayLike, energy_power: int) -> RealOrRealArray:
    r"""Convert a natural-unit quantity from GeV**power to MeV**power.

    Parameters
    ----------
    value : float or array_like
        Finite real values expressed in GeV**energy_power.
    energy_power : int
        Explicit energy dimension. Examples: +1 for energy/width, -2 for
        cross section or sigma*v, -1 for dN/dE, and 0 for a count. An amplitude
        or spectral function requires its own declared energy dimension.

    Returns
    -------
    float or numpy.ndarray
        Values in MeV**energy_power, multiplied by 1000**energy_power;
        array shape is preserved and arrays have floating-point dtype.

    Raises
    ------
    ValueError
        For nonreal/nonfinite values, noninteger dimensions, or conversions
        exceeding the nonzero finite floating-point domain.

    Notes
    -----
    This conversion supplies no charge, spin, identical-particle, population,
    bin-width, or velocity factors.
    """
    return _convert(value, energy_power, 1000.0)


def mev_to_gev(value: ArrayLike, energy_power: int) -> RealOrRealArray:
    """Convert MeV**energy_power to GeV**energy_power.

    Parameters
    ----------
    value : float or array_like
        Finite real values expressed in MeV**energy_power.
    energy_power : int
        Explicit integer energy dimension, as in gev_to_mev.

    Returns
    -------
    float or numpy.ndarray
        Values in GeV**energy_power, multiplied by 0.001**energy_power;
        array shape is preserved and arrays have floating-point dtype.

    Raises
    ------
    ValueError
        For invalid inputs or conversions outside the nonzero finite float
        domain. No charge, spin, or rate-normalization factors are supplied.
    """
    return _convert(value, energy_power, 0.001)
