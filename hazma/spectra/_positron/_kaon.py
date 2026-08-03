from typing import Union, overload

import numpy as np
import numpy.typing as npt
import pathlib
from scipy import interpolate

from hazma.parameters import electron_mass as me
from hazma import parameters

RealArray = npt.NDArray[np.float64]


def _load_interp(fname):
    data = np.loadtxt(
        pathlib.Path(__file__).absolute().parent.joinpath("data", fname),
        delimiter=",",
    ).T

    energies = data[0]
    dnde = np.sum(data[1:], axis=0)
    dnde_integrand = dnde / np.sqrt(energies**2 - me**2)

    integrand_interp = interpolate.InterpolatedUnivariateSpline(
        energies, dnde_integrand, k=1
    )

    return integrand_interp


_charged_kaon_integrand_interp = _load_interp("charged_kaon_positron.csv")
_long_kaon_integrand_interp = _load_interp("long_kaon_positron.csv")
_short_kaon_integrand_interp = _load_interp("short_kaon_positron.csv")


def _point(
    ep: float, kaon_energy: float, interp: interpolate.InterpolatedUnivariateSpline, mk
) -> float:
    if kaon_energy < mk or ep < me:
        return 0.0

    if kaon_energy - mk < np.finfo(type(ep)).eps:
        return interp(ep) * np.sqrt(ep**2 - me**2)

    gamma = kaon_energy / mk
    beta = np.sqrt(1.0 - gamma**-2)

    k = np.sqrt(ep * ep - me * me)
    lb = gamma * (ep - beta * k)
    ub = gamma * (ep + beta * k)

    return interp.integral(lb, ub) / (2 * beta * gamma)


def _array(
    ep: RealArray,
    kaon_energy: float,
    interp: interpolate.InterpolatedUnivariateSpline,
    mk,
) -> RealArray:
    if kaon_energy < mk:
        return np.zeros_like(ep)

    if kaon_energy - mk < np.finfo(ep.dtype).eps:
        return interp(ep) * np.sqrt(ep**2 - me**2)

    return np.array([_point(e, kaon_energy, interp, mk) for e in ep])


@overload
def dnde_positron_charged_kaon(positron_energy: float, kaon_energy: float) -> float: ...


@overload
def dnde_positron_charged_kaon(
    positron_energy: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_positron_charged_kaon(
    positron_energy: Union[RealArray, float], kaon_energy: float
) -> Union[RealArray, float]:
    mk = parameters.charged_kaon_mass
    interp = _charged_kaon_integrand_interp

    if kaon_energy < mk:
        return np.zeros_like(positron_energy)

    if isinstance(positron_energy, float):
        return _point(positron_energy, kaon_energy, interp, mk)

    assert hasattr(positron_energy, "__len__"), (
        "Invalid type for positron_energy."
        + f"Expected float or numpy array, got: {type(positron_energy)}"
    )

    return _array(positron_energy, kaon_energy, interp, mk)


@overload
def dnde_positron_long_kaon(positron_energy: float, kaon_energy: float) -> float: ...


@overload
def dnde_positron_long_kaon(
    positron_energy: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_positron_long_kaon(
    positron_energy: Union[RealArray, float], kaon_energy: float
) -> Union[RealArray, float]:
    mk = parameters.neutral_kaon_mass
    interp = _long_kaon_integrand_interp

    if kaon_energy < mk:
        return np.zeros_like(positron_energy)

    if isinstance(positron_energy, float):
        return _point(positron_energy, kaon_energy, interp, mk)

    assert hasattr(positron_energy, "__len__"), (
        "Invalid type for positron_energy."
        + f"Expected float or numpy array, got: {type(positron_energy)}"
    )

    return _array(positron_energy, kaon_energy, interp, mk)


@overload
def dnde_positron_short_kaon(positron_energy: float, kaon_energy: float) -> float: ...


@overload
def dnde_positron_short_kaon(
    positron_energy: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_positron_short_kaon(
    positron_energy: Union[RealArray, float], kaon_energy: float
) -> Union[RealArray, float]:
    mk = parameters.neutral_kaon_mass
    interp = _short_kaon_integrand_interp

    if kaon_energy < mk:
        return np.zeros_like(positron_energy)

    if isinstance(positron_energy, float):
        return _point(positron_energy, kaon_energy, interp, mk)

    assert hasattr(positron_energy, "__len__"), (
        "Invalid type for positron_energy."
        + f"Expected float or numpy array, got: {type(positron_energy)}"
    )

    return _array(positron_energy, kaon_energy, interp, mk)
