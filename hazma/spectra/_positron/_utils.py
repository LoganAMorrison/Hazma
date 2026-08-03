from typing import Union, overload

import numpy as np
import numpy.typing as npt
import pathlib
from scipy import interpolate

from hazma.parameters import electron_mass as me

RealArray = npt.NDArray[np.float64]


def load_interp(fname):
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


def _dnde_positron_point(
    *,
    positron_energy: float,
    parent_energy: float,
    parent_mass,
    interp: interpolate.InterpolatedUnivariateSpline,
) -> float:
    eps = np.finfo(type(positron_energy)).eps

    if parent_energy < parent_mass or positron_energy < me:
        return 0.0

    if parent_energy - parent_mass < eps:
        return interp(positron_energy) * np.sqrt(positron_energy**2 - me**2)

    gamma = parent_energy / parent_mass
    beta = np.sqrt(1.0 - gamma**-2)

    k = np.sqrt(positron_energy**2 - me**2)
    lb = gamma * (positron_energy - beta * k)
    ub = gamma * (positron_energy + beta * k)

    return interp.integral(lb, ub) / (2 * beta * gamma)


def _dnde_positron_array(
    *,
    positron_energy: RealArray,
    parent_energy: float,
    parent_mass: float,
    interp: interpolate.InterpolatedUnivariateSpline,
) -> RealArray:
    eps = np.finfo(positron_energy.dtype).eps

    if parent_energy < parent_mass:
        return np.zeros_like(positron_energy)

    if parent_energy - parent_mass < eps:
        return interp(positron_energy) * np.sqrt(positron_energy**2 - me**2)

    return np.array(
        [
            _dnde_positron_point(
                positron_energy=e,
                parent_energy=parent_energy,
                parent_mass=parent_mass,
                interp=interp,
            )
            for e in positron_energy
        ]
    )


def dnde_positron(
    *,
    positron_energy: Union[RealArray, float],
    parent_energy: float,
    parent_mass: float,
    interp: interpolate.InterpolatedUnivariateSpline,
) -> Union[RealArray, float]:
    if parent_energy < parent_mass:
        return np.zeros_like(positron_energy)

    if isinstance(positron_energy, float):
        return _dnde_positron_point(
            positron_energy=positron_energy,
            parent_energy=parent_energy,
            parent_mass=parent_mass,
            interp=interp,
        )

    assert hasattr(positron_energy, "__len__"), (
        "Invalid type for positron_energy."
        + f"Expected float or numpy array, got: {type(positron_energy)}"
    )

    return _dnde_positron_array(
        positron_energy=positron_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )
