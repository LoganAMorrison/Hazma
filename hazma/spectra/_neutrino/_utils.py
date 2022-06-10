from typing import Union, Optional

import numpy as np
import numpy.typing as npt
import pathlib
from scipy import interpolate

RealArray = npt.NDArray[np.float_]


def load_interp(fname, k=2):
    data = np.loadtxt(
        pathlib.Path(__file__).absolute().parent.joinpath("data", fname),
        delimiter=",",
    ).T

    energies = data[0]
    dnde = np.sum(data[1:], axis=0)
    dnde_integrand = dnde / energies

    integrand_interp = interpolate.InterpolatedUnivariateSpline(
        energies, dnde_integrand, k=k
    )

    return integrand_interp


def _dnde_neutrino_point(
    *,
    neutrino_energy: float,
    parent_energy: float,
    parent_mass,
    interp: interpolate.InterpolatedUnivariateSpline,
) -> float:
    eps = np.finfo(type(neutrino_energy)).eps

    if parent_energy < parent_mass:
        return 0.0

    if parent_energy - parent_mass < eps:
        return interp(neutrino_energy) * neutrino_energy  # type: ignore

    gamma = parent_energy / parent_mass
    beta = np.sqrt(1.0 - gamma**-2)

    k = neutrino_energy
    lb = gamma * (neutrino_energy - beta * k)
    ub = gamma * (neutrino_energy + beta * k)

    return interp.integral(lb, ub) / (2 * beta * gamma)


def _dnde_neutrino_array(
    *,
    neutrino_energy: RealArray,
    parent_energy: float,
    parent_mass: float,
    interp: interpolate.InterpolatedUnivariateSpline,
) -> RealArray:
    eps = np.finfo(neutrino_energy.dtype).eps

    if parent_energy < parent_mass:
        return np.zeros_like(neutrino_energy)

    if parent_energy - parent_mass < eps:
        return interp(neutrino_energy) * neutrino_energy

    return np.array(
        [
            _dnde_neutrino_point(
                neutrino_energy=e,
                parent_energy=parent_energy,
                parent_mass=parent_mass,
                interp=interp,
            )
            for e in neutrino_energy
        ]
    )


def _dnde_neutrino(
    *,
    neutrino_energy: Union[RealArray, float],
    parent_energy: float,
    parent_mass: float,
    interp: interpolate.InterpolatedUnivariateSpline,
) -> Union[RealArray, float]:
    if parent_energy < parent_mass:
        if isinstance(neutrino_energy, float):
            return 0.0
        return np.zeros_like(neutrino_energy)

    if isinstance(neutrino_energy, float):
        return _dnde_neutrino_point(
            neutrino_energy=neutrino_energy,
            parent_energy=parent_energy,
            parent_mass=parent_mass,
            interp=interp,
        )

    assert hasattr(neutrino_energy, "__len__"), (
        "Invalid type for neutrino_energy."
        + f"Expected float or numpy array, got: {type(neutrino_energy)}"
    )

    return _dnde_neutrino_array(
        neutrino_energy=neutrino_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp=interp,
    )


def dnde_neutrino(
    *,
    neutrino_energy: Union[RealArray, float],
    parent_energy: float,
    parent_mass: float,
    interp_e: interpolate.InterpolatedUnivariateSpline,
    interp_mu: interpolate.InterpolatedUnivariateSpline,
    flavor: Optional[str] = None,
) -> Union[RealArray, float]:

    if isinstance(neutrino_energy, float):
        shape = tuple()
    else:
        shape = neutrino_energy.shape

    dnde = np.zeros((3,) + shape, dtype=np.float64)

    if flavor is None or flavor == "e":
        dnde_e = _dnde_neutrino(
            neutrino_energy=neutrino_energy,
            parent_energy=parent_energy,
            parent_mass=parent_mass,
            interp=interp_e,
        )
        if flavor == "e":
            return dnde[0]
        else:
            dnde[0] = dnde_e

    if flavor is None or flavor == "mu":
        dnde_mu = _dnde_neutrino(
            neutrino_energy=neutrino_energy,
            parent_energy=parent_energy,
            parent_mass=parent_mass,
            interp=interp_mu,
        )
        if flavor == "mu":
            return dnde[1]
        else:
            dnde[1] = dnde_mu

    if flavor == "tau":
        return 0.0

    return dnde
