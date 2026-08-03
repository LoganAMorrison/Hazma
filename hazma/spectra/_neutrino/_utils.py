from typing import Union, Optional

import numpy as np
import numpy.typing as npt
import pathlib
from scipy import interpolate

RealArray = npt.NDArray[np.float64]


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


def _dnde_neutrino(
    *,
    neutrino_energy: RealArray,
    parent_energy: float,
    parent_mass: float,
    interp: interpolate.InterpolatedUnivariateSpline,
) -> RealArray:
    if parent_energy < parent_mass:
        return np.zeros_like(neutrino_energy)

    if parent_energy < parent_mass:
        return np.zeros_like(neutrino_energy)

    eps = np.finfo(neutrino_energy.dtype).eps
    if parent_energy - parent_mass < eps:
        return interp(neutrino_energy) * neutrino_energy  # type: ignore

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


def dnde_neutrino(
    *,
    neutrino_energy: Union[RealArray, float],
    parent_energy: float,
    parent_mass: float,
    interp_e: interpolate.InterpolatedUnivariateSpline,
    interp_mu: interpolate.InterpolatedUnivariateSpline,
    flavor: Optional[str] = None,
) -> Union[RealArray, float]:

    scalar = np.isscalar(neutrino_energy)
    enu = np.atleast_1d(neutrino_energy).astype(np.float64)
    dnde = np.zeros((3, *enu.shape), dtype=enu.dtype)

    if flavor == "tau":
        return dnde[2]

    for i, flav, inter in [(0, "e", interp_e), (1, "mu", interp_mu)]:
        if not flavor or flavor == flav:
            dnde_ = _dnde_neutrino(
                neutrino_energy=enu,
                parent_energy=parent_energy,
                parent_mass=parent_mass,
                interp=inter,
            )

            if flavor == flav:
                return dnde_
            else:
                dnde[i] = dnde_

    if scalar:
        return dnde[..., 0]

    return dnde
