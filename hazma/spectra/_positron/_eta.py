from typing import Union, overload

import numpy as np
import numpy.typing as npt

from hazma import parameters

from ._utils import load_interp, dnde_positron_point, dnde_positron_array

RealArray = npt.NDArray[np.float64]

_eta_interp = load_interp("eta_positron.csv")


@overload
def dnde_positron_eta(positron_energy: float, eta_energy: float) -> float:
    ...


@overload
def dnde_positron_eta(positron_energy: RealArray, eta_energy: float) -> RealArray:
    ...


def dnde_positron_eta(
    positron_energy: Union[RealArray, float], eta_energy: float
) -> Union[RealArray, float]:
    mass = parameters.eta_mass
    interp = _eta_interp

    if eta_energy < mass:
        return np.zeros_like(positron_energy)

    if isinstance(positron_energy, float):
        return dnde_positron_point(positron_energy, eta_energy, mass, interp)

    assert hasattr(positron_energy, "__len__"), (
        "Invalid type for positron_energy."
        + f"Expected float or numpy array, got: {type(positron_energy)}"
    )

    return dnde_positron_array(positron_energy, eta_energy, mass, interp)
