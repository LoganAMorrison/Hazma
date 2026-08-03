from typing import List, Dict, Union, overload

import numpy as np
import numpy.typing as npt

from hazma import parameters

RealArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def call_with_kinematic_threshold(f, x, thresholds: List[float]):
    """
    Call a unary function
    """
    if hasattr(x, "__len__"):
        mask = np.array([True] * len(x))
        for t in thresholds:
            mask = np.logical_and(mask, x > t)

        result = np.zeros_like(x)
        if not np.all(~mask):
            result[mask] = f(x[mask])
        return result

    accessible = np.all([x > t for t in thresholds])
    result = 0.0
    if accessible:
        result = f(x)
    return 0.0


STR_TO_MASS: Dict[str, float] = {
    "e": parameters.electron_mass,
    "mu": parameters.muon_mass,
    "pi": parameters.charged_pion_mass,
    "pi0": parameters.neutral_pion_mass,
    "k": parameters.charged_kaon_mass,
    "k0": parameters.neutral_kaon_mass,
    "eta": parameters.eta_mass,
    "etap": parameters.eta_prime_mass,
    "omega": parameters.omega_mass,
    "phi": parameters.phi_mass,
    "ve": 0.0,
    "vm": 0.0,
    "vt": 0.0,
    "gamma": 0.0,
}


@overload
def channel_open(
    cme: float,
    state: str,
    extra_masses: Dict[str, float] = dict(),
    delimiter: str = " ",
) -> bool: ...


@overload
def channel_open(
    cme: RealArray,
    state: str,
    extra_masses: Dict[str, float] = dict(),
    delimiter: str = " ",
) -> BoolArray: ...


def channel_open(
    cme: Union[float, RealArray],
    state: str,
    extra_masses: Dict[str, float] = dict(),
    delimiter: str = " ",
) -> Union[bool, BoolArray]:
    """Return True if the channel is kinematically accessible.

    Parameters
    ----------
    cme: float
        Center-of-mass energy.
    state: str
        String containing the states with the specified delimiter.
    extra_masses: dict[str,float], optional
        Extra masses aside from the SM particles.
    delimiter: str, optional
        Delimiter of the states in state string. Default is a single space `' '`.

    Returns
    -------
    accessible: bool
        True if the channel is open.
    """
    mass_dict: Dict[str, float] = {**STR_TO_MASS, **extra_masses}
    states: List[str] = state.split(delimiter)
    return cme > sum(map(lambda s: mass_dict[s], states))
