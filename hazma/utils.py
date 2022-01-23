import numpy as np
import numpy.typing as npt
from typing import Union

from .parameters import alpha_em

# ===================================================================
# ---- Types --------------------------------------------------------
# ===================================================================

RealArray = npt.NDArray[np.float64]
RealOrRealArray = Union[float, RealArray]

# ===================================================================
# ---- Kinematics ---------------------------------------------------
# ===================================================================


def kinematically_accessable(etot, masses):
    return etot > sum(masses)


def kallen_lambda(a, b, c):
    """
    Returns the Källén kinematic (triangle) polynomial.
    """
    return a ** 2 + b ** 2 + c ** 2 - 2 * a * b - 2 * a * c - 2 * b * c


# ===================================================================
# ---- Altarelli-Parisi ---------------------------------------------
# ===================================================================


def __scalar_splitting(x):
    return 2 * (1 - x) / x


def __fermion_splitting(x):
    return (1 + (1 - x) ** 2) / x


def __dnde_altarelli_parisi(eng, cme, mass, splitting):
    mu = mass / cme

    def f(e):
        x = 2 * e / cme
        if x > 1 - np.exp(1) * mu ** 2:
            return 0.0
        return (
            2
            * alpha_em
            / (np.pi * cme)
            * splitting(x)
            * (np.log((1 - x) / mu ** 2) - 1)
        )

    if hasattr(eng, "__len__"):
        return np.vectorize(f)(eng)
    return f(eng)


def dnde_altarelli_parisi_fermion(energies, cme: float, mf: float):
    """
    Compute the photon spectrum from radiation off a final-state fermion using the
    Altarelli–Parisi approximation.

    Parameters
    ----------
    energies: float or array-like
        Photon energies.
    cme: float
        Center-of-mass energy.
    mf: float
        Mass of the radiating fermion.

    Returns
    -------
    dnde: float or array-like
        Photon spectrum evaluated at the input energies.
    """
    return __dnde_altarelli_parisi(energies, cme, mf, __fermion_splitting)


def dnde_altarelli_parisi_scalar(energies, cme: float, ms: float):
    """
    Compute the photon spectrum from radiation off a final-state scalar using the
    Altarelli–Parisi approximation.

    Parameters
    ----------
    energies: float or array-like
        Photon energies.
    cme: float
        Center-of-mass energy.
    ms: float
        Mass of the radiating scalar.

    Returns
    -------
    dnde: float or array-like
        Photon spectrum evaluated at the input energies.
    """
    return __dnde_altarelli_parisi(energies, cme, ms, __scalar_splitting)
