import functools as ft
import warnings
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from .parameters import alpha_em

# ===================================================================
# ---- Deprecation --------------------------------------------------
# ===================================================================
#


def _force_deprecation_warning(message):
    warnings.simplefilter("always", DeprecationWarning)  # turn off filter
    warnings.warn(message, category=DeprecationWarning, stacklevel=2)
    warnings.simplefilter("default", DeprecationWarning)  # reset filter


def warn_deprecated_module(module: str, alternative: Optional[str] = None):
    r"""Decorator used to raise warning for calling a deprecated function."""

    message = f"{module} is deprecated."

    if alternative is not None:
        message = f"{message} Use {alternative} instead."

    _force_deprecation_warning(message)


def deprecate_fn(fn, alternative: Optional[str] = None):
    r"""Decorator used to raise warning for calling a deprecated function."""

    message = f"{fn.__name__} is deprecated."

    if alternative is not None:
        message = f"{message} Use {alternative} instead."

    @ft.wraps(fn)
    def wrapped(*args, **kwargs):
        _force_deprecation_warning(message)
        return fn(*args, **kwargs)

    return wrapped


# ===================================================================
# ---- Types --------------------------------------------------------
# ===================================================================

RealArray = npt.NDArray[np.float64]
RealOrRealArray = Union[float, RealArray]
ComplexArray = npt.NDArray[np.complex128]
ComplexOrComplexArray = Union[complex, ComplexArray]
RealOrComplexArray = npt.NDArray[Union[np.float64, np.complex128]]

# ===================================================================
# ---- Enums --------------------------------------------------------
# ===================================================================

NeutrinoFlavor = Literal["e", "mu", "tau"]

# ===================================================================
# ---- Kinematics ---------------------------------------------------
# ===================================================================


def kinematically_accessable(etot, masses):
    return etot > sum(masses)


def kallen_lambda(a, b, c):
    """
    Returns the Källén kinematic (triangle) polynomial.
    """
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * a * c - 2 * b * c


def cross_section_prefactor(m1: float, m2: float, cme: float) -> float:
    """
    Returns the prefactor to convert an integral over
    Lorentz-invariant phase-space to a cross section.

    Parameters
    ----------
    m1: float
        Mass of the first incoming particle.
    m2: float
        Mass of the second incoming particle.
    cme: float
        Center-of-mass energy.
    """
    p = np.sqrt(kallen_lambda(cme**2, m1**2, m2**2)) / (2 * cme)
    return 1.0 / (4.0 * p * cme)


def ldot(lv1, lv2, axis: int = 0):
    """
    Compute the Lorenzian scalar product of two arrays.

    Parameters
    ----------
    lv1, lv2: np.ndarray
        Arrays to compute scalar product from.
    axis: int, optional
        Axes containing the four-vectors. The specified axis must be of
        shape 4 for both `lv1` and `lv2`. Default is 0.
    """
    assert (
        lv1.shape[axis] == 4 and lv2.shape[axis] == 4
    ), "Specified axis must be 4-dimenstional."

    p0 = lv1.take(0, axis=axis) * lv2.take(0, axis=axis)
    p1 = lv1.take(1, axis=axis) * lv2.take(1, axis=axis)
    p2 = lv1.take(2, axis=axis) * lv2.take(2, axis=axis)
    p3 = lv1.take(3, axis=axis) * lv2.take(3, axis=axis)

    return p0 - p1 - p2 - p3  # type: ignore


def lnorm_sqr(lv: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the Lorenzian squared-norm of an array.

    Parameters
    ----------
    lv: np.ndarray
        Array to compute Lorenzian norm off.
    axis: int, optional
        Axes containing the four-vectors. The specified axis must be of
        shape 4. Default is 0.
    """
    assert lv.shape[axis] == 4, "Specified axis must be 4-dimenstional."

    return (
        np.square(lv.take(0, axis=axis))
        - np.square(lv.take(1, axis=axis))
        - np.square(lv.take(2, axis=axis))
        - np.square(lv.take(3, axis=axis))
    )


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
        if x > 1 - np.exp(1) * mu**2:
            return 0.0
        return (
            2 * alpha_em / (np.pi * cme) * splitting(x) * (np.log((1 - x) / mu**2) - 1)
        )

    if hasattr(eng, "__len__"):
        return np.vectorize(f)(eng)
    return f(eng)


@ft.partial(deprecate_fn, alternative="hazma.spectra.dnde_photon_ap_fermion")
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


@ft.partial(deprecate_fn, alternative="hazma.spectra.dnde_photon_ap_scalar")
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
