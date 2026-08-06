import functools as ft
import warnings
from collections.abc import Sequence
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

    Notes
    -----
    The expanded form evaluated here cancels to zero whenever the triangle
    degenerates. For the two-body case ``lambda(cme**2, m1**2, m2**2)`` that
    happens at the threshold ``cme = m1 + m2``, where the four terms are each
    of size ``cme**4`` but their sum vanishes, so every significant digit is
    lost. Use `two_body_momentum` rather than taking the square root of this
    function when the two-body momentum is what you actually want.
    """
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * a * c - 2 * b * c


def two_body_momentum(cme: float, m1: float, m2: float) -> float:
    r"""
    Returns the magnitude of the common three-momentum of a two-body state.

    Parameters
    ----------
    cme: float or array-like
        Center-of-mass energy, in MeV.
    m1: float or array-like
        Mass of the first particle, in MeV.
    m2: float or array-like
        Mass of the second particle, in MeV.

    Returns
    -------
    p: float or array-like
        Magnitude of either particle's three-momentum in the center-of-mass
        frame, in MeV. Zero at threshold and NaN below it, where no such
        state exists. The threshold is resolved to the last bit: for
        unequal masses ``m1 + m2`` itself rounds, so passing that sum as
        `cme` gives an exact zero only when the sum is exact (as it is for
        ``m1 == m2``), and otherwise gives either a sub-ulp momentum or NaN
        according to which side of the true threshold the rounded sum fell.

    Notes
    -----
    This is :math:`\sqrt{\lambda(s, m_1^2, m_2^2)} / (2\sqrt{s})`, but
    evaluated through the factored form

    .. math::

        p = \frac{\sqrt{(E - m_1 - m_2)(E - m_1 + m_2)
                        (E + m_1 - m_2)(E + m_1 + m_2)}}{2 E}

    with :math:`E` the center-of-mass energy. The two are algebraically
    identical and differ only in conditioning: `kallen_lambda` cancels to
    zero at threshold (see its notes), while no factor above does anything
    but shrink smoothly.

    The heavier mass is subtracted first so that every difference stays
    exact. Near threshold ``cme <= 2 * max(m1, m2)``, which puts
    ``cme - max(m1, m2)`` inside the Sterbenz range where a floating-point
    subtraction is exact, and its result is then of order ``min(m1, m2)``,
    so the second subtraction is exact too. Subtracting the lighter mass
    first instead leaves an ``ulp(cme)`` absolute error in a difference
    that tends to zero. Measured over 21 mass pairs from {e, mu, pi0, pi+,
    K+, p}, the ordered form holds a relative error of 3e-16 all the way
    to threshold, where the unordered factored form reaches 4e-5 and the
    `kallen_lambda` form 4e-2.
    """
    heavy = np.maximum(m1, m2)
    light = np.minimum(m1, m2)
    lo = (cme - heavy - light) * (cme - heavy + light)
    hi = (cme + heavy - light) * (cme + heavy + light)
    return np.sqrt(lo * hi) / (2 * cme)


def cross_section_prefactor(m1: float, m2: float, cme: float) -> float:
    """
    Returns the prefactor to convert an integral over
    Lorentz-invariant phase-space to a cross section.

    Parameters
    ----------
    m1: float
        Mass of the first incoming particle, in MeV.
    m2: float
        Mass of the second incoming particle, in MeV.
    cme: float
        Center-of-mass energy, in MeV.

    Returns
    -------
    pre: float
        The flux factor ``1 / (4 p cme)``, in MeV^-2, with `p` the
        center-of-mass momentum of the incoming pair. Diverges to positive
        infinity at threshold, where the relative velocity of the pair
        vanishes, and is NaN below it. See `two_body_momentum` for how the
        threshold itself is resolved.
    """
    p = two_body_momentum(cme, m1, m2)
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


def minkowski_dot(fv1: Sequence[float], fv2: Sequence[float]) -> float:
    """
    Compute the west-coast (+,-,-,-) scalar product of two four-vectors.

    Parameters
    ----------
    fv1, fv2: array-like
        Four-vectors of length 4, ordered (E, px, py, pz). The components
        carry whatever units the caller uses (MeV throughout hazma); the
        result carries their square.

    Returns
    -------
    dot: float
        ``fv1[0]*fv2[0] - fv1[1]*fv2[1] - fv1[2]*fv2[2] - fv1[3]*fv2[3]``.

    Notes
    -----
    Pure-Python replacement for the Cython
    ``hazma.field_theory_helper_functions.common_functions.minkowski_dot``
    deleted in the Cython-to-Rust migration. `ldot` is the array-oriented
    generalization of the same product; this keeps the scalar
    four-vector spelling that squared-matrix-element code is written in.
    """
    return fv1[0] * fv2[0] - fv1[1] * fv2[1] - fv1[2] * fv2[2] - fv1[3] * fv2[3]


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
