from typing import Generator, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.special import gamma  # type:ignore

# Pion mass in GeV
MPI_GEV = 0.13957018
# Neutral Kaon mass in GeV
MK0_GEV = 0.497611
# Charged Kaon mass in GeV
MKP_GEV = 0.493677
# Charged Kaon mass in GeV
META_GEV = 0.547862


def beta2(
    s: Union[float, npt.NDArray[np.float64]],
    m1: float,
    m2: float,
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Return the final state momentum times 4 / s.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    m1: float
        Mass of the first final state particle.
    m2: float
        Mass of the second final state particle.

    Returns
    -------
    beta: Union[float, npt.NDArray]
        Final state momentum times 4 / s.
    """
    return np.clip(
        (1.0 - (m1 + m2) ** 2 / s) * (1.0 - (m1 - m2) ** 2 / s), 0.0, None
    )  # type:ignore


def beta(
    s: Union[float, npt.NDArray[np.float64]],
    m1: float,
    m2: float,
):
    """
    Return the final state momentum times 4 / s.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    m1: float
        Mass of the first final state particle.
    m2: float
        Mass of the second final state particle.

    Returns
    -------
    beta: Union[float, npt.NDArray]
        Final state momentum times 4 / s.
    """
    return np.sqrt(beta2(s, m1, m2))


def dhhatds(
    mres: Union[float, npt.NDArray[np.float64]],
    gamma: Union[float, npt.NDArray[np.float64]],
    m1: float,
    m2: float,
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Compute the derivative of the Hhat(s) function for the Gounaris-Sakurai
    Breit-Wigner function evaluated at the resonance mass. See ArXiv:1002.0279
    Eqn.(4) for details.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    mres: Union[float, npt.NDArray]
        Mass of the resonance.
    gamma: Union[float, npt.NDArray]
        Width of the resonance.
    m1: float
        Mass of the first final state particle.
    m2: float
        Mass of the second final state particle.

    Returns
    -------
    dhhat: Union[float, npt.NDArray]
        The value of the the derivative of Hhat(s) evaluated at the resonance
        mass.
    """
    v2 = beta2(mres ** 2, m1, m2)
    v = np.sqrt(v2)
    r = (m1 ** 2 + m2 ** 2) / mres ** 2
    return (
        gamma
        / np.pi
        / mres
        / v2
        * (
            (3.0 - 2.0 * v2 - 3.0 * r) * np.log((1.0 + v) / (1.0 - v))
            + 2.0 * v * (1.0 - r / (1.0 - v2))
        )
    )


def hhat(
    s: Union[float, npt.NDArray[np.float64]],
    mres: Union[float, npt.NDArray[np.float64]],
    gamma: Union[float, npt.NDArray[np.float64]],
    m1: float,
    m2: float,
    reshape=False,
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Compute the Hhat(s) function for the Gounaris-Sakurai Breit-Wigner
    function. See ArXiv:1002.0279 Eqn.(4) for details.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    mres: Union[float, npt.NDArray]
        Mass of the resonance.
    gamma: Union[float, npt.NDArray]
        Width of the resonance.
    m1: float
        Mass of the first final state particle.
    m2: float
        Mass of the second final state particle.
    reshape: Optional[bool]
        If true, a different value is computed for each `s`. This is useful
        for computing form-factors for many squared center-of-mass energies at
        once.

    Returns
    -------
    hhat: Union[float, npt.NDArray]
        The value of the Hhat(s) function.
    """
    vr = beta(mres ** 2, m1, m2)
    v = beta(s, m1, m2)
    if hasattr(s, "__len__") and reshape:
        ss = np.array(s)
        return (
            gamma
            / mres
            / np.pi
            * ss[:, np.newaxis]
            * (v[:, np.newaxis] / vr) ** 3
            * np.log((1.0 + v[:, np.newaxis]) / (1.0 - v[:, np.newaxis]))
        )
    return gamma / mres / np.pi * s * (v / vr) ** 3 * np.log((1.0 + v) / (1.0 - v))


def h(
    s: Union[float, npt.NDArray[np.float64]],
    mres: Union[float, npt.NDArray[np.float64]],
    gamma: Union[float, npt.NDArray[np.float64]],
    m1: float,
    m2: float,
    dh: Union[float, npt.NDArray[np.float64]],
    hres: Union[float, npt.NDArray[np.float64]],
    reshape=False,
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Compute the H(s) function for the Gounaris-Sakurai Breit-Wigner function.
    See ArXiv:1002.0279 Eqn.(3) for details.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    mres: Union[float, npt.NDArray]
        Mass of the resonance.
    gamma: Union[float, npt.NDArray]
        Width of the resonance.
    m1: float
        Mass of the first final state particle.
    m2: float
        Mass of the second final state particle.
    dh: Union[float, npt.NDArray]
        Derivative of the of the H-hat function evaluated at the resonance
        mass.
    hres: Union[float, npt.NDArray]
        Value of the H(s) function at s=mres^2.
    reshape: Optional[bool]
        If true, a different value is computed for each `s`. This is useful
        for computing form-factors for many squared center-of-mass energies at
        once.

    Returns
    -------
    h: Union[float, npt.NDArray]
        The value of the H(s) function.
    """
    if hasattr(s, "__len__") and reshape:
        ss = np.array(s)
        return (
            hhat(ss, mres, gamma, m1, m2, reshape=True)
            - hres
            - (ss[:, np.newaxis] - mres ** 2) * dh
        )

    if s != 0.0:
        return hhat(s, mres, gamma, m1, m2) - hres - (s - mres ** 2) * dh
    else:
        return (
            -2.0 * (m1 + m2) ** 2 / np.pi * gamma / mres / beta(mres ** 2, m1, m2) ** 3
            - hres
            + mres ** 2 * dh
        )


def gamma_p(
    s: Union[float, npt.NDArray[np.float64]],
    mres: Union[float, npt.NDArray[np.float64]],
    gamma: Union[float, npt.NDArray[np.float64]],
    m1: float,
    m2: float,
    reshape: Optional[bool] = False,
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Compute the s-dependent width of the resonance.
    See ArXiv:1002.0279 Eqn.(6) for details.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    mres: Union[float, npt.NDArray]
        Mass of the resonance.
    gamma: Union[float, npt.NDArray]
        Width of the resonance.
    m1: float
        Mass of the first final state particle.
    m2: float
        Mass of the second final state particle.
    reshape: Optional[bool]
        If true, a different value is computed for each `s`. This is useful
        for computing form-factors for many squared center-of-mass energies at
        once.

    Returns
    -------
    gamma: Union[float, npt.NDArray]
        The s-dependent width.
    """
    v2 = beta2(s, m1, m2)
    vr2 = beta2(mres ** 2, m1, m2)
    if hasattr(s, "__len__") and reshape:
        rp = np.sqrt(
            np.clip(
                v2[:, np.newaxis] / vr2,  # type:ignore
                0.0,
                None,
            )
        )
        return np.sqrt(s)[:, np.newaxis] / mres * rp ** 3 * gamma
    rp = np.where(vr2 == 0.0, vr2, np.sqrt(np.clip(v2 / vr2, 0.0, None)))
    return np.sqrt(s) / mres * rp ** 3 * gamma


def breit_wigner_gs(
    s: Union[float, npt.NDArray[np.float64]],
    mres: Union[float, npt.NDArray[np.float64]],
    gamma: Union[float, npt.NDArray[np.float64]],
    m1: float,
    m2: float,
    h0: Union[float, npt.NDArray[np.float64]],
    dh: Union[float, npt.NDArray[np.float64]],
    hres: Union[float, npt.NDArray[np.float64]],
    reshape: Optional[bool] = False,
) -> Union[complex, npt.NDArray[np.complex128]]:
    """
    Compute the Gounaris-Sakurai Breit-Wigner function with pion loop
    corrections included. See ArXiv:1002.0279 Eqn.(2) for details.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    mres: Union[float, npt.NDArray]
        Mass of the resonance.
    gamma: Union[float, npt.NDArray]
        Width of the resonance.
    m1: float
        Mass of the first final state particle.
    m2: float
        Mass of the second final state particle.
    h0: Union[float, npt.NDArray]
        Value of the H(s) function at s=0.
    dh: Union[float, npt.NDArray]
        Derivative of the of the H-hat function evaluated at the resonance
        mass.
    hres: Union[float, npt.NDArray]
        Value of the H(s) function at s=mres^2.
    reshape: Optional[bool]
        If true, a different value is computed for each `s`. This is useful
        for computing form-factors for many squared center-of-mass energies at
        once.

    Returns
    -------
    bw: Union[float, npt.NDArray]
        The Breit-Wigner function.
    """
    mr2 = mres ** 2

    if hasattr(s, "__len__") and reshape:
        ss = np.array(s)
        return (mr2 + h0) / (
            mr2
            - ss[:, np.newaxis]
            + h(ss, mres, gamma, m1, m2, dh, hres, reshape=True)
            - 1j
            * np.sqrt(ss)[:, np.newaxis]
            * gamma_p(ss, mres, gamma, m1, m2, reshape=True)
        )
    return (mr2 + h0) / (
        mr2
        - s
        + h(s, mres, gamma, m1, m2, dh, hres)
        - 1j * np.sqrt(s) * gamma_p(s, mres, gamma, m1, m2)
    )


def breit_wigner_fw(
    s: Union[float, npt.NDArray[np.float64]],
    mres: Union[float, complex, npt.NDArray[np.float64]],
    gamma: Union[float, complex, npt.NDArray[np.float64]],
    reshape: Optional[bool] = False,
) -> Union[complex, npt.NDArray[np.complex128]]:
    """
    Compute the standard Breit-Wigner with a constant width. See
    ArXiv:1002.0279 Eqn.(8) for details.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    mres: Union[float, npt.NDArray]
        Mass of the resonance.
    gamma: Union[float, npt.NDArray]
        Width of the resonance.
    reshape: Optional[bool]
        If true, a different value is computed for each `s`. This is useful
        for computing form-factors for many squared center-of-mass energies at
        once.

    Returns
    -------
    bw: Union[float, npt.NDArray]
        The Breit-Wigner function.
    """
    mr2 = mres ** 2
    if hasattr(s, "__len__") and reshape:
        ss = np.array(s)
        return mr2 / (mr2 - ss[:, np.newaxis] - 1j * mres * gamma)
    return mr2 / (mr2 - s - 1j * mres * gamma)


def breit_wigner_pwave(
    s: Union[float, npt.NDArray[np.float64]],
    mres: Union[float, complex, npt.NDArray[np.complex128]],
    gamma: Union[float, complex, npt.NDArray[np.complex128]],
    m1: float,
    m2: float,
    reshape: Optional[bool] = False,
):
    mr2 = mres ** 2
    if hasattr(s, "__len__") and reshape:
        ss = np.array(s)
        return mr2 / (
            mr2
            - ss[:, np.newaxis]
            - 1j
            * np.sqrt(ss)[:, np.newaxis]
            * gamma_p(ss, mres, gamma, m1, m2, reshape=True)  # type:ignore
        )
    return mr2 / (
        mr2 - s - 1j * np.sqrt(s) * gamma_p(s, mres, gamma, m1, m2)  # type:ignore
    )


def gamma_generator(
    beta: float,
    nmax: int,
) -> Generator[float, None, None]:
    """
    Generator to efficiently compute gamma(2 - beta + n) / gamma(1 + n) for
    values of n less than a specified maximum value. This is done recurrsively
    to avoid roundoff errors.

    Parameters
    ----------
    beta: float
        Value inside the gamma-function in the numerator of:
            gamma(2 - beta + n) / gamma(1 + n)
    nmax: int
        Maximum value to compute the function for.

    Returns
    -------
    gamma_gen: Generator[float, None, None]
        Generator to yield values of gamma(2 - beta + n) / gamma(1 + n).
    """
    val = gamma(2.0 - beta)
    yield val
    n = 1
    while n < nmax:
        val *= (1.0 - beta + n) / n
        n += 1
        yield val
