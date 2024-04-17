from collections.abc import Generator
from typing import TypeAlias

import numpy as np
from scipy.special import gamma

from hazma import parameters

RealArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
ComplexArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.complex128]]

RealOrRealArray: TypeAlias = float | RealArray
ComplexOrComplexArray: TypeAlias = complex | ComplexArray


# Charged Pion mass in GeV
MPI0_GEV = parameters.neutral_pion_mass * 1e-3
# Charged Pion mass in GeV
MPI_GEV = parameters.charged_pion_mass * 1e-3
# Neutral Kaon mass in GeV
MK0_GEV = parameters.neutral_kaon_mass * 1e-3
# Charged Kaon mass in GeV
MK_GEV = parameters.charged_kaon_mass * 1e-3
# Charged Kaon mass in GeV
META_GEV = parameters.eta_mass * 1e-3
# Rho mass in GeV
MRHO_GEV = parameters.rho_mass * 1e-3
# Omega mass in GeV
MOMEGA_GEV = parameters.omega_mass * 1e-3
# Eta' mass in GeV
METAP_GEV = parameters.eta_prime_mass * 1e-3
# Phi mass in GeV
MPHI_GEV = parameters.phi_mass * 1e-3
# fpi in GeV
FPI_GEV = parameters.fpi * 1e-3


def beta2(
    s: RealOrRealArray,
    m1: float,
    m2: float,
) -> RealOrRealArray:
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
    return np.clip((1.0 - (m1 + m2) ** 2 / s) * (1.0 - (m1 - m2) ** 2 / s), 0.0, None)


def beta(
    s: float | RealArray,
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
    mres: float | RealArray,
    gamma: float | RealArray,
    m1: float,
    m2: float,
) -> float | RealArray:
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
    v2 = beta2(mres**2, m1, m2)
    v = np.sqrt(v2)
    r = (m1**2 + m2**2) / mres**2
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
    s: float | RealArray,
    mres: float | RealArray,
    gamma: float | RealArray,
    m1: float,
    m2: float,
    reshape=False,
) -> float | RealArray:
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
    vr = beta(mres**2, m1, m2)
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
    s: float | RealArray,
    mres: float | RealArray,
    gamma: float | RealArray,
    m1: float,
    m2: float,
    dh: float | RealArray,
    hres: float | RealArray,
    reshape=False,
) -> float | RealArray:
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
            - (ss[:, np.newaxis] - mres**2) * dh
        )

    if s != 0.0:
        return hhat(s, mres, gamma, m1, m2) - hres - (s - mres**2) * dh
    else:
        return (
            -2.0 * (m1 + m2) ** 2 / np.pi * gamma / mres / beta(mres**2, m1, m2) ** 3
            - hres
            + mres**2 * dh
        )


def gamma_p(
    s: float | RealArray,
    mres: float | RealArray,
    gamma: float | RealArray,
    m1: float,
    m2: float,
    reshape: bool | None = False,
) -> float | RealArray:
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
    vr2 = beta2(mres**2, m1, m2)
    if hasattr(s, "__len__") and reshape:
        rp = np.sqrt(
            np.clip(
                v2[:, np.newaxis] / vr2,  # type:ignore
                0.0,
                None,
            )
        )
        return np.sqrt(s)[:, np.newaxis] / mres * rp**3 * gamma
    rp = np.where(vr2 == 0.0, vr2, np.sqrt(np.clip(v2 / vr2, 0.0, None)))
    return np.sqrt(s) / mres * rp**3 * gamma


def breit_wigner_gs(
    s: float | RealArray,
    mass: float | RealArray,
    width: float | RealArray,
    m1: float,
    m2: float,
    h0: float | RealArray,
    dh: float | RealArray,
    hres: float | RealArray,
    reshape: bool | None = False,
) -> complex | ComplexArray:
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
    mr2 = mass**2

    if hasattr(s, "__len__") and reshape:
        ss = np.array(s)
        return (mr2 + h0) / (
            mr2
            - ss[:, np.newaxis]
            + h(ss, mass, width, m1, m2, dh, hres, reshape=True)
            - 1j
            * np.sqrt(ss)[:, np.newaxis]
            * gamma_p(ss, mass, width, m1, m2, reshape=True)
        )
    return (mr2 + h0) / (
        mr2
        - s
        + h(s, mass, width, m1, m2, dh, hres)
        - 1j * np.sqrt(s) * gamma_p(s, mass, width, m1, m2)
    )


def breit_wigner_fw(
    s: RealOrRealArray,
    mass: RealOrRealArray,
    width: RealOrRealArray,
    reshape: bool | None = False,
) -> ComplexOrComplexArray:
    """
    Compute the standard Breit-Wigner with a constant width. See
    ArXiv:1002.0279 Eqn.(8) for details.

    Parameters
    ----------
    s: Union[float, npt.NDArray]
        Center-of-mass energy squared.
    mass: Union[float, npt.NDArray]
        Mass of the resonance.
    width: Union[float, npt.NDArray]
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
    mr2 = mass**2
    if hasattr(s, "__len__") and reshape:
        ss = np.array(s)
        return mr2 / (mr2 - ss[:, np.newaxis] - 1j * mass * width)
    return mr2 / (mr2 - s - 1j * mass * width)


def breit_wigner_pwave(
    s: float | RealArray,
    mres: float | complex | RealArray | ComplexArray,
    gamma: float | complex | RealArray | ComplexArray,
    m1: float,
    m2: float,
    reshape: bool | None = False,
):
    mr2 = mres**2
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
