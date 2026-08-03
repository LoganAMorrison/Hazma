from typing import Generator, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.special import gamma  # type:ignore
from scipy import special

from hazma import parameters
from hazma.utils import kallen_lambda

RealArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

RealOrRealArray = Union[float, RealArray]
ComplexOrComplexArray = Union[complex, ComplexArray]


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
    s: Union[float, npt.NDArray[np.float64]],
    mass: Union[float, npt.NDArray[np.float64]],
    width: Union[float, npt.NDArray[np.float64]],
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
    reshape: Optional[bool] = False,
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
    s: Union[float, npt.NDArray[np.float64]],
    mres: Union[float, complex, npt.NDArray[np.float64], npt.NDArray[np.complex128]],
    gamma: Union[float, complex, npt.NDArray[np.float64], npt.NDArray[np.complex128]],
    m1: float,
    m2: float,
    reshape: Optional[bool] = False,
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


def msqrd_lorentz_p_p(s, m1, m2):
    """
    Returns the square of the Lorentz structure of a current, J_mu, consisting
    of two pseudo-scalar mesons: -J_mu J_mu. This current has the form:
        J_mu = -(p1 - p2)_mu.

    Parameters
    ----------
    s: float or array
        Invariant mass of pseudo-scalars: s = (p1 + p2)^2.
    m1: float
        Mass of 1st pseudo-scalar.
    m2: float
        Mass of 2nd pseudo-scalar.
    """
    return s - 2 * (m1**2 + m2**2)


def msqrd_lorentz_p_v(s, mp, mv):
    """
    Returns the square of the Lorentz structure of a current, J_mu, consisting
    of a pseudo-scalar meson P and a vector meson V: -J_mu J_mu. This current has
    the form:
        J_mu = -Eps[mu, a, b, c] epsV_a (pv + pp)_b pp_c
    where:
        - Eps = Levi-Civita tensor,
        - epsV = vector meson polarization,
        - pp = momentum of the pseudo-scalar, and
        - pv = momentum of vector-meson.

    Parameters
    ----------
    s: float or array
        Invariant mass of pseudo-scalars: s = (pp + pv)^2.
    mp: float
        Mass of the pseudo-scalar meson.
    mv: float
        Mass of the vector meson.
    """
    return 0.5 * kallen_lambda(s, mp**2, mv**2)


def msqrd_lorentz_p_p_p(s, t, m, m1, m2, m3):
    """
    Returns the square of the Lorentz structure of a current, J_mu, consisting
    of three pseudo-scalar mesons: -J_mu J_mu. This current has the form:
        J_mu = -Eps[mu, a, b, c] p1_a p2_b p3_c
    where:
        - Eps = Levi-Civita tensor,
        - p1 = momentum of 1st pseudo-scalar,
        - p2 = momentum of 2nd pseudo-scalar, and
        - p3 = momentum of 3rd pseudo-scalar.

    Parameters
    ----------
    s: float
        Invariant mass of 2nd and 3rd mesons: s = (p2 + p3)^2
    t: float
        Invariant mass of 1st and 3rd mesons: t = (p1 + p3)^2.
    m: float
        Invariant mass of all mesons: m^2 = (p1+p2+p3)^2
    m1: float
        Mass of 1st meson.
    m2: float
        Mass of 2nd meson.
    m3: float
        Mass of 3rd meson.
    """
    return (
        -(m1**4 * m2**2)
        - m**4 * m3**2
        + m**2
        * (
            -(m3**4)
            + m1**2 * (m2**2 + m3**2 - s)
            + m3**2 * s
            + m2**2 * (m3**2 - t)
            + m3**2 * t
            + s * t
        )
        - s * (m2**2 * (m3**2 - t) + t * (-(m3**2) + s + t))
        + m1**2 * (-(m2**4) + (-(m3**2) + s) * t + m2**2 * (m3**2 + s + t))
    ) / 4.0


def msqrd_lorentz_p_p_p_t_coeffs(s, m, m1, m2, m3):
    """
    Returns the coefficents of powers of mandelstam t from the
    square of the Lorentz structure of a current, J_mu, consisting
    of three pseudo-scalar mesons: -J_mu J_mu.

    Parameters
    ----------
    s: float
        Invariant mass of 2nd and 3rd mesons: s = (p2 + p3)^2
    m: float
        Invariant mass of all mesons: m^2 = (p1+p2+p3)^2
    m1: float
        Mass of 1st meson.
    m2: float
        Mass of 2nd meson.
    m3: float
        Mass of 3rd meson.
    """
    t0 = (
        -((m1 * m2 - m * m3) * (m1 * m2 + m * m3) * (-(m**2) + m1**2 + m2**2 - m3**2))
        + (-m + m2) * (m + m2) * (m1 - m3) * (m1 + m3) * s
    ) / 4.0

    t1 = (
        -((m - m1) * (m + m1) * (m2 - m3) * (m2 + m3))
        + (m**2 + m1**2 + m2**2 + m3**2) * s
        - s**2
    ) / 4.0
    t2 = -s / 4.0
    return [t0, t1, t2]


def integral_power_div_simple_monomial(n, z, lb, ub):
    """
    Integral of a power divided by a monomial of degree m:
        ∫ x^n / (x - z) dx
    """
    # x^n / x^m
    if z == 0:
        if n == 0:
            return np.log(ub / lb)
        p = n - 1
        return (ub**p - lb**p) / p

    zu = ub - z
    zl = lb - z

    return z**n * np.log(zu / zl) + sum(
        [
            special.binom(n, k) * z ** (n - k) * (zu**k - zl**k) / k
            for k in range(1, n + 1)
        ]
    )


def integral_power_div_monomial(n, m, z, lb, ub):
    """
    Integral of a power divided by a monomial of degree m:
        ∫ x^n / (x - z)^m dx
    """
    # x^n / x^m
    if z == 0:
        if n == m - 1:
            return np.log(ub / lb)
        p = n - m + 1
        return (ub**p - lb**p) / p

    def term(k):
        if k == m - 1:
            return z**n * np.log((ub - z) / (lb - z))
        return (
            special.binom(n, k)
            * z ** (n - k)
            * ((ub - z) ** (k - m + 1) - (lb - z) ** (k - m + 1))
            / (k - m + 1)
        )

    return sum([term(k) for k in range(n + 1)])


def integral_poly_div_simple_factor(poly_coeffs, z, lb, ub):
    """
    Integral of a polynomial divided by a factored polynomial with simple roots:
        ∫ (a0 + a1*x + ... + an * x^n) / (x - z) dx
    """
    return sum(
        [
            c * integral_power_div_simple_monomial(n, z, lb, ub)
            for n, c in enumerate(poly_coeffs)
        ]
    )


def integral_poly_div_simple_factored(poly_coeffs, simple_zeros, lb, ub):
    """
    Integral of a polynomial divided by a factored polynomial with simple roots:
        ∫ (a0 + a1*t + ... + an * t^n) / ((t-b1) * ... * (t-bn))
    """
    num = np.polynomial.Polynomial(poly_coeffs)
    den = np.polynomial.Polynomial.fromroots(simple_zeros)
    den_prime = den.deriv(1)
    partial_frac_coeffs = [num(x) / den_prime(x) for x in simple_zeros]
    return sum(
        [
            c * integral_poly_div_simple_factor(poly_coeffs, z, lb, ub)
            for c, z in zip(partial_frac_coeffs, simple_zeros)
        ]
    )
