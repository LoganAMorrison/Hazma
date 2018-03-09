import os
import numpy as np
from .parameters import charged_pion_mass as mpi
from .parameters import charged_kaon_mass as mk
from .parameters import neutral_pion_mass as mpi0
from .parameters import neutral_kaon_mass as mk0
from .parameters import fpi

mPI = (mpi + mpi0) / 2. + 0j
mK = (mk + mk0) / 2. + 0j
FPI = fpi + 0j
Lam = 1.1 * 10**3 + 0j  # cut-off scale taken to be 1.1 GeV
q_max = np.sqrt(Lam**2 - mK**2)
eps = 1.0 * 10**-10


# #######################
# Bethe Salpeter Equation
# #######################


def bubble_loop(cme, m):
    """
    Returns value of bubble loop

    Returns
        G_{11} = i\int_{0}^{\infty}\frac{d^4q}{(2\pi)^4}
            \frac{1}{q^2-m_{K}^2}\frac{1}{(P-q)^2-m_{K}^2}
    The integral is regulated with a finite momentum cutoff equal to 1.1 GeV.
    """
    s = cme**2 + eps * 1j

    return ((np.sqrt(4 * m**2 - s) *
             np.arctan((np.sqrt(s) * q_max) /
                       np.sqrt((4 * m**2 - s) * (m**2 + q_max**2)))) /
            np.sqrt(s) + np.log(m / (q_max + np.sqrt(m**2 + q_max**2)))) / \
        (8. * np.pi**2)


def __G11(cme):
    """
    Returns kaon bubble loop factor.

    Returns
        G_{11} = i\int_{0}^{\infty}\frac{d^4q}{(2\pi)^4}
            \frac{1}{q^2-m_{K}^2}\frac{1}{(P-q)^2-m_{K}^2}
    The integral is regulated with a finite momentum cutoff equal to 1.1 GeV.
    """
    s = cme**2. + 1.0j * eps

    p1 = np.sqrt(s - 4. * mK**2) / 2.
    p2 = np.sqrt(s - 4. * mPI**2) / 2.

    g11 = bubble_loop(cme, mK)
    return g11

    if np.imag(p1) > 0. and np.imag(p2) > 0.:
        return g11
    if np.imag(p1) > 0. and np.imag(p2) < 0.:
        return g11
    if np.imag(p1) < 0. and np.imag(p2) < 0.:
        return g11 - 2.0j * np.imag(g11)
    if np.imag(p1) < 0. and np.imag(p2) > 0.:
        return g11 - 2.0j * np.imag(g11)


def __G22(cme):
    """
    Returns pion bubble loop factor.

    Returns
        G_{22} = i\int_{0}^{\infty}\frac{d^4q}{(2\pi)^4}
            \frac{1}{q^2-m_{\pi}^2}\frac{1}{(P-q)^2-m_{\pi}^2}
    The integral is regulated with a finite momentum cutoff equal to 1.1 GeV.
    """
    s = cme**2. + 1.0j * eps

    p1 = np.sqrt(s - 4. * mK**2) / 2.
    p2 = np.sqrt(s - 4. * mPI**2) / 2.

    g22 = bubble_loop(cme, mPI)

    return g22

    if np.imag(p1) > 0. and np.imag(p2) > 0.:
        return g22
    if np.imag(p1) > 0. and np.imag(p2) < 0.:
        return g22
    if np.imag(p1) < 0. and np.imag(p2) < 0.:
        return g22 - 2.0j * np.imag(g22)
    if np.imag(p1) < 0. and np.imag(p2) > 0.:
        return g22 - 2.0j * np.imag(g22)


def __v11(cme):
    s = cme**2. + 1.0j * eps
    return - (1 / 4. / fpi**2) * (3 * s)


def __v12(cme):
    s = cme**2. + 1.0j * eps
    return - (1. / 3.0 / np.sqrt(12.) / fpi**2) * (9.0 / 2. * s)


def __v22(cme):
    s = cme**2. + 1.0j * eps
    return - (1. / 9. / fpi**2) * (9. * s + 15. * mPI**2 / 2. - 12. * mPI**2)


def __delta_pi(cme):
    return 1.0 - __v22(cme) * __G22(cme)


def __delta_k(cme):
    return 1.0 - __v11(cme) * __G11(cme)


def __delta_c(cme):
    return __delta_pi(cme) * __delta_k(cme) - \
        __v12(cme)**2 * __G11(cme) * __G22(cme)


def __delta(cme):
    return __delta_pi(cme) * __delta_c(cme)


def amp_bethe_salpeter_kk_to_kk(cme):
    """
    Returns the unitarized matrix element for kk -> kk in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    Q: float
        Invariant mass of the pions.

    Returns
    -------
    Mu: float
        Unitarized matrix element for kk -> kk in the zero isospin
        channel.
    """

    Q = cme + 0.0j
    return (__delta_pi(Q) * __v11(Q) +
            __v12(Q)**2 * __G22(Q)) / __delta_c(Q) + 0.0j


def amp_bethe_salpeter_pipi_to_kk(cme):
    """
    Returns the unitarized matrix element for pipi -> kk in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    Q: float
        Invariant mass of the pions.

    Returns
    -------
    Mu: float
        Unitarized matrix element for pipi -> kk in the zero isospin
        channel.
    """

    Q = cme + 0.0j
    return (__v12(Q) * __G11(Q) * __v11(Q) +
            __delta_k(Q) * __v12(Q)) / __delta_c(Q) + 0.0j


def amp_bethe_salpeter_pipi_to_pipi(cme):
    """
    Returns the unitarized matrix element for pipi -> pipi in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    Q: float
        Invariant mass of the pions.

    Returns
    -------
    Mu: float
        Unitarized matrix element for pipi -> pipi in the zero isospin
        channel.
    """

    Q = cme + 0.0j
    return __v22(Q) / __delta_pi(Q) + \
        __v12(Q)**2 * __G11(Q) / (__delta_pi(Q) * __delta_c(Q)) + 0.0j


def __phase_shift_pipi_to_pipi(cme, deg=False):
    s = cme**2. + 1.0j * eps
    p2 = np.sqrt(s - 4 * mPI**2) / 2.

    z = 1.0 - 2.0j * p2 * \
        amp_bethe_salpeter_pipi_to_pipi(np.sqrt(s)) / (8 * np.pi * np.sqrt(s))

    x, y = np.real(z), np.imag(z)
    theta = np.abs(np.arctan(np.abs(y / x)))

    if x > 0. and y > 0.:
        theta = theta
    if x < 0. and y > 0.:
        theta = np.pi - theta
    if x < 0. and y < 0.:
        theta = np.pi + theta
    if x > 0. and y < 0.:
        theta = 2. * np.pi - theta

    if deg is True:
        theta = theta * 180. / np.pi

    return theta / 2.0


def phase_shift_pipi_to_pipi(cmes, deg=False):
    if hasattr(cmes, '__len__'):
        return np.array([__phase_shift_pipi_to_pipi(cme, deg=deg)
                         for cme in cmes])
    else:
        return __phase_shift_pipi_to_pipi(cmes, deg=deg)


# ########################
# Inverse Amplitude Method
# ########################

# Import data for IAM

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "unitarization_data",
                         "final_state_int_unitarizated.dat")

unitarizated_data = np.loadtxt(DATA_PATH, delimiter=',', dtype=complex)
unitarizated_data_x = np.real(unitarizated_data[:, 0])
unitarizated_data_y = unitarizated_data[:, 1]


def amp_pipi_to_pipi_I0(cme):
    """
    Lowest order pion scattering squared matrix element in the isospin I = 0
    channel.

    Parameters
    ----------
    cme: double
        Invariant mass of the two charged pions.
    """
    return (2 * cme**2 - mpi0**2) / (32. * fpi**2 * np.pi)


def amp_inverse_amplitude_pipi_to_pipi(cme):
    """
    Unitarized pion scattering squared matrix element in the isopin I = 0
    channel.

    Unitarization was computed using the inverse amplitude method(IAM) with
    only pion contributions.

    Parameters
    ----------
    cme: double
        Invariant mass of the two charged pions.
    """
    return np.interp(cme, unitarizated_data_x, unitarizated_data_y)


def ratio_pipi_to_pipi_unitarized_tree(cme):
    """
    Returns the unitarized squared matrix element for: math: `\pi\pi\to\pi\pi`
    divided by the leading order, ununitarized squared matrix element for
    : math: `\pi\pi\to\pi\pi`.

    This was computed using the inverse amplitude method(IAM) with only

    Parameters
    ----------
    cme: double
        Invariant mass of the two charged pions.

    Results
    -------
    __unit_matrix_elem_sqrd: double
        The unitarized matrix element for: math: `\pi\pi\to\pi\pi`, | t_u | ^2,
        divided by the un - unitarized squared matrix element for
        : math: `\pi\pi\to\pi\pi`, | t | ^2; | t_u | ^2 / |t | ^2.
    """
    return amp_inverse_amplitude_pipi_to_pipi(cme) / \
        amp_pipi_to_pipi_I0(cme)


def msqrd_bethe_salpeter_pipi_to_pipi(Q):
    """
    Returns the square of the unitarized pipi -> pipi amplitude.

    Parameters
    ----------
    Q: float
        Invariant mass of the pions.

    Returns
    -------
    Mu2: float
        Unitarized squared matrix element for pipi -> pipi in the zero isospin
        channel.
    """
    return np.abs(amp_bethe_salpeter_pipi_to_pipi(Q))**2
