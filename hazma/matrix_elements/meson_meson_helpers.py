from cmath import sqrt, log, pi, abs

from ..parameters import fpi
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import eta_mass as meta
from ..parameters import rho_mass as mrho

from scipy.integrate import quad
from scipy.special import lpmv


mPI = (mpi + mpi0) / 2.
mK = (mk + mk0) / 2.

# ####################
# Low energy Constants
# ####################

# All low energy constants are for NLO ChiPT, evaluated at mu = mrho.
# These are:
# L1 = + ( 0.56 +- 0.10 ) * 1.0e-3
# L2 = + ( 1.21 +- 0.10 ) * 1.0e-3
# L3 = - ( 2.79 +- 0.14 ) * 1.0e-3
# L4 = - ( 0.36 +- 0.17 ) * 1.0e-3
# L5 = + ( 1.4  +- 0.5  ) * 1.0e-3
# L6 = + ( 0.07 +- 0.08 ) * 1.0e-3
# L7 = - ( 0.44 +- 0.15 ) * 1.0e-3
# L8 = + ( 0.78 +- 0.18 ) * 1.0e-3

mu = mrho
Lr1 = 0.56 * 1.0e-3
Lr2 = 1.21 * 1.0e-3
L3 = -2.79 * 1.0e-3
Lr4 = -0.36 * 1.0e-3
Lr5 = 1.4 * 1.0e-3
Lr6 = 0.07 * 1.0e-3
L7 = -0.44 * 1.0e-3
Lr8 = 0.78 * 1.0e-3

# ####################################
# LO Meson-Meson Scattering amplutudes
# ####################################


def amp_pipi_to_pipi_LO(s, t, u):
    """
    Returns the leading-order pion scattering amplitude.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    t : float
        Mandelstam variable (p1 - p3)^2.
    u : float
        Mandelstam variable (p1 - p4)^2.

    Returns
    -------
    M : float
        Leading order pion scattering amplitude.
    """
    return (s - mPI**2) / fpi**2


def amp_pipi_to_pipi_LO_I0(s, t):
    """
    Returns the leading-order pion scattering amplitude in the iso-scalar
    channel (I = 0).

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    t : float
        Mandelstam variable (p1 - p3)^2.
    u : float
        Mandelstam variable (p1 - p4)^2.

    Returns
    -------
    M : float
        Leading order pion scattering amplitude in the iso-scalar channel
        (I = 0).
    """
    u = 4. * mPI**2 - s - t
    A2stu = amp_pipi_to_pipi_LO(s, t, u)
    A2tsu = amp_pipi_to_pipi_LO(s, t, u)
    A2uts = amp_pipi_to_pipi_LO(s, t, u)

    return 0.5 * (3. * A2stu + A2tsu + A2uts)


def msqrd_pipi_to_pipi_LO_I0(s):
    """
    Returns the integrated, leading order, squared pion scattering amplitude.

    Parameters
    ----------
    s : float
        Squared center of mass energy.

    Returns
    -------
    M2_LO : float
        Integrated (over t), leading order, squared pion scattering amplitude.
    """
    def f(x):
        return abs(amp_pipi_to_pipi_LO_I0(s, mandlestam_t(x, s)))**2

    return quad(f, -1., 1.)[0]


def partial_wave_pipi_to_pipi_LO_I0(s, l):
    """
    Returns the lth partial wave of the leading order pion scattering
    amplitude.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    l : int
        Partial wave index.

    Returns
    -------
    Ml_LO : complex

    """
    def f_real(x):
        return amp_pipi_to_pipi_LO_I0(s, mandlestam_t(x, s)).real * \
            lpmv(0, l, x)

    def f_imag(x):
        return amp_pipi_to_pipi_LO_I0(s, mandlestam_t(x, s)).imag * \
            lpmv(0, l, x)

    return 0.5 * (quad(f_real, -1., 1.)[0] + 1j * quad(f_imag, -1., 1.)[0])


# #####################################
# NLO Meson-Meson Scattering amplutudes
# #####################################


def _v(m, s):
    return sqrt(1. - 4. * m**2 / s)


def _kP(m):
    return (1. + log(m**2 / mu**2)) / (32. * pi**2)


def _Jbar(m, s):
    num = complex(_v(m, s) - 1.)
    den = complex(_v(m, s) + 1.)
    return (2. + _v(m, s) * log(num / den)) / (16. * pi**2)


def _Jr(m, s):
    return _Jbar(m, s) - 2. * _kP(m)


def _Mr(m, s):
    return (1.0 / 12.) * _Jbar(m, s) * _v(m, s)**2 - \
        (1. / 6.0) * _kP(m) + 1. / (288. * pi**2)


def _B4(s, t, u):
    return ((mPI**4. / 18.) * _Jr(meta, s) +
            (1. / 2.) * (s**2 - mPI**4) * _Jr(mPI, s) +
            (1. / 8.) * s**2 * _Jr(mK, s) +
            (1. / 4.) * (t - 2 * mPI**2)**2 * _Jr(mPI, t) +
            t * (s - u) * (_Mr(mPI, t) + 0.5 * _Mr(mK, t)) +
            (1. / 4.) * (u - 2 * mPI**2)**2 * _Jr(mPI, u) +
            u * (s - t) * (_Mr(mPI, u) + 0.5 * _Mr(mK, u))) / fpi**4


def _C4(s, t, u):
    return (4. * ((8. * Lr6 + 4. * Lr8) * mPI**4. +
                  (4. * Lr4 + 2. * Lr5) * mPI**2. *
                  (s - 2. * mPI**2.) +
                  (L3 + 2. * Lr1) * (s - 2. * mPI**2)**2. +
                  Lr2 * ((t - 2. * mPI**2)**2 +
                         (u - 2. * mPI**2)**2))) / fpi**4


def amp_pipi_to_pipi_NLO(s, t, u):
    """
    Returns the next-to-leading-order pion scattering amplitude.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    t : float
        Mandelstam variable (p1 - p3)^2.
    u : float
        Mandelstam variable (p1 - p4)^2.

    Returns
    -------
    M : float
        Next-to-leading order pion scattering amplitude.
    """
    return amp_pipi_to_pipi_NLO(s, t, u) + _B4(s, t, u) + _C4(s, t, u)


def amp_pipi_to_pipi_NLO_I0(s, t):
    """
    Returns the next-to-leading-order pion scattering amplitude in the
    iso-scalar channel (I = 0).

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    t : float
        Mandelstam variable (p1 - p3)^2.
    u : float
        Mandelstam variable (p1 - p4)^2.

    Returns
    -------
    M : float
        Next-to-leading order pion scattering amplitude in the iso-scalar
        channel (I = 0).
    """
    u = 4. * mPI**2 - s - t
    A4stu = amp_pipi_to_pipi_NLO(s, t, u)
    A4tsu = amp_pipi_to_pipi_NLO(t, s, u)
    A4uts = amp_pipi_to_pipi_NLO(u, t, s)

    return 0.5 * (3. * A4stu + A4tsu + A4uts)


def mandlestam_t(x, s):
    """
    Returns mandlestam variable t in terms on the cosine of the angle between
    initial and final state and the squared center of mass energy.

    Parameters
    ----------
    x : float
        Cosine of the angle between initial and final states.
    s : float
        Squared center of mass energy.

    Returns
    -------
    t : float
        Mandlestam variable t.
    """
    return 0.5 * (4. * mPI**2 - s) * (1. - x)


def msqrd_pipi_to_pipi_NLO_I0(s):
    """
    Returns the integrated, next to leading order, squared pion scattering
    amplitude.

    Parameters
    ----------
    s : float
        Squared center of mass energy.

    Returns
    -------
    M2_NLO : float
        Integrated (over t), next to leading order, squared pion scattering
        amplitude.
    """
    def integrand(x):
        return abs(amp_pipi_to_pipi_NLO_I0(s, mandlestam_t(x, s)))**2
    return quad(integrand, -1., 1.)[0]


def partial_wave_pipi_to_pipi_NLO_I0(s, l):
    """
    Returns the lth partial wave of the next-to-leading-order pion scattering
    amplitude.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    l : int
        Partial wave index.

    Returns
    -------
    Ml_NLO : complex
    """
    def f_real(x):
        return amp_pipi_to_pipi_NLO_I0(s, mandlestam_t(x, s)).real * \
            lpmv(0, l, x)

    def f_imag(x):
        return amp_pipi_to_pipi_NLO_I0(s, mandlestam_t(x, s)).imag * \
            lpmv(0, l, x)

    NLOreal = quad(f_real, -1., 1.)[0]
    NLOimag = quad(f_imag, -1., 1.)[0]

    return 0.5 * (NLOreal + 1.0j * NLOimag)


def high_L_partial_wave_pipi_to_pipi_NLO_I0(s):
    """
    Returns the sum of all partial waves of the next-to-leading-order pion
    scattering amplitude for l > 0.

    Parameters
    ----------
    s : float
        Squared center of mass energy.

    Returns
    -------
    Ml_NLO : float
        Sum of all partial waves of the next-to-leading-order pion
        scattering amplitude for l > 0
    """
    return msqrd_pipi_to_pipi_NLO_I0(s) - 2. * \
        partial_wave_pipi_to_pipi_NLO_I0(s, 0)**2
