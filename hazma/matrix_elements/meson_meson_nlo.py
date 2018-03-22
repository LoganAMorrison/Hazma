from cmath import sqrt, log, pi

import numpy as np

from ..parameters import fpi
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import eta_mass as meta

from ..parameters import LECS, nlo_lec_mu

from scipy.integrate import quad
from scipy.special import lpmv

from .meson_meson_lo import __amp_pipi_pipi_LO

mPI = (mpi + mpi0) / 2.
mK = (mk + mk0) / 2.


def _v(m, s):
    s = complex(s)

    return sqrt(1. - 4. * m**2 / s)


def _kP(m):
    return (1. + log(m**2 / nlo_lec_mu**2)) / (32. * pi**2)


def _Jbar(m, s):
    num = complex(_v(m, s) - 1.)
    den = complex(_v(m, s) + 1.)

    return (2. + _v(m, s) * log(num / den)) / (16. * pi**2)


def _Jr(m, s):
    s = complex(s)

    return _Jbar(m, s) - 2. * _kP(m)


def _Mr(m, s):
    s = complex(s)

    return (1.0 / 12.) * _Jbar(m, s) * _v(m, s)**2 - \
        (1. / 6.0) * _kP(m) + 1. / (288. * pi**2)


def _B4(s, t, u, su=3):
    s = complex(s)

    if su == 2:
        return (1. / 6. / fpi**4) * \
            (3. * (s**2 - mPI**4) * _Jbar(mPI, s) +
             (t * (t - u) - 2 * mPI**2 * t + 4 * mPI**2 * u - 2 * mPI**4) *
             _Jbar(mPI, t) +
             (u * (u - t) - 2 * mPI**2 * u + 4 * mPI**2 * t - 2 * mPI**4) *
             _Jbar(mPI, u) -
             (1. / 96. / pi**2) *
             (21 * s**2 + 5 * (t - u)**2 + 8 * mPI**2 * s - 26. * mPI**4))
    if su == 3:
        return ((mPI**4. / 18.) * _Jr(meta, s) +
                (1. / 2.) * (s**2 - mPI**4) * _Jr(mPI, s) +
                (1. / 8.) * s**2 * _Jr(mK, s) +
                (1. / 4.) * (t - 2 * mPI**2)**2 * _Jr(mPI, t) +
                t * (s - u) * (_Mr(mPI, t) + 0.5 * _Mr(mK, t)) +
                (1. / 4.) * (u - 2 * mPI**2)**2 * _Jr(mPI, u) +
                u * (s - t) * (_Mr(mPI, u) + 0.5 * _Mr(mK, u))) / fpi**4
    else:
        raise ValueError("Invalid value for 'su'. 'su' must be 2 or 3.")


def _C4(s, t, u, su=3):
    s = complex(s)

    if su == 2:
        return (LECS["SU2_Gr"] / 4. / fpi**4) * (-2 * (s - 2 * mPI**2) ** 2 +
                                                 (t - 2 * mPI**2)**2 +
                                                 (t - 2 * mPI**2)**2) + \
            (LECS["SU2_Gr"] / fpi**4) * (s - 2 * mPI**2)**2
    if su == 3:
        return (4. * ((8. * LECS["6"] + 4. * LECS["8"]) * mPI**4. +
                      (4. * LECS["4"] + 2. * LECS["5"]) * mPI**2. *
                      (s - 2. * mPI**2.) +
                      (LECS["3"] + 2. * LECS["1"]) * (s - 2. * mPI**2)**2. +
                      LECS["2"] * ((t - 2. * mPI**2)**2 +
                                   (u - 2. * mPI**2)**2))) / fpi**4


def __amp_pipi_pipi_NLO(s, t, u, su=3):
    """
    Returns the next-to-leading-order pion scattering amplitude in either SU(2)
    or SU(3) ChiPT.
    """
    s = complex(s)

    return __amp_pipi_pipi_LO(s, t, u) + _B4(s, t, u, su=su) + \
        _C4(s, t, u, su=su)


def __amp_pipi_pipi_NLO_I(s, t, iso=0, su=3):
    """
    Returns the next-to-leading-order pion scattering amplitude in a definite
    isospin channel in either SU(2) or SU(3) ChiPT.
    """
    s = complex(s)

    u = 4. * mPI**2 - s - t
    A4stu = __amp_pipi_pipi_NLO(s, t, u, su=su)
    A4tsu = __amp_pipi_pipi_NLO(t, s, u, su=su)
    A4uts = __amp_pipi_pipi_NLO(u, t, s, su=su)

    if iso == 0:
        return 0.5 * (3. * A4stu + A4tsu + A4uts)
    if iso == 1:
        return 0.5 * (A4tsu - A4uts)
    if iso == 2:
        return 0.5 * (A4tsu + A4uts)
    else:
        raise ValueError("Invalid isospin index. 'iso' must be 0, 1 or 2.")


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
    s = complex(s)

    return 0.5 * (4. * mPI**2 - s) * (1. - x)


def __msqrd_pipi_pipi_NLO_I(s, iso=0, su=3):
    """
    Returns the integrated, next to leading order, squared pion scattering
    amplitude in a definite isospin channel in either SU(2) or SU(3) ChiPT.
    """
    s = complex(s)

    def integrand(x):
        return abs(__amp_pipi_pipi_NLO_I(s, mandlestam_t(x, s), iso=iso,
                                         su=su))**2
    return quad(integrand, -1., 1.)[0]


def __pw_pipi_pipi_NLO_I(s, l, iso=0, su=3):
    """
    Returns the lth partial wave of the next-to-leading-order pion scattering
    amplitude in a definite isospin channel in either SU(2) or SU(3) ChiPT.
    """
    s = complex(s)

    def f_real(x):
        return __amp_pipi_pipi_NLO_I(s, mandlestam_t(x, s), iso=iso,
                                     su=su).real * lpmv(0, l, x)

    def f_imag(x):
        return __amp_pipi_pipi_NLO_I(s, mandlestam_t(x, s), iso=iso,
                                     su=su).imag * lpmv(0, l, x)

    NLOreal = quad(f_real, -1., 1.)[0]
    NLOimag = quad(f_imag, -1., 1.)[0]

    return 0.5 * (NLOreal + 1.0j * NLOimag)


def __high_L_partial_wave_pipi_pipi_NLO_I(s, iso=0, su=3):
    """
    Returns the sum of all partial waves of the next-to-leading-order pion
    scattering amplitude for l > 0 in a definite isospin channel in either
    SU(2) or SU(3) ChiPT.
    """
    s = complex(s)

    return __msqrd_pipi_pipi_NLO_I(s, iso=iso, su=su) - 2. * \
        __pw_pipi_pipi_NLO_I(s, 0, iso=iso, su=su)**2


def amp_pipi_to_pipi_NLO(s, t, u, su=3):
    """
    Returns the next-to-leading-order pion scattering amplitude in either SU(2)
    or SU(3) ChiPT.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    t : float
        Mandelstam variable (p1 - p3)^2.
    u : float
        Mandelstam variable (p1 - p4)^2.
    su : int, optional {3}
        SU version of ChiPT to use.

    Returns
    -------
    M : float
        Next-to-leading order pion scattering amplitude.
    """
    isVec = hasattr(s, "__len__") * hasattr(t, "__len__") * \
        hasattr(u, "__len__")

    if isVec:
        s, t, u = np.array(s), np.array(t), np.array(u)

        if s.shape != t.shape or s.shape != u.shape or t.shape != u.shape:
            raise ValueError("s, t and u must have the same shape.")

        return np.array([__amp_pipi_pipi_NLO(s[i], t[i], u[i], su=su)
                         for i in range(len(s))])
    if hasattr(s, "__len__") or hasattr(t, "__len__") or hasattr(u, "__len__"):
        raise ValueError("s, t and u must have the same shape.")

    return __amp_pipi_pipi_NLO(s, t, u, su=su)


def amp_pipi_to_pipi_NLO_I(s, t, iso=0, su=3):
    """
    Returns the next-to-leading-order pion scattering amplitude in a definite
    isospin channel in either SU(2) or SU(3) ChiPT.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    t : float
        Mandelstam variable (p1 - p3)^2.
    u : float
        Mandelstam variable (p1 - p4)^2.
    iso : int, optional {0}
        Isospin
    su : int, optional {3}
        SU version of ChiPT to use.

    Returns
    -------
    M : float
        Next-to-leading order pion scattering amplitude in a definite isospin
        channel.
    """
    isVec = hasattr(s, "__len__") * hasattr(t, "__len__")

    if isVec:
        s, t = np.array(s), np.array(t)

        if s.shape != t.shape:
            raise ValueError("s and t must have the same shape.")

        return np.array([__amp_pipi_pipi_NLO_I(s[i], t[i], iso=iso, su=su)
                         for i in range(len(s))])
    if hasattr(s, "__len__") or hasattr(t, "__len__"):
        raise ValueError("s and u must have the same shape.")

    return __amp_pipi_pipi_NLO_I(s, t, iso=iso, su=su)


def msqrd_pipi_to_pipi_NLO_I(s, iso=0, su=3):
    """
    Returns the integrated, next to leading order, squared pion scattering
    amplitude in a definite isospin channel in either SU(2) or SU(3) ChiPT.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    iso : int, optional {0}
        Isospin.
    su : int, optional {3}
        SU version of ChiPT to use.

    Returns
    -------
    MI2_NLO : float
        Integrated (over t), next to leading order, squared pion scattering
        amplitude in a definite isospin channel.
    """

    if hasattr(s, "__len__"):

        return np.array([__msqrd_pipi_pipi_NLO_I(s[i], iso=iso, su=su)
                         for i in range(len(s))])

    return __msqrd_pipi_pipi_NLO_I(s, iso=iso, su=su)


def partial_wave_pipi_to_pipi_NLO_I(s, l, iso=0, su=3):
    """
    Returns the lth partial wave of the next-to-leading-order pion scattering
    amplitude in a definite isospin channel in either SU(2) or SU(3) ChiPT.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    l : int
        Partial wave index.
    iso : int, optional {0}
        Isospin
    su : int, optional {3}
        SU version of ChiPT to use.


    Returns
    -------
    MIL_NLO : complex
    """

    if hasattr(s, "__len__"):

        return np.array([__pw_pipi_pipi_NLO_I(s[i], l, iso=iso, su=su)
                         for i in range(len(s))])

    return __pw_pipi_pipi_NLO_I(s, l, iso=iso, su=su)


def high_L_partial_wave_pipi_to_pipi_NLO_I(s, iso=0, su=3):
    """
    Returns the sum of all partial waves of the next-to-leading-order pion
    scattering amplitude for l > 0 in a definite isospin channel in either
    SU(2) or SU(3) ChiPT.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    iso : int, optional {0}
        Isospin

    Returns
    -------
    MIL_NLO : float
        Sum of all partial waves of the next-to-leading-order pion
        scattering amplitude for l > 0 in a definite isospin channel.
    """
    if hasattr(s, "__len__"):

        return np.array([
            __high_L_partial_wave_pipi_pipi_NLO_I(s[i], iso=iso, su=su)
            for i in range(len(s))])

    return __high_L_partial_wave_pipi_pipi_NLO_I(s, iso=iso, su=su)
