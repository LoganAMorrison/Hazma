import numpy as np

from ..parameters import fpi
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0

from scipy.integrate import quad
from scipy.special import lpmv

mPI = (mpi + mpi0) / 2.
mK = (mk + mk0) / 2.


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


def __amp_pipi_pipi_LO(s, t, u):
    """
    Returns the leading-order pion scattering amplitude. Use
    'amp_pipi_to_pipi_LO' instead.
    """
    return (s - mPI**2) / fpi**2


def __amp_pipi_pipi_LO_I(s, t, iso=0):
    """
    Returns the leading-order pion scattering amplitude in a definite isospin
    channel. Use 'amp_pipi_to_pipi_LO_I' instead.
    """
    u = 4. * mPI**2 - s - t
    A2stu = __amp_pipi_pipi_LO(s, t, u)
    A2tsu = __amp_pipi_pipi_LO(s, t, u)
    A2uts = __amp_pipi_pipi_LO(s, t, u)

    if iso == 0:
        return 0.5 * (3. * A2stu + A2tsu + A2uts)
    if iso == 1:
        return 0.5 * (A2tsu - A2uts)
    if iso == 2:
        return 0.5 * (A2tsu + A2uts)
    else:
        raise ValueError("Invalid isospin index. 'iso' must be 0, 1 or 2.")


def __msqrd_pipi_pipi_LO_I(s, iso=0):
    """
    Returns the integrated, leading order, squared pion scattering amplitude.
    Use 'msqrd_pipi_to_pipi_LO_I' instead.
    """
    def f(x):
        return abs(__amp_pipi_pipi_LO_I(s, mandlestam_t(x, s), iso=iso))**2

    return quad(f, -1., 1.)[0]


def __pw_pipi_pipi_LO_I(s, l, iso=0):
    """
    Returns the lth partial wave of the leading order pion scattering
    amplitude. Use 'partial_wave_pipi_to_pipi_LO_I' instead.
    """
    def f_real(x):
        return __amp_pipi_pipi_LO_I(s, mandlestam_t(x, s), iso=iso).real * \
            lpmv(0, l, x)

    def f_imag(x):
        return __amp_pipi_pipi_LO_I(s, mandlestam_t(x, s), iso=iso).imag * \
            lpmv(0, l, x)

    return 0.5 * (quad(f_real, -1., 1.)[0] + 1j * quad(f_imag, -1., 1.)[0])


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

    isVec = hasattr(s, "__len__") * hasattr(t, "__len__") * \
        hasattr(u, "__len__")

    if isVec:
        s, t, u = np.array(s), np.array(t), np.array(u)

        if s.shape != t.shape or s.shape != u.shape or t.shape != u.shape:
            raise ValueError("s, t and u must have the same shape.")

        return np.array([__amp_pipi_pipi_LO(s[i], t[i], u[i])
                         for i in range(len(s))])
    if hasattr(s, "__len__") or hasattr(t, "__len__") or hasattr(u, "__len__"):
        raise ValueError("s, t and u must have the same shape.")

    return __amp_pipi_pipi_LO(s, t, u)


def amp_pipi_to_pipi_LO_I(s, t, iso=0):
    """
    Returns the leading-order pion scattering amplitude in a definite isospin
    channel.

    Parameters
    ----------
    s : float
        Squared center of mass energy.
    t : float
        Mandelstam variable (p1 - p3)^2.
    u : float
        Mandelstam variable (p1 - p4)^2.
    iso : int
        Isospin.

    Returns
    -------
    M : float
        Leading order pion scattering amplitude in the iso-scalar channel
        (I = 0).
    """
    isVec = hasattr(s, "__len__") * hasattr(t, "__len__")

    if isVec:
        s, t = np.array(s), np.array(t)

        if s.shape != t.shape:
            raise ValueError("s and t must have the same shape.")

        return np.array([__amp_pipi_pipi_LO_I(s[i], t[i], iso=iso)
                         for i in range(len(s))])
    if hasattr(s, "__len__") or hasattr(t, "__len__"):
        raise ValueError("s and u must have the same shape.")

    return __amp_pipi_pipi_LO_I(s, t, iso=iso)


def msqrd_pipi_to_pipi_LO_I(s, iso=0):
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
    if hasattr(s, "__len__"):

        return np.array([__msqrd_pipi_pipi_LO_I(s[i], iso=iso)
                         for i in range(len(s))])

    return __msqrd_pipi_pipi_LO_I(s, iso=iso)


def partial_wave_pipi_to_pipi_LO_I(s, l, iso=0):
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
    if hasattr(s, "__len__"):
        return np.array([__pw_pipi_pipi_LO_I(s[i], l, iso=iso)
                         for i in range(len(s))])

    return __pw_pipi_pipi_LO_I(s, l, iso=iso)
