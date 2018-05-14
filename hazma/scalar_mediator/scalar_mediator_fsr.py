import numpy as np
from cmath import sqrt, pi, atanh
import warnings

from .scalar_mediator_cross_sections import sigma_xx_to_s_to_pipi

from ..field_theory_helper_functions.three_body_phase_space import E1_to_s
from ..field_theory_helper_functions.three_body_phase_space import t_integral
from ..field_theory_helper_functions.common_functions import \
    cross_section_prefactor
from ..field_theory_helper_functions.three_body_phase_space import \
    phase_space_prefactor, t_lim1, t_lim2

from ..parameters import alpha_em
from ..parameters import charged_pion_mass as mpi
from ..parameters import rho_mass as mrho
from ..parameters import rho_width
from ..parameters import fpi, fv, gv, qe, vh

from ..hazma_errors import NegativeSquaredMatrixElementWarning


def __msqrd_xx_s_ffg(Q, s, t, mf, params):
    gsxx = params.gsxx
    gsff = params.gsff
    mx = params.mx
    ms = params.ms
    return (2 * qe**2 * (-4 * mx**2 + Q**2) *
            (8 * mf**6 * s + (Q**4 + s**2) * (Q**2 - s - t) * t +
             mf**4 * (7 * Q**4 - 8 * Q**2 * s - s * (s + 16 * t)) +
             mf**2 * (-3 * Q**6 + Q**4 * (5 * s + 2 * t) - Q**2 * s *
                      (3 * s + 8 * t) +
                      s * (s**2 + 10 * s * t +
                           8 * t**2))) * gsff**2 * gsxx**2) / \
        ((-ms**2 + Q**2)**2 * (mf**2 + Q**2 - s - t)**2 *
         (-mf**2 + t)**2)


def __dnde_xx_to_s_to_ffg(egam, Q, mf, params):
    """ Unvectorized dnde_xx_to_s_to_ffg """

    e, m, s = egam / Q, mf / Q, Q**2 - 2. * Q * egam

    mx = params.mx

    if 2. * mf < Q and 4. * mf**2 < s < Q**2 and 2. * mx < Q:
        ret_val = (alpha_em *
                   (2 * (-1 + 4 * m**2) *
                    sqrt((-1 + 2 * e) * (-1 + 2 * e + 4 * m**2)) +
                    4 * (1 + 2 * (-1 + e) * e - 6 * m**2 + 8 * e * m**2 +
                         8 * m**4) *
                    atanh(sqrt(1 + (4 * m**2) / (-1 + 2 * e))))) / \
            (e * (1 - 4 * m**2)**1.5 * pi * Q)

        return ret_val.real
    else:
        return 0.0


def dnde_xx_to_s_to_ffg(egam, Q, mf, params):
    """Return the fsr spectra for fermions from decay of scalar mediator.

    Computes the final state radiaton spectrum value dNdE from a scalar
    mediator given a gamma ray energy of `eng_gam`, center of mass
    energy `cme`
    and final state fermion mass `mass_f`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell scalar mediator.
    mf : float
        Mass of the final state fermion.
    params: namedtuple
        Namedtuple of the model parameters.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from scalar mediator.

    """
    if hasattr(egam, '__len__'):
        return np.array([__dnde_xx_to_s_to_ffg(e, Q, mf, params)
                         for e in egam])
    else:
        return __dnde_xx_to_s_to_ffg(egam, Q, mf, params)


# ######################
""" Charged Pion FSR """
# ######################


def __msqrd_xx_to_s_to_pipig(Q, s, t, params):
    """
    Returns the squared matrix element for two fermions annihilating
    into two charged pions and a photon.

    Parameters
    ----------
    Q : double
        Center of mass energy, or sqrt((ppip + ppim + pg)^2).
    s : double
        Mandelstam variable associated with the photon, defined as
        (P-pg)^2, where P = ppip + ppim + pg.
    t : double
        Mandelstam variable associated with the charged pion, defined as
        (P-ppip)^2, where P = ppip + ppim + pg.
    params: namedtuple
        Namedtuple of the model parameters.

    Returns
    -------
    Returns matrix element squared for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma`.

    Notes
    -----
    The matrix element for this process, M, is related to the
    form factor by |M|^2. = s Re[E(s,t,u) E^*(s,u,t)] - m_PI^2.
    |E(s,t,u) + E(s,u,t)|^2.
    """

    ret_val = 0.0

    mx = params.mx
    gsff = params.gsff
    gsGG = params.gsGG
    vs = params.vs
    mrhoT = params.mrhoT
    gsxx = params.gsxx
    ms = params.ms

    if 2. * mpi < Q and 4. * mpi**2 < s < Q**2 and 2. * mx < Q:
        if t_lim1(s, 0.0, mpi, mpi, Q) < t < t_lim2(s, 0.0, mpi, mpi, Q):

            ret_val = (-4 * gsxx * qe**2 *
                       (mpi**4 * s + s * t * (-Q**2 + s + t) +
                        mpi**2 * (Q**4 - Q**2 * s - 2 * s * t)) *
                       (fv * gsGG * gv * (mpi**2 - t) *
                        (mpi**2 + Q**2 - s - t) *
                        (3 * vh + 3 * gsff * vs + 2 * gsGG * vs) *
                        (9 * vh + 4 * gsGG * vs) *
                        (-9 * mrhoT * s * vh - 2 * mrho**3 *
                         (9 * vh + 2 * gsGG * vs) +
                         mrho * s * (9 * vh + 2 * gsGG * vs)) +
                        3 * fpi**2 * mrho * mrhoT**2 * (mrho**2 - s) * vh *
                        (-729 * gsff * mpi**2 * vh**2 +
                         128 * gsGG**3 * mpi**2 * vs**2 +
                         36 * gsGG**2 * vs *
                         (6 * mpi**2 * vh - 3 * Q**2 * vh +
                          2 * mpi**2 * (vh - 8 * gsff * vs)) -
                         162 * gsGG * vh *
                         (-2 * mpi**2 * (vh + gsff * vs) + Q**2 *
                          (vh + gsff * vs) +
                          mpi**2 * (3 * vh + 10 * gsff * vs))))**2) / \
                (6561. * fpi**4 * mrho**2 * mrhoT**4 *
                 (mrho**4 + mrho**2 * (rho_width**2 - 2 * s) + s**2) *
                 (-ms**2 + s)**2 * (mpi**2 - t)**2 *
                 (mpi**2 + Q**2 - s - t)**2 * vh**4 *
                 (3 * vh + 3 * gsff * vs + 2 * gsGG * vs)**2 *
                 (9 * vh + 4 * gsGG * vs)**2)

    if ret_val <= 0.0:
        msg = ""
        warnings.warn(msg, NegativeSquaredMatrixElementWarning)
        ret_val = 0.0

    return ret_val


def __dnde_xx_to_s_to_pipig(eng_gam, Q, params):
    """Unvectorized dnde_xx_to_s_to_pipig"""
    mx = params.mx

    s = Q**2 - 2. * Q * eng_gam

    ret_val = 0.0

    if 2. * mpi < Q and 4. * mpi**2 <= s <= Q**2 and 2. * mx < Q:

        s = E1_to_s(eng_gam, 0., Q)

        def mat_elem_sqrd(t):
            return __msqrd_xx_to_s_to_pipig(Q, s, t, params)

        prefactor1 = phase_space_prefactor(Q)
        prefactor2 = 2. * Q / sigma_xx_to_s_to_pipi(Q, params)
        prefactor3 = cross_section_prefactor(mx, mx, Q)

        prefactor = prefactor1 * prefactor2 * prefactor3

        int_val, _ = t_integral(s, 0., mpi, mpi, Q, mat_elem_sqrd)

        ret_val = prefactor * int_val

    assert ret_val >= 0.

    return ret_val


def dnde_xx_to_s_to_pipig(eng_gams, Q, params):
    """
    Returns the gamma ray energy spectrum for two fermions annihilating
    into two charged pions and a photon.

    Parameters
    ----------
    eng_gam : numpy.ndarray or double
        Gamma ray energy.
    Q : double
        Center of mass energy, or sqrt((ppip + ppim + pg)^2).

    Returns
    -------
    Returns gamma ray energy spectrum for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma` evaluated at the gamma
    ray energy(ies).

    """

    if hasattr(eng_gams, '__len__'):
        return np.array([__dnde_xx_to_s_to_pipig(eng_gam, Q, params)
                         for eng_gam in eng_gams])
    else:
        return __dnde_xx_to_s_to_pipig(eng_gams, Q, params)
