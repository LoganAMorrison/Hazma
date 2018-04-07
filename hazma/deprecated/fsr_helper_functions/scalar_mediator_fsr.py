"""Module for computing fsr spectrum from a scalar mediator.

@author - Logan Morrison and Adam Coogan.
@data - December 2017

"""
import numpy as np
from cmath import sqrt, pi

from ..field_theory_helper_functions.common_functions import \
    cross_section_prefactor
from ..field_theory_helper_functions.three_body_phase_space import E1_to_s
from ..field_theory_helper_functions.three_body_phase_space import t_integral
from ..field_theory_helper_functions.three_body_phase_space import \
    phase_space_prefactor

from ..parameters import rho_mass as mrho
from ..parameters import down_quark_mass as mdq
from ..parameters import up_quark_mass as muq
from ..parameters import alpha_em, b0, vh, fv, gv, qe, fpi

from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import rho_width

from ..unitarization import amp_bethe_salpeter_pipi_to_pipi
from .chrg_pi import msqrd_xx_s_pipig


mPI = (mpi0 + mpi) / 2.
mK = (mk0 + mk) / 2.


def __dnde_xx_to_s_to_ffg(egam, Q, mf):
    """Return the fsr spectra for fermions from decay of scalar mediator.

    Computes the final state radiaton spectrum value dNdE from a scalar
    mediator given a gamma ray energy of `eng_gam`, center of mass energy `cme`
    and final state fermion mass `mass_f`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell scalar mediator.
    mf : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from scalar mediator.

    """
    e, m = egam / Q, mf / Q

    if 0 < e and e < 0.5 * (1.0 - 4.0 * m**2):
        return (alpha_em *
                (2 * (-1 + 4 * m**2) *
                 sqrt((-1 + 2 * e) * (-1 + 2 * e + 4 * m**2)) +
                 4 * (1 + 2 * (-1 + e) * e - 6 * m**2 + 8 * e * m**2 +
                      8 * m**4) *
                 np.arctanh(sqrt(1 + (4 * m**2) / (-1 + 2 * e))))) / \
            (e * (1 - 4 * m**2)**1.5 * pi * Q)
    else:
        return 0.0


def dnde_xx_to_s_to_ffg(egam, Q, mf):
    """Return the fsr spectra for fermions from decay of scalar mediator.

    Computes the final state radiaton spectrum value dNdE from a scalar
    mediator given a gamma ray energy of `eng_gam`, center of mass energy `cme`
    and final state fermion mass `mass_f`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell scalar mediator.
    mf : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from scalar mediator.

    """
    if hasattr(egam, '__len__'):
        return np.array([__dnde_xx_to_s_to_ffg(e, Q, mf) for e in egam])
    else:
        return __dnde_xx_to_s_to_ffg(egam, Q, mf)


""" Charged Pion FSR """


def __msqrd_xx_to_s_to_pipig(s, t, Q, mx, ms, gsxx, gsff, gsGG, vs):
    """
    Returns the squared matrix element for two fermions annihilating into two
    charged pions and a photon.

    Parameters
    ----------
    s : double
        Mandelstam variable associated with the photon, defined as (P-pg)^2,
        where P = ppip + ppim + pg.
    t : double
        Mandelstam variable associated with the charged pion, defined as
        (P-ppip)^2, where P = ppip + ppim + pg.
    Q : double
        Center of mass energy, or sqrt((ppip + ppim + pg)^2).
    mx : double
        Mass of the initial state fermion.
    ms : double
        Mass of the scalar mediator.
    gsxx : double
        Coupling of the initial state fermion to the scalar mediator.
    gsff : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    gsGG : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.

    Returns
    -------
    Returns matrix element squared for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma`.

    u = u_to_st(0.0, mpi, mpi, Q, s, t)

    mat_elem_sqrd = (-16 * gsxx**2 *
                     (-4 * mx**2 + Q**2) *
                     (4 * mpi**6 - s * t * u + mpi**2 *
                         (t + u) * (s + t + u) - mpi**4 * (s + 4 * (t + u))) *
                     (2 * gsGG * (4 * mpi**2 - s - t - u) *
                         (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
                         (9 * vh + 8 * gsGG * vs) +
                         b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
                         (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
                          (9 * vh + 16 * gsGG * vs)))**2 * e**2) / \
        ((ms**2 - Q**2)**2 *
         (mpi**2 - t)**2 * (mpi**2 - u)**2 *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    return mat_elem_sqrd * amp_bethe_salpeter_pipi_to_pipi(sqrt(s))
    """
    return msqrd_xx_s_pipig(Q, s, t, mx, ms, gsxx, gsff, gsGG, vs)


def __sigma_xx_to_s_to_pipi(cme, mx, ms, gsxx, gsff, gsGG, vs):
    """
    Returns the cross section for two fermions annihilating into two
    charged pions.

    Parameters
    ----------
    cme : double
        Center of mass energy.
    mx : double
        Mass of the initial state fermion.
    ms : double
        Mass of the scalar mediator.
    gsxx : double
        Coupling of the initial state fermion to the scalar mediator.
    gsff : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    gsGG : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.

    Returns
    -------
    Returns cross section for :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}`.

    """
    if cme < 2 * mpi:
        return 0

    mat_elem_sqrd = (-2 * gsxx**2 * (4 * mx**2 - cme**2) *
                     (2 * gsGG * (2 * mpi**2 - cme**2) *
                      (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
                      (9 * vh + 8 * gsGG * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
                      (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
                       (9 * vh + 16 * gsGG * vs)))**2) / \
        ((ms**2 - cme**2)**2 *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs) ** 2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    p_i = sqrt(cme**2 / 4.0 - mx**2)
    p_f = sqrt(cme**2 / 4.0 - mpi**2)

    prefactor = 1.0 / (16. * pi * cme**2) * p_f / p_i

    return prefactor * amp_bethe_salpeter_pipi_to_pipi(sqrt(cme**2)) *\
        mat_elem_sqrd


def __dnde_xx_to_s_to_pipig(eng_gam, cme, mx, ms, gsxx, gsff, gsGG, vs):
    """
    Returns the gamma ray energy spectrum for two fermions annihilating into
    two charged pions and a photon.

    Parameters
    ----------
    eng_gam : double
        Gamma ray energy.
    cme : double
        Center of mass energy, or sqrt((ppip + ppim + pg)^2).
    mx : double
        Mass of the initial state fermion.
    ms : double
        Mass of the scalar mediator.
    gsxx : double
        Coupling of the initial state fermion to the scalar mediator.
    gsff : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    gsGG : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.

    Returns
    -------
    Returns gamma ray energy spectrum for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma`.

    """
    if cme < 2 * mpi:
        return 0

    s = E1_to_s(eng_gam, 0.0, cme)

    def mat_elem_sqrd(t):
        return __msqrd_xx_to_s_to_pipig(s, t, cme, mx, ms, gsxx,
                                        gsff, gsGG, vs)

    prefactor1 = phase_space_prefactor(cme)
    prefactor2 = 2 * cme / \
        __sigma_xx_to_s_to_pipi(cme, mx, ms, gsxx, gsff, gsGG, vs)
    prefactor3 = cross_section_prefactor(mx, mx, cme)

    prefactor = prefactor1 * prefactor2 * prefactor3

    int_val, _ = t_integral(s, 0.0, mpi, mpi, cme,
                            mat_elem_sqrd=mat_elem_sqrd)

    return prefactor * int_val


def dnde_xx_to_s_to_pipig(eng_gams, cme, mx, ms, gsxx, gsff, gsGG):
    """
    Returns the gamma ray energy spectrum for two fermions annihilating into
    two charged pions and a photon.

    Parameters
    ----------
    eng_gam : numpy.ndarray or double
        Gamma ray energy.
    cme : double
        Center of mass energy, or sqrt((ppip + ppim + pg)^2).
    mx : double
        Mass of the initial state fermion.
    ms : double
        Mass of the scalar mediator.
    gsxx : double
        Coupling of the initial state fermion to the scalar mediator.
    gsff : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    gsGG : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.

    Returns
    -------
    Returns gamma ray energy spectrum for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma` evaluated at the gamma ray
    energy(ies).

    """
    vs = 0.0  # vs_solver(gsff, gsGG, ms)

    if hasattr(eng_gams, '__len__'):
        return [__dnde_xx_to_s_to_pipig(eng_gam, cme, mx, ms, gsxx, gsff,
                                        gsGG, vs)
                for eng_gam in eng_gams]
    else:
        return __dnde_xx_to_s_to_pipig(eng_gams, cme, mx, ms, gsxx,
                                       gsff, gsGG, vs)


# #########################
""" NO FSI CALCULATIONS """
# #########################


def __sigma_xx_to_s_to_pipi_no_fsi(cme, mx, ms, gsxx, gsff, gsGG, vs):
    """
    Returns the cross section for two fermions annihilating into two
    charged pions **WITHOUT FSI**.
    """
    if cme < 2 * mpi:
        return 0

    mat_elem_sqrd = (-2 * gsxx**2 * (4 * mx**2 - cme**2) *
                     (2 * gsGG * (2 * mpi**2 - cme**2) *
                      (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
                      (9 * vh + 8 * gsGG * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
                      (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
                       (9 * vh + 16 * gsGG * vs)))**2) / \
        ((ms**2 - cme**2)**2 *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs) ** 2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    p_i = sqrt(cme**2 / 4.0 - mx**2)
    p_f = sqrt(cme**2 / 4.0 - mpi**2)

    prefactor = 1.0 / (16. * pi * cme**2) * p_f / p_i

    return prefactor * mat_elem_sqrd


def __xx_s_pipig_no_FSI_E(s, t, Q, mx, ms, gsxx, gsff, gsGG, vs):

    mrhoT = sqrt(mrho**2 - 2 * gsGG * vs / 9. / vh)

    return (complex(0., 0.037037037037037035) * sqrt(2.) * gsxx *
            sqrt(-4. * mx**2 + Q**2) * qe *
            (3. * fv * gsGG * gv * s * (mPI**2 - t) * (mPI**2 + Q**2 - s - t) *
             (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
             (9. * vh + 4. * gsGG * vs) -
             fpi**2 * mrhoT**2 *
             (mrho**2 - complex(0., 1.) * mrho * rho_width - s) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))))) / \
        (fpi**2 * mrhoT**2 * (-ms**2 + Q**2) *
         (mrho**2 - complex(0., 1.) * mrho * rho_width - s) *
         (mPI**2 - t) * vh *
         (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
         (9. * vh + 4. * gsGG * vs))


def __msqrd_xx_to_s_to_pipig_no_fsi(s, t, Q, mx, ms, gsxx, gsff, gsGG, vs):
    """Compute matrix element squared for xbar x -> s^* -> pi+ pi- g.

    Notes
    -----
    The matrix element for this process, M, is related to the form factor by
        |M|^2. = s Re[E(s,t,u) E^*(s,u,t)] - m_PI^2. |E(s,t,u) + E(s,u,t)|^2..
    """
    if Q < 2 * mpi:
        return 0

    u = Q**2 + 2. * mPI**2 - s - t

    E_t = __xx_s_pipig_no_FSI_E(s, t, Q, mx, ms, gsxx, gsff, gsGG, vs)
    E_u = __xx_s_pipig_no_FSI_E(s, u, Q, mx, ms, gsxx, gsff, gsGG, vs)

    return s * np.real(E_t * np.conj(E_u)) - mPI**2 * np.abs(E_t + E_u)**2


def __dnde_xx_to_s_to_pipig_no_fsi(eng_gam, cme, mx, ms, gsxx, gsff, gsGG, vs):
    """
    Returns the gamma ray energy spectrum for two fermions annihilating into
    two charged pions and a photon.
    """
    if cme < 2 * mpi:
        return 0

    s = E1_to_s(eng_gam, 0.0, cme)

    def mat_elem_sqrd(t):
        return __msqrd_xx_to_s_to_pipig_no_fsi(s, t, cme, mx, ms, gsxx,
                                               gsff, gsGG, vs)

    prefactor1 = phase_space_prefactor(cme)
    prefactor2 = 2 * cme / \
        __sigma_xx_to_s_to_pipi_no_fsi(cme, mx, ms, gsxx, gsff, gsGG, vs)
    prefactor3 = cross_section_prefactor(mx, mx, cme)

    prefactor = prefactor1 * prefactor2 * prefactor3

    int_val, _ = t_integral(s, 0.0, mpi, mpi, cme,
                            mat_elem_sqrd=mat_elem_sqrd)

    return prefactor * int_val


def dnde_xx_to_s_to_pipig_no_fsi(eng_gams, cme, mx, ms, gsxx, gsff, gsGG):
    """
    Returns the gamma ray energy spectrum for two fermions annihilating into
    two charged pions and a photon **WITHOUT FSI**.
    """
    vs = 0.0  # vs_solver(gsff, gsGG, ms)

    if hasattr(eng_gams, '__len__'):
        return [__dnde_xx_to_s_to_pipig_no_fsi(eng_gam, cme, mx, ms, gsxx,
                                               gsff, gsGG, vs)
                for eng_gam in eng_gams]
    else:
        return __dnde_xx_to_s_to_pipig_no_fsi(
            eng_gams, cme, mx, ms, gsxx, gsff, gsGG, vs)
