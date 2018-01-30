from ..parameters import vh, b0, alpha_em, fpi
from ..parameters import charged_pion_mass as mpi
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq

from ..field_theory_helper_functions.common_functions import minkowski_dot

import numpy as np
from scipy.optimize import newton

import os


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "unitarization",
                         "final_state_int_unitarizated.dat")

unitarizated_data = np.genfromtxt(DATA_PATH, delimiter=',')


def vs_eqn(vs, cffs, cggs, ms):
    RHS = (27 * b0 * (3 * cffs + 2 * cggs) * fpi**2 *
           (mdq + msq + muq) * vh) / \
        (ms**2 * (9 * vh + 9 * cffs * vs - 2 * cggs * vs) *
         (9 * vh + 8 * cggs * vs))

    return RHS - vs


def vs_solver(cffs, cggs, ms):
    if ms == 0:
        return 27.0 * vh * (3.0 * cffs + 2.0 * cggs) / \
            (16.0 * cggs * (2.0 * cggs - 9.0 * cffs))
    else:
        return newton(vs_eqn, 0.0, args=(cffs, cggs, ms))


def mass_s(cffs, cggs, ms, vs):

    ms_new_sqrd = ms**2 + 16.0 * b0 * fpi**2 * cggs * (mdq + msq + muq) * \
        (9.0 * cffs - 2.0 * cggs) / (8.0 * cggs * vs + 9.0 * vh) / \
        (9.0 * cffs * vs - 2.0 * cggs * vs + 9.0 * vh)

    return np.sqrt(ms_new_sqrd)


def unit_matrix_elem_sqrd(cme):
    """
    Returns the unitarized squared matrix element for :math:`\pi\pi\to\pi\pi`
    divided by the leading order, ununitarized squared matrix element for
    :math:`\pi\pi\to\pi\pi`.

    Parameters
    ----------
    cme : double
        Invariant mass of the two charged pions.

    Results
    -------
    __unit_matrix_elem_sqrd : double
        The unitarized matrix element for :math:`\pi\pi\to\pi\pi`, |t_u|^2,
        divided by the un-unitarized squared matrix element for
        :math:`\pi\pi\to\pi\pi`, |t|^2; |t_u|^2 / |t|^2.
    """
    t_mod_sqrd = np.interp(cme, unitarizated_data[0], unitarizated_data[1])
    additional_factor = (2 * cme**2 - mpi**2) / (32. * fpi**2 * np.pi)
    return t_mod_sqrd / additional_factor**2


def msqrd_xx_to_s_to_pipi(moms, mx, ms, cxxs, cffs, cggs):
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
    cxxs : double
        Coupling of the initial state fermion to the scalar mediator.
    cffs : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    cggs : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.

    Returns
    -------
    Returns matrix element squared for :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}`.
    """
    vs = vs_solver(cffs, cggs, ms)

    ms = mass_s(cffs, cggs, ms, vs)

    pp, pm = moms

    s = minkowski_dot(pp + pm, pp + pm)

    mat_elem_sqrd = (-2 * cxxs**2 * (4 * mx**2 - s) *
                     (2 * cggs * (2 * mpi**2 - s) *
                      (-9 * vh - 9 * cffs * vs + 2 * cggs * vs) *
                      (9 * vh + 8 * cggs * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * cggs * vs) *
                      (54 * cggs * vh - 32 * cggs**2 * vs + 9 * cffs *
                         (9 * vh + 16 * cggs * vs)))**2) / \
        ((ms**2 - s)**2 *
         (9 * vh + 9 * cffs * vs - 2 * cggs * vs) ** 2 *
         (9 * vh + 4 * cggs * vs)**2 * (9 * vh + 8 * cggs * vs)**2)

    return mat_elem_sqrd * unit_matrix_elem_sqrd(np.sqrt(s))


def msqrd_xx_to_s_to_pipi_no_fsi(moms, mx, ms, cxxs, cffs, cggs):
    """Returns the cross section for two fermions annihilating into two
    charged pions WITHOUT includeing final state interactions.

    Parameters
    ----------
    cme : double
        Center of mass energy.
    mx : double
        Mass of the initial state fermion.
    ms : double
        Mass of the scalar mediator.
    cxxs : double
        Coupling of the initial state fermion to the scalar mediator.
    cffs : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    cggs : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.

    Returns
    -------
    Returns matrix element squared for :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}`.
    """
    vs = vs_solver(cffs, cggs, ms)

    ms = mass_s(cffs, cggs, ms, vs)

    pp, pm = moms

    s = minkowski_dot(pp + pm, pp + pm)

    mat_elem_sqrd = (-2 * cxxs**2 * (4 * mx**2 - s) *
                     (2 * cggs * (2 * mpi**2 - s) *
                      (-9 * vh - 9 * cffs * vs + 2 * cggs * vs) *
                      (9 * vh + 8 * cggs * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * cggs * vs) *
                      (54 * cggs * vh - 32 * cggs**2 * vs + 9 * cffs *
                         (9 * vh + 16 * cggs * vs)))**2) / \
        ((ms**2 - s)**2 *
         (9 * vh + 9 * cffs * vs - 2 * cggs * vs) ** 2 *
         (9 * vh + 4 * cggs * vs)**2 * (9 * vh + 8 * cggs * vs)**2)

    return mat_elem_sqrd


def msqrd_xx_to_s_to_pipig(moms, mx, ms, cxxs, cffs, cggs):
    """Returns the squared matrix element for two fermions annihilating into two
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
    cxxs : double
        Coupling of the initial state fermion to the scalar mediator.
    cffs : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    cggs : double
        Coupling of the scalar to gluons.

    Returns
    -------
    Returns matrix element squared for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma`.
    """
    vs = vs_solver(cffs, cggs, ms)

    ms = mass_s(cffs, cggs, ms, vs)

    pp, pm, pg = moms

    Q = (np.sum(moms, 0))[0]

    s = minkowski_dot(pp + pm, pp + pm)
    t = minkowski_dot(pg + pm, pg + pm)
    u = minkowski_dot(pg + pp, pg + pp)

    e = np.sqrt(4 * np.pi * alpha_em)

    mat_elem_sqrd = (-16 * cxxs**2 *
                     (-4 * mx**2 + Q**2) *
                     (4 * mpi**6 - s * t * u + mpi**2 *
                      (t + u) * (s + t + u) - mpi**4 * (s + 4 * (t + u))) *
                     (2 * cggs * (4 * mpi**2 - s - t - u) *
                      (-9 * vh - 9 * cffs * vs + 2 * cggs * vs) *
                      (9 * vh + 8 * cggs * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * cggs * vs) *
                      (54 * cggs * vh - 32 * cggs**2 * vs + 9 * cffs *
                       (9 * vh + 16 * cggs * vs)))**2 * e**2) / \
        ((ms**2 - Q**2)**2 *
         (mpi**2 - t)**2 * (mpi**2 - u)**2 *
         (9 * vh + 9 * cffs * vs - 2 * cggs * vs)**2 *
         (9 * vh + 4 * cggs * vs)**2 * (9 * vh + 8 * cggs * vs)**2)

    pions_inv_mass = np.sqrt(minkowski_dot(pp + pm, pp + pm))

    return unit_matrix_elem_sqrd(pions_inv_mass) * mat_elem_sqrd


def msqrd_xx_to_s_to_pipig_no_fsi(moms, mx, ms, cxxs, cffs, cggs):
    """Returns the squared matrix element for two fermions annihilating into two
    charged pions and a photon WITHOUT including final state interactions.

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
    cxxs : double
        Coupling of the initial state fermion to the scalar mediator.
    cffs : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    cggs : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.

    Returns
    -------
    Returns matrix element squared for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma`.
    """
    vs = vs_solver(cffs, cggs, ms)

    ms = mass_s(cffs, cggs, ms, vs)

    pp, pm, pg = moms

    Q = (np.sum(moms, 0))[0]

    s = minkowski_dot(pp + pm, pp + pm)
    t = minkowski_dot(pg + pm, pg + pm)
    u = minkowski_dot(pg + pp, pg + pp)

    e = np.sqrt(4 * np.pi * alpha_em)

    mat_elem_sqrd = (-16 * cxxs**2 *
                     (-4 * mx**2 + Q**2) *
                     (4 * mpi**6 - s * t * u + mpi**2 *
                      (t + u) * (s + t + u) - mpi**4 * (s + 4 * (t + u))) *
                     (2 * cggs * (4 * mpi**2 - s - t - u) *
                      (-9 * vh - 9 * cffs * vs + 2 * cggs * vs) *
                      (9 * vh + 8 * cggs * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * cggs * vs) *
                      (54 * cggs * vh - 32 * cggs**2 * vs + 9 * cffs *
                       (9 * vh + 16 * cggs * vs)))**2 * e**2) / \
        ((ms**2 - Q**2)**2 *
         (mpi**2 - t)**2 * (mpi**2 - u)**2 *
         (9 * vh + 9 * cffs * vs - 2 * cggs * vs)**2 *
         (9 * vh + 4 * cggs * vs)**2 * (9 * vh + 8 * cggs * vs)**2)

    return mat_elem_sqrd
