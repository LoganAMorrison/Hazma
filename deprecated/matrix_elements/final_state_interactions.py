from ..parameters import vh, b0, alpha_em, fpi
from ..parameters import charged_pion_mass as mpi
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq
from ..unitarization import unit_matrix_elem_sqrd

from ..field_theory_helper_functions.common_functions import minkowski_dot

import numpy as np
from scipy.optimize import newton


def vs_eqn(vs, gsff, gsGG, ms):
    RHS = (27 * b0 * (3 * gsff + 2 * gsGG) * fpi**2 *
           (mdq + msq + muq) * vh) / \
        (ms**2 * (9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))

    return RHS - vs


def vs_solver(gsff, gsGG, ms):
    if ms == 0:
        return 27.0 * vh * (3.0 * gsff + 2.0 * gsGG) / \
            (16.0 * gsGG * (2.0 * gsGG - 9.0 * gsff))
    else:
        return newton(vs_eqn, 0.0, args=(gsff, gsGG, ms))


def mass_s(gsff, gsGG, ms, vs):

    ms_new_sqrd = ms**2 + 16.0 * b0 * fpi**2 * gsGG * (mdq + msq + muq) * \
        (9.0 * gsff - 2.0 * gsGG) / (8.0 * gsGG * vs + 9.0 * vh) / \
        (9.0 * gsff * vs - 2.0 * gsGG * vs + 9.0 * vh)

    return np.sqrt(ms_new_sqrd)


def msqrd_xx_to_s_to_pipi(moms, mx, ms, gsxx, gsff, gsGG):
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
    Returns matrix element squared for :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}`.
    """
    vs = vs_solver(gsff, gsGG, ms)

    ms = mass_s(gsff, gsGG, ms, vs)

    pp, pm = moms

    s = minkowski_dot(pp + pm, pp + pm)

    mat_elem_sqrd = (-2 * gsxx**2 * (4 * mx**2 - s) *
                     (2 * gsGG * (2 * mpi**2 - s) *
                      (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
                      (9 * vh + 8 * gsGG * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
                      (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
                         (9 * vh + 16 * gsGG * vs)))**2) / \
        ((ms**2 - s)**2 *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs) ** 2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    return mat_elem_sqrd * unit_matrix_elem_sqrd(np.sqrt(s))


def msqrd_xx_to_s_to_pipi_no_fsi(moms, mx, ms, gsxx, gsff, gsGG):
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
    Returns matrix element squared for :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}`.
    """
    vs = vs_solver(gsff, gsGG, ms)

    ms = mass_s(gsff, gsGG, ms, vs)

    pp, pm = moms

    s = minkowski_dot(pp + pm, pp + pm)

    mat_elem_sqrd = (-2 * gsxx**2 * (4 * mx**2 - s) *
                     (2 * gsGG * (2 * mpi**2 - s) *
                      (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
                      (9 * vh + 8 * gsGG * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
                      (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
                         (9 * vh + 16 * gsGG * vs)))**2) / \
        ((ms**2 - s)**2 *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs) ** 2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    return mat_elem_sqrd


def msqrd_xx_to_s_to_pipig(moms, mx, ms, gsxx, gsff, gsGG):
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
    gsxx : double
        Coupling of the initial state fermion to the scalar mediator.
    gsff : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    gsGG : double
        Coupling of the scalar to gluons.

    Returns
    -------
    Returns matrix element squared for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma`.
    """
    vs = vs_solver(gsff, gsGG, ms)

    ms = mass_s(gsff, gsGG, ms, vs)

    pp, pm, pg = moms

    Q = (np.sum(moms, 0))[0]

    s = minkowski_dot(pp + pm, pp + pm)
    t = minkowski_dot(pg + pm, pg + pm)
    u = minkowski_dot(pg + pp, pg + pp)

    e = np.sqrt(4 * np.pi * alpha_em)

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

    pions_inv_mass = np.sqrt(minkowski_dot(pp + pm, pp + pm))

    return unit_matrix_elem_sqrd(pions_inv_mass) * mat_elem_sqrd


def msqrd_xx_to_s_to_pipig_no_fsi(moms, mx, ms, gsxx, gsff, gsGG):
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
    """
    vs = vs_solver(gsff, gsGG, ms)

    ms = mass_s(gsff, gsGG, ms, vs)

    pp, pm, pg = moms

    Q = (np.sum(moms, 0))[0]

    s = minkowski_dot(pp + pm, pp + pm)
    t = minkowski_dot(pg + pm, pg + pm)
    u = minkowski_dot(pg + pp, pg + pp)

    e = np.sqrt(4 * np.pi * alpha_em)

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

    return mat_elem_sqrd
