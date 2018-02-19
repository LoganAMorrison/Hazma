"""Module containing squared matrix elements from scalar mediator models.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

import numpy as np
from ..parameters import vh, qe, b0
from ..parameters import charged_kaon_mass as mk
from ..parameters import charged_pion_mass as mpi
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq

from ..unitarization import unit_matrix_elem_sqrd

from ..field_theory_helper_functions.common_functions import minkowski_dot

# from ..parameters import neutral_pion_mass as mpi0
# from ..parameters import neutral_kaon_mass as mk0
# from ..parameters import eta_mass as meta


# ###################################
# ####### NON-RADIATIVE #############
# ###################################

def msqrd_xx_to_s_to_ff(moms, mx, mf, ms, gsxx, gsff):
    """Returns the spin-averaged, squared matrix element for a pair of fermions,
    *x*, annihilating into a pair of fermions, *f*, through a scalar mediator
    in the s-channel.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    mx : float
        Mass of incoming fermions.
    mf : float
        Mass of final state fermions.
    ms : float
        Mass of scalar mediator.
    gsxx : float
        Coupling of initial state fermions with the scalar mediator.
    gsff : float
        Coupling of final state fermions with the scalar mediator.

    Returns
    -------
    mat_elem_sqrd : float
        Spin averaged, squared matrix element for x + x -> S^* -> f + f.
    """
    p3 = moms[0]
    p4 = moms[1]

    Q = p3[0] + p4[0]

    return gsff**2 * gsxx**2 * mf**2 * (Q**2 - 4.0 * mf**2) * \
        (Q**2 - 4.0 * mx**2) / (ms**2 - Q**2)**2 / vh**2


# ###############################
# ####### RADIATIVE #############
# ###############################


def msqrd_xx_to_s_to_ffg(moms, mx, mf, ms, qf, gsxx, gsff):
    """Returns the spin-averaged, squared matrix element for a pair of fermions,
    *x*, annihilating into a pair of fermions, *f*, and a photon through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    mx : float
        Mass of incoming fermions.
    mf : float
        Mass of final state fermions.
    ms : float
        Mass of scalar mediator.
    gsxx : float
        Coupling of initial state fermions with the scalar mediator.
    gsff : float
        Coupling of final state fermions with the scalar mediator.

    Returns
    -------
    mat_elem_sqrd : float
        Spin averaged, squared matrix element for x + x -> s* -> f + f + g.
    """
    pf1, pf2, pg = moms

    cme = pf1[0] + pf2[0] + pg[0]

    s = minkowski_dot(pf1 + pf2, pf1 + pf2)
    t = minkowski_dot(pg + pf2, pg + pf2)

    mat_elem = (2 * gsff**2 * gsxx**2 * mf**2 *
                (cme**2 - 4 * mx**2) * qf * qe**2 *
                (cme**6 * (-3 * mf**2 + t) + cme**4 *
                 (7 * mf**4 - t * (s + t) + mf**2 * (5 * s + 2 * t)) +
                 cme**2 * s * (-8 * mf**4 + s * t - mf**2 * (3 * s + 8 * t)) +
                 s * (8 * mf**6 - s * t * (s + t) - mf**4 * (s + 16 * t) +
                      mf**2 * (s**2 + 10 * s * t + 8 * t**2)))) / \
        ((cme**2 - ms**2)**2 * (mf**2 - t)**2 *
         (cme**2 + mf**2 - s - t)**2 * vh**2)

    return mat_elem


def msqrd_xx_to_s_to_kkg(moms, mx, ms, gsxx, gsff, gsGG, vs):
    """Returns the spin-averaged, squared matrix element for a pair of fermions,
    *x*, annihilating into a pair of charged kaons, and a photon through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    mx : float
        Mass of incoming fermions.
    ms : float
        Mass of scalar mediator.
    gsxx : float
        Coupling of initial state fermions with the scalar mediator.
    gsff : float
        Coupling of final state fermions with the scalar mediator.
    gsGG : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.


    Returns
    -------
    mat_elem_sqrd : float
        Spin averaged, squared matrix element for x + x -> s* -> k+ + k- + g.
    """
    pf1, pf2, pg = moms

    cme = pf1[0] + pf2[0] + pg[0]

    s = minkowski_dot(pf1 + pf2, pf1 + pf2)
    t = minkowski_dot(pg + pf2, pg + pf2)

    mat_elem = (-2 * gsxx**2 * (cme - 2 * mx) * (cme + 2 * mx) * qe**2 *
                (cme**4 * mk**2 - cme**2 * s * (mk**2 + t) +
                 s * (mk**4 - 2 * mk**2 * t + t * (s + t))) *
                (729 * b0 * gsff * (msq + muq) * vh**2 +
                 64 * gsGG**3 * (mk**2 - 2 * b0 * (msq + muq)) * vs**2 +
                 2 * cme**2 * gsGG *
                 (9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
                 (9 * vh + 8 * gsGG * vs) -
                 72 * gsGG**2 * vs * (b0 * (msq + muq) * (vh - 8 * gsff * vs) +
                                      mk**2 * (3 * vh + 4 * gsff * vs)) -
                 162 * gsGG * vh *
                 (2 * mk**2 * (vh + gsff * vs) - b0 * (msq + muq) *
                  (3 * vh + 10 * gsff * vs)))**2) / \
        ((cme**2 - ms**2)**2 * (mk**2 - t)**2 * (cme**2 + mk**2 - s - t)**2 *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 *
         (9 * vh + 8 * gsGG * vs)**2)

    return mat_elem


def msqrd_xx_to_s_to_pipig(moms, mx, ms, gsxx, gsff, gsGG, vs):
    """Returns the spin-averaged, squared matrix element for a pair of fermions,
    *x*, annihilating into a pair of charged pion, and a photon through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    mx : float
        Mass of incoming fermions.
    ms : float
        Mass of scalar mediator.
    gsxx : float
        Coupling of initial state fermions with the scalar mediator.
    gsff : float
        Coupling of final state fermions with the scalar mediator.
    gsGG : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.


    Returns
    -------
    mat_elem_sqrd : float
        Spin averaged, squared matrix element for x + x -> s* -> pi+ + pi- + g.
    """
    pf1, pf2, pg = moms

    cme = pf1[0] + pf2[0] + pg[0]

    s = minkowski_dot(pf1 + pf2, pf1 + pf2)
    t = minkowski_dot(pg + pf2, pg + pf2)

    mat_elem = (-2 * gsxx**2 * (cme - 2 * mx) *
                (cme + 2 * mx) * qe**2 *
                (cme**4 * mpi**2 - cme**2 * s * (mpi**2 + t) +
                 s * (mpi**4 - 2 * mpi**2 * t + t * (s + t))) *
                (2 * cme**2 * gsGG *
                 (9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
                 (9 * vh + 8 * gsGG * vs) +
                 4 * gsGG * mpi**2 *
                 (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
                 (9 * vh + 8 * gsGG * vs) +
                 b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
                 (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
                  (9 * vh + 16 * gsGG * vs)))**2) / \
        ((cme**2 - ms**2)**2 * (mpi**2 - t)**2 * (cme**2 + mpi**2 - s - t)**2 *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 *
         (9 * vh + 8 * gsGG * vs)**2)

    return mat_elem * unit_matrix_elem_sqrd(np.sqrt(s))
