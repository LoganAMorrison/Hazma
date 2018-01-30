"""Module containing squared matrix elements.

@author - Logan Morrison and Adam Coogan
@date - December 2017

TODO: msqrd_kl_to_pienug is returning negative squared matrix elements. FIX!!
"""

import warnings
import numpy as np

from ..parameters import alpha_em, GF, Vus
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import charged_pion_mass as mpi
from ..parameters import electron_mass as me
from ..parameters import muon_mass as mmu

from ..field_theory_helper_functions.common_functions import minkowski_dot


def msqrd_kl_to_pienu(moms):
    """
    Returns matrix element squared for kl -> pi  + e  + nu.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.

    Returns
    -------
    mat_elem_sqrd : float
        Squared matrix element for kl -> pi  + e  + nu.
    """
    pp, pe, pn = moms

    pk = np.sum(moms, 0)

    # s, t, and u are defined as:
    # s = (p1 + p2)^2
    s = minkowski_dot(pk - pp, pk - pp)
    # t = (p1 + p3)^2
    t = minkowski_dot(pk - pe, pk - pe)
    # u = (p1 + p4)^2
    u = minkowski_dot(pk - pn, pk - pn)
    # where p1 = pk, p2 = -pp, p3 = -pe and p4 = -pn

    return GF**2 * (mk0**4 + (mpi**2 - s)**2 +
                    me**2 * (2 * (mk0**2 + mpi**2) - s) -
                    2 * mk0**2 * (mpi**2 + s) - (t - u)**2) * Vus**2


def msqrd_kl_to_pienug(moms):
    """
    Returns matrix element squared for kl -> pi  + e  + nu + gam.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.

    Returns
    -------
    mat_elem_sqrd : float
        Squared matrix element for kl -> pi  + e  + nu + gam.
    """
    pp, pe, pn, pg = moms

    pk = np.sum(moms, 0)

    pkDOTpn = minkowski_dot(pk, pn)
    pkDOTpp = minkowski_dot(pk, pp)
    pgDOTpk = minkowski_dot(pk, pg)
    peDOTpk = minkowski_dot(pe, pk)
    peDOTpg = minkowski_dot(pe, pg)
    peDOTpn = minkowski_dot(pe, pn)
    pgDOTpn = minkowski_dot(pn, pg)
    pnDOTpp = minkowski_dot(pp, pn)
    peDOTpp = minkowski_dot(pp, pe)
    pgDOTpp = minkowski_dot(pp, pg)

    mat_elem_sqrd =\
        (8 * alpha_em * GF**2 * np.pi *
         (-2 * mpi**2 * peDOTpg**3 * (pgDOTpn + pkDOTpn + pnDOTpp) +
          me**2 * pgDOTpp**2 *
          (peDOTpn * (mk0**2 + mpi**2 + 2 * pkDOTpp) +
           pgDOTpn * (mk0**2 + mpi**2 + 2 * pkDOTpp) -
           2 * (peDOTpk + peDOTpp + pgDOTpk + pgDOTpp) * (pkDOTpn + pnDOTpp)) +
          peDOTpg**2 *
          (-2 * mpi**2 * peDOTpp * pgDOTpn - 2 * mpi**2 * pgDOTpn * pgDOTpp -
           2 * mpi**2 * peDOTpp * pkDOTpn - 2 * mpi**2 * pgDOTpp * pkDOTpn +
           2 * peDOTpp * pgDOTpp * pkDOTpn - 2 * pgDOTpn * pgDOTpp * pkDOTpp -
           2 * pgDOTpp * pkDOTpn * pkDOTpp + mpi**2 * peDOTpn *
           (mk0**2 + mpi**2 + 2 * pgDOTpk + 2 * pgDOTpp + 2 * pkDOTpp) -
           2 * mpi**2 * peDOTpp * pnDOTpp + mk0**2 * pgDOTpp * pnDOTpp -
           mpi**2 * pgDOTpp * pnDOTpp +
           2 * peDOTpp * pgDOTpp * pnDOTpp + 2 * pgDOTpk * pgDOTpp * pnDOTpp +
           2 * pgDOTpp**2 * pnDOTpp -
           2 * mpi**2 * peDOTpk * (pgDOTpn + pkDOTpn + pnDOTpp)) -
          peDOTpg * pgDOTpp *
          (peDOTpn *
           (2 * peDOTpp *
            (mk0**2 + mpi**2 + pgDOTpk + pgDOTpp + 2 * pkDOTpp) +
            pgDOTpp * (mk0**2 + mpi**2 +
                       2 * pgDOTpk + 2 * pgDOTpp + 2 * pkDOTpp)) -
           2 * peDOTpp**2 * (pgDOTpn + 2 * (pkDOTpn + pnDOTpp)) - pgDOTpp *
           (-(pgDOTpn * (mk0**2 + mpi**2 - 2 * peDOTpk + 2 * pkDOTpp)) +
            2 * (peDOTpk + pgDOTpk + pgDOTpp) * (pkDOTpn + pnDOTpp)) +
           peDOTpp * (pgDOTpn * (mk0**2 + mpi**2 - 2 * peDOTpk -
                                 2 * pgDOTpp + 2 * pkDOTpp) -
                      2 * (2 * peDOTpk + pgDOTpk + 2 * pgDOTpp) *
                      (pkDOTpn + pnDOTpp)))) * Vus**2) / \
        (peDOTpg**2 * pgDOTpp**2)

    return mat_elem_sqrd


def msqrd_kl_to_pimunu(moms):
    """
    Matrix element squared for kl -> pi  + mu  + nu.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    """
    pp, pmu, pn = moms

    pk = np.sum(moms, 0)

    # s, t, and u are defined as:
    # s = (p1 + p2)^2
    s = minkowski_dot(pk - pp, pk - pp)
    # t = (p1 + p3)^2
    t = minkowski_dot(pk - pmu, pk - pmu)
    # u = (p1 + p4)^2
    u = minkowski_dot(pk - pn, pk - pn)
    # where p1 = pk, p2 = -pp, p3 = -pmu and p4 = -pn

    return GF**2 * (mk0**4 + (mpi**2 - s)**2 +
                    mmu**2 * (2 * (mk0**2 + mpi**2) - s) -
                    2 * mk0**2 * (mpi**2 + s) - (t - u)**2) * Vus**2


def msqrd_kl_to_pi0pi0pi0(moms):
    """
    Matrix element squared for kl -> pi0 + pi0  + pi0.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    """
    warnings.warn("""kl -> pi0 + pi0  + pi0 matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0


def msqrd_kl_to_pipipi0(moms):
    """
    Matrix element squared for kl -> pi  + pi  + pi0.

    Parameters
    ----------
    moms : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    """
    warnings.warn("""kl -> pi  + pi  + pi0 matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0
