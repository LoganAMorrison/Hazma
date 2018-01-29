"""Module containing squared matrix elements.

@author - Logan Morrison and Adam Coogan
@date - December 2017

TODO: msqrd_kl_to_pienug is returning negative squared matrix elements. FIX!!
"""

import warnings
import numpiy as np

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
    # t = (p1 + p3)^2
    # u = (p1 + p4)^2
    # where p1 = pk, p2 = -pp, p3 = -pe and p4 = -pn

    s = minkowski_dot(pk - pp, pk - pp)
    t = minkowski_dot(pk - pe, pk - pe)
    u = minkowski_dot(pk - pn, pk - pn)

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
    pp = moms[0]
    pe = moms[1]
    pn = moms[2]
    pg = moms[3]

    Q = pp[0] + pe[0] + pn[0] + pg[0]

    pk = np.array([Q, 0., 0., 0])

    pkDOTpn = minkowski_dot(pk, pn)
    pkDOTpp = minkowski_dot(pk, pp)
    pkDOTpe = minkowski_dot(pk, pe)
    pkDOTpg = minkowski_dot(pk, pg)
    peDOTpk = minkowski_dot(pe, pk)
    peDOTpg = minkowski_dot(pe, pg)
    peDOTpn = minkowski_dot(pe, pn)
    pnDOTpg = minkowski_dot(pn, pg)
    pnDOTpk = minkowski_dot(pn, pk)
    ppDOTpn = minkowski_dot(pp, pn)
    ppDOTpe = minkowski_dot(pp, pe)
    ppDOTpg = minkowski_dot(pp, pg)

    mat_elem_sqrd =\
        (-8 * alpha_em * GF**2 * np.pi *
         (-(mpi**4 * peDOTpg**2 * peDOTpn) +
          ppDOTpg *
          (-(mk0**2 * (me**2 * (peDOTpn + pnDOTpg) * ppDOTpg +
                       peDOTpg *
                       (2 * peDOTpn * ppDOTpe + pnDOTpg * ppDOTpe +
                        peDOTpn * ppDOTpg - pnDOTpg * ppDOTpg) + peDOTpg**2 *
                       (peDOTpn - ppDOTpn))) + peDOTpg**3 *
           (3 * pkDOTpn + 2 * pnDOTpg + 3 * ppDOTpn) + peDOTpg**2 *
           (2 * pkDOTpe * pkDOTpn - 2 * pkDOTpn * pkDOTpp + 3 *
            pkDOTpe * pnDOTpg - 2 * pkDOTpp * pnDOTpg + 2 * pkDOTpe *
            pnDOTpk + 6 * pkDOTpn * ppDOTpe + 3 * pnDOTpg * ppDOTpe -
            peDOTpn * (3 * pkDOTpg + 4 * pkDOTpp + 3 * ppDOTpg) + 2 *
            (2 * pkDOTpe + pkDOTpg + 3 * ppDOTpe + ppDOTpg) *
            ppDOTpn) +
           2 * peDOTpg *
           (2 * pkDOTpe * pkDOTpn * ppDOTpe + pkDOTpg *
            pkDOTpn * ppDOTpe + pkDOTpe * pnDOTpg * ppDOTpe -
            pkDOTpp * pnDOTpg * ppDOTpe + 2 * pkDOTpn * ppDOTpe**2 +
            pnDOTpg * ppDOTpe**2 + pkDOTpe * pkDOTpn * ppDOTpg -
            pkDOTpg * pkDOTpn * ppDOTpg +
            pkDOTpe * pnDOTpg * ppDOTpg +
            pkDOTpp * pnDOTpg * ppDOTpg +
            2 * pkDOTpn * ppDOTpe * ppDOTpg + pnDOTpg * ppDOTpe *
            ppDOTpg - pkDOTpn * ppDOTpg**2 -
            peDOTpn * ((pkDOTpg + 2 * pkDOTpp) * ppDOTpe +
                       (pkDOTpg + pkDOTpp + ppDOTpe) * ppDOTpg +
                       ppDOTpg**2) +
            (ppDOTpe * (2 * pkDOTpe + pkDOTpg + 2 * ppDOTpe) +
                       (pkDOTpe - pkDOTpg + 2 * ppDOTpe) * ppDOTpg -
             ppDOTpg**2) * ppDOTpn) +
           2 * me**2 * ppDOTpg *
           (-(pkDOTpp * (peDOTpn + pnDOTpg)) +
            (pkDOTpe + pkDOTpg + ppDOTpe + ppDOTpg) *
            (pkDOTpn + ppDOTpn))) +
          mpi**2 * (-(me**2 * (peDOTpn + pnDOTpg) * ppDOTpg**2) -
                    peDOTpg * ppDOTpg *
                    (peDOTpk * peDOTpn + pnDOTpg * (ppDOTpe - ppDOTpg) +
                     peDOTpn * (-pkDOTpe + 2 * ppDOTpe + ppDOTpg)) + 2 *
                    peDOTpg**3 * (pkDOTpn + pnDOTpg + ppDOTpn) +
                    peDOTpg**2 *
                    (-2 * peDOTpn * (pkDOTpg + pkDOTpp + 2 * ppDOTpg) -
                     ppDOTpg * (2 * (pkDOTpn + pnDOTpg) + ppDOTpn) +
                     2 * (ppDOTpe * (pkDOTpn + pnDOTpg + ppDOTpn) + pkDOTpe *
                          (pnDOTpg + pnDOTpk + ppDOTpn))))) * Vus**2) \
        / (peDOTpg**2 * ppDOTpg**2)

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
    # t = (p1 + p3)^2
    # u = (p1 + p4)^2
    # where p1 = pk, p2 = -pp, p3 = -pmu and p4 = -pn

    s = minkowski_dot(pk - pp, pk - pp)
    t = minkowski_dot(pk - pmu, pk - pmu)
    u = minkowski_dot(pk - pn, pk - pn)

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
