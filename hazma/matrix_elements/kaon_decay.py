"""Module containing squared matrix elements.

@author - Logan Morrison and Adam Coogan
@date - December 2017

TODO: correct matrix elements by removing decay width factors.
"""

import warnings
import numpy as np
from ..parameters import alpha_em, GF, Vus
from ..parameters import charged_pion_mass, electron_mass, neutral_kaon_mass


def __fv_dot_prod(fv1, fv2):
    return fv1[0] * fv2[0] - np.dot(fv1[1:4], fv2[1:4])


def kl_to_pienu(kList):
    """
    """
    pp = kList[0]
    pe = kList[1]
    pn = kList[2]

    Q = pp[0] + pe[0] + pn[0]

    pk = np.array([Q, 0., 0., 0.])

    peDOTpn = __fv_dot_prod(pe, pn)
    peDOTpk = __fv_dot_prod(pe, pk)
    pkDOTpn = __fv_dot_prod(pk, pn)
    peDOTpp = __fv_dot_prod(pe, pp)
    pkDOTpp = __fv_dot_prod(pk, pp)
    pnDOTpp = __fv_dot_prod(pn, pp)

    mk = neutral_kaon_mass
    mp = charged_pion_mass

    mat_elem_sqrd = -2 * GF**2 * \
        (peDOTpn * (mk**2 + mp**2 + 2 * pkDOTpp) -
         2 * (peDOTpk + peDOTpp) * (pkDOTpn + pnDOTpp)) * Vus**2

    return mat_elem_sqrd


def kl_to_pienug(kList):
    """
    Matrix element squared for kl -> pi  + e  + nu + gam.
    """
    pp = kList[0]
    pe = kList[1]
    pn = kList[2]
    pg = kList[3]

    Q = pp[0] + pe[0] + pn[0] + pg[0]

    pk = np.array([Q, 0., 0., 0.])

    pkDOTpn = __fv_dot_prod(pk, pn)
    pkDOTpp = __fv_dot_prod(pk, pp)
    peDOTpk = __fv_dot_prod(pe, pk)
    peDOTpg = __fv_dot_prod(pe, pg)
    peDOTpn = __fv_dot_prod(pe, pn)
    peDOTpp = __fv_dot_prod(pe, pp)
    pgDOTpp = __fv_dot_prod(pg, pp)
    pgDOTpk = __fv_dot_prod(pg, pk)
    pgDOTpn = __fv_dot_prod(pg, pn)
    pnDOTpp = __fv_dot_prod(pn, pp)

    e = np.sqrt(4 * np.pi * alpha_em)
    mp = charged_pion_mass
    mk = neutral_kaon_mass
    me = electron_mass

    mat_elem_sqrd = (2 * e**2 * GF**2 *
                     (mp**4 * peDOTpg**2 * peDOTpn + mk**2 *
                      (mp**2 * peDOTpg**2 * peDOTpn + pgDOTpp *
                       (me**2 * (peDOTpn + pgDOTpn) * pgDOTpp - peDOTpg *
                        (peDOTpp * (2 * peDOTpn + pgDOTpn) +
                           (peDOTpn + pgDOTpn) * pgDOTpp) + peDOTpg**2 *
                        pnDOTpp)) - mp**2 *
                      (-(me**2 * (peDOTpn + pgDOTpn) * pgDOTpp**2) +
                       peDOTpg * pgDOTpp *
                       (pgDOTpn * (peDOTpp + pgDOTpp) +
                        peDOTpn * (2 * peDOTpp + pgDOTpp)) +
                          2 * peDOTpg**3 * (pgDOTpn + pkDOTpn + pnDOTpp) +
                          peDOTpg**2 *
                          (2 * (peDOTpk + peDOTpp + pgDOTpp) *
                           (pgDOTpn + pkDOTpn) - 2 *
                           peDOTpn * (pgDOTpk + pgDOTpp + pkDOTpp) +
                           (2 * (peDOTpk + peDOTpp) + pgDOTpp) * pnDOTpp)) +
                      pgDOTpp *
                      (-(peDOTpg**3 * (2 * pgDOTpn + pkDOTpn + pnDOTpp)) +
                       2 * peDOTpg *
                          (peDOTpk * peDOTpp * pgDOTpn + peDOTpp**2 * pgDOTpn +
                           peDOTpk * pgDOTpn * pgDOTpp + peDOTpp * pgDOTpn *
                           pgDOTpp - peDOTpn * (peDOTpp + pgDOTpp) *
                           (pgDOTpk + pgDOTpp) +
                           2 * peDOTpk * peDOTpp * pkDOTpn +
                           2 * peDOTpp**2 * pkDOTpn +
                           peDOTpp * pgDOTpk * pkDOTpn + peDOTpk * pgDOTpp *
                           pkDOTpn +
                           2 * peDOTpp * pgDOTpp * pkDOTpn +
                           pgDOTpk * pgDOTpp * pkDOTpn +
                           pgDOTpp**2 * pkDOTpn - peDOTpp * pgDOTpn * pkDOTpp -
                           pgDOTpn * pgDOTpp * pkDOTpp - peDOTpn *
                           (2 * peDOTpp + pgDOTpp) * pkDOTpp +
                           (peDOTpp * (2 * (peDOTpk + peDOTpp) + pgDOTpk) +
                            (peDOTpk + 2 * peDOTpp + pgDOTpk) *
                            pgDOTpp + pgDOTpp**2) *
                           pnDOTpp) - 2 * me**2 * pgDOTpp *
                          (-((peDOTpn + pgDOTpn) * pkDOTpp) +
                           (peDOTpk + peDOTpp + pgDOTpk + pgDOTpp) *
                           (pkDOTpn + pnDOTpp)) +
                          peDOTpg**2 *
                          (-(peDOTpk * pgDOTpn) - peDOTpp * pgDOTpn - 2 *
                           (pgDOTpn + pkDOTpn) * pkDOTpp + 2 * peDOTpp *
                           (pkDOTpn + pnDOTpp) + (pgDOTpk + pgDOTpp) *
                           (peDOTpn + 2 * pnDOTpp)))) * Vus**2) / \
        (peDOTpg**2 * pgDOTpp**2)

    return mat_elem_sqrd


def kl_to_pimunu(kList):
    """
    Matrix element squared for kl -> pi  + mu  + nu.
    """
    warnings.warn("""kl -> pi  + mu  + nu matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0


def kl_to_pi0pi0pi0(kList):
    """
    Matrix element squared for kl -> pi0 + pi0  + pi0.
    """
    warnings.warn("""kl -> pi0 + pi0  + pi0 matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0


def kl_to_pipipi0(kList):
    """
    Matrix element squared for kl -> pi  + pi  + pi0.
    """
    warnings.warn("""kl -> pi  + pi  + pi0 matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0
