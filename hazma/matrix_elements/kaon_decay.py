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

    peDOTpn = __fv_dot_prod(pe, pn)
    ppDOTpe = __fv_dot_prod(pp, pe)
    ppDOTpn = __fv_dot_prod(pp, pn)

    mk0 = neutral_kaon_mass
    mp = charged_pion_mass
    me = charged_pion_mass

    mat_elem_sqrd = 2 * GF**2 * \
        (peDOTpn * (2 * me**2 - mk0**2 - 3 * mp**2 + 2 * (
            peDOTpn + ppDOTpe)) + 2 *
         (2 * me**2 + peDOTpn + 4 * ppDOTpe) * ppDOTpn) * Vus**2

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

    pk = np.array([Q, 0., 0., 0])

    pkDOTpn = __fv_dot_prod(pk, pn)
    pkDOTpp = __fv_dot_prod(pk, pp)
    pkDOTpe = __fv_dot_prod(pk, pe)
    pkDOTpg = __fv_dot_prod(pk, pg)
    peDOTpk = __fv_dot_prod(pe, pk)
    peDOTpg = __fv_dot_prod(pe, pg)
    peDOTpn = __fv_dot_prod(pe, pn)
    pnDOTpg = __fv_dot_prod(pn, pg)
    pnDOTpk = __fv_dot_prod(pn, pk)
    ppDOTpn = __fv_dot_prod(pp, pn)
    ppDOTpe = __fv_dot_prod(pp, pe)
    ppDOTpg = __fv_dot_prod(pp, pg)

    mp = charged_pion_mass
    mk0 = neutral_kaon_mass
    me = electron_mass

    mat_elem_sqrd =\
        (-8 * alpha_em * GF**2 * np.pi *
         (-(mp**4 * peDOTpg**2 * peDOTpn) +
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
          mp**2 * (-(me**2 * (peDOTpn + pnDOTpg) * ppDOTpg**2) -
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
