from hazma.field_theory_helper_functions.common_functions import \
    minkowski_dot as MDot

from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import b0, vh, fpi, qe

import numpy as np


def msqrd_xx_to_p_to_pm0(momenta, params):
    """
    Returns the squared matrix element for dark matter annihilating into
    two charged pions and a netural pion through a scalar mediator.
    This function is used for RAMBO.

    Parameters
    ----------
    momenta : numpy.array
        List of momenta of the final state particles. The first momentum is
        for the positively charged pion, the second is for the negatively
        charged pion and the third is for the neutral pion.
    params : PseudoScalarMediatorParameters or PseudoScalarMediator object
        Parameter object for the pseudo-scalar mediator theory.

    Returns
    -------
    msqrd : float
        The squared matrix element for a given set of momenta.
    """
    p1 = momenta[0]
    p2 = momenta[1]
    p3 = momenta[2]
    P = sum(momenta)

    mx = params.mx
    mp = params.mp
    gpxx = params.gpxx
    gpuu = params.gpuu
    gpdd = params.gpdd
    gpGG = params.gpGG
    beta = params.beta
    widthp = params.width_p

    pxmag = np.sqrt(P[0]**2 / 4. - mx**2)

    px = np.array([P[0] / 2., 0., 0., pxmag])
    pxbar = np.array([P[0] / 2., 0., 0., -pxmag])

    """ Order beta^0 """
    beta0 = (-(b0**2 * gpxx**2 * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)**2 *
               (mx**2 + MDot(px, pxbar))) /
             (9. * fpi**2 * vh**2 *
              (mp**2 - 2 * mpi**2 - mpi0**2 +
               1.0j * mp * widthp - 2 * MDot(p1, p2) -
               2 * MDot(p1, p3) - 2 * MDot(p2, p3)) *
              (-mp**2 + 2 * mpi**2 + mpi0**2 + 1.0j * mp * widthp +
               2 * MDot(p1, p2) + 2 * MDot(p1, p3) + 2 * MDot(p2, p3))))

    """ Order beta^1 """
    beta1 = ((b0 * beta * gpxx**2 * (mp - mpi0) * (mp + mpi0) *
              (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) *
              (2 * (mpi - mpi0) *
               (mpi + mpi0) + b0 * (mdq + muq) + 4 * MDot(p1, p2) -
               2 * MDot(p1, p3) - 2 * MDot(p2, p3)) *
              (mx**2 + MDot(px, pxbar))) /
             (9. * fpi**3 * vh *
              (mp**2 - 2 * mpi**2 - mpi0**2 + 1.0j * mp * widthp -
               2 * MDot(p1, p2) - 2 * MDot(p1, p3) - 2 * MDot(p2, p3)) *
              (mpi**2 + MDot(p1, p2) + MDot(p1, p3) + MDot(p2, p3)) *
              (-mp**2 + 2 * mpi**2 + mpi0**2 + 1.0j * mp * widthp +
               2 * MDot(p1, p2) + 2 * MDot(p1, p3) + 2 * MDot(p2, p3))))

    """ Order beta^2 """

    beta2 = 0.

    msqrd = beta0 + beta1 + beta2

    assert msqrd.imag == 0.0

    return msqrd.real


def msqrd_xx_to_p_to_pm0g(momenta, params):
    """
    Returns the squared matrix element for dark matter annihilating into
    two charged pions, a netural pion and a photon through a scalar mediator.
    This function is used for RAMBO.

    Parameters
    ----------
    momenta : numpy.array
        List of momenta of the final state particles. The first momentum is
        for the positively charged pion, the second is for the negatively
        charged pion, the third is for the neutral pion and the last is
        for the photon.
    params : PseudoScalarMediatorParameters or PseudoScalarMediator object
        Parameter object for the pseudo-scalar mediator theory.

    Returns
    -------
    msqrd : float
        The squared matrix element for a given set of momenta.
    """
    mx = params.mx
    mp = params.mp
    gpxx = params.gpxx
    gpuu = params.gpuu
    gpdd = params.gpdd
    gpGG = params.gpGG
    beta = params.beta

    p1 = momenta[0]
    p2 = momenta[1]
    p3 = momenta[2]
    k = momenta[3]

    P = sum(momenta)

    pxmag = np.sqrt(P[0]**2 / 4. - mx**2)

    px = np.array([P[0] / 2., 0., 0., pxmag])
    pxbar = np.array([P[0] / 2., 0., 0., -pxmag])

    beta0 = -(gpxx**2 * qe**2 *
              (b0 * gpGG * mdq - b0 * gpGG * muq +
               b0 * gpdd * vh - b0 * gpuu * vh)**2 *
              (mpi**2 * MDot(k, p1)**2 + mpi**2 * MDot(k, p2)**2 -
               2 * MDot(k, p1) * MDot(k, p2) * MDot(p1, p2)) *
              (mx**2 + MDot(px, pxbar))) / \
        (9. * fpi**2 * vh**2 * MDot(k, p1)**2 * MDot(k, p2)**2 *
         (-mp**2 + 2 * mpi**2 + mpi0**2 +
          2 * MDot(k, p1) + 2 * MDot(k, p2) + 2 * MDot(k, p3) +
          2 * MDot(p1, p2) + 2 * MDot(p1, p3) + 2 * MDot(p2, p3))**2)

    beta1 = (beta * gpxx**2 * qe**2 *
             (b0 * gpGG * mdq - b0 * gpGG * muq +
              b0 * gpdd * vh - b0 * gpuu * vh) *
             (mpi**2 * MDot(k, p1)**2 + mpi**2 * MDot(k, p2)**2 -
              2 * MDot(k, p1) * MDot(k, p2) * MDot(p1, p2)) *
             (b0 * mdq * mp**2 + 2 * mp**2 * mpi**2 - b0 * mdq * mpi0**2 -
              2 * mp**2 * mpi0**2 - 2 * mpi**2 * mpi0**2 + 2 * mpi0**4 +
              b0 * mp**2 * muq - b0 * mpi0**2 * muq +
              4 * mp**2 * MDot(k, p1) - 4 * mpi0**2 * MDot(k, p1) +
                 4 * mp**2 * MDot(k, p2) - 4 * mpi0**2 * MDot(k, p2) -
                 2 * mp**2 * MDot(k, p3) + 2 * mpi0**2 * MDot(k, p3) +
                 4 * mp**2 * MDot(p1, p2) - 4 * mpi0**2 * MDot(p1, p2) -
                 2 * mp**2 * MDot(p1, p3) + 2 * mpi0**2 * MDot(p1, p3) -
              2 * mp**2 * MDot(p2, p3) + 2 * mpi0**2 * MDot(p2, p3)) *
             (mx**2 + MDot(px, pxbar))) / \
        (9. * fpi**3 * vh * MDot(k, p1)**2 * MDot(k, p2)**2 *
         (mpi**2 + MDot(k, p1) + MDot(k, p2) + MDot(k, p3) +
          MDot(p1, p2) + MDot(p1, p3) + MDot(p2, p3)) *
         (-mp**2 + 2 * mpi**2 + mpi0**2 + 2 * MDot(k, p1) + 2 * MDot(k, p2) +
          2 * MDot(k, p3) + 2 * MDot(p1, p2) + 2 * MDot(p1, p3) +
          2 * MDot(p2, p3))**2)

    beta2 = (beta**2 * gpxx**2 * qe**2 *
             (mpi**2 * MDot(k, p1)**2 + mpi**2 * MDot(k, p2)**2 -
              2 * MDot(k, p1) * MDot(k, p2) * MDot(p1, p2)) *
             (8 * (b0 * fpi * gpGG * mdq * mp**2 -
                   b0 * fpi * gpGG * mdq * mpi0**2 -
                   b0 * fpi * gpGG * mp**2 * muq +
                   b0 * fpi * gpGG * mpi0**2 * muq +
                   b0 * fpi * gpdd * mp**2 * vh -
                   b0 * fpi * gpuu * mp**2 * vh -
                   b0 * fpi * gpdd * mpi0**2 * vh +
                   b0 * fpi * gpuu * mpi0**2 * vh) *
              (b0 * fpi * gpGG * mdq * mpi**2 -
               b0 * fpi * gpGG * mpi**2 * muq +
               b0 * fpi * gpdd * mpi**2 * vh -
               b0 * fpi * gpuu * mpi**2 * vh +
               b0 * fpi * gpGG * mdq * MDot(k, p1) -
               b0 * fpi * gpGG * muq * MDot(k, p1) +
               b0 * fpi * gpdd * vh * MDot(k, p1) -
               b0 * fpi * gpuu * vh * MDot(k, p1) +
               b0 * fpi * gpGG * mdq * MDot(k, p2) -
               b0 * fpi * gpGG * muq * MDot(k, p2) +
               b0 * fpi * gpdd * vh * MDot(k, p2) -
               b0 * fpi * gpuu * vh * MDot(k, p2) +
               b0 * fpi * gpGG * mdq * MDot(k, p3) -
               b0 * fpi * gpGG * muq * MDot(k, p3) +
               b0 * fpi * gpdd * vh * MDot(k, p3) -
               b0 * fpi * gpuu * vh * MDot(k, p3) +
               b0 * fpi * gpGG * mdq * MDot(p1, p2) -
               b0 * fpi * gpGG * muq * MDot(p1, p2) +
               b0 * fpi * gpdd * vh * MDot(p1, p2) -
               b0 * fpi * gpuu * vh * MDot(p1, p2) +
               b0 * fpi * gpGG * mdq * MDot(p1, p3) -
               b0 * fpi * gpGG * muq * MDot(p1, p3) +
               b0 * fpi * gpdd * vh * MDot(p1, p3) -
               b0 * fpi * gpuu * vh * MDot(p1, p3) +
               b0 * fpi * gpGG * mdq * MDot(p2, p3) -
               b0 * fpi * gpGG * muq * MDot(p2, p3) +
               b0 * fpi * gpdd * vh * MDot(p2, p3) -
               b0 * fpi * gpuu * vh * MDot(p2, p3)) +
              4 * (b0 * fpi * gpGG * mdq * mpi**2 -
                   b0 * fpi * gpGG * mpi**2 * muq +
                   b0 * fpi * gpdd * mpi**2 * vh -
                   b0 * fpi * gpuu * mpi**2 * vh +
                   b0 * fpi * gpGG * mdq * MDot(k, p1) -
                   b0 * fpi * gpGG * muq * MDot(k, p1) +
                   b0 * fpi * gpdd * vh * MDot(k, p1) -
                   b0 * fpi * gpuu * vh * MDot(k, p1) +
                   b0 * fpi * gpGG * mdq * MDot(k, p2) -
                   b0 * fpi * gpGG * muq * MDot(k, p2) +
                   b0 * fpi * gpdd * vh * MDot(k, p2) -
                   b0 * fpi * gpuu * vh * MDot(k, p2) +
                   b0 * fpi * gpGG * mdq * MDot(k, p3) -
                   b0 * fpi * gpGG * muq * MDot(k, p3) +
                   b0 * fpi * gpdd * vh * MDot(k, p3) -
                   b0 * fpi * gpuu * vh * MDot(k, p3) +
                   b0 * fpi * gpGG * mdq * MDot(p1, p2) -
                   b0 * fpi * gpGG * muq * MDot(p1, p2) +
                   b0 * fpi * gpdd * vh * MDot(p1, p2) -
                   b0 * fpi * gpuu * vh * MDot(p1, p2) +
                   b0 * fpi * gpGG * mdq * MDot(p1, p3) -
                   b0 * fpi * gpGG * muq * MDot(p1, p3) +
                   b0 * fpi * gpdd * vh * MDot(p1, p3) -
                   b0 * fpi * gpuu * vh * MDot(p1, p3) +
                   b0 * fpi * gpGG * mdq * MDot(p2, p3) -
                   b0 * fpi * gpGG * muq * MDot(p2, p3) +
                   b0 * fpi * gpdd * vh * MDot(p2, p3) -
                   b0 * fpi * gpuu * vh * MDot(p2, p3))**2 -
              (-(b0 * mdq * mp**2 * vh) - 2 * mp**2 * mpi**2 * vh +
               b0 * mdq * mpi0**2 * vh + 2 * mp**2 * mpi0**2 * vh +
               2 * mpi**2 * mpi0**2 * vh - 2 * mpi0**4 * vh -
               b0 * mp**2 * muq * vh + b0 * mpi0**2 * muq * vh -
               4 * mp**2 * vh * MDot(k, p1) +
                 4 * mpi0**2 * vh * MDot(k, p1) -
                 4 * mp**2 * vh * MDot(k, p2) +
                 4 * mpi0**2 * vh * MDot(k, p2) +
                 2 * mp**2 * vh * MDot(k, p3) -
                 2 * mpi0**2 * vh * MDot(k, p3) -
                 4 * mp**2 * vh * MDot(p1, p2) +
                 4 * mpi0**2 * vh * MDot(p1, p2) +
                 2 * mp**2 * vh * MDot(p1, p3) -
                 2 * mpi0**2 * vh * MDot(p1, p3) +
                 2 * mp**2 * vh * MDot(p2, p3) -
                 2 * mpi0**2 * vh * MDot(p2, p3))**2) *
             (mx**2 + MDot(px, pxbar))) / \
        (36. * fpi**4 * vh**2 * MDot(k, p1)**2 * MDot(k, p2)**2 *
         (mpi**2 + MDot(k, p1) + MDot(k, p2) + MDot(k, p3) +
          MDot(p1, p2) + MDot(p1, p3) + MDot(p2, p3))**2 *
         (-mp**2 + 2 * mpi**2 + mpi0**2 + 2 * MDot(k, p1) +
          2 * MDot(k, p2) + 2 * MDot(k, p3) +
          2 * MDot(p1, p2) + 2 * MDot(p1, p3) +
          2 * MDot(p2, p3))**2)

    return beta0 + beta1 + beta2


def msqrd_xx_to_p_to_000(momenta, params):
    """
    Returns the squared matrix element for dark matter annihilating into
    three neutral pions through a scalar mediator. This function is used for
    RAMBO.

    Parameters
    ----------
    momenta : numpy.array
        List of momenta of the final state particles, the three netural pions.
    params : PseudoScalarMediatorParameters or PseudoScalarMediator object
        Parameter object for the pseudo-scalar mediator theory.

    Returns
    -------
    msqrd : float
        The squared matrix element for a given set of momenta.
    """
    mx = params.mx
    mp = params.mp
    gpxx = params.gpxx
    gpuu = params.gpuu
    gpdd = params.gpdd
    gpGG = params.gpGG
    beta = params.beta
    widthp = params.width_p

    p1 = momenta[0]
    p2 = momenta[1]
    p3 = momenta[2]

    P = sum(momenta)

    pxmag = np.sqrt(P[0]**2 / 4. - mx**2)

    px = np.array([P[0] / 2., 0., 0., pxmag])
    pxbar = np.array([P[0] / 2., 0., 0., -pxmag])

    beta0 = (-((b0**2 * gpxx**2 *
                (gpGG * (mdq - muq) +
                 (gpdd - gpuu) * vh)**2 * (mx**2 + MDot(px, pxbar))) /
               (fpi**2 * vh**2 * (mp**2 - 2 * mpi**2 - mpi0**2 +
                                  1.0j * mp * widthp - 2 * MDot(p1, p2) -
                                  2 * MDot(p1, p3) - 2 * MDot(p2, p3)) *
                (-mp**2 + 2 * mpi**2 + mpi0**2 +
                 1.0j * mp * widthp + 2 * MDot(p1, p2) +
                 2 * MDot(p1, p3) + 2 * MDot(p2, p3)))))

    beta1 = ((b0**2 * beta * gpxx**2 * (mp - mpi0) * (mp + mpi0) *
              (mdq + muq) * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh) *
              (mx**2 + MDot(px, pxbar))) /
             (fpi**3 * vh * (mp**2 - 2 * mpi**2 - mpi0**2 +
                             1.0j * mp * widthp - 2 * MDot(p1, p2) -
                             2 * MDot(p1, p3) - 2 * MDot(p2, p3)) *
              (mpi**2 + MDot(p1, p2) + MDot(p1, p3) + MDot(p2, p3)) *
              (-mp**2 + 2 * mpi**2 + mpi0**2 + 1.0j * mp * widthp +
               2 * MDot(p1, p2) + 2 * MDot(p1, p3) + 2 * MDot(p2, p3))))

    beta2 = ((b0**2 * beta**2 * gpxx**2 *
              ((mdq + muq)**2 / (mpi**2 + MDot(p1, p2) +
                                 MDot(p1, p3) + MDot(p2, p3))**2 +
               (2 * ((mdq + muq)**2 * vh**2 -
                     4 * fpi**2 * (gpGG * (mdq - muq) +
                                   (gpdd - gpuu) * vh)**2)) /
               (vh**2 * (mp**2 - 2 * mpi**2 - mpi0**2 + 1.0j * mp * widthp -
                         2 * MDot(p1, p2) - 2 * MDot(p1, p3) -
                         2 * MDot(p2, p3)) *
                (mpi**2 + MDot(p1, p2) + MDot(p1, p3) + MDot(p2, p3))) -
               (4 * (mdq + muq)**2 * vh**2 -
                44 * fpi**2 * (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)**2) /
               (vh**2 * (mp**2 - 2 * mpi**2 - mpi0**2 +
                         1.0j * mp * widthp - 2 * MDot(p1, p2) -
                         2 * MDot(p1, p3) - 2 * MDot(p2, p3)) *
                (-mp**2 + 2 * mpi**2 + mpi0**2 +
                 1.0j * mp * widthp + 2 * MDot(p1, p2) +
                 2 * MDot(p1, p3) + 2 * MDot(p2, p3))) -
               (2 * ((mdq + muq)**2 * vh**2 -
                     4 * fpi**2 * (gpGG * (mdq - muq) +
                                   (gpdd - gpuu) * vh)**2)) /
               (vh**2 * (mpi**2 + MDot(p1, p2) + MDot(p1, p3) +
                         MDot(p2, p3)) *
                (-mp**2 + 2 * mpi**2 + mpi0**2 +
                 1.0j * mp * widthp + 2 * MDot(p1, p2) +
                 2 * MDot(p1, p3) + 2 * MDot(p2, p3)))) *
              (mx**2 + MDot(px, pxbar))) / (4. * fpi**4))

    msqrd = beta0 + beta1 + beta2

    assert msqrd.imag == 0.0

    return msqrd.real
