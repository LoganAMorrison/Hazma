import numpy as np

from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import up_quark_mass as muq
from hazma.parameters import down_quark_mass as mdq
from hazma.parameters import b0, vh, fpi, qe
from hazma.utils import ldot

from ._proto import PseudoScalarMediatorBase


def msqrd_xx_to_p_to_pm0(momenta, model: PseudoScalarMediatorBase):
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
    model : PseudoScalarMediatorParameters or PseudoScalarMediator object
        Parameter object for the pseudo-scalar mediator theory.

    Returns
    -------
    msqrd : float
        The squared matrix element for a given set of momenta.
    """
    p1 = momenta[0]
    p2 = momenta[1]
    p3 = momenta[2]
    P = np.sum(momenta, axis=1)

    mx = model.mx
    mp = model.mp
    gpxx = model.gpxx
    gpuu = model.gpuu
    gpdd = model.gpdd
    gpGG = model.gpGG
    beta = model.beta
    widthp = model.width_p

    pxmag = np.sqrt(P[0] ** 2 / 4.0 - mx**2)

    px = np.array([P[0] / 2.0, 0.0, 0.0, pxmag])
    pxbar = np.array([P[0] / 2.0, 0.0, 0.0, -pxmag])

    """ Order beta^0 """
    beta0 = (
        gpxx**2
        * (
            -(b0 * fpi * gpGG * mdq)
            + b0 * fpi * gpGG * muq
            - b0 * fpi * gpdd * vh
            + b0 * fpi * gpuu * vh
        )
        ** 2
        * (mx**2 + ldot(px, pxbar))
    ) / (
        9.0
        * fpi**4
        * vh**2
        * (
            (-(mp**2) + 2 * mpi**2 + mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (
                mp**2
                - 2 * mpi**2
                - mpi0**2
                - ldot(p1, p2)
                - ldot(p1, p3)
                - ldot(p2, p3)
            )
            * (ldot(p1, p2) + ldot(p1, p3) + ldot(p2, p3))
        )
    )

    """ Order beta^1 """
    beta1 = (
        2
        * beta
        * gpxx**2
        * (
            -(b0 * fpi * gpGG * mdq)
            + b0 * fpi * gpGG * muq
            - b0 * fpi * gpdd * vh
            + b0 * fpi * gpuu * vh
        )
        * (
            b0 * mdq * vh
            + 2 * mpi**2 * vh
            - 2 * mpi0**2 * vh
            + b0 * muq * vh
            + 4 * vh * ldot(p1, p2)
            - 2 * vh * ldot(p1, p3)
            - 2 * vh * ldot(p2, p3)
        )
        * (mx**2 + ldot(px, pxbar))
    ) / (
        9.0
        * fpi**4
        * vh**2
        * (
            (-(mp**2) + 2 * mpi**2 + mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (
                mp**2
                - 2 * mpi**2
                - mpi0**2
                - ldot(p1, p2)
                - ldot(p1, p3)
                - ldot(p2, p3)
            )
            * (ldot(p1, p2) + ldot(p1, p3) + ldot(p2, p3))
        )
    )

    """ Order beta^2 """

    beta2 = (
        beta**2
        * gpxx**2
        * (
            2
            * (
                2 * b0 * fpi * gpGG * mdq
                - 2 * b0 * fpi * gpGG * muq
                + 2 * b0 * fpi * gpdd * vh
                - 2 * b0 * fpi * gpuu * vh
            )
            * (
                -(b0 * fpi * gpGG * mdq)
                + b0 * fpi * gpGG * muq
                - b0 * fpi * gpdd * vh
                + b0 * fpi * gpuu * vh
            )
            - (
                -(b0 * fpi * gpGG * mdq)
                + b0 * fpi * gpGG * muq
                - b0 * fpi * gpdd * vh
                + b0 * fpi * gpuu * vh
            )
            ** 2
            + (
                b0 * mdq * vh
                + 2 * mpi**2 * vh
                - 2 * mpi0**2 * vh
                + b0 * muq * vh
                + 4 * vh * ldot(p1, p2)
                - 2 * vh * ldot(p1, p3)
                - 2 * vh * ldot(p2, p3)
            )
            ** 2
        )
        * (mx**2 + ldot(px, pxbar))
    ) / (
        9.0
        * fpi**4
        * vh**2
        * (
            (-(mp**2) + 2 * mpi**2 + mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (
                mp**2
                - 2 * mpi**2
                - mpi0**2
                - ldot(p1, p2)
                - ldot(p1, p3)
                - ldot(p2, p3)
            )
            * (ldot(p1, p2) + ldot(p1, p3) + ldot(p2, p3))
        )
    )

    msqrd = beta0 + beta1 + beta2

    return msqrd


def msqrd_xx_to_p_to_pm0g(momenta, model: PseudoScalarMediatorBase):
    """
    Returns the squared matrix element for dark matter annihilating into
    two charged pions, a netural pion and a photon through a scalar
    mediator. This function is used for RAMBO.

    Parameters
    ----------
    momenta : numpy.array
        List of momenta of the final state particles. The first momentum is
        for the positively charged pion, the second is for the negatively
        charged pion, the third is for the neutral pion and the last is
        for the photon.
    model : PseudoScalarMediatorParameters or PseudoScalarMediator object
        Parameter object for the pseudo-scalar mediator theory.

    Returns
    -------
    msqrd : float
        The squared matrix element for a given set of momenta.
    """
    mx = model.mx
    mp = model.mp
    gpxx = model.gpxx
    gpuu = model.gpuu
    gpdd = model.gpdd
    gpGG = model.gpGG
    beta = model.beta
    widthp = model.width_p

    p1 = momenta[0]
    p2 = momenta[1]
    p3 = momenta[2]
    k = momenta[3]

    P = np.sum(momenta, axis=1)

    pxmag = np.sqrt(P[0] ** 2 / 4.0 - mx**2)

    px = np.array([P[0] / 2.0, 0.0, 0.0, pxmag])
    pxbar = np.array([P[0] / 2.0, 0.0, 0.0, -pxmag])

    beta0 = -(
        gpxx**2
        * qe**2
        * (
            -(b0 * fpi * gpGG * mdq)
            + b0 * fpi * gpGG * muq
            - b0 * fpi * gpdd * vh
            + b0 * fpi * gpuu * vh
        )
        ** 2
        * (
            mpi**2 * ldot(k, p1) ** 2
            + mpi**2 * ldot(k, p2) ** 2
            - 2 * ldot(k, p1) * ldot(k, p2) * ldot(p1, p2)
        )
        * (mx**2 + ldot(px, pxbar))
    ) / (
        9.0
        * fpi**4
        * vh**2
        * ldot(k, p1) ** 2
        * ldot(k, p2) ** 2
        * (
            (-(mp**2) + 2 * mpi**2 + mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (
                mp**2
                - 2 * mpi**2
                - mpi0**2
                - ldot(k, p1)
                - ldot(k, p2)
                - ldot(k, p3)
                - ldot(p1, p2)
                - ldot(p1, p3)
                - ldot(p2, p3)
            )
            * (
                ldot(k, p1)
                + ldot(k, p2)
                + ldot(k, p3)
                + ldot(p1, p2)
                + ldot(p1, p3)
                + ldot(p2, p3)
            )
        )
    )

    beta1 = (
        -2
        * beta
        * gpxx**2
        * qe**2
        * (
            -(b0 * fpi * gpGG * mdq)
            + b0 * fpi * gpGG * muq
            - b0 * fpi * gpdd * vh
            + b0 * fpi * gpuu * vh
        )
        * (
            mpi**2 * ldot(k, p1) ** 2
            + mpi**2 * ldot(k, p2) ** 2
            - 2 * ldot(k, p1) * ldot(k, p2) * ldot(p1, p2)
        )
        * (
            b0 * mdq * vh
            + 2 * mpi**2 * vh
            - 2 * mpi0**2 * vh
            + b0 * muq * vh
            + 4 * vh * ldot(k, p1)
            + 4 * vh * ldot(k, p2)
            - 2 * vh * ldot(k, p3)
            + 4 * vh * ldot(p1, p2)
            - 2 * vh * ldot(p1, p3)
            - 2 * vh * ldot(p2, p3)
        )
        * (mx**2 + ldot(px, pxbar))
    ) / (
        9.0
        * fpi**4
        * vh**2
        * ldot(k, p1) ** 2
        * ldot(k, p2) ** 2
        * (
            (-(mp**2) + 2 * mpi**2 + mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (
                mp**2
                - 2 * mpi**2
                - mpi0**2
                - ldot(k, p1)
                - ldot(k, p2)
                - ldot(k, p3)
                - ldot(p1, p2)
                - ldot(p1, p3)
                - ldot(p2, p3)
            )
            * (
                ldot(k, p1)
                + ldot(k, p2)
                + ldot(k, p3)
                + ldot(p1, p2)
                + ldot(p1, p3)
                + ldot(p2, p3)
            )
        )
    )

    beta2 = (
        beta**2
        * gpxx**2
        * qe**2
        * (
            mpi**2 * ldot(k, p1) ** 2
            + mpi**2 * ldot(k, p2) ** 2
            - 2 * ldot(k, p1) * ldot(k, p2) * ldot(p1, p2)
        )
        * (
            -2
            * (
                2 * b0 * fpi * gpGG * mdq
                - 2 * b0 * fpi * gpGG * muq
                + 2 * b0 * fpi * gpdd * vh
                - 2 * b0 * fpi * gpuu * vh
            )
            * (
                -(b0 * fpi * gpGG * mdq)
                + b0 * fpi * gpGG * muq
                - b0 * fpi * gpdd * vh
                + b0 * fpi * gpuu * vh
            )
            + (
                -(b0 * fpi * gpGG * mdq)
                + b0 * fpi * gpGG * muq
                - b0 * fpi * gpdd * vh
                + b0 * fpi * gpuu * vh
            )
            ** 2
            - (
                b0 * mdq * vh
                + 2 * mpi**2 * vh
                - 2 * mpi0**2 * vh
                + b0 * muq * vh
                + 4 * vh * ldot(k, p1)
                + 4 * vh * ldot(k, p2)
                - 2 * vh * ldot(k, p3)
                + 4 * vh * ldot(p1, p2)
                - 2 * vh * ldot(p1, p3)
                - 2 * vh * ldot(p2, p3)
            )
            ** 2
        )
        * (mx**2 + ldot(px, pxbar))
    ) / (
        9.0
        * fpi**4
        * vh**2
        * ldot(k, p1) ** 2
        * ldot(k, p2) ** 2
        * (
            (-(mp**2) + 2 * mpi**2 + mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (
                mp**2
                - 2 * mpi**2
                - mpi0**2
                - ldot(k, p1)
                - ldot(k, p2)
                - ldot(k, p3)
                - ldot(p1, p2)
                - ldot(p1, p3)
                - ldot(p2, p3)
            )
            * (
                ldot(k, p1)
                + ldot(k, p2)
                + ldot(k, p3)
                + ldot(p1, p2)
                + ldot(p1, p3)
                + ldot(p2, p3)
            )
        )
    )

    return beta0 + beta1 + beta2


def msqrd_xx_to_p_to_000(momenta, model: PseudoScalarMediatorBase):
    """
    Returns the squared matrix element for dark matter annihilating into
    three neutral pions through a scalar mediator. This function is used
    for RAMBO.

    Parameters
    ----------
    momenta : numpy.array
        List of momenta of the final state particles, the three netural
        pions.
    model : PseudoScalarMediatorParameters or PseudoScalarMediator object
        Parameter object for the pseudo-scalar mediator theory.

    Returns
    -------
    msqrd : float
        The squared matrix element for a given set of momenta.
    """
    mx = model.mx
    mp = model.mp
    gpxx = model.gpxx
    gpuu = model.gpuu
    gpdd = model.gpdd
    gpGG = model.gpGG
    beta = model.beta
    widthp = model.width_p

    p1 = momenta[0]
    p2 = momenta[1]
    p3 = momenta[2]

    P = np.sum(momenta, axis=1)

    pxmag = np.sqrt(P[0] ** 2 / 4.0 - mx**2)

    px = np.array([P[0] / 2.0, 0.0, 0.0, pxmag])
    pxbar = np.array([P[0] / 2.0, 0.0, 0.0, -pxmag])

    beta0 = (
        b0**2
        * gpxx**2
        * (fpi * gpGG * mdq - fpi * gpGG * muq + fpi * gpdd * vh - fpi * gpuu * vh) ** 2
        * (mx**2 + ldot(px, pxbar))
    ) / (
        fpi**4
        * vh**2
        * (
            (mp**2 - 3 * mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (mp**2 - 3 * mpi0**2 - ldot(p1, p2) - ldot(p1, p3) - ldot(p2, p3))
            * (ldot(p1, p2) + ldot(p1, p3) + ldot(p2, p3))
        )
    )

    beta1 = (
        -2
        * b0**2
        * beta
        * gpxx**2
        * (fpi * gpGG * mdq - fpi * gpGG * muq + fpi * gpdd * vh - fpi * gpuu * vh)
        * (mdq * vh + muq * vh)
        * (mx**2 + ldot(px, pxbar))
    ) / (
        fpi**4
        * vh**2
        * (
            (mp**2 - 3 * mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (mp**2 - 3 * mpi0**2 - ldot(p1, p2) - ldot(p1, p3) - ldot(p2, p3))
            * (ldot(p1, p2) + ldot(p1, p3) + ldot(p2, p3))
        )
    )

    beta2 = (
        b0**2
        * beta**2
        * gpxx**2
        * (
            -3
            * (fpi * gpGG * mdq - fpi * gpGG * muq + fpi * gpdd * vh - fpi * gpuu * vh)
            ** 2
            + 2
            * (fpi * gpGG * mdq - fpi * gpGG * muq + fpi * gpdd * vh - fpi * gpuu * vh)
            * (
                -4 * fpi * gpGG * mdq
                + 4 * fpi * gpGG * muq
                - 4 * fpi * gpdd * vh
                + 4 * fpi * gpuu * vh
            )
            + (-(mdq * vh) - muq * vh) ** 2
        )
        * (mx**2 + ldot(px, pxbar))
    ) / (
        fpi**4
        * vh**2
        * (
            (mp**2 - 3 * mpi0**2) ** 2
            + mp**2 * widthp**2
            - 4
            * (mp**2 - 3 * mpi0**2 - ldot(p1, p2) - ldot(p1, p3) - ldot(p2, p3))
            * (ldot(p1, p2) + ldot(p1, p3) + ldot(p2, p3))
        )
    )

    msqrd = beta0 + beta1 + beta2

    return msqrd
