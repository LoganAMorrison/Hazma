"""Module containing squared matrix elements from axial-vector mediator models.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

import numpy as np
from hazma.field_theory_helper_functions.common_functions import minkowski_dot
from hazma.parameters import alpha_em


def msqrd_xx_to_a_to_ff(moms, mx, mf, ma, cxxa, cffa):
    """Returns the spin-averaged, squared matrix element for a pair of fermions,
    *x*, annihilating into a pair of fermions, *f*, through a axial-vector
    mediator in the s-channel.

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
    ma : float
        Mass of axial-vector mediator.
    qf : float
        Electric charge of final state fermion.
    cxxa : float
        Coupling of initial state fermions with the axial-vector mediator.
    cffa : float
        Coupling of final state fermions with the axial-vector mediator.

    Returns
    -------
    mat_elem_sqrd : float
        Spin averaged, squared matrix element for x + x -> A^* -> f + f.
    """
    p3 = moms[0]
    p4 = moms[1]

    Q = p3[0] + p4[0]

    E = Q / 2.0
    p = np.sqrt(E**2 - mx**2)

    p1 = np.array([E, 0.0, 0.0, p])
    p2 = np.array([E, 0.0, 0.0, -p])

    p1DOTp4 = minkowski_dot(p1, p4)
    p2DOTp3 = minkowski_dot(p2, p3)
    p1DOTp3 = minkowski_dot(p1, p3)
    p2DOTp4 = minkowski_dot(p2, p4)

    return (
        4
        * cffa**2
        * cxxa**2
        * (
            2 * mf**2 * mx**2
            + 6 * mf**2 * mx**2
            + 2 * p1DOTp4 * p2DOTp3
            + 2 * p1DOTp3 * p2DOTp4
            - mf**2 * Q**2
            - mx**2 * Q**2
        )
    ) / (-(ma**2) + Q**2) ** 2


def msqrd_xx_to_a_to_ffg(moms, mx, mf, ma, qf, cxxa, cffa):
    """Returns the spin-averaged, squared matrix element for a pair of fermions,
    *x*, annihilating into a pair of fermions, *f*, and a photon through a
    axial-vector mediator in the s-channel.

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
    ma : float
        Mass of axial-vector mediator.
    qf : float
        Electric charge of final state fermion.
    cxxa : float
        Coupling of initial state fermions with the axial-vector mediator.
    cffa : float
        Coupling of final state fermions with the axial-vector mediator.

    Returns
    -------
    mat_elem_sqrd : float
        Spin averaged, squared matrix element for x + x -> a^* -> f + f + g.
    """
    p3 = moms[0]
    p4 = moms[1]
    k = moms[2]

    Q = p3[0] + p4[0] + k[0]

    E = Q / 2
    p = np.sqrt(E**2 - mx**2)

    p1 = np.array([E, 0, 0, p])
    p2 = np.array([E, 0, 0, -p])

    kDOTp3 = minkowski_dot(k, p3)
    kDOTp4 = minkowski_dot(k, p4)
    p3DOTp4 = minkowski_dot(p3, p4)

    kDOTp1 = minkowski_dot(k, p1)
    p1DOTp3 = minkowski_dot(p1, p3)
    p1DOTp4 = minkowski_dot(p1, p4)

    kDOTp2 = minkowski_dot(k, p2)
    p2DOTp3 = minkowski_dot(p2, p3)
    p2DOTp4 = minkowski_dot(p2, p4)

    e = np.sqrt(4 * np.pi * alpha_em)

    return (
        4
        * cffa**2
        * cxxa**2
        * e**2
        * (
            2
            * kDOTp3
            * kDOTp4
            * (
                kDOTp1 * kDOTp3 * p2DOTp3
                - 2 * kDOTp4 * p1DOTp3 * p2DOTp3
                + kDOTp3 * p1DOTp4 * p2DOTp3
                + kDOTp4 * p1DOTp4 * p2DOTp3
                + kDOTp1 * kDOTp4 * p2DOTp4
                + kDOTp3 * p1DOTp3 * p2DOTp4
                + kDOTp4 * p1DOTp3 * p2DOTp4
                - 2 * kDOTp3 * p1DOTp4 * p2DOTp4
                + ((kDOTp1 + 2 * p1DOTp4) * p2DOTp3 + (kDOTp1 + 2 * p1DOTp3) * p2DOTp4)
                * p3DOTp4
                + kDOTp2
                * (kDOTp3 * p1DOTp3 + kDOTp4 * p1DOTp4 + (p1DOTp3 + p1DOTp4) * p3DOTp4)
                - mx**2
                * (
                    kDOTp3**2
                    + kDOTp4**2
                    + 2 * (kDOTp3 + kDOTp4) * p3DOTp4
                    + 2 * p3DOTp4**2
                )
            )
            + (kDOTp3**2 + kDOTp4**2) * mf**4 * (-6 * mx**2 + Q**2)
            + 2
            * mf**2
            * (
                -(kDOTp2 * kDOTp3**2 * p1DOTp3)
                - kDOTp2 * kDOTp4**2 * p1DOTp4
                - kDOTp3**2 * p1DOTp4 * p2DOTp3
                - kDOTp4**2 * p1DOTp4 * p2DOTp3
                - kDOTp3**2 * p1DOTp3 * p2DOTp4
                - kDOTp4**2 * p1DOTp3 * p2DOTp4
                + kDOTp1
                * (
                    2 * kDOTp2 * kDOTp3 * kDOTp4
                    - kDOTp3**2 * p2DOTp3
                    - kDOTp4**2 * p2DOTp4
                )
                + mx**2
                * (
                    kDOTp3**3
                    + kDOTp3**2 * (kDOTp4 + p3DOTp4)
                    + kDOTp4**2 * (kDOTp4 + p3DOTp4)
                    + kDOTp3 * kDOTp4 * (kDOTp4 + 6 * p3DOTp4)
                )
                - kDOTp3 * kDOTp4 * p3DOTp4 * Q**2
            )
        )
        * qf**2
    ) / (
        kDOTp3**2
        * kDOTp4**2
        * (ma**2 - 2 * mf**2 - 2 * (kDOTp3 + kDOTp4 + p3DOTp4)) ** 2
    )
