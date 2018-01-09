"""
Module containing squared matrix elements for a scalar mediator.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

import numpy as np


def xx_to_ffg(kList, mx, mf, ms, CXXS, CffS):
    """
    Returns the matrix element squared for two identical fermions "x" to two
    identical fermions "f" and a photon.

    Parameters
    ----------
    kList : numpy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        kList must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    mx : float
        Mass of incoming fermions.
    mf : float
        Mass of final state fermions.
    ms : float
        Mass of scalar mediator.
    CXXS : float
        Coupling of initial state fermions with the scalar mediator.
    CffS : float
        Coupling of final state fermions with the scalar mediator.

    Returns
    -------
    mat_elem_sqrd : float
        Modulus squared of the matrix element for x + x -> S^* -> f + f + p.
    """
    e = np.sqrt(4 * np.pi / 137.)
    Qf = 1.0
    CffS = 1.
    CXXS = 1.

    p3 = kList[0]
    p4 = kList[1]
    k = kList[2]

    Q = p3[0] + p4[0] + k[0]

    mfp = mf / Q
    mxp = mx / Q
    msp = ms / Q

    kDOTp3 = k[0] * p3[0] - k[1] * p3[1] - k[2] * p3[2] - k[3] * p3[3]
    kDOTp4 = k[0] * p4[0] - k[1] * p4[1] - k[2] * p4[2] - k[3] * p4[3]
    p3DOTp4 = p4[0] * p3[0] - p4[1] * p3[1] - p4[2] * p3[2] - p4[3] * p3[3]

    mat_elem = (-8 * CffS**2 * CXXS**2 * e**2 * (-1 + 4 * mxp**2) *
                (kDOTp3 * kDOTp4 *
                 ((kDOTp3 + kDOTp4)**2 + 2 * (kDOTp3 + kDOTp4) * p3DOTp4 +
                  2 * p3DOTp4**2) - (kDOTp3 + kDOTp4) * mfp**2 *
                 (kDOTp3**2 + kDOTp3 * p3DOTp4 +
                  kDOTp4 * (kDOTp4 + p3DOTp4)) * Q**2 +
                 (kDOTp3**2 + kDOTp4**2) * mfp**4 * Q**4) * Qf**2) / \
        (kDOTp3**2 * kDOTp4**2 * (-1 + msp**2)**2 * Q**2)

    return mat_elem / (2.0 * Q**2 * np.sqrt(1. - 4.0 * mx**2 / Q**2))
