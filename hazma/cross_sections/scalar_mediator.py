"""
Module containing cross sections.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""
import numpy as np


def xx_to_ff(kList, mx, mf, ms, CXXS, CffS):
    """
    Returns the cross section for two identical fermions "x" to two
    identical fermions "f".

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
    cross_section : float
        Cross section for x + x -> S^* -> f + f.
    """
    p3 = kList[0]
    p4 = kList[1]
    k = kList[2]

    Q = p3[0] + p4[0] + k[0]
    return (CffS**2 * CXXS**2 * (-4 * mf**2 + Q**2)**1.5 *
            np.sqrt(-4 * mx**2 + Q**2)) / \
        (16. * np.pi * (-(ms**2 * Q) + Q**3)**2)
