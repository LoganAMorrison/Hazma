"""Module containing cross sections for the pseudo-scalar mediator.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

from numpy import sqrt, pi


def sigma_xx_to_p_to_ff(cme, mx, mf, mp, cxxp, cffp):
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
    mp : float
        Mass of pseudo-scalar mediator.
    cxxp : float
        Coupling of initial state fermions with the pseudo-scalar mediator.
    cffp : float
        Coupling of final state fermions with the pseudo-scalar mediator.

    Returns
    -------
    cross_section : float
        Cross section for x + x -> P^* -> f + f.
    """

    return (cffp**2 * cxxp**2 * cme**2 *
            sqrt(-4 * mf**2 + cme**2)) /\
        (16. * pi * (mp**2 - cme**2)**2 *
         sqrt(-4 * mx**2 + cme**2))
