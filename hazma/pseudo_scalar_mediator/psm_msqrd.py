"""Module containing squared matrix elements from pseudo-scalar mediator
models.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""
from numpy import sqrt, pi
from ..field_theory_helper_functions.common_functions import minkowski_dot
from ..parameters import alpha_em


def msqrd_xx_to_p_to_ff(moms, mx, mf, mp, cxxp, cffp):
    """Returns the spin-averaged, squared matrix element for a pair of fermions,
    *x*, annihilating into a pair of fermions, *f*, through a pseudo-scalar
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
    mp : float
        Mass of pseudo-scalar mediator.
    qf : float
        Electric charge of final state fermion.
    cxxp : float
        Coupling of initial state fermions with the pseudo-scalar mediator.
    cffp : float
        Coupling of final state fermions with the pseudo-scalar mediator.

    Returns
    -------
    mat_elem_sqrd : float
        Spin averaged, squared matrix element for x + x -> P^* -> f + f.
    """
    p3 = moms[0]
    p4 = moms[1]

    Q = p3[0] + p4[0]

    return (cffp**2 * cxxp**2 * Q**4) / (mp**2 - Q**2)**2


def msqrd_xx_to_p_to_ffg(moms, mx, mf, mp, qf, cxxp, cffp):
    """Returns the spin-averaged, squared matrix element for a pair of fermions,
    *x*, annihilating into a pair of fermions, *f*, and a photon through a
    pseudo-scalar mediator in the s-channel.

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
    mp : float
        Mass of pseudo-scalar mediator.
    qf : float
        Electric charge of final state fermion.
    cxxp : float
        Coupling of initial state fermions with the pseudo-scalar mediator.
    cffp : float
        Coupling of final state fermions with the pseudo-scalar mediator.

    Returns
    -------
    mat_elem_sqrd : float
        Spin averaged, squared matrix element for x + x -> p^* -> f + f + g.
    """
    p3 = moms[0]
    p4 = moms[1]
    k = moms[2]

    Q = p3[0] + p4[0] + k[0]

    mfp = mf / Q
    mpp = mp / Q

    kDOTp3 = minkowski_dot(k, p3)
    kDOTp4 = minkowski_dot(k, p4)
    p3DOTp4 = minkowski_dot(p3, p4)

    e = sqrt(4 * pi * alpha_em)

    mat_elem = (-2 * cffp**2 * cxxp**2 * e**2 * Q**2 *
                (kDOTp3 * kDOTp4 *
                 (-(kDOTp3 + kDOTp4)**2 -
                  2 * (kDOTp3 + kDOTp4) * p3DOTp4 - 2 * p3DOTp4**2) + mfp**2 *
                 (kDOTp3**3 + kDOTp3 * kDOTp4 * (kDOTp4 - 2 * p3DOTp4) +
                  kDOTp3**2 * (kDOTp4 + p3DOTp4) +
                  kDOTp4**2 * (kDOTp4 + p3DOTp4)) * Q**2 +
                 (kDOTp3**2 + kDOTp4**2) * mfp**4 * Q**4) * qf**2) /\
        (kDOTp3**2 * kDOTp4**2 *
         (2 * (kDOTp3 + kDOTp4 + p3DOTp4) + (2 * mfp**2 - mpp**2) * Q**2)**2)

    return mat_elem
