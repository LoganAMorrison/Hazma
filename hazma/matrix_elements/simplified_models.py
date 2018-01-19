"""
Module containing squared matrix elements from simplified models.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

import numpy as np

e = np.sqrt(4 * np.pi / 137.)


def minkowski_dot(fv1, fv2):
    return fv1[0] * fv2[0] - np.dot(fv1[1:4], fv2[1:4])


"""TREE-LEVEL SQUARED MATRIX ELEMENTS."""


def xx_to_s_to_ff(moms, mx, mf, ms, qf, cxxs, cffs):
    """
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
    ms : float
        Mass of scalar mediator.
    cxxs : float
        Coupling of initial state fermions with the scalar mediator.
    cffs : float
        Coupling of final state fermions with the scalar mediator.
    """
    p3 = moms[0]
    p4 = moms[1]

    Q = p3[0] + p4[0]

    return (cffs**2 * cxxs**2 * (-2 * mf + Q) * (2 * mf + Q) *
            (-2 * mx + Q) * (2 * mx + Q)) / (ms**2 - Q**2)**2


def xx_to_p_to_ff(moms, mx, mf, mp, qf, cxxp, cffp):
    """
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
    """
    p3 = moms[0]
    p4 = moms[1]

    Q = p3[0] + p4[0]

    return (cffp**2 * cxxp**2 * Q**4) / (mp**2 - Q**2)**2


def xx_to_v_to_ff(moms, mx, mf, mv, qf, cxxv, cffv):
    """
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
    mv : float
        Mass of vector mediator.
    qf : float
        Electric charge of final state fermion.
    cxxv : float
        Coupling of initial state fermions with the vector mediator.
    cffv : float
        Coupling of final state fermions with the vector mediator.
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

    return (4 * cffv**2 * cxxv**2 *
            (2 * p1DOTp4 * p2DOTp3 + 2 * p1DOTp3 * p2DOTp4 +
             (mf**2 + mx**2) * Q**2)) / (mv**2 - Q**2)**2


def xx_to_a_to_ff(moms, mx, mf, ma, qf, cxxs, cffs):
    pass


"""RADIATIVE SQUARED MATRIX ELEMENTS."""


def xx_to_s_to_ffg(moms, mx, mf, ms, qf, cxxs, cffs):
    """
    Returns the matrix element squared for two identical fermions "x" to two
    identical fermions "f" and a photon.

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
    ms : float
        Mass of scalar mediator.
    cxxs : float
        Coupling of initial state fermions with the scalar mediator.
    cffs : float
        Coupling of final state fermions with the scalar mediator.

    Returns
    -------
    mat_elem_sqrd : float
        Modulus squared of the matrix element for x + x -> S^* -> f + f + p.
    """
    p3 = moms[0]
    p4 = moms[1]
    k = moms[2]

    Q = p3[0] + p4[0] + k[0]

    mfp = mf / Q
    mxp = mx / Q
    msp = ms / Q

    kDOTp3 = minkowski_dot(k, p3)
    kDOTp4 = minkowski_dot(k, p4)
    p3DOTp4 = minkowski_dot(p3, p4)

    mat_elem = (-8 * cffs**2 * cxxs**2 * e**2 * (-1 + 4 * mxp**2) *
                (kDOTp3 * kDOTp4 *
                 ((kDOTp3 + kDOTp4)**2 + 2 * (kDOTp3 + kDOTp4) * p3DOTp4 +
                  2 * p3DOTp4**2) - (kDOTp3 + kDOTp4) * mfp**2 *
                 (kDOTp3**2 + kDOTp3 * p3DOTp4 +
                  kDOTp4 * (kDOTp4 + p3DOTp4)) * Q**2 +
                 (kDOTp3**2 + kDOTp4**2) * mfp**4 * Q**4) * qf**2) / \
        (kDOTp3**2 * kDOTp4**2 * (-1 + msp**2)**2 * Q**2)

    return mat_elem


def xx_to_p_to_ffg(moms, mx, mf, mp, qf, cxxp, cffp):
    """
    Returns the matrix element squared for two identical fermions "x" to two
    identical fermions "f" and a photon.

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
        Modulus squared of the matrix element for x + x -> P^* -> f + f + g.
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


def xx_to_v_to_ffg(moms, mx, mf, mv, qf, cxxs, cffs):
    """
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
    mv : float
        Mass of vector mediator.
    qf : float
        Electric charge of final state fermion.
    cxxv : float
        Coupling of initial state fermions with the vector mediator.
    cffv : float
        Coupling of final state fermions with the vector mediator.
    """
    pass


def xx_to_a_to_ffg(moms, mx, mf, ma, qf, CXXA, CffA):
    """
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
    """
    pass
