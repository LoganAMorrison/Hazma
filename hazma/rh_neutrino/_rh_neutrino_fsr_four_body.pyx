"""
This file contains cythonized functions for computing gamma-ray spectra
from 4-body final states occuring in the right-handed neutrino model.
"""

from hazma.gamma_ray_helper_functions.gamma_ray_fsr cimport c_gamma_ray_fsr
from libcpp.vector cimport vector
from libcpp.functional cimport function
from libc.math cimport sqrt, M_PI
import cython

cdef double MW = 80.379e3
cdef double MH = 125.10e3
cdef double mpi0 = 134.9766
cdef double mpi = 139.57018
cdef double alpha_em = 1.0 / 137.04
cdef double GF = 1.1663787e-11
cdef double vh = 246.22795e3
cdef double qe = sqrt(4.0 * M_PI * alpha_em)
cdef double sw2 = 0.22290
cdef double sw = sqrt(sw2)
cdef double cw = sqrt(1.0 - sw2)
cdef double Vud = 0.974267

cdef double electron_mass = 0.510998928
cdef double muon_mass = 105.6583715
cdef double tau_mass = 1776.86
cdef vector[double] lepton_masses = [electron_mass, muon_mass, tau_mass]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ldot(vector[double] &fv1,vector[double] &fv2):
    """
    Compute the scalar product between two four-vectors.

    Parameters
    ----------
    fv1, fv2: vector[double]
       Four-vectors to compute scalar product of.

    Returns
    -------
    dot: double
        The scalar-product of `fv1` and `fv2`.
    """
    return fv1[0] * fv2[0] - fv1[1] * fv2[1] - fv1[2] * fv2[2] - fv1[3] * fv2[3]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double msqrd_nu_pi_pi_g(vector[vector[double]] &momenta, vector[double]&params):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    neutrino, two charged pions and a photon at leading order in the Fermi
    constant. Momenta are ordered as follows: {nu,pi+,pi-,photon}.

    Parameters
    ----------
    momenta: np.ndarray
        List of NumPy arrays storing the four-momenta of the final state
        particles.
    mx: double
        Mass of the right-handed neutrino
    smix: double
        Right-handed neutrino mixing angle.
    ml: double
        Mass of the lepton corresponding the neutrino flavor which RH neutrino
        mixes with.

    Returns
    -------
    msqrd: float
        The matrix element for N -> nu + pi + pi + gamma for the given model
        and four-momenta.
    """
    cdef double mx = params[0]
    cdef double smix = params[1]
    cdef double ml = params[2]
    cdef vector[double] k1 = momenta[0]
    cdef vector[double] k2 = momenta[1]
    cdef vector[double] k3 = momenta[2]
    cdef vector[double] k4 = momenta[3]

    return (
        -16
        * GF ** 2
        * qe ** 2
        * (-1 + smix ** 2)
        * (smix - 2 * smix * sw ** 2) ** 2
        * (
            ldot(k1, k2) ** 2
            * (
                mpi ** 2 * ldot(k2, k4) ** 2
                - 2 * ldot(k2, k3) * ldot(k2, k4) * ldot(k3, k4)
                + mpi ** 2 * ldot(k3, k4) ** 2
            )
            + ldot(k1, k3) ** 2
            * (
                mpi ** 2 * ldot(k2, k4) ** 2
                - 2 * ldot(k2, k3) * ldot(k2, k4) * ldot(k3, k4)
                + mpi ** 2 * ldot(k3, k4) ** 2
            )
            + ldot(k1, k2)
            * (
                mpi ** 2 * ldot(k2, k4) ** 3
                + mpi ** 2
                * ldot(k3, k4) ** 2
                * (
                    -(mpi ** 2)
                    - 2 * ldot(k1, k3)
                    + 2 * ldot(k1, k4)
                    + ldot(k2, k3)
                    + ldot(k3, k4)
                )
                - ldot(k2, k4) ** 2
                * (
                    mpi ** 4
                    + 2 * mpi ** 2 * ldot(k1, k3)
                    + 2 * mpi ** 2 * ldot(k1, k4)
                    - mpi ** 2 * ldot(k2, k3)
                    + 3 * mpi ** 2 * ldot(k3, k4)
                    + 2 * ldot(k2, k3) * ldot(k3, k4)
                )
                + ldot(k2, k4)
                * ldot(k3, k4)
                * (
                    -2 * ldot(k2, k3) ** 2
                    + mpi ** 2 * ldot(k3, k4)
                    + 2
                    * ldot(k2, k3)
                    * (mpi ** 2 + 2 * ldot(k1, k3) + ldot(k3, k4))
                )
            )
            + ldot(k1, k4)
            * (
                mpi ** 2 * ldot(k2, k4) ** 3
                + mpi ** 2
                * ldot(k3, k4) ** 2
                * (-(mpi ** 2) + ldot(k1, k4) + ldot(k2, k3) + ldot(k3, k4))
                + ldot(k2, k4)
                * ldot(k3, k4)
                * (
                    -2 * ldot(k2, k3) ** 2
                    + mpi ** 2 * ldot(k3, k4)
                    + 2
                    * ldot(k2, k3)
                    * (mpi ** 2 + ldot(k1, k4) + ldot(k3, k4))
                )
                + ldot(k2, k4) ** 2
                * (
                    -(mpi ** 4)
                    + mpi ** 2 * ldot(k1, k4)
                    + mpi ** 2 * ldot(k3, k4)
                    + ldot(k2, k3) * (mpi ** 2 + 2 * ldot(k3, k4))
                )
            )
            + ldot(k1, k3)
            * (
                mpi ** 2 * ldot(k2, k4) ** 3
                + mpi ** 2
                * ldot(k3, k4) ** 2
                * (
                    -(mpi ** 2)
                    - 2 * ldot(k1, k4)
                    + ldot(k2, k3)
                    + ldot(k3, k4)
                )
                - ldot(k2, k4)
                * ldot(k3, k4)
                * (
                    2 * ldot(k2, k3) ** 2
                    - 2 * ldot(k2, k3) * (mpi ** 2 - ldot(k3, k4))
                    + 3 * mpi ** 2 * ldot(k3, k4)
                )
                + ldot(k2, k4) ** 2
                * (
                    -(mpi ** 4)
                    + 2 * mpi ** 2 * ldot(k1, k4)
                    + mpi ** 2 * ldot(k3, k4)
                    + ldot(k2, k3) * (mpi ** 2 + 2 * ldot(k3, k4))
                )
            )
        )
    ) / ((-1 + sw ** 2) * ldot(k2, k4) ** 2 * ldot(k3, k4) ** 2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double msqrd_l_pi_pi0_g(vector[vector[double]] &momenta, vector[double]& params):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    charged lepton, neutral pion, a charged pion and a photon at leading order
    in the Fermi constant. Momenta are ordered as follows: {l-,pi+,pi0,photon}.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> l + pi + pi0 + gamma for the given model
        and four-momenta.
    """
    cdef double mx = params[0]
    cdef double smix = params[1]
    cdef double ml = params[2]
    cdef vector[double] k1 = momenta[0]
    cdef vector[double] k2 = momenta[1]
    cdef vector[double] k3 = momenta[2]
    cdef vector[double] k4 = momenta[3]

    return (
        -8
        * GF ** 2
        * qe ** 2
        * smix ** 2
        * Vud ** 2
        * (
            2 * mpi ** 2 * ldot(k1, k4) ** 4
            + ldot(k1, k4) ** 3
            * (
                mpi ** 4
                - 3 * mpi ** 2 * mpi0 ** 2
                - 4 * mpi ** 2 * ldot(k1, k3)
                + 2 * mpi ** 2 * ldot(k2, k3)
                + 4 * ldot(k1, k2) * (mpi ** 2 - ldot(k2, k4))
                + 4 * mpi ** 2 * ldot(k2, k4)
                - 6 * ldot(k2, k3) * ldot(k2, k4)
                - 2 * ldot(k2, k4) ** 2
                + 2 * mpi ** 2 * ldot(k3, k4)
            )
            + ml ** 2
            * ldot(k2, k4) ** 2
            * (
                -(ml ** 2 * mpi ** 2)
                - ml ** 2 * mpi0 ** 2
                + 2 * ldot(k1, k2) ** 2
                + 2 * ldot(k1, k3) ** 2
                + 2 * ml ** 2 * ldot(k2, k3)
                + mpi ** 2 * ldot(k2, k4)
                - 3 * mpi0 ** 2 * ldot(k2, k4)
                + 2 * ldot(k2, k3) * ldot(k2, k4)
                + 2 * ldot(k2, k4) ** 2
                + ldot(k1, k2)
                * (
                    mpi ** 2
                    - 3 * mpi0 ** 2
                    - 4 * ldot(k1, k3)
                    + 2 * ldot(k2, k3)
                    + 4 * ldot(k2, k4)
                    - 4 * ldot(k3, k4)
                )
                - 3 * mpi ** 2 * ldot(k3, k4)
                + mpi0 ** 2 * ldot(k3, k4)
                + 2 * ldot(k2, k3) * ldot(k3, k4)
                - 4 * ldot(k2, k4) * ldot(k3, k4)
                + 2 * ldot(k3, k4) ** 2
                + ldot(k1, k3)
                * (
                    -3 * mpi ** 2
                    + mpi0 ** 2
                    + 2 * ldot(k2, k3)
                    - 4 * ldot(k2, k4)
                    + 4 * ldot(k3, k4)
                )
            )
            + ldot(k1, k4) ** 2
            * (
                -(ml ** 2 * mpi ** 4)
                - ml ** 2 * mpi ** 2 * mpi0 ** 2
                + 2 * mpi ** 2 * ldot(k1, k3) ** 2
                + 2 * ml ** 2 * mpi ** 2 * ldot(k2, k3)
                + 2 * ldot(k1, k2) ** 2 * (mpi ** 2 - 4 * ldot(k2, k4))
                + mpi ** 4 * ldot(k2, k4)
                - 3 * mpi ** 2 * mpi0 ** 2 * ldot(k2, k4)
                - 2 * ml ** 2 * ldot(k2, k3) * ldot(k2, k4)
                - mpi ** 2 * ldot(k2, k3) * ldot(k2, k4)
                + mpi0 ** 2 * ldot(k2, k3) * ldot(k2, k4)
                + 2 * ldot(k2, k3) ** 2 * ldot(k2, k4)
                + mpi ** 2 * ldot(k2, k4) ** 2
                + 3 * mpi0 ** 2 * ldot(k2, k4) ** 2
                - 8 * ldot(k2, k3) * ldot(k2, k4) ** 2
                - 4 * ldot(k2, k4) ** 3
                + 2 * ml ** 2 * mpi ** 2 * ldot(k3, k4)
                + 2 * mpi ** 2 * ldot(k2, k4) * ldot(k3, k4)
                + 2 * ldot(k2, k3) * ldot(k2, k4) * ldot(k3, k4)
                + 4 * ldot(k2, k4) ** 2 * ldot(k3, k4)
                + ldot(k1, k2)
                * (
                    mpi ** 4
                    - 3 * mpi ** 2 * mpi0 ** 2
                    + 2 * ldot(k2, k3) * (mpi ** 2 - 5 * ldot(k2, k4))
                    - 4 * ldot(k1, k3) * (mpi ** 2 - 2 * ldot(k2, k4))
                    + 2 * mpi ** 2 * ldot(k2, k4)
                    + 6 * mpi0 ** 2 * ldot(k2, k4)
                    - 10 * ldot(k2, k4) ** 2
                    + 2 * mpi ** 2 * ldot(k3, k4)
                    + 2 * ldot(k2, k4) * ldot(k3, k4)
                )
                + ldot(k1, k3)
                * (
                    -4 * mpi ** 2 * ldot(k2, k4)
                    + 4 * ldot(k2, k4) ** 2
                    + 2 * ldot(k2, k3) * (mpi ** 2 + ldot(k2, k4))
                    + mpi ** 2 * (-3 * mpi ** 2 + mpi0 ** 2 + 2 * ldot(k3, k4))
                )
            )
            - ldot(k1, k4)
            * ldot(k2, k4)
            * (
                4 * ldot(k1, k2) ** 3
                + 2
                * ldot(k1, k2) ** 2
                * (
                    mpi ** 2
                    - 3 * mpi0 ** 2
                    - 4 * ldot(k1, k3)
                    + 2 * ldot(k2, k3)
                    + 4 * ldot(k2, k4)
                    - ldot(k3, k4)
                )
                + ldot(k1, k2)
                * (
                    -2 * ml ** 2 * mpi ** 2
                    - 2 * ml ** 2 * mpi0 ** 2
                    + 4 * ldot(k1, k3) ** 2
                    - 4 * ml ** 2 * ldot(k2, k4)
                    + 2 * mpi ** 2 * ldot(k2, k4)
                    - 6 * mpi0 ** 2 * ldot(k2, k4)
                    + 6 * ldot(k2, k4) ** 2
                    + 2 * ml ** 2 * ldot(k3, k4)
                    - 3 * mpi ** 2 * ldot(k3, k4)
                    + mpi0 ** 2 * ldot(k3, k4)
                    - 8 * ldot(k2, k4) * ldot(k3, k4)
                    + 2 * ldot(k3, k4) ** 2
                    + 2
                    * ldot(k2, k3)
                    * (2 * ml ** 2 + 2 * ldot(k2, k4) + ldot(k3, k4))
                    + 2
                    * ldot(k1, k3)
                    * (
                        -3 * mpi ** 2
                        + mpi0 ** 2
                        + 2 * ldot(k2, k3)
                        - 7 * ldot(k2, k4)
                        + 3 * ldot(k3, k4)
                    )
                )
                + ldot(k2, k4)
                * (
                    -(ml ** 2 * mpi ** 2)
                    + 3 * ml ** 2 * mpi0 ** 2
                    + 2 * ldot(k1, k3) ** 2
                    - 2 * ml ** 2 * ldot(k2, k4)
                    + mpi ** 2 * ldot(k2, k4)
                    - 3 * mpi0 ** 2 * ldot(k2, k4)
                    + 2 * ldot(k2, k4) ** 2
                    + 2 * ml ** 2 * ldot(k3, k4)
                    - 3 * mpi ** 2 * ldot(k3, k4)
                    + mpi0 ** 2 * ldot(k3, k4)
                    - 4 * ldot(k2, k4) * ldot(k3, k4)
                    + 2 * ldot(k3, k4) ** 2
                    + 2
                    * ldot(k2, k3)
                    * (-(ml ** 2) + ldot(k2, k4) + ldot(k3, k4))
                    + ldot(k1, k3)
                    * (
                        2 * ml ** 2
                        - 3 * mpi ** 2
                        + mpi0 ** 2
                        + 2 * ldot(k2, k3)
                        - 4 * ldot(k2, k4)
                        + 4 * ldot(k3, k4)
                    )
                )
            )
        )
    ) / (ldot(k1, k4) ** 2 * ldot(k2, k4) ** 2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double msqrd_nu_l_l_g(vector[vector[double]] &momenta, vector[double]& params):
    """
    Compute the squared matrix-element for a RH neutrino decaying into a
    neutrino, two charged leptons and a photon at leading order in the
    Fermi constant. Momenta are ordered as follows: {nu, l+, l-, photon}.

    Parameters
    ----------
    momenta: List
        List of NumPy arrays storing the four-momenta of the final state
        particles.

    Returns
    -------
    msqrd: float
        The matrix element for N -> nu + l + l + gamma for the given model and
        four-momenta.
    """
    cdef double mx = params[0]
    cdef double smix = params[1]
    cdef double ml = params[2]
    cdef vector[double] k1 = momenta[0]
    cdef vector[double] k2 = momenta[1]
    cdef vector[double] k3 = momenta[2]
    cdef vector[double] k4 = momenta[3]

    return (
        16
        * GF ** 2
        * qe ** 2
        * smix ** 2
        * (-1 + smix ** 2)
        * (
            2
            * (1 + 4 * sw ** 2 + 8 * sw ** 4)
            * ldot(k1, k3) ** 2
            * ldot(k2, k4) ** 2
            * ldot(k3, k4)
            + 2
            * (1 + 4 * sw ** 2 + 8 * sw ** 4)
            * ldot(k1, k2) ** 2
            * ldot(k2, k4)
            * ldot(k3, k4) ** 2
            + ldot(k1, k4)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * ldot(k2, k4) ** 3
                * (ml ** 2 - ldot(k3, k4))
                + ml ** 2
                * ldot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k2, k3)
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k3, k4)
                )
                + ldot(k2, k4) ** 2
                * (
                    (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * ldot(k2, k3)
                    * (ml ** 2 - 2 * ldot(k3, k4))
                    + ml ** 2
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (-1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k3, k4)
                    )
                )
                - ldot(k2, k4)
                * ldot(k3, k4)
                * (
                    -8 * ml ** 2 * sw ** 2 * (1 + 2 * sw ** 2) * ldot(k1, k4)
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k2, k3) ** 2
                    + 2
                    * ldot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k3, k4)
                    )
                    + ldot(k3, k4)
                    * (
                        ml ** 2 * (1 - 4 * sw ** 2 - 8 * sw ** 4)
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k3, k4)
                    )
                )
            )
            + ldot(k1, k3)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * ldot(k2, k4) ** 3
                * (ml ** 2 - ldot(k3, k4))
                + ml ** 2
                * ldot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k1, k4)
                    + ldot(k2, k3)
                    + ldot(k3, k4)
                    + 4
                    * sw ** 2
                    * (1 + 2 * sw ** 2)
                    * (ldot(k2, k3) + ldot(k3, k4))
                )
                + ldot(k2, k4) ** 2
                * (
                    ml ** 4 * (1 + 4 * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (
                        ldot(k2, k3) * (ml ** 2 - 2 * ldot(k3, k4))
                        + ml ** 2 * ldot(k3, k4)
                    )
                )
                - ldot(k2, k4)
                * ldot(k3, k4)
                * (
                    2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k2, k3) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * ldot(k3, k4)
                    * (-(ml ** 2) + 2 * ldot(k1, k4) + ldot(k3, k4))
                    + 2
                    * ldot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k1, k4)
                        + (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k3, k4)
                    )
                )
            )
            + ldot(k1, k2)
            * (
                (1 + 4 * sw ** 2 + 8 * sw ** 4)
                * ldot(k2, k4) ** 3
                * (ml ** 2 - ldot(k3, k4))
                + ldot(k2, k4)
                * ldot(k3, k4)
                * (
                    -2
                    * ldot(k2, k3)
                    * (
                        (ml + 4 * ml * sw ** 2) ** 2
                        + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k1, k3)
                        + ldot(k1, k4)
                        + ldot(k2, k3)
                        + 4
                        * sw ** 2
                        * (1 + 2 * sw ** 2)
                        * (ldot(k1, k4) + ldot(k2, k3))
                    )
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (ml ** 2 - 2 * ldot(k1, k3) - 2 * ldot(k2, k3))
                    * ldot(k3, k4)
                    - (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k3, k4) ** 2
                )
                + ml ** 2
                * ldot(k3, k4) ** 2
                * (
                    (ml + 4 * ml * sw ** 2) ** 2
                    + 2 * (1 + 4 * sw ** 2 + 8 * sw ** 4) * ldot(k1, k3)
                    + ldot(k2, k3)
                    + ldot(k3, k4)
                    + 4
                    * sw ** 2
                    * (1 + 2 * sw ** 2)
                    * (ldot(k2, k3) + ldot(k3, k4))
                )
                + ldot(k2, k4) ** 2
                * (
                    ml ** 4 * (1 + 4 * sw ** 2) ** 2
                    + (1 + 4 * sw ** 2 + 8 * sw ** 4)
                    * (
                        ml ** 2
                        * (2 * (ldot(k1, k3) + ldot(k1, k4)) + ldot(k2, k3))
                        + (
                            ml ** 2
                            - 2 * ldot(k1, k3)
                            - 2 * ldot(k1, k4)
                            - 2 * ldot(k2, k3)
                        )
                        * ldot(k3, k4)
                    )
                )
            )
        )
    ) / (ldot(k2, k4) ** 2 * ldot(k3, k4) ** 2)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _dnde_nu_l_l_fsr(vector[double] photon_energies, double mx, double smix, double width, int genv, int genl1, int genl2):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    an active neutrino and two charged leptons.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_energies: float or np.array
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float or np.array
        Photon spectrum.
    """
    cdef vector[double] isp_masses = [mx]
    cdef vector[double] fsp_masses = [0.0, ml, ml]
    cdef vector[double] params = [mx, smix, ml]
    cdef int nevents = 10000

    return c_gamma_ray_fsr(
        photon_energies,
        mx,
        isp_masses,
        fsp_masses,
        width,
        msqrd_nu_l_l_g,
        nevents,
        params,
    ).first


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _dnde_l_pi_pi0_fsr(vector[double] photon_energies, double mx, double smix, double ml, double width):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    a neutral pion, charged pion and charged lepton.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_energies: float or np.array
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float or np.array
        Photon spectrum.
    """
    cdef vector[double] isp_masses = [mx]
    cdef vector[double] fsp_masses = [ml, mpi, mpi0]
    cdef vector[double] params = [mx, smix, ml]
    cdef int nevents = 5000

    return c_gamma_ray_fsr(
        photon_energies,
        mx,
        isp_masses,
        fsp_masses,
        width,
        msqrd_l_pi_pi0_g,
        nevents,
        params,
    ).first


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _dnde_nu_pi_pi_fsr(vector[double] photon_energies, double mx, double smix, double ml, double width):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    an active neutrino and two charged pions.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_energies: float or np.array
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float or np.array
        Photon spectrum.
    """
    cdef vector[double] isp_masses = [mx]
    cdef vector[double] fsp_masses = [0.0, mpi, mpi]
    cdef vector[double] params = [mx, smix, ml]
    cdef int nevents = 1000

    return c_gamma_ray_fsr(
        photon_energies,
        mx,
        isp_masses,
        fsp_masses,
        width,
        msqrd_nu_pi_pi_g,
        nevents,
        params,
    ).first


def dnde_nu_l_l_fsr(self, photon_energies, width, gens=None):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    an active neutrino and two charged leptons.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_energies: float or np.array
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float or np.array
        Photon spectrum.
    """
    return _dnde_nu_l_l_fsr(photon_energies, self.mx, self.theta, self.ml, width)


def dnde_l_pi_pi0_fsr(self, photon_energies, width):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    a neutral pion, charged pion and charged lepton.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_energies: float or np.array
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float or np.array
        Photon spectrum.
    """
    return _dnde_l_pi_pi0_fsr(photon_energies, self.mx, self.theta, self.ml, width)


def dnde_nu_pi_pi_fsr(self, photon_energies,width):
    """
    Compute the FSR spectra from a right-handed neutrino decaying into
    an active neutrino and two charged pions.

    Parameters
    ----------
    self: object
        Instance of the `RHNeutrino` class. (see
        `hazma.rh_neutrino.__init__.py`)
    photon_energies: float or np.array
       The energy of the final state photon in MeV.

    Returns
    -------
    dnde: float or np.array
        Photon spectrum.
    """
    return _dnde_nu_pi_pi_fsr(photon_energies, self.mx, self.theta, self.ml, width)
