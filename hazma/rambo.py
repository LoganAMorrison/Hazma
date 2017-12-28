"""
High level module to generate relativistic phase space points.

* Author - Logan A. Morrison and Adam Coogan
* Date - December 2017
"""
from .phase_space_helper_functions import phase_space_point_generator as pspg


def generate_phase_space_point(masses, cme, mat_elem_sqrd=lambda klist: 1):
    """
    Generate a phase space point given a set of
    final state particles and a given center of mass energy.

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    mat_elem_sqrd : (double)(numpy.ndarray) {lambda klist: 1}
        Function for the matrix element squared.

    Returns
    -------
    phase_space_points : numpy.ndarray
        List of four momenta and a event weight. The returned numpy array is of
        the form {ke1, kx1, ky1, kz1, ..., keN, kxN, kyN, kzN, weight}.
    """
    point = pspg.generate_phase_space_point(masses, cme)


def generate_phase_space():
    """
    Generate a specified number of phase space points given a set of
    final state particles and a given center of mass energy.

    Parameters
    ----------
    num_ps_pts : int
        Total number of phase space points to generate.
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    mat_elem_sqrd : (double)(numpy.ndarray) {lambda klist: 1}
        Function for the matrix element squared.

    Returns
    -------
    phase_space_points : numpy.ndarray
        List of phase space points. The phase space points are in the form
        {{ke11, kx11, ky11, kz11, ..., keN1, kxN1, kyN1, kzN1, weight1},
            .
            .
            .
         {ke1N, kx1N, ky1N, kz1N, ..., keNN, kxNN, kyNN, kzNN, weightN}}
    """
