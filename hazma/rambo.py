"""
High level module to generate relativistic phase space points.

* Author - Logan A. Morrison and Adam Coogan
* Date - December 2017
"""
from .phase_space_helper_functions import generator
from .phase_space_helper_functions import histogram
from .phase_space_helper_functions.modifiers import normalize_weights
from .phase_space_helper_functions.modifiers import apply_matrix_elem
import numpy as np


def split_point(l, num_fsp):
    kList = np.zeros((num_fsp, 4), dtype=np.float64)
    for i in xrange(num_fsp):
        for j in xrange(4):
            kList[i, j] = l[4 * i + j]
    return kList


def generate_phase_space_point(masses, cme):
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
    return generator.generate_point(masses, cme)


def generate_phase_space(num_ps_pts, masses, cme,
                         mat_elem_sqrd=lambda klist: 1):
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
    num_fsp = len(masses)

    points = np.array([generate_phase_space_point(masses, cme)
                       for _ in range(num_ps_pts)])
    # points = generator.generate_space(num_ps_pts, masses, cme)
    # points = normalize_weights(points, num_ps_pts, num_fsp)
    points = apply_matrix_elem(points, num_ps_pts, num_fsp, mat_elem_sqrd)
    points[:, 4 * num_fsp] = points[:, 4 * num_fsp] * (1.0 / num_ps_pts)
    return points


def generate_energy_histogram(num_ps_pts, masses, cme,
                              mat_elem_sqrd=lambda klist: 1, num_bins=25):
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
    num_fsp = len(masses)

    pts = generate_phase_space(num_ps_pts, masses, cme, mat_elem_sqrd)

    return histogram.space_to_energy_hist(pts, num_ps_pts, num_fsp, num_bins)
