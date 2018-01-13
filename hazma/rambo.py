"""
High level module to generate relativistic phase space points.

* Author - Logan A. Morrison and Adam Coogan
* Date - December 2017
"""
from .phase_space_helper_functions import generator
from .phase_space_helper_functions import histogram
from .phase_space_helper_functions.modifiers import apply_matrix_elem
import numpy as np
import multiprocessing as mp


def generate_phase_space_point(masses, cme, num_fsp):
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
    return generator.generate_point(masses, cme, num_fsp)


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

    Examples
    --------

    Generate 100000 phase space points for a 3 body final state.

    >>> from hazma import rambo
    >>> import numpy as np
    >>> masses = np.array([100., 200., 0.0])
    >>> cme = 1000.
    >>> num_ps_pts = 100000
    >>> num_fsp = len(masses)
    >>>
    >>> pts = rambo.generate_phase_space(num_ps_pts, masses, cme)
    """
    num_fsp = len(masses)
    # Use 75% of the cpu power.
    num_cpus = int(np.floor(mp.cpu_count() * 0.75))
    # Instantiate `num_cpus` number of workers and divide num_ps_pts among the
    # the workers to speed up phase space generation.
    pool = mp.Pool(num_cpus)
    num_ps_pts_per_cpu = num_ps_pts / num_cpus
    # If num_ps_pts % num_cpus !=0, then we need to compute the actual number
    # of phase space points.
    actual_num_ps_pts = num_ps_pts_per_cpu * num_cpus
    # Create a container to store the results from the workers
    job_results = []
    # Run the jobs on 75% of the cpus.
    for i in range(num_cpus):
        job_results.append(pool.apply_async(generator.generate_space,
                                            (num_ps_pts_per_cpu,
                                             masses, cme, num_fsp)))
    # Put results in a numpy array.
    points = np.array([result.get() for result in job_results])
    # Flatten the outer axis to have a list of phase space points.
    points = points.reshape(actual_num_ps_pts, 4 * num_fsp + 1)
    # Resize the weights to have the correct cross section.
    points = apply_matrix_elem(
        points, actual_num_ps_pts, num_fsp, mat_elem_sqrd)
    # Divide the weights by the number of phase space points used.
    points[:, 4 * num_fsp] = points[:, 4 * num_fsp] * (1.0 / actual_num_ps_pts)

    return points


def generate_energy_histogram(num_ps_pts, masses, cme,
                              mat_elem_sqrd=lambda klist: 1, num_bins=25):
    """
    Generate energy histograms for each of the final state particles.

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
    num_bins : int
        Number of energy bins to use for each of the final state particles.

    Returns
    -------
    energy_histograms : numpy.ndarray
        List of energies and dsigma/dE's. The resulting array has the shape
        (num_fsp, 2, num_bins). The array is formatted such that
        energy_histograms =
        {{{E11, E12, ....}, {hist11, hist12, ...}},
                    .,
                    .,
                    .,
        {{EN1, EN2, ....}, {histM1, histN2, ...}}}.

    Examples
    --------
    Making energy histograms for 4 final state particles and plotting their
    energy spectra.

    >>> from hazma import rambo
    >>> import numpy as np
    >>> num_ps_pts = 100000
    >>> masses = np.array([100., 100., 0.0, 0.0])
    >>> cme = 1000.
    >>> num_bins = 100
    >>>
    >>> eng_hist = rambo.generate_energy_histogram(num_ps_pts, masses, cme,
    ...                                            num_bins=num_bins)
    >>> import matplotlib as plt
    >>> for i in range(len(masses)):
    ...     plt.loglog(pts[i, 0], pts[i, 1])
    """
    num_fsp = len(masses)

    pts = generate_phase_space(num_ps_pts, masses, cme, mat_elem_sqrd)

    return histogram.space_to_energy_hist(pts, num_ps_pts, num_fsp, num_bins)
