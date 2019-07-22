"""
# Module for computing relativistic phase space points.

# Authors: Logan Morrison and Adam Coogan
# Date: December 2017

TODO: Code up specific functions for cross-section
      functions for 2->2 processes.
TODO: Code up specific functions for cross-section
      functions for 2->3 processes.

"""
from hazma.phase_space_helper_functions import generator
from hazma.phase_space_helper_functions import histogram
from hazma.phase_space_helper_functions.modifiers import apply_matrix_elem
import numpy as np
import multiprocessing as mp
import warnings

from hazma.hazma_errors import RamboCMETooSmall

from hazma.field_theory_helper_functions.common_functions import cross_section_prefactor


def generate_phase_space_point(masses, cme):
    """
    Generate a phase space point given a set of final state particles and a
    given center of mass energy.

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.

    Returns
    -------
    phase_space_points : numpy.ndarray
        List of four momenta and a event weight. The returned numpy array is of
        the form::

            [ke1, kx1, ky1, kz1, ..., keN, kxN, kyN, kzN, weight]

    """

    if not hasattr(masses, "__len__"):
        masses = [masses]

    masses = np.array(masses)
    return generator.generate_point(masses, cme, len(masses))


def generate_phase_space(
    masses, cme, num_ps_pts=10000, mat_elem_sqrd=lambda klist: 1, num_cpus=None
):
    """
    Generate a specified number of phase space points given a set of
    final state particles and a given center of mass energy.
    Note that the weights are not normalized.

    Parameters
    ----------
    masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    num_ps_pts : int {10000]
        Total number of phase space points to generate.
    mat_elem_sqrd : (double)(numpy.ndarray) {lambda klist: 1]
        Function for the matrix element squared.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.

    Returns
    -------
    phase_space_points : numpy.ndarray
        List of phase space points. The phase space points are in the form::

            [[ke11, kx11, ky11, kz11, ..., keN1, kxN1, kyN1, kzN1, weight1],
             ...
             [ke1N, kx1N, ky1N, kz1N, ..., keNN, kxNN, kyNN, kzNN, weightN]]

    Examples
    --------

    Generate 100000 phase space points for a 3 body final state::

        from hazma import rambo
        import numpy as np
        masses = np.array([100., 200., 0.0])
        cme = 10.0 * sum(masses)
        num_ps_pts = 100000
        rambo.generate_phase_space(masses, cme, num_ps_pts=num_ps_pts)

    """
    if not hasattr(masses, "__len__"):
        masses = [masses]

    if cme < sum(masses):
        raise RamboCMETooSmall()

    num_fsp = len(masses)
    # If the user doesn't specify the number of cpus to use,
    # use 75% of them.
    if num_cpus is not None:
        if num_cpus > num_ps_pts:
            num_cpus = num_ps_pts
        if num_cpus > mp.cpu_count():
            num_cpus = int(np.floor(mp.cpu_count() * 0.75))
            warnings.warn(
                """You only have {] cpus.
                          Using {] cpus instead.
                          """.format(
                    mp.cpu_count(), num_cpus
                )
            )
    if num_cpus is None:
        # Use 75% of the cpu power.
        num_cpus = int(np.floor(mp.cpu_count() * 0.75))
        # If user wants a number of phase space points which is less
        # than the number of cpus available, use num_ps_pts cpus instead.
        if num_cpus > num_ps_pts:
            num_cpus = num_ps_pts
    # Instantiate `num_cpus` number of workers and divide num_ps_pts among the
    # the workers to speed up phase space generation.
    pool = mp.Pool(num_cpus)
    num_ps_pts_per_cpu = int(num_ps_pts / num_cpus)
    # If num_ps_pts % num_cpus !=0, then we need to compute the actual number
    # of phase space points.
    actual_num_ps_pts = num_ps_pts_per_cpu * num_cpus
    # Create a container to store the results from the workers
    job_results = []
    # Run the jobs on 75% of the cpus.
    for i in range(num_cpus):
        job_results.append(
            pool.apply_async(
                generator.generate_space, (num_ps_pts_per_cpu, masses, cme, num_fsp)
            )
        )
    # Close the pool and wait for results to finish
    pool.close()
    pool.join()
    # Put results in a numpy array.
    points = np.array([result.get() for result in job_results])
    # Flatten the outer axis to have a list of phase space points.
    points = points.reshape(actual_num_ps_pts, 4 * num_fsp + 1)
    # Resize the weights to have the correct cross section.
    points = apply_matrix_elem(points, actual_num_ps_pts, num_fsp, mat_elem_sqrd)

    return points


def generate_energy_histogram(
    masses,
    cme,
    num_ps_pts=10000,
    mat_elem_sqrd=lambda klist: 1,
    num_bins=25,
    num_cpus=None,
    density=False,
):
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
    mat_elem_sqrd : (double)(numpy.ndarray) {lambda klist: 1]
        Function for the matrix element squared.
    num_bins : int
        Number of energy bins to use for each of the final state particles.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.
    density: Bool
        If true, the histograms will be normalized to have unit area underneath
        the curves, i.e. they will be probability density functions.

    Returns
    -------
    energy_histograms : numpy.ndarray
        List of energies and dsigma/dE's. The resulting array has the shape
        (num_fsp, 2, num_bins). The array is formatted as::

            [[[E11, E12, ....], [hist11, hist12, ...]],
             ...
             [[EN1, EN2, ....], [histM1, histN2, ...]]]

    Examples
    --------

    Making energy histograms for 4 final state particles and plotting their
    energy spectra::

        from hazma import rambo
        import numpy as np
        num_ps_pts = 100000
        masses = np.array([100., 100., 0.0, 0.0])
        cme = 1000.
        num_bins = 100
        eng_hist = rambo.generate_energy_histogram(masses, cme,
                                                   num_ps_pts=num_ps_pts
                                                   num_bins=num_bins)
        import matplotlib.pyplot as plt
        for i in range(len(masses)):
            plt.loglog(eng_hist[i, 0], eng_hist[i, 1])

    """
    if not hasattr(masses, "__len__"):
        masses = [masses]

    if cme < sum(masses):
        raise RamboCMETooSmall()

    num_fsp = len(masses)

    pts = generate_phase_space(masses, cme, num_ps_pts, mat_elem_sqrd, num_cpus)

    actual_num_ps_pts = pts.shape[0]

    return histogram.space_to_energy_hist(
        pts, actual_num_ps_pts, num_fsp, num_bins, density=density
    )


def integrate_over_phase_space(
    fsp_masses, cme, num_ps_pts=10000, mat_elem_sqrd=lambda momenta: 1, num_cpus=None
):
    """
    Returns the integral over phase space given a squared matrix element, a
    set of final state particle masses and a given energy.

    Parameters
    ----------
    fsp_masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    num_ps_pts : int {10000]
        Total number of phase space points to generate.
    mat_elem_sqrd : (double)(numpy.ndarray) {lambda momenta: 1]
        Function for the matrix element squared.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.

    Returns
    -------
    integral : float
        The result of the integral over phase space.
    std : float
        The estimated error in the integral over phase space.

    """
    if not hasattr(fsp_masses, "__len__"):
        fsp_masses = [fsp_masses]

    if cme < sum(fsp_masses):
        raise RamboCMETooSmall()

    num_fsp = len(fsp_masses)
    points = generate_phase_space(fsp_masses, cme, num_ps_pts, mat_elem_sqrd, num_cpus)
    actual_num_ps_pts = len(points[:, 4 * num_fsp])
    weights = points[:, 4 * num_fsp]
    integral = np.average(weights)
    std = np.std(weights) / np.sqrt(actual_num_ps_pts)

    return integral, std


def compute_annihilation_cross_section(
    isp_masses,
    fsp_masses,
    cme,
    num_ps_pts=10000,
    mat_elem_sqrd=lambda momenta: 1,
    num_cpus=None,
):
    """
    Computes the cross section for a given process.

    Parameters
    ----------
    isp_masses : numpy.ndarray
        List of masses of the initial state particles.
    fsp_masses : numpy.ndarray
        List of masses of the final state and particles.
    cme : double
        Center-of-mass-energy of the process.
    num_ps_pts : int
        Total number of phase space points to generate.
    mat_elem_sqrd : (double)(numpy.ndarray) {lambda momenta: 1]
        Function for the matrix element squared.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.

    Returns
    -------
    cross_section : double
        Cross section for X -> final state particles(fsp), where the fsp have
        masses `masses` and the process X -> fsp has a squared matrix element
        of `mat_elem_sqrd`.
    std : double
        Estimated error in cross section.

    Examples
    --------

    Compute the cross section for electrons annihilating into muons through a
    photon. First, we construct a function for the matrix element::

        from hazma.parameters import electron_mass as me
        from hazma.parameters import muon_mass as mmu
        from hazma.parameters import qe
        MDot = lambda p1, p2: (p1[0] * p2[0] - p1[1] * p1[1] - p1[2] * p1[2] -
                               p1[3] * p1[3])
        def msqrd(momenta):
            Q = sum(momenta)[0] # center-of-mass energy
            # Momenta of the incoming electrons
            p1 = np.array([Q / 2., 0., 0., np.sqrt(Q**2 / 4. - me**2)])
            p2 = np.array([Q / 2., 0., 0., -np.sqrt(Q**2 / 4. - me**2)])
            # Momenta for the outgoing muons
            p3 = momenta[0]
            p4 = momenta[1]
            # Mandelstam variables
            s = MDot(p1 + p2, p1 + p2)
            t = MDot(p1 - p3, p1 - p3)
            u = MDot(p1 - p4, p1 - p4)
            return (2 * qe**4 * (t**2 + u**2 - 4 * (t + u)* me**2 +
                    6 * me**4 + 4 * s * mmu**2 + 2 * mmu**4)) / s**2

    Next we integrate over phase space using RAMBO::

        from hazma.rambo import compute_annihilation_cross_section
        import numpy as np
        isp_masses = np.array([me, me])
        fsp_masses = np.array([mmu, mmu])
        cme = 1000.
        compute_annihilation_cross_section(
            isp_masses, fsp_masses, cme, num_ps_pts=5000, mat_elem_sqrd=msqrd)

    """
    integral, std = integrate_over_phase_space(
        fsp_masses,
        cme,
        num_ps_pts=num_ps_pts,
        mat_elem_sqrd=mat_elem_sqrd,
        num_cpus=num_cpus,
    )

    m1 = isp_masses[0]
    m2 = isp_masses[1]

    cross_section = integral * cross_section_prefactor(m1, m2, cme)
    error = cross_section_prefactor(m1, m2, cme) * std

    return cross_section, error


def compute_decay_width(
    fsp_masses, cme, num_ps_pts=10000, mat_elem_sqrd=lambda momenta: 1, num_cpus=None
):
    r"""
    Computes the decay width for a given process.

    Parameters
    ----------
    fsp_masses : numpy.ndarray
        List of masses of the final state particles.
    cme : double
        Center-of-mass-energy of the process.
    num_ps_pts : int
        Total number of phase space points to generate.
    mat_elem_sqrd : (double)(numpy.ndarray) {lambda momenta: 1]
        Function for the matrix element squared.
    num_cpus : int {None]
        Number of cpus to use in parallel with rambo. If not specified, 75% of
        the cpus will be used.

    Returns
    -------
    cross_section : double
        Cross section for X -> final state particles (FSPs), where the FSPs have
        masses `masses` and the process X -> FSPs has a squared matrix element
        of `mat_elem_sqrd`.
    std : double
        Estimated error in cross section.

    Examples
    --------

    In this example we compute the partial decay width of the muon for
    :math:`\mu~\to~e\nu\nu`

    First, we declare the matrix element::

        from hazma.parameters import GF
        MDot = lambda p1, p2: (p1[0] * p2[0] - p1[1] * p1[1] - p1[2] * p1[2] -
                               p1[3] * p1[3])
        def msqrd(momenta):
            pe = momenta[0]
            pve = momenta[1]
            pvmu = momenta[2]
            pmu = sum(momenta)
            return 64. * GF**2 * MDot(pe, pvmu) * MDot(pmu, pve)

    Next, we compute the decay width::

        from hazma.rambo import compute_decay_width
        from hazma.parameters import electron_mass as me
        from hazma.parameters import muon_mass as mmu
        import numpy as np
        fsp_masses = np.array([me, 0.0, 0.0])
        cme = mmu
        compute_decay_width(fsp_masses, cme, mat_elem_sqrd=msqrd)
    """
    integral, std = integrate_over_phase_space(
        fsp_masses,
        cme,
        num_ps_pts=num_ps_pts,
        mat_elem_sqrd=mat_elem_sqrd,
        num_cpus=num_cpus,
    )

    cross_section = integral / (2.0 * cme)
    error = std / (2.0 * cme)

    return cross_section, error
