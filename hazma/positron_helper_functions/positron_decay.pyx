"""Module for computing positron spectra from a many-particle final state.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""
import numpy as np
cimport numpy as np
import cython
import multiprocessing as mp

from hazma import rambo
from hazma.rambo import compute_annihilation_cross_section
from hazma.rambo import compute_decay_width

from hazma.positron_helper_functions import positron_muon
from hazma.positron_helper_functions import positron_charged_pion

include "../decay_helper_functions/parameters.pxd"

cdef spec_dict = {'muon': positron_muon.Spectrum,
                  'charged_pion': positron_charged_pion.Spectrum}

cdef mass_dict = {'muon': MASS_MU, 'charged_pion': MASS_PI,
                  'neutral_pion': MASS_PI0, 'electron': MASS_E}

cdef dict cspec_dict = spec_dict
cdef dict cmass_dict = mass_dict

cdef np.ndarray names_to_masses(np.ndarray names):
    """Returns the masses of particles given a list of their names.

    Parameters
    ----------
    names : np.ndarray
        List of names of the final state particles.

    Returns
    -------
    masses : np.ndarray
        List of masses of the final state particles. For
        example, ['electron', 'muon'] -> [0.510998928, 105.6583715].
    """
    cdef int i
    cdef int size = len(names)
    cdef np.ndarray masses = np.zeros(size, dtype=np.float64)

    for i in range(size):
        masses[i] = cmass_dict[names[i]]
    return masses

def __gen_spec(name, eng, eng_ps, norm, verbose=False):
    """
    c-function used by ``gamma`` and ``gamma_point`` to generate spectrum
    values.
    """
    if verbose is True:
        print("creating {} spectrum with energy {}".format(name, eng))
    try:
        return norm * cspec_dict[name](eng_ps, eng)
    except:
        return np.zeros(len(eng_ps), dtype=np.float64)

def __gen_spec_2body(particles, cme, eng_ps):
    masses = names_to_masses(particles)

    m1 = masses[0]
    m2 = masses[1]

    E1 = (cme * cme + m1 * m1 - m2 * m2) / (2 * cme)
    E2 = (cme * cme - m1 * m1 + m2 * m2) / (2 * cme)

    spec = cspec_dict[particles[0]](eng_ps, E1)
    spec += cspec_dict[particles[1]](eng_ps, E2)

    return spec

@cython.boundscheck(False)
@cython.wraparound(False)
def positron(np.ndarray particles, double cme,
             np.ndarray eng_ps, mat_elem_sqrd=lambda k_list: 1.0,
             int num_ps_pts=10000, int num_bins=25, verbose=False):
    """Returns total gamma ray spectrum from final state particles.

    Parameters
    ----------
    particles : np.ndarray[string, ndim=1]
        1-D array of strings containing the final state particle names. The
        accepted particle names are: "muon", and "charged_pion".
    cme : double
        Center of mass energy of the final state.
    eng_gams : np.ndarray[double, ndim=1]
        List of gamma ray energies to compute spectra at.
    mat_elem_sqrd : double(*)(np.ndarray)
        Function for the matrix element squared of the proccess. Must be
        a function taking in a list of four momenta of size (num_fsp, 4).
        Default value is a flat matrix element.
    num_ps_pts : int
        Number of phase space points to use.
    num_bins : int
        Number of bins to use.

    Returns
    -------
    spec : np.ndarray[double, ndim=1]
        1-D array of total gamma ray spectrum from all final state particles.
    """

    if len(particles) == 1:
        return cspec_dict[particles[0]](eng_ps, cme)

    if len(particles) == 2:
        return __gen_spec_2body(particles, cme, eng_ps)

    cdef int i, j
    cdef int num_fsp
    cdef int num_engs
    cdef np.ndarray masses
    cdef np.ndarray hist
    cdef double prefactor

    masses = names_to_masses(particles)

    num_fsp = len(masses)
    num_engs = len(eng_ps)

    hist = rambo.generate_energy_histogram(masses, cme, num_ps_pts, mat_elem_sqrd, num_bins, density=True)[0]

    cpdef int num_cpus = int(np.floor(mp.cpu_count() * 0.75))

    p = mp.Pool(num_cpus)
    specs = []

    for i in range(num_bins):
        for j in range(num_fsp):
            part = particles[j]  # particle name
            part_eng = hist[j, 0, i]  # particle energy

            # Normalize spectrum: Need to multiply by the probability of
            # particle having energy part_eng. Since we are essentially
            # integrating over the energy probability distribution, we need to
            # multiply by (b - a) / N, where a = e_min, b = e_max and
            # N = num_bins.
            norm = (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins * hist[j, 1, i]

            specs.append(p.apply_async(__gen_spec,
                                       (part, part_eng, eng_ps, norm, verbose)))

    p.close()
    p.join()

    return sum([spec.get() for spec in specs])

@cython.boundscheck(False)
@cython.wraparound(False)
def positron_point(np.ndarray particles, double cme,
                   double eng_p, mat_elem_sqrd=lambda k_list: 1.0,
                   int num_ps_pts=1000, int num_bins=25):
    """
    Returns gamma ray spectrum at single gamma ray enegy from final state
    particles.

    Parameters
    ----------
    particles : np.ndarray[string, ndim=1]
        1-D array of strings containing the final state particle names. The
        accepted particle names are: "muon", and "charged_pion".
    cme : double
        Center of mass energy of the final state.
    eng_gam : double
        Gamma ray energy to evaluate spectrum at.
    mat_elem_sqrd : double(*)(np.ndarray)
        Function for the matrix element squared of the proccess. Must be
        a function taking in a list of four momenta of size (num_fsp, 4).
        Default value is a flat matrix element.
    num_ps_pts : int
        Number of phase space points to use.
    num_bins : int
        Number of bins to use.

    Returns
    -------
    spec : double
        Spectrum evaluated at ``eng_gam``.
    """
    cdef int i, j
    cdef int num_fsp
    cdef np.ndarray masses
    cdef np.ndarray hist
    cdef double spec_val = 0.0
    cdef double prefactor

    masses = names_to_masses(particles)

    num_fsp = len(masses)

    hist = rambo.generate_energy_histogram(masses, cme, num_ps_pts, mat_elem_sqrd, num_bins)[0]

    for i in range(num_bins):
        for j in range(num_fsp):
            if particles[j] is 'muon':
                spec_val += hist[j, 1, i] * \
                            positron_muon.SpectrumPoint(eng_p, hist[j, 0, i]) * \
                            (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins
            if particles[j] is 'charged_pion':
                spec_val += hist[j, 1, i] * \
                            positron_charged_pion.Spectrum(eng_p, hist[j, 0, i],
                                                           "total") * \
                            (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins
            else:
                spec_val += 0.0

    return spec_val * prefactor
