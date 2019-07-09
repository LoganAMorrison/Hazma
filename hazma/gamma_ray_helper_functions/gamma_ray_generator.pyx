"""Module for computing gamma ray spectra from a many-particle final state.

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

from hazma.decay import muon as dm
from hazma.decay import electron as de

from hazma.decay import neutral_pion as dnp
from hazma.decay import charged_pion as dcp

from hazma.decay import charged_kaon as dck
from hazma.decay import long_kaon as dlk
from hazma.decay import short_kaon as dsk

include "../decay_helper_functions/parameters.pxd"

cdef spec_dict = {'muon': dm, 'electron': de, 'neutral_pion': dnp,
                  'charged_pion': dcp, 'charged_kaon': dck,
                  'long_kaon': dlk, 'short_kaon': dsk}

cdef mass_dict = {'muon': MASS_MU, 'electron': MASS_E,
                  'neutral_pion': MASS_PI0, 'charged_pion': MASS_PI, 'charged_kaon': MASS_K, 'long_kaon': MASS_K0,
                  'short_kaon': MASS_K0}

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

def __gen_spec(name, eng, eng_gams, norm, verbose=False):
    """
    c-function used by ``gamma`` and ``gamma_point`` to generate spectrum
    values.
    """
    if verbose is True:
        print("creating {} spectrum with energy {}".format(name, eng))
    return norm * cspec_dict[name](eng_gams, eng)

def __gen_spec_2body(particles, cme, eng_gams):
    masses = names_to_masses(particles)

    m1 = masses[0]
    m2 = masses[1]

    E1 = (cme**2 + m1**2 - m2**2) / (2 * cme)
    E2 = (cme**2 - m1**2 + m2**2) / (2 * cme)

    spec = cspec_dict[particles[0]](eng_gams, E1)
    spec += cspec_dict[particles[1]](eng_gams, E2)

    return spec

@cython.boundscheck(False)
@cython.wraparound(False)
def gamma(np.ndarray particles, double cme,
          np.ndarray eng_gams, mat_elem_sqrd=lambda k_list: 1.0,
          int num_ps_pts=10000, int num_bins=25, verbose=False):
    """Returns total gamma ray spectrum from final state particles.

    Parameters
    ----------
    particles : np.ndarray[string, ndim=1]
        1-D array of strings containing the final state particle names. The
        accepted particle names are: "muon", "electron", "neutral_pion",
        "charged_pion", "long_kaon", "short_kaon" and "charged_kaon".
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
        return cspec_dict[particles[0]](eng_gams, cme)

    if len(particles) == 2:
        return __gen_spec_2body(particles, cme, eng_gams)

    cdef int i, j
    cdef int num_fsp
    cdef int num_engs
    cdef np.ndarray masses
    cdef np.ndarray hist
    cdef double prefactor

    masses = names_to_masses(particles)

    num_fsp = len(masses)
    num_engs = len(eng_gams)

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

            specs.append(p.apply_async(__gen_spec, \
                                       (part, part_eng, eng_gams, norm, verbose)))

    p.close()
    p.join()

    return sum([spec.get() for spec in specs])

@cython.boundscheck(False)
@cython.wraparound(False)
def gamma_point(np.ndarray particles, double cme,
                double eng_gam, mat_elem_sqrd=lambda k_list: 1.0,
                int num_ps_pts=1000, int num_bins=25):
    """Returns total gamma ray spectrum from final state particles.

    Parameters
    ----------
    particles : np.ndarray[string, ndim=1]
        1-D array of strings containing the final state particle names. The
        accepted particle names are: "muon", "electron", "neutral_pion",
        "charged_pion", "long_kaon", "short_kaon" and "charged_kaon".
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
            if particles[j] is 'electron':
                spec_val += 0.0
            if particles[j] is 'muon':
                spec_val += hist[j, 1, i] * \
                            dm(eng_gam, hist[j, 0, i]) * \
                            (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins
            if particles[j] is 'charged_pion':
                spec_val += hist[j, 1, i] * \
                            dcp(eng_gam, hist[j, 0, i], "total") * \
                            (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins
            if particles[j] is 'neutral_pion':
                spec_val += hist[j, 1, i] * \
                            dnp(eng_gam, hist[j, 0, i]) * \
                            (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins
            if particles[j] is 'charged_kaon':
                spec_val += hist[j, 1, i] * \
                            dck(eng_gam, hist[j, 0, i], "total") * \
                            (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins
            if particles[j] is 'short_kaon':
                spec_val += hist[j, 1, i] * \
                            dsk(eng_gam, hist[j, 0, i], "total") * \
                            (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins
            if particles[j] is 'long_kaon':
                spec_val += hist[j, 1, i] * \
                            dlk(eng_gam, hist[j, 0, i], "total") * \
                            (hist[j, 0, -1] - hist[j, 0, 0]) / num_bins

    return spec_val * prefactor
