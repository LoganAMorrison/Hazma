"""
Module for computing gamma ray spectra from a many-particle final state.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

import numpy as np
cimport numpy as np
import cython
import multiprocessing as mp

from ..phase_space_generator cimport rambo

from ..decay_helper_functions import decay_muon as dm
from ..decay_helper_functions import decay_electron as de

from ..decay_helper_functions import decay_neutral_pion as dnp
from ..decay_helper_functions import decay_charged_pion as dcp

from ..decay_helper_functions import decay_charged_kaon as dck
from ..decay_helper_functions import decay_long_kaon as dlk
from ..decay_helper_functions import decay_short_kaon as dsk

include "../decay_helper_functions/parameters.pxd"


cdef np.ndarray names_to_masses(np.ndarray names):
    """
    Returns the masses of particles given a list of their names.

    Parameters:
        names (np.ndarray) : List of names of the final state particles.

    Returns:
        masses (np.ndarray) : List of masses of the final state particles. For
        example, ['electron', 'muon'] -> [0.510998928, 105.6583715].
    """
    cdef int i
    cdef int size = len(names)
    cdef np.ndarray masses = np.zeros(size, dtype=np.float64)

    for i in range(size):
        if names[i] is 'electron':
            masses[i] = MASS_E
        if names[i] is 'muon':
            masses[i] = MASS_MU
        if names[i] is 'charged_pion':
            masses[i] = MASS_PI
        if names[i] is 'neutral_pion':
            masses[i] = MASS_PI0
        if names[i] is 'charged_kaon':
            masses[i] = MASS_K
        if names[i] is 'short_kaon':
            masses[i] = MASS_K0
        if names[i] is 'long_kaon':
            masses[i] = MASS_K0
    return masses


def __gen_spec(name, prob, eng, eng_gams):

    if name == 'electron':
        return prob * de.Spectrum(eng_gams, eng)
    if name == 'muon':
        return prob * dm.Spectrum(eng_gams, eng)
    if name == 'charged_pion':
        return prob * dcp.Spectrum(eng_gams, eng)
    if name == 'neutral_pion':
        return prob * dnp.Spectrum(eng_gams, eng)
    if name == 'charged_kaon':
        return prob * dck.Spectrum(eng_gams, eng)
    if name == 'short_kaon':
        return prob * dsk.Spectrum(eng_gams, eng)
    if name == 'long_kaon':
        return prob * dlk.Spectrum(eng_gams, eng)


@cython.boundscheck(False)
@cython.wraparound(False)
def gamma(np.ndarray particles, double cme, np.ndarray eng_gams,
          mat_elem_sqrd=lambda k_list : 1.0,
          int num_ps_pts=1000, int num_bins=25):
    """
    Returns total gamma ray spectrum from final state particles.

    Parameters:
        particles (np.ndarray[string, ndim=1]) :
            List of particle names.
        cme (double) :
            Center of mass energy of the final state.
        eng_gams (np.ndarray[double, ndim=1]) :
            List of gamma ray energies to compute spectra at.
        mat_elem_sqrd (double(np.ndarray)) :
            Function for the matrix element squared of the proccess. Must be
            a function taking in a list of four momenta of size (num_fsp, 4).
            Default value is a flat matrix element.
        num_ps_pts (int) :
            Number of phase space points to use.
        num_bins (int) :
            Number of bins to use.

    Returns:
        spec (np.ndarray) :
            Total gamma ray spectrum from all final state particles.
    """
    cdef int i, j
    cdef int __num_fsp
    cdef int __num_engs
    cdef np.ndarray __masses
    cdef np.ndarray __probs
    cdef np.ndarray __spec
    cdef rambo.Rambo __ram

    __masses = names_to_masses(particles)

    __num_fsp = len(__masses)
    __num_engs = len(eng_gams)

    __ram = rambo.Rambo()

    __probs = __ram.generate_energy_histogram(num_ps_pts, __masses,
                                              cme, num_bins)

    __spec = np.zeros(__num_engs, dtype=np.float64)

    p = mp.Pool(4)
    specs = []

    for i in range(num_bins):
        for j in range(__num_fsp):
            specs.append(p.apply_async(__gen_spec, (particles[j],
                                                      __probs[j, 1, i], \
                                                      __probs[j, 0, i], eng_gams)))


    __spec = sum([spec.get() for spec in specs])


    """
    for i in range(num_bins):
        for j in range(__num_fsp):
            if particles[j] == 'electron':
                __spec += __probs[j, 1, i] * \
                    de.CSpectrum(eng_gams, __probs[j, 0, i])
            if particles[j] == 'muon':
                __spec += __probs[j, 1, i] * \
                    dm.CSpectrum(eng_gams, __probs[j, 0, i])
            if particles[j] == 'charged_pion':
                __spec += __probs[j, 1, i] * \
                    dcp.CSpectrum(eng_gams, __probs[j, 0, i])
            if particles[j] == 'neutral_pion':
                __spec += __probs[j, 1, i] * \
                    dnp.CSpectrum(eng_gams, __probs[j, 0, i])
            if particles[j] == 'charged_kaon':
                __spec += __probs[j, 1, i] * \
                    dck.Spectrum(eng_gams, __probs[j, 0, i])
            if particles[j] == 'short_kaon':
                __spec += __probs[j, 1, i] * \
                    dsk.Spectrum(eng_gams, __probs[j, 0, i])
            if particles[j] == 'long_kaon':
                __spec += __probs[j, 1, i] * \
                    dlk.Spectrum(eng_gams, __probs[j, 0, i])
    """
    return __spec


@cython.boundscheck(False)
@cython.wraparound(False)
def gamma_point(np.ndarray particles, double cme, double eng_gam,
          mat_elem_sqrd=lambda k_list : 1.0,
          int num_ps_pts=1000, int num_bins=25):
    """
    Returns total gamma ray spectrum from final state particles.

    Parameters:
        particles (np.ndarray[string, ndim=1]) :
            List of particle names.
        cme (double) :
            Center of mass energy of the final state.
        eng_gam (double) :
            Gamma ray energy to compute spectrum at.
        mat_elem_sqrd (double(np.ndarray)) :
            Function for the matrix element squared of the proccess. Must be
            a function taking in a list of four momenta of size (num_fsp, 4).
            Default value is a flat matrix element.
        num_ps_pts (int) :
            Number of phase space points to use.
        num_bins (int) :
            Number of bins to use.

    Returns:
        spec (np.ndarray) :
            Total gamma ray spectrum from all final state particles.
    """
    cdef int i, j
    cdef int __num_fsp
    cdef np.ndarray __masses
    cdef np.ndarray __probs
    cdef double __spec_val = 0.0
    cdef rambo.Rambo __ram

    __masses = names_to_masses(particles)

    __num_fsp = len(__masses)

    __ram = rambo.Rambo()

    __probs = __ram.generate_energy_histogram(num_ps_pts, __masses,
                                            cme, num_bins)

    for i in range(num_bins):
        for j in range(__num_fsp):
            if particles[j] is 'electron':
                __spec += __probs[j, 1, i] * \
                    de.CSpectrumPoint(eng_gam, __probs[j, 0, i])
            if particles[j] is 'muon':
                __spec += __probs[j, 1, i] * \
                    dm.CSpectrumPoint(eng_gam, __probs[j, 0, i])
            if particles[j] is 'charged_pion':
                __spec += __probs[j, 1, i] * \
                    dcp.CSpectrumPoint(eng_gam, __probs[j, 0, i])
            if particles[j] is 'neutral_pion':
                __spec += __probs[j, 1, i] * \
                    dnp.CSpectrumPoint(eng_gam, __probs[j, 0, i])
            if particles[j] is 'charged_kaon':
                __spec += __probs[j, 1, i] * \
                    dck.SpectrumPoint(eng_gam, __probs[j, 0, i])
            if particles[j] is 'short_kaon':
                __spec += __probs[j, 1, i] * \
                    dsk.SpectrumPoint(eng_gam, __probs[j, 0, i])
            if particles[j] is 'long_kaon':
                __spec += __probs[j, 1, i] * \
                    dlk.SpectrumPoint(eng_gam, __probs[j, 0, i])

    return __spec_val
