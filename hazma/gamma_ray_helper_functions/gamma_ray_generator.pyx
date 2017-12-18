import numpy as np
cimport numpy as np
import cython

from ..phase_space_generator cimport rambo
from .. import electron
from .. import muon

from .. import neutral_pion
from .. import charged_pion

from .. import charged_kaon
from .. import short_kaon
from .. import long_kaon

include "../decay_helper_functions/parameters.pxd"

def name_to_mass(name):
    if name is 'electron':
        return MASS_E
    if name is 'muon':
        return MASS_MU
    if name is 'charged_pion':
        return MASS_PI
    if name is 'neutral_pion':
        return MASS_PI0
    if name is 'charged_kaon':
        return MASS_K
    if name is 'short_kaon':
        return MASS_K0
    if name is 'long_kaon':
        return MASS_K0


def name_to_func(name):
    if name is 'electron':
        return electron.decay_spectra
    if name is 'muon':
        return muon.decay_spectra
    if name is 'charged_pion':
        return charged_pion.decay_spectra
    if name is 'neutral_pion':
        return neutral_pion.decay_spectra
    if name is 'charged_kaon':
        return charged_kaon.decay_spectra
    if name is 'short_kaon':
        return short_kaon.decay_spectra
    if name is 'long_kaon':
        return long_kaon.decay_spectra


def name_to_func_point(name):
    if name is 'electron':
        return electron.decay_spectra_point
    if name is 'muon':
        return muon.decay_spectra_point
    if name is 'charged_pion':
        return charged_pion.decay_spectra_point
    if name is 'neutral_pion':
        return neutral_pion.decay_spectra_point
    if name is 'charged_kaon':
        return charged_kaon.decay_spectra_point
    if name is 'short_kaon':
        return short_kaon.decay_spectra_point
    if name is 'long_kaon':
        return long_kaon.decay_spectra_point


@cython.boundscheck(False)
@cython.wraparound(False)
def gamma(particles, double cme, np.ndarray eng_gams,
          mat_elem_sqrd=lambda k_list : 1.0,
          int num_ps_pts=1000, int num_bins=25):

    cdef int i, j
    cdef int __num_fsp
    cdef int __num_engs
    cdef np.ndarray __masses
    cdef np.ndarray __probs
    cdef np.ndarray __spec
    cdef rambo.Rambo __ram

    __masses = np.array([name_to_mass(name) for name in particles])

    __num_fsp = len(__masses)
    __num_engs = len(eng_gams)

    __ram = rambo.Rambo()

    __probs = __ram.generate_energy_histogram(num_ps_pts, __masses,
                                              cme, num_bins)

    __spec = np.zeros(__num_engs, dtype=np.float64)

    __funcs = np.array([name_to_func(name) for name in particles])

    for i in range(num_bins):
        for j in range(__num_fsp):
            __spec += __probs[j, 1, i] * __funcs[j](eng_gams, __probs[j, 0, i])

    return __spec

@cython.boundscheck(False)
@cython.wraparound(False)
def gamma_point(particles, double cme, double eng_gam,
          mat_elem_sqrd=lambda k_list : 1.0,
          int num_ps_pts=1000, int num_bins=25):

    cdef int i, j
    cdef int __num_fsp
    cdef np.ndarray __masses
    cdef np.ndarray __probs
    cdef double __spec_val = 0.0
    # cdef np.ndarray __funcs
    cdef rambo.Rambo __ram

    __masses = np.array([name_to_mass(name) for name in particles])

    __num_fsp = len(__masses)

    __ram = rambo.Rambo()

    __probs = __ram.generate_energy_histogram(num_ps_pts, __masses,
                                            cme, num_bins)

    __funcs = np.array([name_to_func_point(name) for name in particles])

    for i in range(num_bins):
        for j in range(__num_fsp):
            __spec_val += \
                __probs[j, 1, i] * __funcs[j](eng_gam, __probs[j, 0, i])

    return __spec_val
