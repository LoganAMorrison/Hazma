import numpy as np
cimport numpy as np
import cython

from ..phase_space_generator import rambo
from . cimport _electron
from . cimport _muon

from . cimport _neutral_pion
from . cimport _charged_pion

from . cimport _charged_kaon
from . cimport _short_kaon
from . cimport _long_kaon

include "../decay_helper_functions/paramaters.pxd"


cdef double name_to_mass(name):
    if name is 'electron':
        return MASS_E
    if name is 'muon':
        return MASS_MU
    if name is 'charged pion':
        return MASS_PI
    if name is 'neutral pion':
        return MASS_PI0
    if name is 'charged kaon':
        return MASS_K
    if name is 'short kaon':
        return MASS_K0
    if name is 'long kaon':
        return MASS_K0


cdef name_to_func(name):
    if name is 'electron':
        return _electron.decay_spectra
    if name is 'muon':
        return _muon.decay_spectra
    if name is 'charged pion':
        return _charged_pion.decay_spectra
    if name is 'neutral pion':
        return _neutral_pion.decay_spectra
    if name is 'charged kaon':
        return _charged_kaon.decay_spectra
    if name is 'short kaon':
        return _short_kaon.decay_spectra
    if name is 'long kaon':
        return _long_kaon.decay_spectra


cdef name_to_func_point(name):
    if name is 'electron':
        return _electron.decay_spectra_point
    if name is 'muon':
        return _muon.decay_spectra_point
    if name is 'charged pion':
        return _charged_pion.decay_spectra_point
    if name is 'neutral pion':
        return _neutral_pion.decay_spectra_point
    if name is 'charged kaon':
        return _charged_kaon.decay_spectra_point
    if name is 'short kaon':
        return _short_kaon.decay_spectra_point
    if name is 'long kaon':
        return _long_kaon.decay_spectra_point


@cython.boundscheck(False)
@cython.wraparound(False)
def gamma(particles, double cme, np.ndarray eng_gams,
          mat_elem_sqrd=double lambda np.ndarray k_list : 1.0,
          int num_ps_pts=1000, int num_bins=25):

    __masses = np.array([name_to_mass(name) for name in particles])

    __num_fsp = len(__masses)
    __num_engs = len(eng_gams)

    __ram  = rambo.Rambo()

    __probs = __ram.generate_energy_histogram(num_ps_pts, __masses,
                                              cme, num_bins)

    __spec = np.zeros((__num_fsp, __num_engs), dtype=np.float64)

    __funcs = np.array([name_to_func(name) for name in particles])

    cdef int i, j

    for i in range(num_bins):
        for j in range(__num_fsp):
            __spec += __probs[j, 1, i] * __funcs[j](eng_gams, __probs[j, 0, i])

    return __spec

@cython.boundscheck(False)
@cython.wraparound(False)
def gamma_point(particles, double cme, double eng_gam,
          mat_elem_sqrd=double lambda np.ndarray k_list : 1.0,
          int num_ps_pts=1000, int num_bins=25):

    __masses = np.array([name_to_mass(name) for name in particles])

    __num_fsp = len(__masses)
    __num_engs = len(eng_gams)

    __ram  = rambo.Rambo()

    __probs = __ram.generate_energy_histogram(num_ps_pts, __masses,
                                              cme, num_bins)

    __spec = np.zeros((__num_fsp, __num_engs), dtype=np.float64)

    __funcs = np.array([name_to_func_point(name) for name in particles])

    cdef int i, j

    for i in range(num_bins):
        for j in range(__num_fsp):
            __spec += __probs[j, 1, i] * __funcs[j](eng_gam, __probs[j, 0, i])

    return __spec
