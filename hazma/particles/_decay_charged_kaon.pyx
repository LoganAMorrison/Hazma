from ._decay_muon import Muon
from ._decay_charged_pion import ChargedPion
from ._decay_neutral_pion import NeutralPion
from ..rambo.rambo import Rambo
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from libc.math cimport exp, log, M_PI, log10, sqrt
import cython
from functools import partial
include "parameters.pxd"



cdef int __num_ps_pts = 1000
cdef int __num_bins = 25

# Mass arrays
cdef np.ndarray __msPiPiPi = np.array([MASS_PI, MASS_PI, MASS_PI], \
    dtype=np.float64)
cdef np.ndarray __msPi0MuNu = np.array([MASS_PI0, MASS_MU, 0.0],\
    dtype=np.float64)
cdef np.ndarray __mssPi0ENu = np.array([MASS_PI0, MASS_E, 0.0],\
    dtype=np.float64)
cdef np.ndarray __msPiPiPi0Pi0 = np.array([MASS_PI, MASS_PI, MASS_PI0, \
    MASS_PI0], dtype=np.float64)

# Energy probability distributions
cdef np.ndarray __probsPiPiPi = np.zero((3, 2, __num_bins), dtype=float)
cdef np.ndarray __probsPi0MuNu = np.zero((3, 2, __num_bins), dtype=float)
cdef np.ndarray __probsPi0ENu = np.zero((3, 2, __num_bins), dtype=float)
cdef np.ndarray __probsPiPiPi0Pi0 = np.zero((3, 2, __num_bins), dtype=float)

__ram = Rambo.Rambo()

__probsPiPiPi \
    = __ram.generate_energy_histogram(__num_ps_pts, __msPiPiPi, MASS_K, 25)

__probsPi0MuNu \
    = __ram.generate_energy_histogram(__num_ps_pts, __msPi0MuNu, MASS_K, 25)

__probsPi0ENu \
    = __ram.generate_energy_histogram(__num_ps_pts, __mssPi0ENu, MASS_K, 25)

__probsPiPiPi0Pi0 \
    = __ram.generate_energy_histogram(__num_ps_pts,__msPiPiPi0Pi0, MASS_K, 25)



cdef class ChargedKaon:
    """
    Class for computing the photon spectrum from radiative kaon decay.
    """

    def __init__(self):
        pass
