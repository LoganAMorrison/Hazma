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
# include "parameters.pxd


"""
cdef int __num_phase_space_pts = 1000
cdef np.ndarray __massesPiPiPi
cdef np.ndarray __massesPi0MuNu
cdef np.ndarray __massesPi0ENu
cdef np.ndarray __massesPiPiPi0Pi0

cdef np.ndarray __engsPiPiPi = np.zero(__num_phase_space_pts, dtype=float)
cdef np.ndarray __engsPi0MuNu = np.zero(__num_phase_space_pts, dtype=float)
cdef np.ndarray __engsPi0ENu = np.zero(__num_phase_space_pts, dtype=float)
cdef np.ndarray __engsPiPiPi0Pi0 = np.zero(__num_phase_space_pts, dtype=float)

__ramboPiPiPi = Rambo.Rambo()
__ramboPi0MuNu = Rambo.Rambo()
__ramboPi0ENu = Rambo.Rambo()
__ramboPiPiPi0Pi0 = Rambo.Rambo()

__massesPiPiPi = np.array([MASS_PI, MASS_PI, MASS_PI], dtype=np.float64)
__massesPi0MuNu = np.array([MASS_PI0, MASS_MU, 0.0], dtype=np.float64)
__massesPi0ENu = np.array([MASS_PI0, MASS_E, 0.0], dtype=np.float64)
__massesPiPiPi0Pi0 = np.array([MASS_PI, MASS_PI, MASS_PI0, MASS_PI0],\
    dtype=np.float64)

__psPiPiPi, __weightsPiPiPi \
    = ramboPiPiPi.generate_phase_space(__num_phase_space_pts, \
                                       __massesPiPiPi, MASS_K)

__psPi0MuNu, __weightsPi0MuNu \
    = ramboPi0MuNu.generate_phase_space(__num_phase_space_pts, \
                                        __massesPi0MuNu, MASS_K)

__psPiPiPi, __weightsPiPiPi \
    = ramboPiPiPi.generate_phase_space(__num_phase_space_pts, \
                                       __massesPiPiPi, MASS_K)

__psPiPiPi, __weightsPiPiPi \
    = ramboPiPiPi.generate_phase_space(__num_phase_space_pts, \
                                       __massesPiPiPi, MASS_K)
"""



cdef class ChargedKaon:
    """
    Class for computing the photon spectrum from radiative kaon decay.
    """



    def __init__(self):
        pass
