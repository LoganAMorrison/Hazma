from ..phases_space_generator cimport rambo

cimport decay_muon
cimport decay_charged_pion
cimport decay_neutral_pion

import numpy as np
cimport numpy as np

cdef class ChargedKaon:
    """ Class variables """
    cdef int __num_ps_pts
    cdef int __num_bins

    cdef np.ndarray __msPiPiPi
    cdef np.ndarray __msPi0MuNu

    cdef np.ndarray __probsPiPiPi
    cdef np.ndarray __probsPi0MuNu

    cdef np.ndarray __funcsPiPiPi
    cdef np.ndarray __funcsPi0MuNu

    cdef rambo.Rambo __ram

    cdef decay_muon.Muon __muon
    cdef decay_neutral_pion.NeutralPion __neuPion
    cdef decay_charged_pion.ChargedPion __chrgpi

    """ Class functions """
    cdef double __integrand2(self, double cl, double eng_gam, double eng_k)
    cdef double __integrand3(self, double cl, double eng_gam, double eng_k)
    cdef double __integrand(self, double cl, double eng_gam, double eng_k)
