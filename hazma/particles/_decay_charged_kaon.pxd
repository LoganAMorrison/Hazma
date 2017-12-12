from ..rambo cimport rambo

cimport _decay_muon
cimport _decay_charged_pion
cimport _decay_neutral_pion

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

    cdef _decay_muon.Muon __muon
    cdef _decay_neutral_pion.NeutralPion __neuPion
    cdef _decay_charged_pion.ChargedPion __chrgpi

    """ Class functions """
    cdef float __integrand2(self, float cl, float eng_gam, float eng_k)
    cdef float __integrand3(self, float cl, float eng_gam, float eng_k)
    cdef float __integrand(self, float cl, float eng_gam, float eng_k)
