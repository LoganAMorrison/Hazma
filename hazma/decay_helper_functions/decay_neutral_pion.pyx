from libc.math cimport sqrt
import numpy as np
cimport numpy as np
import cython
include "parameters.pxd"


cdef class NeutralPion:
    """
    Class for computing the decay spectrum from neutral pion.
    """

    def __init__(self):
        pass

    @cython.cdivision(True)
    def SpectrumPoint(self, eng_gam, float eng_pi):
        """
        Returns decay spectrum for pi0 -> g g.
        """
        cdef float beta = sqrt(1.0 - (MASS_PI0 / eng_pi)**2)

        cdef float ret_val = 0.0

        if eng_pi * (1 - beta) / 2.0 <= eng_gam and \
                eng_gam <= eng_pi * (1 + beta) / 2.0:
            ret_val = BR_PI0_TO_GG * 2.0 / (eng_pi * beta)

        return ret_val

    @cython.cdivision(True)
    def Spectrum(self, np.ndarray eng_gam, float eng_pi):
        """
        Returns decay spectrum for pi0 -> g g.
        """
        cdef float beta = sqrt(1.0 - (MASS_PI0 / eng_pi)**2)
        cdef int numpts = len(eng_gam)
        cdef np.ndarray spec = np.zeros(numpts, float)
        cdef int i
        cdef float ret_val

        for i in range(numpts):

            ret_val = 0.0

            if eng_pi * (1 - beta) / 2.0 <= eng_gam[i] and \
                    eng_gam[i] <= eng_pi * (1 + beta) / 2.0:
                ret_val = BR_PI0_TO_GG * 2.0 / (eng_pi * beta)
            spec[i] = ret_val

        return spec
