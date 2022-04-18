import numpy as np
cimport numpy as np
from hazma.spectra._neutrino._neutrino cimport NeutrinoSpectrumPoint

cdef NeutrinoSpectrumPoint c_muon_decay_spectrum_point(double, double)
cdef np.ndarray[np.float64_t,ndim=2] c_muon_decay_spectrum_array(double[:], double)
