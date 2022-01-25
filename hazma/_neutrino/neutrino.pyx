from hazma._neutrino.neutrino cimport NeutrinoSpectrumPoint

cdef NeutrinoSpectrumPoint new_neutrino_spectrum_point():
    cdef NeutrinoSpectrumPoint res
    res.electron = 0.0
    res.muon = 0.0
    res.tau = 0.0
    
    return res

