# Structure to hold the spectrum decomposed into neutrino flavors
cdef struct NeutrinoSpectrumPoint:
    double electron
    double muon
    double tau


cdef NeutrinoSpectrumPoint new_neutrino_spectrum_point()
