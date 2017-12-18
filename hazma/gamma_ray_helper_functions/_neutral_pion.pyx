from ..decay_helper_functions cimport decay_neutral_pion


cdef double decay_spectra_point(double eng_gam, double eng_pi):
    """
    Compute dNdE from neutral pion decay.

    Compute dNdE from decay pi0 -> gamma gamma in the laborartory frame given
    a gamma ray engergy of `eng_gam` and neutral pion energy of `eng_pi`.

    Paramaters
        eng_gam (float/numpy.ndarray) :
            Gamma ray energy(ies) in laboratory frame.
        eng_pi (float) :
            Neutral pion energy in laboratory frame.

    Returns
        spec (np.ndarray) :
            List of gamma ray spectrum values, dNdE, evaluated at
            `eng_gams` given neutral pion energy `eng_pi`.
    """
    return decay_neutral_pion.CSpectrumPoint(eng_gam, eng_pi)



cdef np.ndarray decay_spectra(np.ndarray eng_gam, double eng_k):
    """
    Compute dNdE from neutral pion decay.

    Compute dNdE from decay pi0 -> gamma gamma in the laborartory frame given
    a gamma ray engergy of `eng_gam` and neutral pion energy of `eng_pi`.

    Paramaters
        eng_gam (float/numpy.ndarray) :
            Gamma ray energy(ies) in laboratory frame.
        eng_pi (float) :
            Neutral pion energy in laboratory frame.

    Returns
        spec (np.ndarray) :
            List of gamma ray spectrum values, dNdE, evaluated at
            `eng_gams` given neutral pion energy `eng_pi`.
    """
    return decay_neutral_pion.CSpectrum(eng_gam, eng_pi)


cdef fsr(eng_gam, cme, mediator='scalar'):
    """
    Returns zero.
    """
    return 0.0
