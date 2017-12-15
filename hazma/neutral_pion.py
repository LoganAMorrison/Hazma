from decay_helper_functions import decay_neutral_pion


def decay_spectra(eng_gam, eng_pi):
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
    if hasattr(eng_gam, '__len__'):
        return decay_neutral_pion.Spectrum(eng_gam, eng_pi)
    return decay_neutral_pion.SpectrumPoint(eng_gam, eng_pi)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    Returns zero.
    """
    return 0.0
