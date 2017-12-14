from decay_helper_functions import decay_charged_kaon


def decay_spectra(eng_gam, eng_k):
    """
    Compute dNdE from charged kaon decay.

    Compute dNdE from decay of charged kaon through the following decay modes:
        k -> mu  + nu
        k -> pi  + pi0
        k -> pi  + pi  + pi
        k -> pi0 + e   + nu
        k -> pi0 + mu  + nu
        k -> pi  + pi0 + pi0
    in the laboratory frame given a gamma ray engergy of `eng_gam` and charged
    kaon energy of `eng_k`.

    Paramaters
        eng_gam (float or numpy.ndarray) : Gamma ray energy(ies) in laboratory
        frame.
        eng_k (float) : Charged kaon energy in laboratory frame.

    Returns
        spec (np.ndarray) : List of gamma ray spectrum values, dNdE, evaluated
        at `eng_gams` given muon energy `eng_mu`.
    """
    charged_k = decay_charged_kaon.ChargedKaon()
    if hasattr(eng_gam, "__len__"):
        return charged_k.Spectrum(eng_gam, eng_k)
    return charged_k.SpectrumPoint(eng_gam, eng_k)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    NOT YET IMPLEMENTED!
    """
    raise ValueError('FSR spectrum for charged pion is not yet available')
