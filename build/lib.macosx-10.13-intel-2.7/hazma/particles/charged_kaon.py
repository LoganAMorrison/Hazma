from . import _decay_charged_kaon


def decay_spectra(eng_gam, eng_k):
    charged_k = _decay_charged_kaon.ChargedKaon()
    if hasattr(eng_gam, "__len__"):
        return charged_k.Spectrum(eng_gam, eng_k)
    return charged_k.SpectrumPoint(eng_gam, eng_k)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    NOT YET IMPLEMENTED!
    """
    raise ValueError('FSR spectrum for charged pion is not yet available')
