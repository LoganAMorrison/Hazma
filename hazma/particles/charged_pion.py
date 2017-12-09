from . import _decay_charged_pion


def decay_spectra(eng_gam, eng_pi):
    charged_pi = _decay_charged_pion.ChargedPion()
    if hasattr(eng_gam, "__len__"):
        return charged_pi.Spectrum(eng_gam, eng_pi)
    return charged_pi.SpectrumPoint(eng_gam, eng_pi)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    NOT YET IMPLEMENTED!
    """
    raise ValueError('FSR spectrum for charged pion is not yet available')
