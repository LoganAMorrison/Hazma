from . import _decay_neutral_pion


def decay_spectra(eng_gam, eng_pi):
    """
    Returns zero. Electron is stable.
    """
    pi0 = _decay_neutral_pion.NeutralPion()

    if hasattr(eng_gam, '__len__'):
        return pi0.Spectrum(eng_gam, eng_pi)
    return pi0.SpectrumPoint(eng_gam, eng_pi)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    Returns zero.
    """
    return 0.0
