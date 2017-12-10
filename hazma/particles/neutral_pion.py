from . import _decay_neutral_pion


def decay_spectra(eng_gam, eng_pi):
    """
    Returns zero. Electron is stable.
    """
    pi0 = _decay_neutral_pion.NeutralPion()

    return pi0.decay_spectra()


def fsr(eng_gam, cme, mediator='scalar'):
    """
    Returns zero.
    """
    return 0.0
