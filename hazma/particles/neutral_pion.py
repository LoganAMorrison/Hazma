import numpy as np

MASS_PI0 = 134.9766
BR_PI0_TO_GG = 0.9882


def decay_spectra(eng_gam, eng_pi):
    """
    Returns zero. Electron is stable.
    """
    beta = np.sqrt(1.0 - (MASS_PI0 / eng_pi)**2)

    ret_val = 0.0

    if eng_pi * (1 + beta) / 2.0 <= eng_gam and \
            eng_gam <= eng_pi * (1 + beta) / 2.0:
        ret_val = BR_PI0_TO_GG * 2.0 / (eng_pi * beta)

    return ret_val


def fsr(eng_gam, cme, mediator='scalar'):
    """
    Returns zero.
    """
    return 0.0
