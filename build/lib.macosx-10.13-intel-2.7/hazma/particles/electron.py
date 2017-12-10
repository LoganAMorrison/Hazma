from ..fsr_helper_functions.scalar_mediator_fsr import fermion
import numpy as np

MASS_E = 0.510998928


def decay_spectra(eng_gam, eng_mu):
    """
    Returns zero. Electron is stable.
    """
    return 0.0


def fsr(eng_gam, cme, mediator='scalar'):
    """
    Compute final state radiation spectrum dN_{\gamma}/dE_{\gamma} an
    off-shell scalar decaying into an on-shell electron and an off-shell
    electron, which decays to an on-shell electron and photon: S* -> e* e -> g
    e e.

    Keyword arguments::
        eng_gam (float or np.ndarray)-- Gamma ray energy(ies) in laboratory
                                        frame.
        cme (float) -- Electron energy in laboratory frame.
        mediator (string) -- Mediator type
    """
    if mediator == 'scalar':
        if hasattr(eng_gam, "__len__"):
            fermionvec = np.vectorize(fermion)
            return fermionvec(eng_gam, cme, MASS_E)
        return fermion(eng_gam, cme, MASS_E)
    if mediator == 'psuedo-scalar':
        raise ValueError('psuedo-scalar mediator not yet availible.')
    if mediator == 'vector':
        raise ValueError('vector mediator not yet availible.')
    if mediator == 'axial-vector':
        raise ValueError('axial-vector mediator not yet availible.')
    else:
        raise ValueError(
            '''Invalid mediator: {}. Use scalar, psuedo-scalar,\
               vector or axial-vector.''')
