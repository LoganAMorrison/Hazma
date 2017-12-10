from . import _decay_muon
from ..fsr_helper_functions import scalar_mediator_fsr,\
    pseudo_scalar_mediator_fsr
from ..fsr_helper_functions import vector_mediator
import numpy as np

MASS_MU = 105.6583715


def decay_spectra(eng_gam, eng_mu):
    """
    Compute dN_{\gamma}/dE_{\gamma} from decay mu -> e nu nu gamma in the
    laborartory frame.

    Keyword arguments::
        engGam (float or numpy.ndarray)-- Gamma ray energy in laboratory frame.
        engMu (float) -- Muon energy in laboratory frame.

    Returns:
        Returns a float or numpy.ndarray containing spectrum.
    """
    mu = _decay_muon.Muon()
    if hasattr(eng_gam, "__len__"):
        return mu.Spectrum(eng_gam, eng_mu)
    return mu.SpectrumPoint(eng_gam, eng_mu)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    Compute final state radiation spectrum dN_{\gamma}/dE_{\gamma} an
    off-shell scalar decaying into an on-shell muon and an off-shell muon,
    which decays to an on-shell muon and photon: S* -> mu* mu -> g mu mu.

    Keyword arguments::
        engGam (float or np.ndarray)-- Gamma ray energy in laboratory frame.
        engMu (float) -- Muon energy in laboratory frame.
        mediator (string) -- Mediator type
    """
    if mediator == 'scalar':
        if hasattr(eng_gam, "__len__"):
            smvec = np.vectorize(scalar_mediator_fsr.fermion)
            return smvec(eng_gam, cme, MASS_MU)
        return scalar_mediator_fsr.fermion(eng_gam, cme, MASS_MU)
    if mediator == 'psuedo-scalar':
        if hasattr(eng_gam, "__len__"):
            psm = np.vectorize(pseudo_scalar_mediator_fsr.fermion)
            return psm(eng_gam, cme, MASS_MU)
        return pseudo_scalar_mediator_fsr.fermion(eng_gam, cme, MASS_MU)
    if mediator == 'vector':
        if hasattr(eng_gam, "__len__"):
            vm = np.vectorize(vector_mediator.fermion)
            return vm(eng_gam, cme, MASS_MU)
        return vector_mediator.fermion(eng_gam, cme, MASS_MU)
    if mediator == 'axial-vector':
        raise ValueError('axial-vector mediator not yet availible.')
    else:
        raise ValueError(
            '''Invalid mediator: {}. Use scalar, psuedo-scalar,\
               vector or axial-vector.''')
