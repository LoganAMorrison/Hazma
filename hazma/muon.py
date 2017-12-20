"""
Module for computing gamma ray spectra from a muon.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""
from .decay_helper_functions import decay_muon
from .fsr_helper_functions import scalar_mediator_fsr,\
    pseudo_scalar_mediator_fsr
from .fsr_helper_functions import vector_mediator
import numpy as np

MASS_MU = 105.6583715


def decay_spectra(eng_gam, eng_mu):
    """
    Compute dNdE from muon decay.

    Compute dNdE from decay mu -> e nu nu gamma in the laborartory frame given
    a gamma ray engergy of `eng_gam` and muon energy of `eng_mu`.

    Paramaters
        eng_gam (float or numpy.ndarray) :
            Gamma ray energy(ies) in laboratory frame.
        eng_mu (float) :
            Muon energy in laboratory frame.

    Returns
        spec (np.ndarray) List of gamma ray spectrum values, dNdE, evaluated at
        `eng_gams` given muon energy `eng_mu`.

    Examples
        Calculate spectrum for single gamma ray energy

        >>> from hazma import muon
        >>> eng_gam, eng_mu = 200., 1000.
        >>> spec = muon.decay_spectra(eng_gam, eng_mu)

        Calculate spectrum for array of gamma ray energies

        >>> from hazma import muon
        >>> import numpy as np
        >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
        >>> eng_mu = 1000.
        >>> spec = muon.decay_spectra(eng_gams, eng_mu)
    """
    # mu = decay_muon.Muon()
    if hasattr(eng_gam, "__len__"):
        return decay_muon.Spectrum(eng_gam, eng_mu)
    return decay_muon.SpectrumPoint(eng_gam, eng_mu)


def fsr(eng_gam, cme, mediator='scalar'):
    """
    Compute muon fsr spectrum.

    Compute final state radiation spectrum dN/dE from decay of an off-shell
    mediator (scalar, psuedo-scalar, vector or axial-vector) into a pair of
    muons.

    Paramaters
        eng_gam (float or np.ndarray) : Gamma ray energy(ies) in laboratory
        frame.
        cme (float) : Center of mass energy or mass of the off-shell mediator.
        mediator (string) : Mediator type : scalar, psuedo-scalar, vector or
        axial-vector.

    Returns
        spec (np.ndarray) : List of gamma ray spectrum values, dNdE, evaluated
        at `eng_gams` given a center of mass energy `cme`.

    Examples
        dNdE for a single gamma ray energy from scalar mediator.

        >>> from hazma import muon
        >>> eng_gam, cme = 200., 1000.
        >>> spec = muon.fsr(eng_gam, cme, 'scalar')

        dNdE for list of gamma ray energies from vector mediator.

        >>> from hazma import muon
        >>> eng_gams = np.logspace(0.0, 3.0, num=1000, dtype=float)
        >>> cme = 1000.
        >>> spec = muon.fsr(eng_gams, cme, 'scalar')
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
