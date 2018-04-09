import numpy as np

from ..decay import muon

from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me

from .vector_mediator_fsr import dnde_xx_to_v_to_ffg

# from .vector_mediator_cross_sections import branching_fractions


def dnde_ee(egams, cme, params, type='All'):
    fsr = np.vectorize(dnde_xx_to_v_to_ffg)

    if type == 'All':
        return fsr(egams, cme / 2., me, params)
    if type == 'FSR':
        return fsr(egams, cme / 2., me, params)
    if type == 'Decay':
        return np.arrary([0.0 for _ in range(len(egams))])
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(type))


def dnde_mumu(egams, cme, params, type='All'):
    fsr = np.vectorize(dnde_xx_to_v_to_ffg)
    decay = np.vectorize(muon)

    if type == 'All':
        mu_decay = decay(egams, cme / 2.0)
        mu_fsr = fsr(egams, cme / 2., mmu, params)
        return 2. * mu_decay + mu_fsr
    if type == 'FSR':
        return fsr(egams, cme / 2., mmu, params)
    if type == 'Decay':
        return 2. * decay(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(type))


def spectra(egams, cme, params):
    """
    Compute the total spectrum from two fermions annihilating through a
    scalar mediator to mesons and leptons.

    Parameters
    ----------
    cme : float
        Center of mass energy.
    egams : array-like, optional
        Gamma ray energies to evaluate the spectrum at.

    Returns
    -------
    specs : dictionary
        Dictionary of the spectra. The keys are 'total', 'mu mu', 'e e',
        'pi0 pi0', 'pi pi', 'k k', 'k0 k0'.
    """

    # Compute branching fractions
    # bfs = branching_fractions(cme, params)

    # Leptons
    # muons = bfs['mu mu'] * dnde_mumu(egams, cme, params)
    # electrons = bfs['e e'] * dnde_ee(egams, cme, params)

    muons = dnde_mumu(egams, cme, params)
    electrons = dnde_ee(egams, cme, params)

    # Compute total spectrum
    total = muons + electrons

    # Define dictionary for spectra
    specs = {'total': total, 'mu mu': muons, 'e e': electrons}

    return specs
