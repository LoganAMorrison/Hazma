import numpy as np

from ..decay import muon
from ..decay import neutral_pion, charged_pion
from ..decay import short_kaon, long_kaon, charged_kaon

from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me

from .scalar_mediator_fsr import dnde_xx_to_s_to_ffg
from .scalar_mediator_fsr import dnde_xx_to_s_to_pipig
from .scalar_mediator_fsr import dnde_xx_to_s_to_pipig_no_fsi
from .scalar_mediator_cross_sections import branching_fractions


def dnde_ee(egams, cme, params, type='All'):
    fsr = np.vectorize(dnde_xx_to_s_to_ffg)

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
    fsr = np.vectorize(dnde_xx_to_s_to_ffg)
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


def dnde_neutral_pion(egams, cme, params, type='All'):
    decay = np.vectorize(neutral_pion)

    if type == 'All':
        return 2.0 * decay(egams, cme / 2.0)
    if type == 'FSR':
        return np.arrary([0.0 for _ in range(len(egams))])
    if type == 'Decay':
        return 2.0 * decay(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(type))


def dnde_charged_pion(egams, cme, params, type='All', fsi=True):

    if type == 'All':
        cpi_decay = charged_pion(egams, cme / 2.0)
        if fsi is True:
            cpi_fsr = dnde_xx_to_s_to_pipig(egams, cme, params)
        if fsi is False:
            cpi_fsr = dnde_xx_to_s_to_pipig_no_fsi(egams, cme, params)
        return 2. * cpi_decay + cpi_fsr
    if type == 'FSR':
        if fsi is True:
            return dnde_xx_to_s_to_pipig(egams, cme, params)
        if fsi is False:
            return dnde_xx_to_s_to_pipig_no_fsi(egams, cme, params)
    if type == 'Decay':
        return 2. * charged_pion(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(type))


def dnde_neutral_kaon(egams, cme, params, type='All'):
    decay1 = np.vectorize(long_kaon)
    decay2 = np.vectorize(short_kaon)

    if type == 'All':
        return decay1(egams, cme / 2.0) + decay2(egams, cme / 2.0)
    if type == 'FSR':
        return np.arrary([0.0 for _ in range(len(egams))])
    if type == 'Decay':
        return decay1(egams, cme / 2.0) + decay2(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(type))


def dnde_charged_kaon(egams, cme, params, type='All'):
    decay = charged_kaon

    if type == 'All':
        return 2. * decay(egams, cme / 2.0)
    if type == 'FSR':
        return np.arrary([0.0 for _ in range(len(egams))])
    if type == 'Decay':
        return 2. * decay(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(type))


def spectra(egams, cme, params, fsi=True):
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
    bfs = branching_fractions(cme, params)

    # Pions
    npions = bfs['pi0 pi0'] * dnde_neutral_pion(egams, cme, params)
    cpions = bfs['pi pi'] * dnde_charged_pion(egams, cme, params, fsi=fsi)

    # Leptons
    muons = bfs['mu mu'] * dnde_mumu(egams, cme, params)
    electrons = bfs['e e'] * dnde_ee(egams, cme, params)

    # Kaons
    nkaons = bfs['k0 k0'] * dnde_neutral_kaon(egams, cme, params)
    ckaons = bfs['k k'] * dnde_charged_kaon(egams, cme, params)

    # Comput total spectrum
    total = muons + electrons + npions + cpions + nkaons + ckaons

    # Define dictionary for spectra
    specs = {'total': total, 'mu mu': muons, 'e e': electrons,
             'pi0 pi0': npions, 'pi pi': cpions, 'k0 k0': nkaons,
             'k k': ckaons}

    return specs
