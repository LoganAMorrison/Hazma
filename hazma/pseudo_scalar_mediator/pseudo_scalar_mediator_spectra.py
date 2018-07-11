import numpy as np

from .pseudo_scalar_mediator_fsr import dnde_xx_to_p_to_ffg
from .pseudo_scalar_mediator_cross_sections import branching_fractions

from ..decay import muon, charged_pion, neutral_pion

from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me


def dnde_ee(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_ee(egams, cme, params, 'FSR') +
                dnde_ee(egams, cme, params, 'Decay'))
    elif spectrum_type == 'FSR':
        return dnde_xx_to_p_to_ffg(egams, cme, me, params)
    elif spectrum_type == 'Decay':
        return np.array([0.0 for _ in range(len(egams))])
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_mumu(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_mumu(egams, cme, params, 'FSR') +
                dnde_mumu(egams, cme, params, 'Decay'))
    elif spectrum_type == 'FSR':
        return dnde_xx_to_p_to_ffg(egams, cme, mmu, params)
    elif spectrum_type == 'Decay':
        return 2. * muon(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_pi0pipi(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_pi0pipi(egams, cme, params, 'FSR') +
                dnde_pi0pipi(egams, cme, params, 'Decay'))
    elif spectrum_type == 'FSR':
        # TODO: implement this using Low's theorem or something
        return np.array([0.0 for _ in range(len(egams))])
    elif spectrum_type == 'Decay':
        return 2. * charged_pion(egams, cme / 2.0) + neutral_pion
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def spectra(egams, cme, params):
    """Compute the total spectrum from two fermions annihilating through a
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

    # Leptons
    muons = bfs['mu mu'] * dnde_mumu(egams, cme, params)
    electrons = bfs['e e'] * dnde_ee(egams, cme, params)

    # Comput total spectrum
    total = muons + electrons

    # Define dictionary for spectra
    specs = {'total': total, 'mu mu': muons, 'e e': electrons}

    return specs


def gamma_ray_lines(cme, params):
    bf = branching_fractions(cme, params)["g g"]

    return {"g g": {"energy": cme / 2.0, "bf": bf}}
