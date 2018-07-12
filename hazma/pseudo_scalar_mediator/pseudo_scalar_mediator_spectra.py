import numpy as np

from .pseudo_scalar_mediator_fsr import dnde_xx_to_p_to_ffg
from .pseudo_scalar_mediator_cross_sections import branching_fractions

from ..decay import muon, charged_pion, neutral_pion

from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me


# TODO: pp spectrum. Gonna need Logan to do this since it requires cython...
def dnde_pp(egams, Q, params, mode="total"):
    eng_p = Q / 2.

    pass


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


# TODO: figure this out!
def dnde_pi0pipi(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_pi0pipi(egams, cme, params, 'FSR') +
                dnde_pi0pipi(egams, cme, params, 'Decay'))
    elif spectrum_type == 'FSR':
        # Either use rambo with the 4-body FSR matrix element or Low's theorem
        pass
    elif spectrum_type == 'Decay':
        # Will need to use rambo for this
        pass
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


# TODO: figure this out!
def dnde_pi0pi0pi0(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return dnde_pi0pipi(egams, cme, params, 'Decay')
    elif spectrum_type == 'FSR':
        return np.array([0.0 for _ in range(len(egams))])
    elif spectrum_type == 'Decay':
        # Will need rambo for this
        pass
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

    # Only compute the spectrum if the channel's branching fraction is nonzero
    def spec_helper(bf, specfn):
        if bf != 0:
            return bf * specfn(egams, cme, params)
        else:
            return np.zeros(egams.shape)

    # Pions. TODO: use rambo to compute this.
    pi0pipi_spec = bfs['pi0 pi pi'] * dnde_pi0pipi
    pi0pi0pi0_spec = bfs['pi0 pi0 pi0'] * dnde_pi0pi0pi0

    # Leptons
    mumu_spec = spec_helper(bfs['mu mu'], dnde_pi0pipi)
    ee_spec = spec_helper(bfs['e e'], dnde_pi0pipi)

    # Mediator
    pp_spec = spec_helper(bfs['p p'], dnde_pp)

    # Compute total spectrum
    total = pi0pipi_spec + mumu_spec + ee_spec + pp_spec + pi0pi0pi0_spec

    # Define dictionary for spectra
    specs = {'total': total,
             'pi0 pi pi': pi0pipi_spec,
             'pi0 pi0 pi0': pi0pi0pi0_spec,
             'mu mu': mumu_spec,
             'e e': ee_spec,
             'p p': pp_spec}

    return specs


def gamma_ray_lines(cme, params):
    bf = branching_fractions(cme, params)["g g"]

    return {"g g": {"energy": cme / 2.0, "bf": bf}}
