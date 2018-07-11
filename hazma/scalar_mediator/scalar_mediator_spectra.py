import numpy as np

from ..decay import muon
from ..decay import neutral_pion, charged_pion

from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me

from .scalar_mediator_fsr import dnde_xx_to_s_to_ffg
from .scalar_mediator_fsr import dnde_xx_to_s_to_pipig
from .scalar_mediator_cross_sections import branching_fractions
from .scalar_mediator_decay_spectrum import dnde_decay_s, dnde_decay_s_pt
from .scalar_mediator_widths import partial_widths


def dnde_ee(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_ee(egams, cme, params, 'FSR') +
                dnde_ee(egams, cme, params, 'Decay'))
    elif spectrum_type == 'FSR':
        return dnde_xx_to_s_to_ffg(egams, cme, me, params)
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
        return dnde_xx_to_s_to_ffg(egams, cme, mmu, params)
    elif spectrum_type == 'Decay':
        return 2. * muon(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_neutral_pion(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_neutral_pion(egams, cme, params, 'FSR') +
                dnde_neutral_pion(egams, cme, params, 'Decay'))
    if spectrum_type == 'FSR':
        return np.array([0.0 for _ in range(len(egams))])
    if spectrum_type == 'Decay':
        return 2.0 * neutral_pion(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_charged_pion(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_charged_pion(egams, cme, params, 'FSR') +
                dnde_charged_pion(egams, cme, params, 'Decay'))
    elif spectrum_type == 'FSR':
        return dnde_xx_to_s_to_pipig(egams, cme, params)
    elif spectrum_type == 'Decay':
        return 2. * charged_pion(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_s(egams, eng_s, params, mode="total"):
    ms = params.ms
    pws = partial_widths(params)
    pw_array = np.zeros(5, dtype=float)

    pw_array[0] = pws["e e"] / pws["total"]
    pw_array[1] = pws["mu mu"] / pws["total"]
    pw_array[2] = pws["pi0 pi0"] / pws["total"]
    pw_array[3] = pws["pi pi"] / pws["total"]
    pw_array[4] = pws["g g"] / pws["total"]

    if hasattr(egams, "__len__"):
        return 2. * dnde_decay_s(egams, eng_s, ms, pw_array, mode)
    return 2. * dnde_decay_s_pt(egams, eng_s, ms, pw_array, mode)


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
        'pi0 pi0', 'pi pi'
    """

    # Compute branching fractions
    bfs = branching_fractions(cme, params)

    # Only compute the spectrum if the channel's branching fraction is nonzero
    def spec_helper(bf, specfn):
        if bf != 0:
            return bf * specfn(egams, cme, params)
        else:
            return np.zeros(egams.shape)

    # Pions
    npions = spec_helper(bfs['pi0 pi0'], dnde_neutral_pion)
    cpions = spec_helper(bfs['pi pi'], dnde_charged_pion)

    # Leptons
    muons = spec_helper(bfs['mu mu'], dnde_mumu)
    electrons = spec_helper(bfs['e e'], dnde_ee)

    # mediator
    mediator = spec_helper(bfs['s s'], dnde_s)

    # Compute total spectrum
    total = muons + electrons + npions + cpions + mediator

    # Define dictionary for spectra
    specs = {'total': total,
             'mu mu': muons,
             'e e': electrons,
             'pi0 pi0': npions,
             'pi pi': cpions,
             's s': mediator}

    return specs


def gamma_ray_lines(cme, params):
    bf = branching_fractions(cme, params)["g g"]

    return {"g g": {"energy": cme / 2.0, "bf": bf}}
