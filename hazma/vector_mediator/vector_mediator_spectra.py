import numpy as np

from ..decay import muon
from ..decay import neutral_pion, charged_pion

from ..parameters import neutral_pion_mass as mpi0

from .vector_mediator_fsr import dnde_xx_to_v_to_ffg, dnde_xx_to_v_to_pipig

from .vector_mediator_cross_sections import branching_fractions

from .vector_mediator_decay_spectrum import dnde_decay_v, dnde_decay_v_pt
from .vector_mediator_widths import partial_widths


def dnde_ee(egams, cme, params, spectrum_type='All'):
    fsr = np.vectorize(dnde_xx_to_v_to_ffg)

    if spectrum_type == 'All':
        return (dnde_ee(egams, cme, params, "FSR") +
                dnde_ee(egams, cme, params, "Decay"))
    elif spectrum_type == 'FSR':
        return fsr(egams, cme, "e", params)
    elif spectrum_type == 'Decay':
        return np.array([0.0 for _ in range(len(egams))])
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_mumu(egams, cme, params, spectrum_type='All'):
    fsr = np.vectorize(dnde_xx_to_v_to_ffg)  # todo: this line
    decay = np.vectorize(muon)

    if spectrum_type == 'All':
        return (dnde_mumu(egams, cme, params, "FSR") +
                dnde_mumu(egams, cme, params, "Decay"))
    elif spectrum_type == 'FSR':
        return fsr(egams, cme, "mu", params)
    elif spectrum_type == 'Decay':
        return 2. * decay(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_pi0g(egams, cme, params, spectrum_type="All"):
    if spectrum_type == 'All':
        return (dnde_pi0g(egams, cme, params, "FSR") +
                dnde_pi0g(egams, cme, params, "Decay"))
    elif spectrum_type == 'FSR':
        return np.array([0.0 for _ in range(len(egams))])
    elif spectrum_type == 'Decay':
        # Neutral pion's energy
        e_pi0 = (cme**2 + mpi0**2) / (2. * cme)

        return neutral_pion(egams, e_pi0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_pipi(egams, cme, params, spectrum_type="All"):
    if spectrum_type == 'All':
        return (dnde_pipi(egams, cme, params, "FSR") +
                dnde_pipi(egams, cme, params, "Decay"))
    elif spectrum_type == 'FSR':
        return dnde_xx_to_v_to_pipig(egams, cme, params)
    elif spectrum_type == 'Decay':
        return 2. * charged_pion(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_v(egams, eng_v, params, mode="total"):
    mv = params.mv
    pws = partial_widths(params)
    pw_array = np.zeros(5, dtype=float)

    pw_array[0] = pws["e e"] / pws["total"]
    pw_array[1] = pws["mu mu"] / pws["total"]
    pw_array[2] = pws["pi0 g"] / pws["total"]
    pw_array[3] = pws["pi pi"] / pws["total"]

    if hasattr(egams, "__len__"):
        return 2. * dnde_decay_v(egams, eng_v, mv, pw_array, mode)
    return 2. * dnde_decay_v_pt(egams, eng_v, mv, pw_array, mode)


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
    bfs = branching_fractions(cme, params)

    # Only compute the spectrum if the channel's branching fraction is nonzero
    def spec_helper(bf, specfn):
        if bf != 0:
            return bf * specfn(egams, cme, params)
        else:
            return np.zeros(egams.shape)

    # Leptons
    muons = spec_helper(bfs['mu mu'], dnde_mumu)
    electrons = spec_helper(bfs['e e'], dnde_ee)

    # Pions
    pi0g = spec_helper(bfs["pi0 g"], dnde_pi0g)
    pipi = spec_helper(bfs["pi pi"], dnde_pipi)

    # mediator
    mediator = spec_helper(bfs['pi0 pi0'], dnde_v)

    # Compute total spectrum
    total = muons + electrons + pi0g + pipi + mediator

    # Define dictionary for spectra
    specs = {'total': total,
             'mu mu': muons,
             'e e': electrons,
             "pi0 g": pi0g,
             "pi pi": pipi,
             "v v": mediator}

    return specs
