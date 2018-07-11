import numpy as np

from ..positron_spectra import muon as pspec_muon
from ..positron_spectra import charged_pion as pspec_charged_pion
from .pseudo_scalar_mediator_cross_sections import branching_fractions


def positron_spectra(eng_ps, cme, params):
    """Computes continuum part of positron spectrum from DM annihilation.

    Parameters
    ----------
    eng_ps : array-like
        Positron energies at which to compute the spectrum.
    cme : float
        Center of mass energy.

    Returns
    -------
    specs : dict
        Dictionary of positron spectra. The keys are the final states producing
        contributions to the continuum positron spectrum and 'total'.
    """
    # Compute branching fractions
    bfs = branching_fractions(cme, params)

    # Only compute the spectrum if the channel's branching fraction is nonzero
    def spec_helper(bf, specfn):
        if bf != 0:
            return bf * specfn(eng_ps, cme / 2.)
        else:
            return np.zeros(eng_ps.shape)

    mumu_spec = spec_helper(bfs['mu mu'], pspec_muon)

    # TODO: figure this out. Since the matrix element is constant, we just need
    # to average the pion energies over phase space! Should end up with a
    # 1-parameter integral for each pion species.
    pi0pipi_spec = 0.

    total = mumu_spec

    return {"total": total,
            "mu mu": mumu_spec,
            "pi0 pi pi": pi0pipi_spec}


def positron_lines(cme, params):
    bf = branching_fractions(cme, params)["e e"]

    return {"e e": {"energy": cme / 2., "bf": bf}}
