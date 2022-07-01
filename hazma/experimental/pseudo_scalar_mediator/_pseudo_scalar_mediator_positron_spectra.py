import numpy as np

from hazma import spectra
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0


def dnde_pos_pi0pipi(self, positron_energies, cme):
    if cme <= 2.0 * mpi + mpi0:
        return np.zeros(len(positron_energies), dtype=float)

    return spectra.dnde_positron(
        positron_energies,
        cme,
        final_states=["pi", "pi", "pi0"],
        msqrd=self.msqrd_xx_to_p_to_pm0,
    )


def dnde_pos_mumu(eng_ps, cme):
    return pspec_muon(eng_ps, cme / 2.0)


def positron_spectra(self, eng_ps, cme):
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
        Dictionary of positron spectra. The keys are the final states
        producing contributions to the continuum positron spectrum and
        'total'.
    """
    # Compute branching fractions
    bfs = self.annihilation_branching_fractions(cme)

    # Only compute the spectrum if the channel's branching fraction is
    # nonzero
    def spec_helper(bf, specfn):
        if bf != 0:
            return bf * specfn(eng_ps, cme)
        else:
            return np.zeros(eng_ps.shape)

    mumu_spec = spec_helper(bfs["mu mu"], self.dnde_pos_mumu)
    pi0pipi_spec = spec_helper(bfs["pi0 pi pi"], self.dnde_pos_pi0pipi)

    total = mumu_spec + pi0pipi_spec

    return {"total": total, "mu mu": mumu_spec, "pi0 pi pi": pi0pipi_spec}


def positron_lines(self, cme):
    bf = self.annihilation_branching_fractions(cme)["e e"]

    return {"e e": {"energy": cme / 2.0, "bf": bf}}
