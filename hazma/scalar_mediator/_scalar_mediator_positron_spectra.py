import numpy as np

from ..positron_spectra import muon as pspec_muon
from ..positron_spectra import charged_pion as pspec_charged_pion


class ScalarMediatorPositronSpectra:
    def positron_spectra(self, eng_ps, cme):
        """
        """

        # Compute branching fractions
        bfs = self.branching_fractions(cme)

        # Only compute the spectrum if the channel's branching fraction is
        # nonzero

        def spec_helper(bf, specfn):
            if bf != 0:
                return bf * specfn(eng_ps, cme / 2.)
            else:
                return np.zeros(eng_ps.shape)

        cpions = spec_helper(bfs['pi pi'], pspec_charged_pion)
        muons = spec_helper(bfs['mu mu'], pspec_muon)

        total = cpions + muons

        return {"total": total, "mu mu": muons, "pi pi": cpions}

    def positron_lines(self, cme):
        bf = self.branching_fractions(cme)["e e"]

        return {"e e": {"energy": cme / 2., "bf": bf}}
