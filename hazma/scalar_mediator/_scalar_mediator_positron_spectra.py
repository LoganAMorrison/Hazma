import numpy as np

from hazma.positron_spectra import muon as pspec_muon
from hazma.positron_spectra import charged_pion as pspec_charged_pion


class ScalarMediatorPositronSpectra:
    def dnde_pos_pipi(self, eng_ps, cme):
        return pspec_charged_pion(eng_ps, cme / 2.)

    def dnde_pos_mumu(self, eng_ps, cme):
        return pspec_muon(eng_ps, cme / 2.)

    def positron_spectra(self, eng_ps, cme):
        """Computes total continuum positron spectrum.
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

        muon_spec = spec_helper(bfs['mu mu'], self.dnde_pos_mumu)
        pipi_spec = spec_helper(bfs['pi pi'], self.dnde_pos_pipi)

        total = pipi_spec + muon_spec

        return {"total": total, "mu mu": muon_spec, "pi pi": pipi_spec}

    def positron_lines(self, cme):
        bf = self.annihilation_branching_fractions(cme)["e e"]

        return {"e e": {"energy": cme / 2., "bf": bf}}
