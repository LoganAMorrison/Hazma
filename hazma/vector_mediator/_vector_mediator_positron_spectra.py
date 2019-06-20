import numpy as np

from hazma.positron_spectra import charged_pion as pspec_charged_pion
from hazma.positron_spectra import muon as pspec_muon


class VectorMediatorPositronSpectra:
    def dnde_pos_pipi(self, e_ps, e_cm):
        return pspec_charged_pion(e_ps, e_cm / 2.0)

    def dnde_pos_mumu(self, e_ps, e_cm):
        return pspec_muon(e_ps, e_cm / 2.0)

    def positron_spectra(self, e_ps, e_cm):
        """Computes total continuum positron spectrum.
        """
        bfs = self.annihilation_branching_fractions(e_cm)

        def spec_helper(bf, specfn):
            if bf != 0:
                return bf * specfn(e_ps, e_cm)
            else:
                return np.zeros(e_ps.shape)

        muon_spec = spec_helper(bfs["mu mu"], self.dnde_pos_mumu)
        pipi_spec = spec_helper(bfs["pi pi"], self.dnde_pos_pipi)

        total = pipi_spec + muon_spec

        return {"total": total, "mu mu": muon_spec, "pi pi": pipi_spec}

    def positron_lines(self, e_cm):
        bf = self.annihilation_branching_fractions(e_cm)["e e"]

        return {"e e": {"energy": e_cm / 2.0, "bf": bf}}
