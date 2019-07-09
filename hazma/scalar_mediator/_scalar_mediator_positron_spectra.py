import numpy as np

from hazma.positron_spectra import charged_pion as pspec_charged_pion
from hazma.positron_spectra import muon as pspec_muon
from hazma.scalar_mediator.scalar_mediator_positron_spec import dnde_decay_s


class ScalarMediatorPositronSpectra:
    # positron decay spectrum for chi chibar -> pi pi
    def dnde_pos_pipi(self, e_ps, e_cm):
        return pspec_charged_pion(e_ps, e_cm / 2.0)

    # positron decay spectrum for chi chibar -> mu mu
    def dnde_pos_mumu(self, e_ps, e_cm):
        return pspec_muon(e_ps, e_cm / 2.0)

    # positron decay spectrum for chi chibar -> s s
    def dnde_pos_ss(self, e_ps, e_cm, ms, pws, fs="total"):
        return dnde_decay_s(e_ps, e_cm / 2.0, ms, pws, fs)

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

        # Handle the chi chibar-> S S seperately.
        if bfs["s s"] != 0.0:
            pws = self.partial_widths()
            if pws["total"] != 0.0:
                pw_array = np.zeros(3, dtype=float)
                pw_array[0] = pws["e e"] / pws["total"]
                pw_array[1] = pws["mu mu"] / pws["total"]
                pw_array[2] = pws["pi pi"] / pws["total"]
                ss_spec = bfs["s s"] * self.dnde_pos_ss(e_ps, e_cm, self.ms,
                                                        pw_array, "total")
            else:
                ss_spec = np.zeros(e_ps.shape)
        else:
            ss_spec = np.zeros(e_ps.shape)

        total = pipi_spec + muon_spec + ss_spec

        return {"total": total, "mu mu": muon_spec, "pi pi": pipi_spec,
                "s s": ss_spec}

    def positron_lines(self, e_cm):
        bf = self.annihilation_branching_fractions(e_cm)["e e"]

        return {"e e": {"energy": e_cm / 2.0, "bf": bf}}
