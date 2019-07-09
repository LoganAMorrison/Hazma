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
    def dnde_pos_ss(self, e_ps, e_cm, fs="total"):
        # Each scalar gets half the COM energy
        e_s = e_cm / 2.0
        pws = self.partial_widths()

        if pws["total"] != 0:
            pw_array = np.array([pws["e e"], pws["mu mu"], pws["pi pi"]], dtype=float)
            pw_array /= pws["total"]

            # Factor of 2 since S is self-conjugate
            return 2.0 * dnde_decay_s(e_ps, e_s, self.ms, pw_array, fs)
        else:
            return np.zeros_like(e_ps)

    def positron_spectra(self, e_ps, e_cm):
        """Computes total continuum positron spectrum.
        """
        bfs = self.annihilation_branching_fractions(e_cm)

        def spec_helper(bf, spec_fn):
            if bf != 0:
                return bf * spec_fn(e_ps, e_cm)
            else:
                return np.zeros_like(e_ps)

        muon_spec = spec_helper(bfs["mu mu"], self.dnde_pos_mumu)
        pipi_spec = spec_helper(bfs["pi pi"], self.dnde_pos_pipi)
        ss_spec = spec_helper(bfs["s s"], self.dnde_pos_ss)

        total = pipi_spec + muon_spec + ss_spec

        return {"total": total, "mu mu": muon_spec, "pi pi": pipi_spec, "s s": ss_spec}

    def positron_lines(self, e_cm):
        bf = self.annihilation_branching_fractions(e_cm)["e e"]

        return {"e e": {"energy": e_cm / 2.0, "bf": bf}}
