import numpy as np

from hazma.positron_spectra import charged_pion as pspec_charged_pion
from hazma.positron_spectra import muon as pspec_muon
from hazma.vector_mediator.vector_mediator_positron_spec import dnde_decay_v


class VectorMediatorPositronSpectra:
    def dnde_pos_pipi(self, e_ps, e_cm):
        return pspec_charged_pion(e_ps, e_cm / 2.0)

    def dnde_pos_mumu(self, e_ps, e_cm):
        return pspec_muon(e_ps, e_cm / 2.0)

    # positron decay spectrum for chi chibar -> v v
    def dnde_pos_vv(self, e_ps, e_cm, fs="total"):
        # Each scalar gets half the COM energy
        e_v = e_cm / 2.0
        pws = self.partial_widths()

        if pws["total"] != 0:
            pw_array = np.array([pws["e e"], pws["mu mu"], pws["pi pi"]], dtype=float)
            pw_array /= pws["total"]

            # Factor of 2 since there are two V's
            return 2.0 * dnde_decay_v(e_ps, e_v, self.mv, pw_array, fs)
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
                return np.zeros(e_ps.shape)

        muon_spec = spec_helper(bfs["mu mu"], self.dnde_pos_mumu)
        pipi_spec = spec_helper(bfs["pi pi"], self.dnde_pos_pipi)
        vv_spec = spec_helper(bfs["v v"], self.dnde_pos_vv)

        total = pipi_spec + muon_spec + vv_spec

        return {"total": total, "mu mu": muon_spec, "pi pi": pipi_spec, "v v": vv_spec}

    def positron_lines(self, e_cm):
        bf = self.annihilation_branching_fractions(e_cm)["e e"]

        return {"e e": {"energy": e_cm / 2.0, "bf": bf}}
