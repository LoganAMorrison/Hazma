import numpy as np

from hazma import spectra
from hazma.vector_mediator.vector_mediator_positron_spec import dnde_decay_v


class VectorMediatorPositronSpectra:
    def dnde_pos_pipi(self, e_ps, e_cm):
        return spectra.dnde_positron_charged_pion(e_ps, e_cm / 2.0)

    def dnde_pos_mumu(self, e_ps, e_cm):
        return spectra.dnde_photon_muon(e_ps, e_cm / 2.0)

    # positron decay spectrum for chi chibar -> v v
    def dnde_pos_vv(self, e_ps, e_cm, fs="total"):
        # Each scalar gets half the COM energy
        e_v = e_cm / 2.0
        pws = self.partial_widths()  # type: ignore

        if pws["total"] != 0:
            # dnde_decay_v relies on this ordering of the partial widths
            pw_array = np.array([pws["e e"], pws["mu mu"], pws["pi pi"]])
            pw_array /= pws["total"]

            # Factor of 2 since there are two V's
            return 2.0 * dnde_decay_v(e_ps, e_v, self.mv, pw_array, fs)  # type: ignore
        else:
            return np.zeros_like(e_ps)

    def positron_spectrum_funcs(self):
        return {
            "mu mu": self.dnde_pos_mumu,
            "pi pi": self.dnde_pos_pipi,
            "v v": self.dnde_pos_vv,
        }

    def positron_lines(self, e_cm):
        bf = self.annihilation_branching_fractions(e_cm)["e e"]  # type: ignore

        return {"e e": {"energy": e_cm / 2.0, "bf": bf}}
