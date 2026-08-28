import numpy as np

from hazma import spectra

# Served by the Rust extension since cython-to-rust Task 6.3 deleted
# `hazma/vector_mediator/vector_mediator_positron_spec.pyx`. Import
# paths, call signatures and returned values are unchanged; the `_core`
# names differ from these because `hazma._core.vector_mediator` already
# spells `dnde_decay_v` for the *photon* spectrum Task 6.2 ported. See
# `rust/src/kernels/mediator_decay_positron.rs` for the port.
#
# Both names are bound here rather than imported directly, because
# `dnde_decay_v_pt` is a re-export: it was a public `def` of the deleted
# extension and this module is what stands in for it, but nothing below
# calls it.
from hazma._core import vector_mediator as _core

dnde_decay_v = _core.dnde_positron_decay_v
dnde_decay_v_pt = _core.dnde_positron_decay_v_pt


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
