import numpy as np

from hazma.decay import muon
from hazma.decay import neutral_pion, charged_pion

from hazma.parameters import muon_mass as mmu
from hazma.parameters import electron_mass as me

from hazma.scalar_mediator.scalar_mediator_decay_spectrum import (
    dnde_decay_s,
    dnde_decay_s_pt,
)


class ScalarMediatorSpectra:
    def dnde_ee(self, e_gams, e_cm, spectrum_type="all"):
        if spectrum_type == "all":
            return self.dnde_ee(e_gams, e_cm, "fsr") + self.dnde_ee(
                e_gams, e_cm, "decay"
            )
        elif spectrum_type == "fsr":
            return self.dnde_xx_to_s_to_ffg(e_gams, e_cm, me)
        elif spectrum_type == "decay":
            return np.array([0.0 for _ in range(len(e_gams))])
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def dnde_mumu(self, e_gams, e_cm, spectrum_type="all"):
        if spectrum_type == "all":
            return self.dnde_mumu(e_gams, e_cm, "fsr") + self.dnde_mumu(
                e_gams, e_cm, "decay"
            )
        elif spectrum_type == "fsr":
            return self.dnde_xx_to_s_to_ffg(e_gams, e_cm, mmu)
        elif spectrum_type == "decay":
            return 2.0 * muon(e_gams, e_cm / 2.0)
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def dnde_pi0pi0(self, e_gams, e_cm, spectrum_type="all"):
        if spectrum_type == "all":
            return self.dnde_pi0pi0(e_gams, e_cm, "fsr") + self.dnde_pi0pi0(
                e_gams, e_cm, "decay"
            )
        if spectrum_type == "fsr":
            return np.array([0.0 for _ in range(len(e_gams))])
        if spectrum_type == "decay":
            return 2.0 * neutral_pion(e_gams, e_cm / 2.0)
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def dnde_pipi(self, e_gams, e_cm, spectrum_type="all"):
        if spectrum_type == "all":
            return self.dnde_pipi(e_gams, e_cm, "fsr") + self.dnde_pipi(
                e_gams, e_cm, "decay"
            )
        elif spectrum_type == "fsr":
            return self.dnde_xx_to_s_to_pipig(e_gams, e_cm)
        elif spectrum_type == "decay":
            return 2.0 * charged_pion(e_gams, e_cm / 2.0)
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def dnde_ss(self, e_gams, e_cm, fs="total"):
        # Each scalar gets half the COM energy
        e_s = e_cm / 2.0

        ms = self.ms
        pws = self.partial_widths()
        if pws["total"] != 0:
            pw_array = np.array(
                [pws["e e"], pws["mu mu"], pws["pi0 pi0"], pws["pi pi"], pws["g g"]],
                dtype=float,
            )
            pw_array /= pws["total"]

            # Factor of 2 since S is self-conjugate
            return 2.0 * dnde_decay_s(e_gams, e_s, ms, pw_array, fs)
        else:
            return np.zeros_like(e_gams)

    def spectrum_funcs(self):
        """
        Returns a dictionary of all the avaiable spectrum functions for
        a pair of initial state fermions with mass `mx` annihilating into
        each available final state.

        Each argument of the spectrum functions in `eng_gams`, an array
        of the gamma ray energies to evaluate the spectra at and `e_cm`, the
        center of mass energy of the process.
        """
        return {
            "mu mu": self.dnde_mumu,
            "e e": self.dnde_ee,
            "pi0 pi0": self.dnde_pi0pi0,
            "pi pi": self.dnde_pipi,
            "s s": self.dnde_ss,
        }

    def gamma_ray_lines(self, e_cm):
        bf = self.annihilation_branching_fractions(e_cm)["g g"]

        return {"g g": {"energy": e_cm / 2.0, "bf": bf}}
