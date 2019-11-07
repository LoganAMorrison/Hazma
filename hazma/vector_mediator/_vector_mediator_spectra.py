import numpy as np

from hazma.decay import muon
from hazma.decay import neutral_pion, charged_pion

from hazma.parameters import neutral_pion_mass as mpi0

from hazma.vector_mediator.vector_mediator_decay_spectrum import (
    dnde_decay_v,
    dnde_decay_v_pt,
)


class VectorMediatorSpectra:
    def dnde_ee(self, e_gams, e_cm, spectrum_type="all"):
        fsr = np.vectorize(self.dnde_xx_to_v_to_ffg)

        if spectrum_type == "all":
            return self.dnde_ee(e_gams, e_cm, "fsr") + self.dnde_ee(
                e_gams, e_cm, "decay"
            )
        elif spectrum_type == "fsr":
            return fsr(e_gams, e_cm, "e")
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
        fsr = np.vectorize(self.dnde_xx_to_v_to_ffg)  # todo: this line
        decay = np.vectorize(muon)

        if spectrum_type == "all":
            return self.dnde_mumu(e_gams, e_cm, "fsr") + self.dnde_mumu(
                e_gams, e_cm, "decay"
            )
        elif spectrum_type == "fsr":
            return fsr(e_gams, e_cm, "mu")
        elif spectrum_type == "decay":
            return 2.0 * decay(e_gams, e_cm / 2.0)
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def dnde_pi0g(self, e_gams, e_cm, spectrum_type="all"):
        if spectrum_type == "all":
            return self.dnde_pi0g(e_gams, e_cm, "fsr") + self.dnde_pi0g(
                e_gams, e_cm, "decay"
            )
        elif spectrum_type == "fsr":
            return np.array([0.0 for _ in range(len(e_gams))])
        elif spectrum_type == "decay":
            # Neutral pion's energy
            e_pi0 = (e_cm ** 2 + mpi0 ** 2) / (2.0 * e_cm)

            return neutral_pion(e_gams, e_pi0)
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
            return self.dnde_xx_to_v_to_pipig(e_gams, e_cm)
        elif spectrum_type == "decay":
            return 2.0 * charged_pion(e_gams, e_cm / 2.0)
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or \
                             'decay'".format(
                    spectrum_type
                )
            )

    def __dnde_v(self, e_gams, e_v, fs="total"):
        """
        Helper function for computing the spectrum from V decaying with
        arbitrary boost.
        """
        mv = self.mv
        pws = self.partial_widths()
        pw_array = np.zeros(5, dtype=float)

        # Check is the decay width of the vector is zero.
        if pws["total"] == 0.0:
            if hasattr(e_gams, "__len__"):
                return np.array([0.0] * len(e_gams))
            return 0.0

        pw_array[0] = pws["e e"] / pws["total"]
        pw_array[1] = pws["mu mu"] / pws["total"]
        pw_array[2] = pws["pi0 g"] / pws["total"]
        pw_array[3] = pws["pi pi"] / pws["total"]

        if hasattr(e_gams, "__len__"):
            return dnde_decay_v(e_gams, e_v, mv, pw_array, fs)

        return dnde_decay_v_pt(e_gams, e_v, mv, pw_array, fs)

    def dnde_pi0v(self, e_gams, e_cm):
        e_pi0 = (e_cm ** 2 + mpi0 ** 2 - self.mv ** 2) / (2 * e_cm)
        pi0_spec = neutral_pion(e_gams, e_pi0)

        e_v = (e_cm ** 2 - mpi0 ** 2 + self.mv ** 2) / (2 * e_cm)
        v_spec = self.__dnde_v(e_gams, e_v, fs="total")

        return pi0_spec + v_spec

    def dnde_vv(self, e_gams, e_cm, fs="total"):
        # Each vector gets half the COM energy
        return 2.0 * self.__dnde_v(e_gams, e_cm / 2.0, fs)

    def spectrum_funcs(self):
        """
        Returns a dictionary of all the avaiable spectrum functions for
        a pair of initial state fermions with mass `mx` annihilating into
        each available final state.

        Each argument of the spectrum functions in `eng_gams`, an array
        of the gamma ray energies to evaluate the spectra at and `e_cm`, the
        center of mass energy of the process.

        Note
        ----
        This does not return a function for computing the spectrum for the pi0
        pi pi final state since it always contributes orders of magnitude less
        than the pi pi and pi0 g final states.
        """
        return {
            "mu mu": self.dnde_mumu,
            "e e": self.dnde_ee,
            "pi pi": self.dnde_pipi,
            "pi0 g": self.dnde_pi0g,
            "pi0 v": self.dnde_pi0v,
            "v v": self.dnde_vv,
        }

    def gamma_ray_lines(self, e_cm):
        bf = self.annihilation_branching_fractions(e_cm)["pi0 g"]

        return {"pi0 g": {"energy": (e_cm ** 2 - mpi0 ** 2) / (2.0 * e_cm), "bf": bf}}
