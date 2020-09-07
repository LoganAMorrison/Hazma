"""
This file contains the decay spectra from a right-handed neutrino at rest.
"""
import numpy as np

from hazma.decay import (
    neutral_pion as decay_pi0,
    charged_pion as decay_pi,
    muon as decay_mu,
)
from hazma.parameters import (
    neutral_pion_mass as mpi0,
    charged_pion_mass as mpi,
    electron_mass as me,
    muon_mass as mmu,
)


class RHNeutrinoSpectra:
    def dnde_nu_pi0(self, photon_energies, spectrum_type="all"):
        if spectrum_type == "all":
            return self.dnde_nu_pi0(photon_energies, "fsr") + self.dnde_nu_pi0(
                photon_energies, "decay"
            )
        elif spectrum_type == "fsr":
            if hasattr(photon_energies, "__len__"):
                return np.array([0.0 for _ in photon_energies])
            else:
                return 0.0
        elif spectrum_type == "decay":
            epi = (self.mx ** 2 + mpi0 ** 2) / (2.0 * self.mx)
            return decay_pi0(photon_energies, epi)
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(
                    spectrum_type
                )
            )

    def dnde_pi_l(self, photon_energies, spectrum_type="all"):

        if spectrum_type == "all":
            return self.dnde_pi_l(photon_energies, "fsr") + self.dnde_pi_l(
                photon_energies, "decay"
            )
        elif spectrum_type == "fsr":
            return self.dnde_pi_l_fsr(photon_energies)
        elif spectrum_type == "decay":
            epi = (self.mx ** 2 + mpi ** 2 - self.ml ** 2) / (2.0 * self.mx)
            el = (self.mx ** 2 - mpi ** 2 + self.ml ** 2) / (2.0 * self.mx)
            if self.lepton == "e":
                return decay_pi(photon_energies, epi)
            elif self.lepton == "mu":
                return decay_pi(photon_energies, epi) + decay_mu(
                    photon_energies, el
                )
        else:
            raise ValueError(
                "Type {} is invalid. Use 'all', 'fsr' or 'decay'".format(
                    spectrum_type
                )
            )

