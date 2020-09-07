"""
Module for describing a RH-Neutrino dark matter particle derived from a theory
where the RH neutrino interactions with active neutrinos via a Yukawa
interaction.

The Lagrangian above the EW scale contains:
    L > i * Ndag.sibar[mu].del[mu].N - 1/2 MN (N.N + Ndag.Ndag)
        -yl (Ldag.Htilde.N + h.c.)
"""

import numpy as np

from hazma.theory import TheoryDec
from hazma.parameters import (
    electron_mass as me,
    charged_pion_mass as mpi,
    muon_mass as mmu,
)
from hazma.rh_neutrino._rh_neutrino_widths import RHNeutrinoWidths
from hazma.rh_neutrino._rh_neutrino_spectra import RHNeutrinoSpectra
from hazma.rh_neutrino._rh_neutrino_positron_spectrum import (
    RHNeutrinoPositronSpectra,
)
from hazma.rh_neutrino._rh_neutrino_fsr import RHNeutrinoFSR


class RHNeutrino(
    RHNeutrinoWidths,
    RHNeutrinoFSR,
    RHNeutrinoPositronSpectra,
    RHNeutrinoSpectra,
    TheoryDec,
):
    def __init__(self, mx, stheta, lepton="e"):
        """
        Initialize a right-handed neutrino model.

        Parameters
        ----------
        rhn_mass: float
            Right-handed neutrino mass in MeV.
        stheta: float
            Mixing angle between the right-handed neutrino and active neutrino.
        """
        self._stheta = stheta
        self._mx = mx
        # self._ctheta = np.sqrt(1.0 - stheta ** 2)
        # self._lhn_mass = mx * stheta ** 2 / (1.0 - stheta ** 2)

        self.lepton = lepton
        if lepton == "e":
            self.ml = me
        elif lepton == "mu":
            self.ml = mmu
        else:
            raise ValueError(
                "Lepton {} is invalid. Use 'e' or 'mu'.".format(lepton)
            )

    def list_decay_final_states(self):
        return ["pi l", "pi0 nu"]

    def _decay_widths(self):
        """
        Decay width into each final state.
        """
        return {
            "pi l": self.width_pi_l(),
            "pi0 nu": self.width_pi0_nu(),
        }

    def _spectrum_funcs(self):
        """
        Gets a function taking a photon energy and returning the continuum
        gamma ray spectrum dN/dE for each relevant decay final state.
        """

        def dnde_pi_l(es):
            return self.dnde_pi_l(es)

        def dnde_pi0_nu(es):
            return self.dnde_nu_pi0(es)

        return {"pi l": dnde_pi_l, "pi0 nu": dnde_pi0_nu}

    def _gamma_ray_line_energies(self):
        """
        Returns dict of final states and photon energies for final states
        containing monochromatic gamma ray lines.
        """
        return {}

    def _positron_spectrum_funcs(self):
        """
        Returns functions `float -> float` giving the continuum positron
        spectrum for each final state.
        """

        def dnde_pi_l(es):
            return self.dnde_pos_pi_l(es)

        return {"pi l": dnde_pi_l}

    def _positron_line_energies(self):
        return {
            "pi l": (self.mx ** 2 + self.ml ** 2 - mpi ** 2) / (2.0 * self.mx)
        }

    @property
    def mx(self):
        """
        Get the right-handed neutrino mass in MeV.
        """
        return self._mx

    @mx.setter
    def mx(self, val):
        """
        Set the right-handed neutrino mass.

        Parameters
        ----------
        val: float
            New right-handed neutrino mass.
        """
        self._mx = val

    @property
    def stheta(self):
        """
        Get the mixing angle between right-handed neutrino and active neutrino.
        """
        return self._stheta

    @stheta.setter
    def stheta(self, val):
        """
        Set the mixing angle between the right-handed neutrino and active
        neutrino.

        Parameters
        ----------
        val: float
            New mixing angle.
        """
        self._stheta = val
