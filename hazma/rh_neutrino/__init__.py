"""
=================================================
Right-Handed Neutrino (:mod: `hazma.rh_neutrino`)
=================================================

This module provides an object for studying the physics of an MeV-scale
right-handed neutrino. The model for the RH-Neutrino dark matter particle was
derived from a theory where the RH neutrino interactions with active neutrinos
via a Yukawa interaction. The Lagrangian above the EW scale contains:
    L > i * Ndag.sibar[mu].del[mu].N - 1/2 MN (N.N + Ndag.Ndag)
        -yl (Ldag.Htilde.N + h.c.)

Classes
-------
.. autosummary::
    :toctree: generated/

    RHNeutrino

See Also
--------
`hazma.theory`

"""
__all__ = ["RHNeutrino"]

import numpy as np

from hazma.theory import TheoryDec
from hazma.parameters import (
    electron_mass as me,
    charged_pion_mass as mpi,
    muon_mass as mmu,
)


class RHNeutrino(TheoryDec):
    """Model containing an unstable, right-handed (RH), neutrino as the dark
    matter.

    This model assume that the RH neutrino mixes with a single active neutrino
    (either the electron- or muon-neutrino.) The mixing is assumed to arise
    from a Yukawa interaction with the SM Higgs. The Lagrangian is given by:
        L = LSM + i * Ndag.sibar[mu].del[mu].N - 1/2 MN (N.N + Ndag.Ndag)
            -yl (Ldag.Htilde.N + h.c.)

    Attributes
    ----------
    mx: float
        The mass of the RH-neutrino.
    stheta: float
        Mixing angle between active-neutron and the RH neutrino.
    lepton: str
        String specifying which flavor of neutrino the RH neutrino mixes with.
    include_3body: bool
        Flag specifying if 3-body final states should be consider (i.e.
        N->nu+nu+nu, N->nu+l+lbar, etc.)

    """

    from ._rh_neutrino_widths import (
        width_pi_l,
        width_k_l,
        width_pi0_nu,
        width_nu_gamma,
        width_nu_pi0_pi0,
        width_nu_pi_pi,
        width_l_pi_pi0,
        width_nu_nu_nu,
        width_nu_l_l,
    )

    from ._rh_neutrino_matrix_elements import (
        msqrd_nu_l_l,
        msqrd_nu_l_l_g,
        msqrd_pi_pi0_l,
        msqrd_pi_pi0_l_g,
    )

    from ._rh_neutrino_fsr import (
        dnde_pi_l_fsr,
        dnde_k_l_fsr,
        dnde_nu_l_l_fsr,
        dnde_pi_pi0_l_fsr,
    )

    from ._rh_neutrino_spectra import (
        dnde_nu_pi0,
        dnde_pi_l,
        dnde_k_l,
        dnde_nu_l_l,
        dnde_pi_pi0_l,
    )
    from ._rh_neutrino_positron_spectrum import dnde_pos_pi_l

    def __init__(self, mx, stheta, lepton="e", include_3body=False):
        """
        Generate an MeV right-handed object.

        Parameters
        ----------
        rhn_mass: float
            Right-handed neutrino mass in MeV.
        stheta: float
            Mixing angle between the right-handed neutrino and active neutrino.
        lepton: str, optional
            String specifying which flavor of active neutrino the RH neutrino
            mixes with. Options are "e" or "mu". Default is "e".
        include_3body: bool, optional
            Flag specifying if 3-body final states should be consider (i.e.
            N->nu+nu+nu, N->nu+l+lbar, etc.). Default is False.
        """
        self._stheta = stheta
        self._mx = mx
        self.include_3body = include_3body

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
        """
        Returns a list of the 
        """
        return ["pi l", "pi0 nu", "k l"]

    def _decay_widths(self):
        """
        Decay width into each final state.
        """
        return {
            "pi l": self.width_pi_l(),
            "pi0 nu": self.width_pi0_nu(),
            "k l": self.width_k_l(),
            "pi pi nu": self.width_nu_pi_pi(),
            "pi pi0 l": self.width_l_pi_pi0(),
            "nu nu nu": self.width_nu_nu_nu(),
            "nu l l": self.width_nu_l_l(),
            "nu g": self.width_nu_gamma(),
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

        def dnde_k_l(es):
            return self.dnde_k_l(es)

        def dnde_nu_l_l(es):
            return self.dnde_nu_l_l(es)

        def dnde_pi_pi0_l(es):
            return self.dnde_pi_pi0_l(es)

        return {
            "pi l": dnde_pi_l,
            "pi0 nu": dnde_pi0_nu,
            "k l": dnde_k_l,
            "nu l l": dnde_nu_l_l,
            "pi pi0 l": dnde_pi_pi0_l,
        }

    def _gamma_ray_line_energies(self):
        """
        Returns dict of final states and photon energies for final states
        containing monochromatic gamma ray lines.
        """
        return {"nu g": self.mx / 2.0}

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
