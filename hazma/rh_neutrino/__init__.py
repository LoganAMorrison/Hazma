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
        width_nu_g_g,
    )

    from ._rh_neutrino_fsr_four_body import (
        dnde_nu_l_l_fsr as _dnde_nu_l_l_fsr,
        dnde_l_pi_pi0_fsr as _dnde_l_pi_pi0_fsr,
        dnde_nu_pi_pi_fsr as _dnde_nu_pi_pi_fsr,
    )

    from ._rh_neutrino_fsr import (
        _dnde_pi_l_fsr,
        dnde_pi_l_fsr,
        _dnde_k_l_fsr,
        dnde_k_l_fsr,
    )

    from ._rh_neutrino_spectra import (
        dnde_nu_pi0,
        dnde_pi_l,
        dnde_k_l,
        dnde_nu_l_l,
        dnde_l_pi_pi0,
        dnde_nu_pi_pi,
        dnde_nu_g_g,
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

        self._lepton = lepton
        if lepton == "e":
            self._ml = me
        elif lepton == "mu":
            self._ml = mmu
        else:
            raise ValueError("Lepton {} is invalid. Use 'e' or 'mu'.".format(lepton))

    def dnde_nu_l_l_fsr(self, photon_energies):
        """
        Compute the FSR contribution to the gamma-ray spectrum fom the decay of a
        right-handed neutrino into an active neutrino and two charged leptons.

        Parameters
        ----------
        photon_energies: float or array-like
            Photon energies where the spectrum should be computed.

        Returns
        -------
        spectrum: float or array-like
            Gamma-ray spectrum.
        """
        width = self.width_nu_l_l()
        return self._dnde_nu_l_l_fsr(
            photon_energies, self.mx, self.stheta, self.ml, width
        )

    def dnde_l_pi_pi0_fsr(self, photon_energies):
        """
        Compute the FSR contribution to the gamma-ray spectrum fom the decay of a
        right-handed neutrino a charged lepton, charged pion and neutral pion.

        Parameters
        ----------
        photon_energies: float or array-like
            Photon energies where the spectrum should be computed.

        Returns
        -------
        spectrum: float or array-like
            Gamma-ray spectrum.
        """
        width = self.width_l_pi_pi0()
        return self._dnde_l_pi_pi0_fsr(
            photon_energies, self.mx, self.stheta, self.ml, width
        )

    def dnde_nu_pi_pi_fsr(self, photon_energies):
        """
        Compute the FSR contribution to the gamma-ray spectrum fom the decay of a
        right-handed neutrino into an active neutrino and two charged pions.

        Parameters
        ----------
        photon_energies: float or array-like
            Photon energies where the spectrum should be computed.

        Returns
        -------
        spectrum: float or array-like
            Gamma-ray spectrum.
        """
        width = self.width_nu_pi_pi()
        return self._dnde_nu_pi_pi_fsr(
            photon_energies, self.mx, self.stheta, self.ml, width
        )

    def list_decay_final_states(self):
        """
        Returns a list of the availible final states.
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
            "nu pi pi": self.width_nu_pi_pi(),
            "l pi pi0": self.width_l_pi_pi0(),
            "nu nu nu": self.width_nu_nu_nu(),
            "nu l l": self.width_nu_l_l(),
            "nu g": self.width_nu_gamma(),
            "nu g g": self.width_nu_g_g(),
        }

    def _spectrum_funcs(self):
        """
        Gets a function taking a photon energy and returning the continuum
        gamma ray spectrum dN/dE for each relevant decay final state.
        """
        return {
            "pi l": self.dnde_pi_l,
            "pi0 nu": self.dnde_nu_pi0,
            "k l": self.dnde_k_l,
            "nu l l": self.dnde_nu_l_l,
            "l pi pi0": self.dnde_l_pi_pi0,
            "nu pi pi": self.dnde_nu_pi_pi,
            "nu g g": self.dnde_nu_g_g,
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
        return {"pi l": (self.mx ** 2 + self.ml ** 2 - mpi ** 2) / (2.0 * self.mx)}

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

    @property
    def lepton(self):
        """
        Return a string specifying the lepton flavor which the RH-neutrino
        mixes with.
        """
        return self._lepton

    @lepton.setter
    def lepton(self, val):
        """
        Specify the lepton flavor which the RH-neutrino mixes with. Options
        are "e" or "mu".

        Parameters
        ----------
        val: str
           Lepton flavor. Options are "e" or "mu".
        """
        if val in ["e", "mu"]:
            self._lepton = val
            self._ml = me if val == "e" else mmu
        else:
            raise ValueError(f"Invalid lepton {val}. Use 'e' or 'mu'")

    @property
    def ml(self):
        """
        Return the mass of the lepton associated with the neutrino the RH
        neutrino mixes with.
        """
        return self._ml
