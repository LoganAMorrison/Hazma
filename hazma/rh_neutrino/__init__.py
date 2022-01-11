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
    charged_kaon_mass as mk,
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
    theta: float
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

    # TODO: These need to be fixed before they can be used.
    # from ._rh_neutrino_fsr_four_body import dnde_nu_l_l_fsr as _dnde_nu_l_l_fsr
    # from ._rh_neutrino_fsr_four_body import dnde_l_pi_pi0_fsr as _dnde_l_pi_pi0_fsr
    # from ._rh_neutrino_fsr_four_body import dnde_nu_pi_pi_fsr as _dnde_nu_pi_pi_fsr

    from ._rh_neutrino_fsr import (
        dnde_pi_l_fsr,
        dnde_k_l_fsr,
        dnde_nu_l_l_fsr,
        dnde_l_pi_pi0_fsr,
        dnde_nu_pi_pi_fsr,
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

    def __init__(self, mx, theta, lepton="e", include_3body=False):
        """
        Generate an MeV right-handed object.

        Parameters
        ----------
        rhn_mass: float
            Right-handed neutrino mass in MeV.
        theta: float
            Mixing angle between the right-handed neutrino and active neutrino.
        lepton: str, optional
            String specifying which flavor of active neutrino the RH neutrino
            mixes with. Options are "e" or "mu". Default is "e".
        include_3body: bool, optional
            Flag specifying if 3-body final states should be consider (i.e.
            N->nu+nu+nu, N->nu+l+lbar, etc.). Default is False.
        """
        self._theta = theta
        self._mx = mx
        self.include_3body = include_3body

        self._lepton = lepton
        if lepton == "e":
            self._ml = me
            self._gen = 1
        elif lepton == "mu":
            self._ml = mmu
            self._gen = 2
        else:
            raise ValueError("Lepton {} is invalid. Use 'e' or 'mu'.".format(lepton))

    def __nu_l_l_final_states(self):
        i = self._gen
        ll = "e" if i == 1 else "mu"
        lp = "mu" if i == 1 else "e"

        return [f"nu{ll} {ll} {ll}", f"nu{ll} {lp} {lp}", f"nu{lp} {ll} {lp}"]

    def __nu_nu_nu_final_states(self):
        i = self._gen
        ll = "e" if i == 1 else "mu"
        lp = "mu" if i == 1 else "e"

        return [f"nu{ll} nu{ll} nu{ll}", f"nu{ll} nu{lp} nu{lp}", f"nu{ll} nutau nutau"]

    def list_decay_final_states(self):
        """
        Returns a list of the availible final states.
        """
        fstates = [
            "pi l",
            "pi0 nu",
            "k l",
            "nu pi pi",
            "l pi pi0",
            "nu g",
            # "nu g g",
        ]
        fstates = fstates + self.__nu_l_l_final_states()
        fstates = fstates + self.__nu_nu_nu_final_states()
        return fstates

    def _decay_widths(self):
        """
        Decay width into each final state.
        """
        widths = {
            "pi l": 2 * self.width_pi_l(),
            "pi0 nu": self.width_pi0_nu(),
            "k l": 2 * self.width_k_l(),
            "nu pi pi": self.width_nu_pi_pi(),
            "l pi pi0": 2 * self.width_l_pi_pi0(),
            "nu g": self.width_nu_gamma(),
        }
        i = self._gen
        lep = self._lepton

        j = 2 if i == 1 else 1
        lepp = "e" if j == 1 else "mu"

        widths[f"nu{lep} nu{lep} nu{lep}"] = self.width_nu_nu_nu(i, i, i)
        widths[f"nu{lep} nu{lepp} nu{lepp}"] = self.width_nu_nu_nu(i, j, j)
        widths[f"nu{lep} nutau nutau"] = self.width_nu_nu_nu(i, 3, 3)

        widths[f"nu{lep} {lep} {lep}"] = self.width_nu_l_l(i, i, i)
        widths[f"nu{lep} {lepp} {lepp}"] = self.width_nu_l_l(i, j, j)
        widths[f"nu{lepp} {lep} {lepp}"] = 2 * self.width_nu_l_l(j, i, j)

        return widths

    def _spectrum_funcs(self):
        """
        Gets a function taking a photon energy and returning the continuum
        gamma ray spectrum dN/dE for each relevant decay final state.
        """
        funcs = {
            "pi l": lambda e: 2 * self.dnde_pi_l(e),  # type: ignore
            "pi0 nu": self.dnde_nu_pi0,
            "k l": lambda e: self.dnde_k_l(e),  # type: ignore
            "l pi pi0": lambda e: 2 * self.dnde_l_pi_pi0(e),  # type: ignore
            "nu pi pi": self.dnde_nu_pi_pi,
            # "nu g g": self.dnde_nu_g_g,
        }

        i = self._gen
        j = 2 if i == 1 else 1
        ll = "e" if i == 1 else "mu"
        lp = "mu" if i == 1 else "e"
        funcs[f"nu{ll} {ll} {ll}"] = lambda es: self.dnde_nu_l_l(es, i, i, i)
        funcs[f"nu{ll} {lp} {lp}"] = lambda es: self.dnde_nu_l_l(es, i, j, j)
        funcs[f"nu{lp} {ll} {lp}"] = lambda es: 2.0 * self.dnde_nu_l_l(es, j, i, j)

        return funcs

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
        # TODO: Add the 3-body final states
        if self.lepton == "e":
            return {
                "pi l": (self.mx ** 2 + self.ml ** 2 - mpi ** 2) / (2.0 * self.mx),
                "k l": (self.mx ** 2 + self.ml ** 2 - mk ** 2) / (2.0 * self.mx),
            }
        return {}

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
    def theta(self):
        """
        Get the mixing angle between right-handed neutrino and active neutrino.
        """
        return self._theta

    @theta.setter
    def theta(self, val):
        """
        Set the mixing angle between the right-handed neutrino and active
        neutrino.

        Parameters
        ----------
        val: float
            New mixing angle.
        """
        self._theta = val

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
        if val == "e":
            self._lepton = "e"
            self._ml = me
            self._gen = 1
        elif val == "mu":
            self._lepton = "mu"
            self._ml = mmu
            self._gen = 2
        else:
            raise ValueError(f"Invalid lepton {val}. Use 'e' or 'mu'")

    @property
    def ml(self):
        """
        Return the mass of the lepton associated with the neutrino the RH
        neutrino mixes with.
        """
        return self._ml
