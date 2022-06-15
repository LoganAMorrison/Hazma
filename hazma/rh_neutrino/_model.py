from typing import Dict, List

from hazma.theory import TheoryDec
from hazma.parameters import (
    electron_mass as me,
    charged_pion_mass as mpi,
    charged_kaon_mass as mk,
    muon_mass as mmu,
)
from hazma.form_factors.vector import VectorFormFactorPiPi

from ._proto import SingleRhNeutrinoModel
from ._utils import three_lepton_fs_generations
from ._utils import three_lepton_fs_strings
from ._widths import (
    width_v_pi0,
    width_v_eta,
    width_l_pi,
    width_l_k,
    width_l_pi0_pi,
    width_v_pi_pi,
    width_v_l_l,
    width_v_v_v,
    width_v_a,
)


class RHNeutrino(SingleRhNeutrinoModel, TheoryDec):
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

    def __init__(self, mx: float, theta: float, flavor: str = "e"):
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

        self._flavor = flavor
        if flavor == "e":
            self._ml = me
            self._gen = 1
            self._vstr = "ve"
            self._lstr = "e"
        elif flavor == "mu":
            self._ml = mmu
            self._gen = 2
            self._vstr = "vmu"
            self._lstr = "mu"
        else:
            raise ValueError("`flavor` {} is invalid. Use 'e' or 'mu'.".format(flavor))

        self._ff_pi = VectorFormFactorPiPi()

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
    def flavor(self):
        """
        Return a string specifying the lepton flavor which the RH-neutrino
        mixes with.
        """
        return self._flavor

    @flavor.setter
    def flavor(self, val):
        """
        Specify the lepton flavor which the RH-neutrino mixes with. Options
        are "e" or "mu".

        Parameters
        ----------
        val: str
           Lepton flavor. Options are "e" or "mu".
        """
        if val == "e":
            self._flavor = "e"
            self._ml = me
            self._gen = 1
            self._vstr = "ve"
            self._lstr = "e"
        elif val == "mu":
            self._flavor = "mu"
            self._ml = mmu
            self._gen = 2
            self._vstr = "vmu"
            self._lstr = "mu"
        else:
            raise ValueError(f"Invalid lepton {val}. Use 'e' or 'mu'")

    @property
    def ml(self):
        """
        Return the mass of the lepton associated with the neutrino the RH
        neutrino mixes with.
        """
        return self._ml

    def list_decay_final_states(self):
        """
        Returns a list of the availible final states.
        """
        ll = self._lstr
        vv = self._vstr

        final_states: List[str] = [
            f"{ll} k",
            f"{ll} pi",
            f"{vv} pi0",
            f"{vv} eta",
            f"{vv} a",
            f"{vv} pi pibar",
            f"{ll} pi pi0",
        ]

        # N -> v1 + l2 + lbar3
        str_tups = three_lepton_fs_strings(self._gen)
        for s1, s2, s3 in str_tups:
            final_states.append(" ".join(["v" + s1, s2, s3 + "bar"]))

        # N -> v1 + v2 + v3
        str_tups = three_lepton_fs_strings(self._gen, unique=True)
        for s1, s2, s3 in str_tups:
            final_states.append(" ".join(["v" + s1, "v" + s2, "v" + s3]))

        return final_states

    def _decay_widths(self):
        """
        Decay width into each final state.
        """
        ll = self._lstr
        vv = self._vstr

        pws: Dict[str, float] = {}
        pws[f"{ll} k"] = 2 * width_l_k(self, self.ml)
        pws[f"{ll} pi"] = 2 * width_l_pi(self, self.ml)
        pws[f"{vv} pi0"] = width_v_pi0(self)
        pws[f"{vv} eta"] = width_v_eta(self)
        pws[f"{vv} a"] = width_v_a(self)
        pws[f"{vv} pi pibar"] = width_v_pi_pi(self, self._ff_pi)
        pws[f"{ll} pi pi0"] = 2 * width_l_pi0_pi(self, self._ml, self._ff_pi)

        # N -> v1 + l2 + lbar3
        gen_tups = three_lepton_fs_generations(self._gen)
        str_tups = three_lepton_fs_strings(self._gen)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, s2, s3 + "bar"])
            pf = 1.0 if g2 == g3 else 2.0
            pws[key] = pf * width_v_l_l(self, g1, g2, g3)

        # N -> v1 + v2 + v3
        gen_tups = three_lepton_fs_generations(self._gen, unique=True)
        str_tups = three_lepton_fs_strings(self._gen, unique=True)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, "v" + s2, "v" + s3])
            pws[key] = width_v_v_v(self, g1, g2)

        return pws

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
                "pi l": (self.mx**2 + self.ml**2 - mpi**2) / (2.0 * self.mx),
                "k l": (self.mx**2 + self.ml**2 - mk**2) / (2.0 * self.mx),
            }
        return {}
