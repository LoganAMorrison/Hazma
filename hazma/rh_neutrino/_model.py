from typing import Callable, Dict, List

from hazma.theory import TheoryDec
from hazma.parameters import standard_model_masses as sm_masses
from hazma.form_factors.vector import VectorFormFactorPiPi

from ._proto import SingleRhNeutrinoModel
from ._utils import three_lepton_fs_generations
from ._utils import three_lepton_fs_strings
from . import _widths as rhn_widths
from . import _spectra as rhn_spectra


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
    flavor: str
        String specifying which flavor of neutrino the RH neutrino mixes with.
    """

    def __init__(self, mx: float, theta: float, flavor: str):
        """
        Generate an MeV right-handed object.

        Parameters
        ----------
        rhn_mass: float
            Right-handed neutrino mass in MeV.
        theta: float
            Mixing angle between the right-handed neutrino and active neutrino.
        flavor: str, optional
            String specifying which flavor of active neutrino the RH neutrino
            mixes with. Options are "e" or "mu". Default is "e".
        """
        self._theta = theta
        self._mx = mx

        self._flavor = flavor
        if flavor == "e":
            self._gen = 1
            self._vstr = "ve"
        elif flavor == "mu":
            self._gen = 2
            self._vstr = "vmu"
        else:
            raise ValueError("`flavor` {} is invalid. Use 'e' or 'mu'.".format(flavor))

        self._ml = sm_masses[flavor]
        self._lstr = flavor
        self._ff_pi = VectorFormFactorPiPi()

    @property
    def mx(self) -> float:
        """
        Get the right-handed neutrino mass in MeV.
        """
        return self._mx

    @mx.setter
    def mx(self, val: float) -> None:
        self._mx = val

    @property
    def theta(self) -> float:
        """
        Get the mixing angle between right-handed neutrino and active neutrino.
        """
        return self._theta

    @theta.setter
    def theta(self, val: float) -> None:
        self._theta = val

    @property
    def flavor(self) -> str:
        """
        Return a string specifying the lepton flavor which the RH-neutrino
        mixes with.
        """
        return self._flavor

    @flavor.setter
    def flavor(self, val: str) -> None:
        if val == "e":
            self._gen = 1
            self._vstr = "ve"
        elif val == "mu":
            self._gen = 2
            self._vstr = "vmu"
        else:
            raise ValueError(f"Invalid lepton {val}. Use 'e' or 'mu'")

        self._flavor = val
        self._lstr = val
        self._ml = sm_masses[val]

    @property
    def ml(self) -> float:
        """
        Return the mass of the lepton associated with the neutrino the RH
        neutrino mixes with.
        """
        return self._ml

    def list_decay_final_states(self) -> List[str]:
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

    def _decay_widths(self) -> Dict[str, float]:
        """
        Decay width into each final state.
        """
        ll = self._lstr
        vv = self._vstr

        pws: Dict[str, float] = {}
        pws[f"{ll} k"] = 2 * rhn_widths.width_l_k(self, self.ml)
        pws[f"{ll} pi"] = 2 * rhn_widths.width_l_pi(self, self.ml)
        pws[f"{vv} pi0"] = rhn_widths.width_v_pi0(self)
        pws[f"{vv} eta"] = rhn_widths.width_v_eta(self)
        pws[f"{vv} a"] = rhn_widths.width_v_a(self)
        pws[f"{vv} pi pibar"] = rhn_widths.width_v_pi_pi(self, self._ff_pi)
        pws[f"{ll} pi pi0"] = 2 * rhn_widths.width_l_pi0_pi(self, self._ff_pi)

        # N -> v1 + l2 + lbar3
        gen_tups = three_lepton_fs_generations(self._gen)
        str_tups = three_lepton_fs_strings(self._gen)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, s2, s3 + "bar"])
            pf = 1.0 if g2 == g3 else 2.0
            pws[key] = pf * rhn_widths.width_v_l_l(self, g1, g2, g3)

        # N -> v1 + v2 + v3
        gen_tups = three_lepton_fs_generations(self._gen, unique=True)
        str_tups = three_lepton_fs_strings(self._gen, unique=True)
        for gen_tup, str_tup in zip(gen_tups, str_tups):
            g1, g2, g3 = gen_tup
            s1, s2, s3 = str_tup
            key = " ".join(["v" + s1, "v" + s2, "v" + s3])
            pws[key] = rhn_widths.width_v_v_v(self, g1, g2)

        return pws

    def __spectrum_funcs(self, product: str) -> Dict[str, Callable]:
        """
        Gets a function taking a photon energy and returning the continuum
        gamma ray spectrum dN/dE for each relevant decay final state.
        """
        # Factor to account for charge conjugation
        pre = 1.0 if product == "positron" else 2.0

        funcs = {}
        funcs["pi l"] = lambda e: pre * rhn_spectra.dnde_l_pi(self, e, product=product)
        funcs["k l"] = lambda e: pre * rhn_spectra.dnde_l_k(self, e, product=product)
        funcs["pi0 nu"] = lambda e: rhn_spectra.dnde_v_pi0(self, e, product=product)
        funcs["eta nu"] = lambda e: rhn_spectra.dnde_v_eta(self, e, product=product)
        funcs["nu pi pi"] = lambda e: rhn_spectra.dnde_nu_pi_pi(
            self, e, product=product, form_factor=self._ff_pi
        )
        funcs["l pi pi0"] = lambda e: pre * rhn_spectra.dnde_nu_pi_pi(
            self, e, product=product, form_factor=self._ff_pi
        )

        # TODO: Added the v + l + l spectra
        # i = self._gen
        # j = 2 if i == 1 else 1
        # ll = "e" if i == 1 else "mu"
        # lp = "mu" if i == 1 else "e"
        # funcs[f"nu{ll} {ll} {ll}"] = lambda es: rhn_spectra.dnde_v_l_l(es, i, i, i)
        # funcs[f"nu{ll} {lp} {lp}"] = lambda es: self.dnde_nu_l_l(es, i, j, j)
        # funcs[f"nu{lp} {ll} {lp}"] = lambda es: 2.0 * self.dnde_nu_l_l(es, j, i, j)

        return funcs

    def _spectrum_funcs(self) -> Dict[str, Callable]:
        """
        Gets a function taking a photon energy and returning the continuum
        gamma ray spectrum dN/dE for each relevant decay final state.
        """
        return self.__spectrum_funcs("photon")

    def _gamma_ray_line_energies(self) -> Dict[str, float]:
        """
        Returns dict of final states and photon energies for final states
        containing monochromatic gamma ray lines.
        """
        return {"nu g": self.mx / 2.0}

    def _positron_spectrum_funcs(self) -> Dict[str, Callable]:
        """
        Returns functions `float -> float` giving the continuum positron
        spectrum for each final state.
        """
        return self.__spectrum_funcs("positron")

    def _positron_line_energies(self) -> Dict[str, float]:
        if self.flavor == "e":
            return {
                "pi l": (self.mx**2 + self.ml**2 - sm_masses["pi"] ** 2)
                / (2.0 * self.mx),
                "k l": (self.mx**2 + self.ml**2 - sm_masses["k"] ** 2)
                / (2.0 * self.mx),
            }
        return {}

    def _neutrino_spectrum_funcs(self) -> Dict[str, Callable]:
        """
        Returns functions `float -> float` giving the continuum positron
        spectrum for each final state.
        """
        return self.__spectrum_funcs("neutrino")

    def _neutrino_line_energies(self) -> Dict[str, float]:
        lines = {}
        nu = self._vstr
        mx = self.mx

        if self.mx > sm_masses["pi0"]:
            lines[f"{nu} pi0"] = (mx**2 - sm_masses["pi0"] ** 2) / (2.0 * mx)

        if self.mx > sm_masses["eta"]:
            lines[f"{nu} pi0"] = (mx**2 - sm_masses["eta"] ** 2) / (2.0 * mx)

        lines[f"{nu} a"] = 0.5 * mx
        return lines

    def constraints(self):
        r"""
        Get a dictionary of all available constraints.

        Subclasses must implement this method.

        Notes
        -----
        Each key in the dictionary is the name of a constraint. Each value is a
        function that is positive when the constraint is satisfied and negative
        when it is not.
        """
        raise NotImplementedError(
            "Constraints have not been implemented for the RH-neutrino model."
        )
