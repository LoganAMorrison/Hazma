"""
Core implementation of the RH-Neutrino model.
"""

from typing import Callable, Dict, List

from hazma.theory import TheoryDec
from hazma.parameters import standard_model_masses as sm_masses
from hazma.form_factors.vector import VectorFormFactorPiPi

from ._proto import SingleRhNeutrinoModel, Generation
from . import _configure as rhn_configure


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
            self.__gen = Generation.Fst
            self._vstr = "ve"
        elif flavor == "mu":
            self.__gen = Generation.Snd
            self._vstr = "vm"
        elif flavor == "tau":
            self.__gen = Generation.Trd
            self._vstr = "vt"
        else:
            raise ValueError(
                "`flavor` {} is invalid. Use 'e', 'mu', or 'tau'.".format(flavor)
            )

        self._ml = sm_masses[flavor]
        self._lstr = flavor
        self._ff_pi = VectorFormFactorPiPi()
        self._config = rhn_configure.configure(self, form_factor=self._ff_pi)

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
            self.__gen = Generation.Fst
            self._vstr = "ve"
        elif val == "mu":
            self.__gen = Generation.Snd
            self._vstr = "vmu"
        elif val == "tau":
            self.__gen = Generation.Snd
            self._vstr = "vmu"
        else:
            raise ValueError(f"Invalid lepton {val}. Use 'e', 'mu', or 'tau'")

        self._flavor = val
        self._lstr = val
        self._ml = sm_masses[val]
        self._config = rhn_configure.configure(self, form_factor=self._ff_pi)

    @property
    def gen(self) -> Generation:
        r"""The generation of the right-handed neutrino."""
        return self.__gen

    @property
    def ml(self) -> float:
        r"""Return the mass of the lepton associated with the neutrino the RH
        neutrino mixes with.
        """
        return self._ml

    def list_decay_final_states(self) -> List[str]:
        r"""Returns a list of the availible final states."""
        return list(self._config.keys())

    def _decay_widths(self) -> Dict[str, float]:
        """
        Decay width into each final state.
        """

        def pre(self_conjugate):
            return 1.0 if self_conjugate else 2.0

        return {
            key: pre(val.self_conjugate) * val.width()
            for key, val in self._config.items()
        }

    def __spectrum_funcs(self, product: str) -> Dict[str, Callable]:
        """
        Gets a function taking a photon energy and returning the continuum
        gamma ray spectrum dN/dE for each relevant decay final state.
        """
        ispositron = product == "positron"
        conjfactor = 1.0 if ispositron else 2.0

        def pre(self_conjugate):
            return 1.0 if self_conjugate else conjfactor

        def mk_dnde(dnde, self_conjugate):
            def new_dnde(energies):
                return pre(self_conjugate) * dnde(
                    product_energies=energies, product=product
                )

            return new_dnde

        return {
            key: mk_dnde(val.dnde, val.self_conjugate)
            for key, val in self._config.items()
        }

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
        vv = self._vstr
        mx = self.mx

        if self.mx > sm_masses["pi0"]:
            lines[f"{vv} pi0"] = (mx**2 - sm_masses["pi0"] ** 2) / (2.0 * mx)

        if self.mx > sm_masses["eta"]:
            lines[f"{vv} pi0"] = (mx**2 - sm_masses["eta"] ** 2) / (2.0 * mx)

        lines[f"{vv} a"] = 0.5 * mx
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
