"""
Implementation of dark matter models annihilating into a single final state.
"""

from typing import Callable, Dict, List
import functools as ft

from hazma.theory import TheoryAnn, TheoryDec
from hazma.parameters import standard_model_masses as sm_masses
from hazma import spectra


class SingleChannelAnn(TheoryAnn):
    """Model with dark matter annihilating into a single final state."""

    def __init__(self, mx: float, fs: str, sigma: float) -> None:
        self._mx = mx
        self._fs = fs
        self.sigma = sigma
        self._final_states = fs.split(" ")
        self._fs_masses = list(map(lambda s: sm_masses[s], self._final_states))

    def __repr__(self) -> str:
        return (
            "SingleChannelAnn("
            + f"mx={self._mx} MeV"
            + f"fs='{self._fs}'"
            + f"sigma={self.sigma} MeV^-2"
            + ")"
        )

    @property
    def fs(self) -> str:
        """The final state."""
        return self._fs

    @fs.setter
    def fs(self, fs) -> None:
        self._fs = fs
        self._final_states = fs.split(" ")
        self._fs_masses = list(map(lambda s: sm_masses[s], self._final_states))

    @property
    def mx(self) -> float:
        """Dark matter mass in MeV."""
        return self._mx

    @mx.setter
    def mx(self, mx) -> None:
        self._mx = mx

    @property
    def fs_mass(self) -> float:
        """Return the threshold for annihilation."""
        return sum(map(lambda s: sm_masses[s], self._final_states))

    def annihilation_cross_section_funcs(self) -> Dict[str, Callable]:
        def xsec(e_cm):
            if e_cm < 2 * self.mx or e_cm < self.fs_mass:
                return 0.0
            return self.sigma

        return {self.fs: xsec}

    def list_annihilation_final_states(  # pylint: disable=arguments-differ
        self,
    ) -> List[str]:
        r"""Return a list of the final state particles DM can annihilate into."""
        return [self.fs]

    def __spectrum_func(self, product: str) -> Callable:
        final_states = self.fs.split(" ")
        if product == "photon":
            dnde_fn = spectra.dnde_photon
        elif product == "positron":
            dnde_fn = spectra.dnde_positron
        elif product in ["neutrino", "ve", "vm", "vt"]:
            dnde_fn = spectra.dnde_neutrino
        else:
            raise ValueError(f"Invalid product {product}")

        return ft.partial(dnde_fn, final_states=final_states)

    def _spectrum_funcs(self) -> Dict[str, Callable]:
        return {self.fs: self.__spectrum_func("photon")}

    def _gamma_ray_line_energies(self, e_cm) -> Dict:
        lines = {}
        if len(self._final_states) == 2:
            s1, s2 = self._final_states
            m1, m2 = self._fs_masses
            energies = [
                (e_cm**2 + m1**2 - m2**2) / (2 * e_cm),
                (e_cm**2 - m1**2 + m2**2) / (2 * e_cm),
            ]

            if s1 == "g" and s2 == "g":
                lines[self.fs] = {"g g": energies[0]}
            elif s1 == "g":
                lines[self.fs] = {"g g": energies[0]}
            elif s2 == "g":
                lines[self.fs] = {"g g": energies[1]}

        return lines

    def _positron_spectrum_funcs(self) -> Dict[str, Callable]:
        return {self.fs: self.__spectrum_func("positron")}

    def _positron_line_energies(self, e_cm) -> Dict[str, float]:
        lines = {}
        if len(self._final_states) == 2:
            s1, s2 = self._final_states
            m1, m2 = self._fs_masses
            energies = [
                (e_cm**2 + m1**2 - m2**2) / (2 * e_cm),
                (e_cm**2 - m1**2 + m2**2) / (2 * e_cm),
            ]

            if s1 == "e":
                lines[self.fs] = {self.fs: energies[0]}
            elif s2 == "e":
                lines[self.fs] = {self.fs: energies[1]}

        return lines

    def partial_widths(self):
        raise NotImplementedError(
            "partial_widths is not implimented for the SingleChannelAnn model."
        )


class SingleChannelDec(TheoryDec):
    r"""Model with dark matter decaying into a single final state."""

    def __init__(self, mx: float, fs: str, width: float) -> None:
        self._mx = mx
        self._fs = fs
        self.width = width
        self._final_states = fs.split(" ")
        self._fs_masses = list(map(lambda s: sm_masses[s], self._final_states))

    def __repr__(self) -> str:
        return (
            "SingleChannelDec("
            + f"mx={self._mx} MeV"
            + f"fs='{self._fs}'"
            + f"width={self.width} MeV"
            + ")"
        )

    @property
    def fs(self) -> str:
        """The final state."""
        return self._fs

    @fs.setter
    def fs(self, fs: str) -> None:
        self._fs = fs

    @property
    def mx(self) -> float:
        """Dark matter mass in MeV."""
        return self._mx

    @mx.setter
    def mx(self, mx: float) -> None:
        self._mx = mx

    @property
    def fs_mass(self) -> float:
        """Return the threshold for annihilation."""
        return sum(map(lambda s: sm_masses[s], self._final_states))

    def list_decay_final_states(self) -> List[str]:  # pylint: disable=arguments-differ
        r"""Return a list of the final state particles DM can decay into."""
        return [self.fs]

    def _decay_widths(self) -> Dict[str, float]:
        return {self.fs: self.width}

    def __spectrum_func(self, product: str) -> Callable:
        final_states = self.fs.split(" ")
        if product == "photon":
            dnde_fn = spectra.dnde_photon
        elif product == "positron":
            dnde_fn = spectra.dnde_positron
        elif product in ["neutrino", "ve", "vm", "vt"]:
            dnde_fn = spectra.dnde_neutrino
        else:
            raise ValueError(f"Invalid product {product}")

        def dnde(product_energies):
            return dnde_fn(product_energies, self.mx, final_states=final_states)

        return dnde

    def _spectrum_funcs(self) -> Dict[str, Callable]:
        return {self.fs: self.__spectrum_func("photon")}

    def _gamma_ray_line_energies(self) -> Dict[str, float]:
        mx = self.mx
        lines = {}
        if len(self._final_states) == 2:
            s1, s2 = self._final_states
            m1, m2 = self._fs_masses
            energies = [
                (mx**2 + m1**2 - m2**2) / (2 * mx),
                (mx**2 - m1**2 + m2**2) / (2 * mx),
            ]

            if s1 == "g" and s2 == "g":
                lines[self.fs] = {"g g": energies[0]}
            elif s1 == "g":
                lines[self.fs] = {"g g": energies[0]}
            elif s2 == "g":
                lines[self.fs] = {"g g": energies[1]}

        return lines

    def _positron_spectrum_funcs(self) -> Dict[str, Callable]:
        return {self.fs: self.__spectrum_func("positron")}

    def _positron_line_energies(self) -> Dict[str, float]:
        mx = self.mx
        lines = {}
        if len(self._final_states) == 2:
            s1, s2 = self._final_states
            m1, m2 = self._fs_masses
            energies = [
                (mx**2 + m1**2 - m2**2) / (2 * mx),
                (mx**2 - m1**2 + m2**2) / (2 * mx),
            ]

            if s1 == "e":
                lines[self.fs] = {self.fs: energies[0]}
            elif s2 == "e":
                lines[self.fs] = {self.fs: energies[1]}

        return lines

    def constraints(self):
        raise NotImplementedError("Constraints are not implimented.")
