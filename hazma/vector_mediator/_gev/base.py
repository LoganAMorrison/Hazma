"""Base class for the GeV vector mediator model."""

# pylint: disable=too-many-instance-attributes

import dataclasses
from abc import abstractmethod
from typing import Union

from hazma.theory import TheoryAnn
from hazma.vector_mediator.form_factors.utils import ComplexArray, RealArray


@dataclasses.dataclass
class VectorMediatorCouplings:
    """Dataclass holding vector couplings to fermions."""

    gvxx: float
    gvuu: float
    gvdd: float
    gvss: float
    gvee: float
    gvmumu: float
    gvveve: float
    gvvmvm: float
    gvvtvt: float


class VectorMediatorGeVBase(TheoryAnn):
    def __init__(
        self,
        mx: float,
        mv: float,
        gvxx: float,
        gvuu: float,
        gvdd: float,
        gvss: float,
        gvee: float,
        gvmumu: float,
        gvveve: float,
        gvvmvm: float,
        gvvtvt: float,
    ) -> None:
        """
        Create a `VectorMediatorGeV` object.

        Parameters
        ----------
        mx : float
            Mass of the dark matter.
        mv : float
            Mass of the vector mediator.
        gvxx : float
            Coupling of vector mediator to dark matter.
        gvuu : float
            Coupling of vector mediator to the up quark.
        gvdd : float
            Coupling of vector mediator to the down quark.
        gvss : float
            Coupling of vector mediator to the strange quark.
        gvee : float
            Coupling of vector mediator to the electron.
        gvmumu : float
            Coupling of vector mediator to the muon.
        gvveve : float
            Coupling of vector mediator to the electron-neutrino.
        gvvmvm : float
            Coupling of vector mediator to the muon-neutrino.
        gvvtvt : float
            Coupling of vector mediator to the tau-neutrino.
        """
        from hazma.vector_mediator.form_factors.eta_gamma import FormFactorEtaGamma
        from hazma.vector_mediator.form_factors.eta_omega import FormFactorEtaOmega
        from hazma.vector_mediator.form_factors.eta_phi import FormFactorEtaPhi
        from hazma.vector_mediator.form_factors.four_pi import FormFactorPiPiPiPi
        from hazma.vector_mediator.form_factors.kk import FormFactorKK
        from hazma.vector_mediator.form_factors.omega_pi import FormFactorOmegaPi0
        from hazma.vector_mediator.form_factors.phi_pi import FormFactorPhiPi0
        from hazma.vector_mediator.form_factors.pi_gamma import FormFactorPiGamma
        from hazma.vector_mediator.form_factors.pi_k_k import (
            FormFactorPi0K0K0,
            FormFactorPi0KpKm,
            FormFactorPiKK0,
        )
        from hazma.vector_mediator.form_factors.pi_pi_eta import FormFactorPiPiEta
        from hazma.vector_mediator.form_factors.pi_pi_etap import FormFactorPiPiEtaP
        from hazma.vector_mediator.form_factors.pi_pi_omega import FormFactorPiPiOmega
        from hazma.vector_mediator.form_factors.pi_pi_pi0 import FormFactorPiPiPi0
        from hazma.vector_mediator.form_factors.pipi import FormFactorPiPi

        # Compute and store the parameters needed to compute form factors.
        # self._ff_pipi_params = _compute_ff_params_pipi(2000)
        self._ff_pi_pi = FormFactorPiPi()
        # self._ff_kk_params = _compute_ff_params_kk(200)
        self._ff_k_k = FormFactorKK()
        self._ff_eta_gamma = FormFactorEtaGamma()
        self._ff_eta_omega = FormFactorEtaOmega()
        self._ff_eta_phi = FormFactorEtaPhi()
        self._ff_pi_gamma = FormFactorPiGamma()
        self._ff_pi_omega = FormFactorOmegaPi0()
        self._ff_pi_phi = FormFactorPhiPi0()

        self._ff_pi_pi_pi0 = FormFactorPiPiPi0()
        self._ff_pi_pi_eta = FormFactorPiPiEta()
        self._ff_pi_pi_etap = FormFactorPiPiEtaP()
        self._ff_pi_pi_omega = FormFactorPiPiOmega()
        self._ff_pi0_k0_k0 = FormFactorPi0K0K0()
        self._ff_pi0_k_k = FormFactorPi0KpKm()
        self._ff_pi_k_k0 = FormFactorPiKK0()
        self._ff_four_pi = FormFactorPiPiPiPi()

        self._mx = mx
        self._mv = mv
        self._gvxx = gvxx
        self._gvuu = gvuu
        self._gvdd = gvdd
        self._gvss = gvss
        self._gvee = gvee
        self._gvmumu = gvmumu
        self._gvveve = gvveve
        self._gvvmvm = gvvmvm
        self._gvvtvt = gvvtvt

        final_states = self.list_annihilation_final_states()
        self._width_cache = {
            name: {"value": 0.0, "valid": False, "kwargs": dict()}
            for name in final_states
        }
        self._width_cache["x x"] = {"value": 0.0, "valid": False, "kwargs": dict()}

    # ========================================================================
    # ---- Cache Control -----------------------------------------------------
    # ========================================================================

    def _invalidate_width_cache(self):
        """Invalidate the entire cache."""
        for key in self._width_cache.keys():
            self._width_cache[key]["valid"] = False

    def _update_width_cache(self, name, val):
        """Invalidate the entire cache."""
        assert name in self._width_cache, f"Invalid cache entry: no entry named {name}."

        self._width_cache[name]["valid"] = True
        self._width_cache[name]["value"] = val

    # ========================================================================
    # ---- Masses ------------------------------------------------------------
    # ========================================================================

    @property
    def mx(self) -> float:
        """Dark matter mass in MeV."""
        return self._mx

    @property
    def mv(self) -> float:
        """Vector mediator mass in MeV."""
        return self._mv

    @mv.setter
    def mv(self, val: float) -> None:
        self._mv = val
        self._invalidate_width_cache()

    # ========================================================================
    # ---- Couplings ---------------------------------------------------------
    # ========================================================================

    @property
    def gvxx(self) -> float:
        """Coupling of vector mediator to the dark matter."""
        return self._gvuu

    @property
    def gvuu(self) -> float:
        """Coupling of vector mediator to the up-quark."""
        return self._gvuu

    @property
    def gvdd(self) -> float:
        """Coupling of vector mediator to the down-quark."""
        return self._gvdd

    @property
    def gvss(self) -> float:
        """Coupling of vector mediator to the strange-quark."""
        return self._gvss

    @property
    def gvee(self) -> float:
        """Coupling of vector mediator to the electron."""
        return self._gvee

    @property
    def gvmumu(self) -> float:
        """Coupling of vector mediator to the muon."""
        return self._gvmumu

    @property
    def gvveve(self) -> float:
        """Coupling of vector mediator to the electron-neutrino."""
        return self._gvveve

    @property
    def gvvmvm(self) -> float:
        """Coupling of vector mediator to the muon-neutrino."""
        return self._gvvmvm

    @property
    def gvvtvt(self) -> float:
        """Coupling of vector mediator to the tau-neutrino."""
        return self._gvvtvt

    # ========================================================================
    # ---- Form Factors ------------------------------------------------------
    # ========================================================================

    def form_factor_pi_pi(
        self, q: Union[float, RealArray], imode: int = 1
    ) -> Union[complex, ComplexArray]:
        """
        Compute the pi-pi-V form factor.

        Parameters
        ----------
        q: Union[float,RealArray
            Center-of-mass energy in MeV.
        imode: Optional[int]
            Iso-spin channel. Default is 1.

        Returns
        -------
        ff: Union[complex,ComplexArray]
            Form factor from pi-pi-V.
        """
        return self._ff_pi_pi.form_factor(
            q=q, gvuu=self.gvuu, gvdd=self.gvdd, imode=imode
        )

    def form_factor_k_k(
        self, q: Union[float, RealArray], imode: int = 1
    ) -> Union[complex, ComplexArray]:
        """
        Compute the pi-pi-V form factor.

        Parameters
        ----------
        s: Union[float,RealArray
            Center-of-mass energy in MeV.
        imode: Optional[int]
            Iso-spin channel. Default is 1.

        Returns
        -------
        ff: Union[complex,ComplexArray]
            Form factor from pi-pi-V.
        """
        return self._ff_k_k.form_factor(
            q=q, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, imode=imode
        )

    def form_factor_pi_gamma(
        self, q: Union[float, RealArray]
    ) -> Union[complex, ComplexArray]:
        return self._ff_pi_gamma.form_factor(
            q=q, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss
        )

    def form_factor_pi_omega(
        self, q: Union[float, RealArray]
    ) -> Union[complex, ComplexArray]:
        return self._ff_pi_omega.form_factor(q=q, gvuu=self.gvuu, gvdd=self.gvdd)

    def form_factor_pi_phi(
        self, q: Union[float, RealArray]
    ) -> Union[complex, ComplexArray]:
        return self._ff_pi_phi.form_factor(q=q, gvuu=self.gvuu, gvdd=self.gvdd)

    def form_factor_eta_gamma(
        self, q: Union[float, RealArray]
    ) -> Union[complex, ComplexArray]:
        return self._ff_eta_gamma.form_factor(
            q=q, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss
        )

    def form_factor_eta_phi(
        self, q: Union[float, RealArray]
    ) -> Union[complex, ComplexArray]:
        return self._ff_eta_phi.form_factor(q=q, gvss=self.gvss)

    def form_factor_eta_omega(
        self, q: Union[float, RealArray]
    ) -> Union[complex, ComplexArray]:
        return self._ff_eta_omega.form_factor(q=q, gvuu=self.gvuu, gvdd=self.gvdd)

    # ========================================================================
    # ---- Other ------------------------------------------------------
    # ========================================================================

    @abstractmethod
    def width_v(self) -> float:
        pass
