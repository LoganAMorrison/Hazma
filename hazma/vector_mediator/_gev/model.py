import warnings
from typing import Callable, Dict, List, Union

import numpy as np
import numpy.typing as npt

from hazma.parameters import Qd, Qe, Qu
from hazma.parameters import charged_kaon_mass as _MK
from hazma.parameters import charged_pion_mass as _MPI
from hazma.parameters import electron_mass as _ME
from hazma.parameters import eta_mass as _META
from hazma.parameters import eta_prime_mass as _METAP
from hazma.parameters import muon_mass as _MMU
from hazma.parameters import neutral_kaon_mass as _MK0
from hazma.parameters import neutral_pion_mass as _MPI0
from hazma.parameters import omega_mass as _MOMEGA
from hazma.parameters import phi_mass as _MPHI
from hazma.parameters import qe
from hazma.theory import TheoryAnn
from hazma.utils import kallen_lambda
from hazma.vector_mediator.form_factors.utils import ComplexArray, RealArray

from . import spectra as gev_spectra


def with_cache(*, cache_name: str, name: str):
    def decorator(f):
        def wrapper(*args, **kwargs):
            cache = getattr(args[0], cache_name)
            assert name in cache

            if cache[name]["valid"] and cache[name]["kwargs"] == kwargs:
                return cache[name]["value"]

            val = f(*args, **kwargs)

            cache[name]["valid"] = True
            cache[name]["value"] = val
            cache[name]["kwargs"] = kwargs

            return val

        return wrapper

    return decorator


class VectorMediatorGeV(TheoryAnn):
    """
    A generic dark matter model where interactions with the SM are mediated via
    an s-channel vector mediator. This model is valid for dark-matter masses
    up to 1 GeV.
    """

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
        from hazma.vector_mediator.form_factors.eta_gamma import \
            FormFactorEtaGamma
        from hazma.vector_mediator.form_factors.eta_omega import \
            FormFactorEtaOmega
        from hazma.vector_mediator.form_factors.eta_phi import FormFactorEtaPhi
        from hazma.vector_mediator.form_factors.four_pi import \
            FormFactorPiPiPiPi
        from hazma.vector_mediator.form_factors.kk import FormFactorKK
        from hazma.vector_mediator.form_factors.omega_pi import \
            FormFactorOmegaPi0
        from hazma.vector_mediator.form_factors.phi_pi import FormFactorPhiPi0
        from hazma.vector_mediator.form_factors.pi_gamma import \
            FormFactorPiGamma
        from hazma.vector_mediator.form_factors.pi_k_k import (
            FormFactorPi0K0K0, FormFactorPi0KpKm, FormFactorPiKK0)
        from hazma.vector_mediator.form_factors.pi_pi_eta import \
            FormFactorPiPiEta
        from hazma.vector_mediator.form_factors.pi_pi_etap import \
            FormFactorPiPiEtaP
        from hazma.vector_mediator.form_factors.pi_pi_omega import \
            FormFactorPiPiOmega
        from hazma.vector_mediator.form_factors.pi_pi_pi0 import \
            FormFactorPiPiPi0
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

    @mx.setter
    def mx(self, val: float) -> None:
        self._mx = val
        self._width_cache["x x"]["value"] = self.width_v_to_x_x()

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

    @gvxx.setter
    def gvxx(self, val: float) -> None:
        self._gvxx = val
        self._update_width_cache("x x", self.width_v_to_x_x())

    @gvuu.setter
    def gvuu(self, val: float) -> None:
        self._gvuu = val
        self._reset_state()
        self._invalidate_width_cache()

    @gvdd.setter
    def gvdd(self, val: float) -> None:
        self._gvdd = val
        self._reset_state()
        self._invalidate_width_cache()

    @gvss.setter
    def gvss(self, val: float) -> None:
        self._gvss = val
        self._reset_state()
        self._invalidate_width_cache()

    @gvee.setter
    def gvee(self, val: float) -> None:
        self._gvee = val
        self._update_width_cache("e e", self.width_v_to_e_e())

    @gvmumu.setter
    def gvmumu(self, val: float) -> None:
        self._gvmumu = val
        self._update_width_cache("mu mu", self.width_v_to_mu_mu())

    @gvveve.setter
    def gvveve(self, val: float) -> None:
        self._gvveve = val
        self._update_width_cache("ve ve", self.width_v_to_ve_ve())

    @gvvmvm.setter
    def gvvmvm(self, val: float) -> None:
        self._gvvmvm = val
        self._update_width_cache("vm vm", self.width_v_to_vm_vm())

    @gvvtvt.setter
    def gvvtvt(self, val: float) -> None:
        self._gvvtvt = val
        self._update_width_cache("vt vt", self.width_v_to_vt_vt())

    def _reset_state(self) -> None:
        """
        Function to reset the state of the derived quantities such as the
        vector width and form-factors.
        """
        pass

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
    # ---- Partial Widths ----------------------------------------------------
    # ========================================================================

    def __width_v_to_f_f(self, mass, coupling) -> float:
        """Width for V -> f + fbar."""
        return (
            coupling**2
            * np.sqrt(np.clip(-4 * mass**2 + self.mv**2, 0.0, None))
            * (2 * mass**2 + self.mv**2)
        ) / (12.0 * self.mv**2 * np.pi)

    @with_cache(cache_name="_width_cache", name="e e")
    def width_v_to_e_e(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        electrons [e⁺ e⁻].
        """
        return self.__width_v_to_f_f(_ME, self.gvee)

    @with_cache(cache_name="_width_cache", name="mu mu")
    def width_v_to_mu_mu(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        muons [μ⁺ μ⁻].
        """
        return self.__width_v_to_f_f(_MMU, self.gvmumu)

    @with_cache(cache_name="_width_cache", name="ve ve")
    def width_v_to_ve_ve(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        electron-neutrinos [νe νe].
        """
        return self.__width_v_to_f_f(0.0, self.gvveve)

    @with_cache(cache_name="_width_cache", name="vm vm")
    def width_v_to_vm_vm(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        muon-neutrinos [νμ νμ].
        """
        return self.__width_v_to_f_f(0.0, self.gvvmvm)

    @with_cache(cache_name="_width_cache", name="vt vt")
    def width_v_to_vt_vt(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        tau-neutrinos [ντ ντ].
        """
        return self.__width_v_to_f_f(0.0, self.gvvtvt)

    @with_cache(cache_name="_width_cache", name="x x")
    def width_v_to_x_x(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two dark matter particles.
        """
        return self.__width_v_to_f_f(self.mx, self.gvxx)

    @with_cache(cache_name="_width_cache", name="pi pi")
    def width_v_to_pi_pi(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        charged pions [𝜋⁺ 𝜋⁻].
        """
        return self._ff_pi_pi.width(mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, imode=1)

    @with_cache(cache_name="_width_cache", name="k0 k0")
    def width_v_to_k0_k0(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        neutral kaons [K⁰ K⁰].
        """
        return self._ff_k_k.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, imode=0
        )

    @with_cache(cache_name="_width_cache", name="k k")
    def width_v_to_k_k(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        charged kaons [K⁺ K⁻].
        """
        return self._ff_k_k.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, imode=1
        )

    @with_cache(cache_name="_width_cache", name="pi0 gamma")
    def width_v_to_pi0_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion [𝜋⁰] and photon [𝛾].
        """
        return self._ff_pi_gamma.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss
        )

    @with_cache(cache_name="_width_cache", name="eta gamma")
    def width_v_to_eta_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion [𝜋⁰] and photon [𝛾].
        """
        return self._ff_eta_gamma.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss
        )

    @with_cache(cache_name="_width_cache", name="pi0 phi")
    def width_v_to_pi0_phi(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an phi [𝜙(1020)] and neutral pion [𝜋⁰].
        """
        return self._ff_pi_phi.width(mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd)

    @with_cache(cache_name="_width_cache", name="eta phi")
    def width_v_to_eta_phi(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜂] and phi [𝜙(1020)].
        """
        return self._ff_eta_phi.width(mv=self.mv, gvss=self.gvss)

    @with_cache(cache_name="_width_cache", name="eta omega")
    def width_v_to_eta_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜂] and omega [𝜔(782)].
        """
        return self._ff_eta_omega.width(mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd)

    @with_cache(cache_name="_width_cache", name="pi0 pi0 gamma")
    def width_v_to_pi0_pi0_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two neutral pions and a photon [𝜋⁰ 𝜋⁰ 𝛾].
        """
        pi0_omega = self._ff_pi_omega.width(mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd)
        br_omega_to_pi0_gamma = 8.34e-2

        return br_omega_to_pi0_gamma * pi0_omega

    @with_cache(cache_name="_width_cache", name="pi pi pi0")
    def width_v_to_pi_pi_pi0(self, *, npts=10_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and a neutral pion.
        """
        return self._ff_pi_pi_pi0.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )

    @with_cache(cache_name="_width_cache", name="pi pi eta")
    def width_v_to_pi_pi_eta(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta.
        """
        return self._ff_pi_pi_eta.width(mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd)

    @with_cache(cache_name="_width_cache", name="pi pi etap")
    def width_v_to_pi_pi_etap(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        return self._ff_pi_pi_etap.width(mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd)

    @with_cache(cache_name="_width_cache", name="pi pi omega")
    def width_v_to_pi_pi_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        return self._ff_pi_pi_omega.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, imode=1
        )

    @with_cache(cache_name="_width_cache", name="pi0 pi0 omega")
    def width_v_to_pi0_pi0_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        return self._ff_pi_pi_omega.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, imode=0
        )

    @with_cache(cache_name="_width_cache", name="pi0 k0 k0")
    def width_v_to_pi0_k0_k0(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two neutral kaons.
        """
        return self._ff_pi0_k0_k0.width(
            m=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )

    @with_cache(cache_name="_width_cache", name="pi0 k k")
    def width_v_to_pi0_k_k(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two charged kaons.
        """
        return self._ff_pi0_k_k.width(
            m=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )

    @with_cache(cache_name="_width_cache", name="pi k k0")
    def width_v_to_pi_k_k0(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two charged kaons.
        """
        return self._ff_pi_k_k0.width(
            m=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )

    @with_cache(cache_name="_width_cache", name="pi pi pi pi")
    def width_v_to_pi_pi_pi_pi(self, *, npts=1 << 14) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into four charged pions.

        Parameters
        ----------
        npts: int, optional
            Number of points to use for Monte-Carlo integration. Default
            is 1<<14 ~ 16_000.
        """
        if self.mv < 4 * _MPI:
            return 0.0
        width, _ = self._ff_four_pi.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, npts=npts, neutral=False
        )
        return width

    @with_cache(cache_name="_width_cache", name="pi pi pi0 pi0")
    def width_v_to_pi_pi_pi0_pi0(self, *, npts: int = 1 << 14) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pion and two neutral pions.

        Parameters
        ----------
        npts: int, optional
            Number of points to use for Monte-Carlo integration. Default
            is 1<<14 ~ 16_000.
        """
        if self.mv < 2 * _MPI + 2 * _MPI0:
            return 0.0
        width, _ = self._ff_four_pi.width(
            mv=self.mv, gvuu=self.gvuu, gvdd=self.gvdd, npts=npts, neutral=True
        )
        return width

    def partial_widths(self, *, npts=50_000) -> Dict[str, float]:
        """
        Compute the partial decay widths of the vector mediator.
        """
        return {
            "e e": self.width_v_to_e_e(),
            "mu mu": self.width_v_to_mu_mu(),
            "ve ve": self.width_v_to_ve_ve(),
            "vm vm": self.width_v_to_vm_vm(),
            "vt vt": self.width_v_to_vt_vt(),
            "x x": self.width_v_to_x_x(),
            "pi pi": self.width_v_to_pi_pi(),
            "k0 k0": self.width_v_to_k0_k0(),
            "k k": self.width_v_to_k_k(),
            "pi0 gamma": self.width_v_to_pi0_gamma(),
            "eta gamma": self.width_v_to_eta_gamma(),
            "pi0 phi": self.width_v_to_pi0_phi(),
            "eta phi": self.width_v_to_eta_phi(),
            "eta omega": self.width_v_to_eta_omega(),
            "pi0 pi0 gamma": self.width_v_to_pi0_pi0_gamma(),
            "pi pi pi0": self.width_v_to_pi_pi_pi0(npts=npts),
            "pi pi eta": self.width_v_to_pi_pi_eta(),
            "pi pi etap": self.width_v_to_pi_pi_etap(),
            "pi pi omega": self.width_v_to_pi_pi_omega(),
            "pi0 pi0 omega": self.width_v_to_pi0_pi0_omega(),
            "pi0 k0 k0": self.width_v_to_pi0_k0_k0(npts=npts),
            "pi0 k k": self.width_v_to_pi0_k_k(npts=npts),
            "pi k k0": self.width_v_to_pi_k_k0(npts=npts),
            "pi pi pi pi": self.width_v_to_pi_pi_pi_pi(npts=npts),
            "pi pi pi0 pi0": self.width_v_to_pi_pi_pi0_pi0(npts=npts),
        }

    def width_v(self) -> float:
        """Compute the total decay width of the vector mediator."""
        return sum(self.partial_widths().values())

    # ========================================================================
    # ---- Cross Sections ----------------------------------------------------
    # ========================================================================

    def list_annihilation_final_states(self) -> List[str]:
        r"""
        Lists annihilation final states.

        Returns
        -------
        fss : list(str)
            Possible annihilation final states.
        """
        return [
            "e e",
            "mu mu",
            "ve ve",
            "vt vt",
            "vm vm",
            "pi pi",
            "k0 k0",
            "k k",
            "pi0 gamma",
            "eta gamma",
            "pi0 phi",
            "eta phi",
            "eta omega",
            "pi0 pi0 gamma",
            "pi pi pi0",
            "pi pi eta",
            "pi pi etap",
            "pi pi omega",
            "pi0 pi0 omega",
            "pi0 k0 k0",
            "pi0 k k",
            "pi k k0",
            "pi pi pi pi",
            "pi pi pi0 pi0",
        ]

    def annihilation_cross_section_funcs(
        self,
    ) -> Dict[str, Callable[[float | RealArray], float | RealArray]]:
        return {
            "e e": self.sigma_xx_to_e_e,
            "mu mu": self.sigma_xx_to_mu_mu,
            "ve ve": self.sigma_xx_to_ve_ve,
            "vt vt": self.sigma_xx_to_vt_vt,
            "vm vm": self.sigma_xx_to_vm_vm,
            "pi pi": self.sigma_xx_to_pi_pi,
            "k0 k0": self.sigma_xx_to_k0_k0,
            "k k": self.sigma_xx_to_k_k,
            "pi0 gamma": self.sigma_xx_to_pi0_gamma,
            "eta gamma": self.sigma_xx_to_eta_gamma,
            "pi0 phi": self.sigma_xx_to_pi0_phi,
            "eta phi": self.sigma_xx_to_eta_phi,
            "eta omega": self.sigma_xx_to_eta_omega,
            "pi0 pi0 gamma": self.sigma_xx_to_pi0_pi0_gamma,
            "pi pi pi0": self.sigma_xx_to_pi_pi_pi0,
            "pi pi eta": self.sigma_xx_to_pi_pi_eta,
            "pi pi etap": self.sigma_xx_to_pi_pi_etap,
            "pi pi omega": self.sigma_xx_to_pi_pi_omega,
            "pi0 pi0 omega": self.sigma_xx_to_pi0_pi0_omega,
            "pi0 k0 k0": self.sigma_xx_to_pi0_k0_k0,
            "pi0 k k": self.sigma_xx_to_pi0_k_k,
            "pi k k0": self.sigma_xx_to_pi_k_k,
            "pi pi pi pi": self.sigma_xx_to_pi_pi_pi_pi,
            "pi pi pi0 pi0": self.sigma_xx_to_pi_pi_pi0_pi0,
        }

    def sigma_xx_to_e_e(self, e_cm):
        s = e_cm**2
        gvff = self.gvee
        gvxx = self.gvxx
        mf = _ME
        return (
            gvff**2
            * gvxx**2
            * np.sqrt(s * (-4 * mf**2 + s))
            * (2 * mf**2 + s)
            * (2 * self.mx**2 + s)
        ) / (
            12.0
            * np.pi
            * s
            * np.sqrt(s * (-4 * self.mx**2 + s))
            * (self.mv**4 + s**2 + self.mv**2 * (-2 * s + self.width_v() ** 2))
        )

    def sigma_xx_to_mu_mu(self, e_cm):
        s = e_cm**2
        gvff = self.gvmumu
        gvxx = self.gvxx
        mf = _MMU
        return (
            gvff**2
            * gvxx**2
            * np.sqrt(s * (-4 * mf**2 + s))
            * (2 * mf**2 + s)
            * (2 * self.mx**2 + s)
        ) / (
            12.0
            * np.pi
            * s
            * np.sqrt(s * (-4 * self.mx**2 + s))
            * (self.mv**4 + s**2 + self.mv**2 * (-2 * s + self.width_v() ** 2))
        )

    def sigma_xx_to_ve_ve(self, e_cm):
        s = e_cm**2
        gvff = self.gvveve
        gvxx = self.gvxx
        return (gvff**2 * gvxx**2 * s * (2 * self.mx**2 + s)) / (
            24.0
            * np.pi
            * np.sqrt(s * (-4 * self.mx**2 + s))
            * (self.mv**4 + s**2 + self.mv**2 * (-2 * s + self.width_v() ** 2))
        )

    def sigma_xx_to_vm_vm(self, e_cm):
        s = e_cm**2
        gvff = self.gvvmvm
        gvxx = self.gvxx
        return (gvff**2 * gvxx**2 * s * (2 * self.mx**2 + s)) / (
            24.0
            * np.pi
            * np.sqrt(s * (-4 * self.mx**2 + s))
            * (self.mv**4 + s**2 + self.mv**2 * (-2 * s + self.width_v() ** 2))
        )

    def sigma_xx_to_vt_vt(self, e_cm):
        s = e_cm**2
        gvff = self.gvvtvt
        gvxx = self.gvxx
        return (gvff**2 * gvxx**2 * s * (2 * self.mx**2 + s)) / (
            24.0
            * np.pi
            * np.sqrt(s * (-4 * self.mx**2 + s))
            * (self.mv**4 + s**2 + self.mv**2 * (-2 * s + self.width_v() ** 2))
        )

    def sigma_xx_to_pi_pi(
        self, e_cm: Union[float, RealArray]
    ) -> Union[float, RealArray]:
        """
        Compute the dark matter annihilation cross section into two charged
        pions.

        Parameters
        ----------
        e_cm: Union[float, np.ndarray]
            Center-of-mass energy.

        Returns
        -------
        sigma: Union[float, np.ndarray]
            Cross section for chi + chibar -> pi + pi.
        """
        return self.gvxx**2 * self._ff_pi_pi.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gamv=self.width_v(),
            imode=1,
        )

    def sigma_xx_to_k_k(self, e_cm: Union[float, RealArray]) -> Union[float, RealArray]:
        """
        Compute the dark matter annihilation cross section into two charged
        pions.

        Parameters
        ----------
        e_cm: Union[float, np.ndarray]
            Center-of-mass energy.

        Returns
        -------
        sigma: Union[float, np.ndarray]
            Cross section for chi + chibar -> pi + pi.
        """
        return self.gvxx**2 * self._ff_k_k.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gvss=self.gvss,
            gamv=self.width_v(),
            imode=1,
        )

    def sigma_xx_to_k0_k0(
        self, e_cm: Union[float, RealArray]
    ) -> Union[float, RealArray]:
        return self.gvxx**2 * self._ff_k_k.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gvss=self.gvss,
            gamv=self.width_v(),
            imode=0,
        )

    def sigma_xx_to_pi0_gamma(
        self, e_cm: Union[float, RealArray]
    ) -> Union[float, RealArray]:
        return self.gvxx**2 * self._ff_pi_gamma.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gvss=self.gvss,
            gamv=self.width_v(),
        )

    def sigma_xx_to_pi0_phi(
        self, e_cm: Union[float, RealArray]
    ) -> Union[float, RealArray]:
        return self.gvxx**2 * self._ff_pi_phi.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gamv=self.width_v(),
        )

    def sigma_xx_to_pi0_omega(
        self, e_cm: Union[float, RealArray]
    ) -> Union[float, RealArray]:
        return self.gvxx**2 * self._ff_pi_omega.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gamv=self.width_v(),
        )

    def sigma_xx_to_eta_gamma(
        self, e_cm: Union[float, RealArray]
    ) -> Union[float, RealArray]:
        return self.gvxx**2 * self._ff_eta_gamma.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gvss=self.gvss,
            gamv=self.width_v(),
        )

    def sigma_xx_to_eta_phi(
        self, e_cm: Union[float, RealArray]
    ) -> Union[float, RealArray]:
        return self.gvxx**2 * self._ff_eta_phi.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvss=self.gvss,
            gamv=self.width_v(),
        )

    def sigma_xx_to_eta_omega(
        self, e_cm: Union[float, RealArray]
    ) -> Union[float, RealArray]:
        return self.gvxx**2 * self._ff_eta_omega.cross_section(
            cme=e_cm,
            mx=self.mx,
            mv=self.mv,
            gvuu=self.gvuu,
            gvdd=self.gvdd,
            gamv=self.width_v(),
        )

    def sigma_xx_to_pi_pi_pi0_pi0(
        self, e_cm: Union[float, RealArray], npts=1 << 14
    ) -> Union[float, RealArray]:
        def cross_section(q: float):
            return (
                self.gvxx**2
                * self._ff_four_pi.cross_section(
                    cme=q,
                    mx=self.mx,
                    mv=self.mv,
                    gvuu=self.gvuu,
                    gvdd=self.gvdd,
                    widthv=self.width_v(),
                    neutral=True,
                    npts=npts,
                )[0]
            )

        if isinstance(e_cm, float):
            return cross_section(e_cm)
        else:
            return np.array([cross_section(q) for q in e_cm])

    def __width_to_cs(self, cme):
        """
        (
            gvxx**2 * (2 * mx**2 + s)
        )/(
            np.sqrt(1 - (4 * mx**2)/s) * cme * ((M**2 - s)**2 + mv**2 * widthv**2)
        )
        """

        gvxx = self.gvxx
        mux2 = (self.mx / cme) ** 2
        mug2 = (self.width_v() / cme) ** 2
        muv2 = (self.mv / cme) ** 2

        num = gvxx**2 * (1 + 2 * mux2)
        den = (
            (1 + (mug2 - 2) * muv2 + muv2**2) * np.sqrt(1.0 - 4.0 * mux2) * cme**1.5
        )
        return num / den

    def sigma_xx_to_pi0_pi0_gamma(self, e_cm):
        r = self.__width_to_cs(e_cm)
        pi0_omega = self._ff_pi_omega.width(mv=e_cm, gvuu=self.gvuu, gvdd=self.gvdd)
        br_omega_to_pi0_gamma = 8.34e-2

        return r * br_omega_to_pi0_gamma * pi0_omega

    def sigma_xx_to_pi_pi_pi0(self, e_cm, *, npts=10_000):
        r = self.__width_to_cs(e_cm)
        return r * self._ff_pi_pi_pi0.width(
            mv=e_cm, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )

    def sigma_xx_to_pi_pi_eta(self, e_cm):
        r = self.__width_to_cs(e_cm)
        return r * self._ff_pi_pi_eta.width(mv=e_cm, gvuu=self.gvuu, gvdd=self.gvdd)

    def sigma_xx_to_pi_pi_etap(self, e_cm):
        r = self.__width_to_cs(e_cm)
        return r * self._ff_pi_pi_etap.width(mv=e_cm, gvuu=self.gvuu, gvdd=self.gvdd)

    def sigma_xx_to_pi_pi_omega(self, e_cm):
        r = self.__width_to_cs(e_cm)
        return r * self._ff_pi_pi_omega.width(
            mv=e_cm, gvuu=self.gvuu, gvdd=self.gvdd, imode=1
        )

    def sigma_xx_to_pi0_pi0_omega(self, e_cm):
        r = self.__width_to_cs(e_cm)
        return r * self._ff_pi_pi_omega.width(
            mv=e_cm, gvuu=self.gvuu, gvdd=self.gvdd, imode=0
        )

    def sigma_xx_to_pi0_k0_k0(self, e_cm, *, npts=1 << 14):
        r = self.__width_to_cs(e_cm)
        return r * self._ff_pi0_k0_k0.width(
            m=e_cm, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )

    def sigma_xx_to_pi0_k_k(self, e_cm, *, npts=1 << 14):
        r = self.__width_to_cs(e_cm)
        return r * self._ff_pi0_k_k.width(
            m=e_cm, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )

    def sigma_xx_to_pi_k_k(self, e_cm, *, npts=1 << 14):
        r = self.__width_to_cs(e_cm)
        return r * self._ff_pi_k_k0.width(
            m=e_cm, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )

    def sigma_xx_to_pi_pi_pi_pi(self, e_cm, *, npts=1 << 14):
        r = self.__width_to_cs(e_cm)
        if self.mv < 4 * _MPI:
            return 0.0
        width, _ = self._ff_four_pi.width(
            mv=e_cm, gvuu=self.gvuu, gvdd=self.gvdd, npts=npts, neutral=False
        )
        return r * width

    # ========================================================================
    # ---- Spectra -----------------------------------------------------------
    # ========================================================================

    def _spectrum_funcs(
        self,
    ) -> Dict[str, Callable[[Union[float, npt.NDArray[np.float64]], float], float]]:
        def dnde_e_e(e, cme: float):
            return gev_spectra.dnde_photon_e_e(self, e, cme)

        def dnde_mu_mu(e, cme: float):
            return gev_spectra.dnde_photon_mu_mu(self, e, cme)

        def dnde_pi_pi(e, cme: float):
            return gev_spectra.dnde_photon_pi_pi(self, e, cme)

        def dnde_k0_k0(e, cme: float):
            return gev_spectra.dnde_photon_k0_k0(self, e, cme)

        def dnde_k_k(e, cme: float):
            return gev_spectra.dnde_photon_k_k(self, e, cme)

        def dnde_pi0_gamma(e, cme: float):
            return gev_spectra.dnde_photon_pi0_gamma(self, e, cme)

        def dnde_eta_gamma(e, cme: float):
            return gev_spectra.dnde_photon_eta_gamma(self, e, cme)

        def dnde_pi0_phi(e, cme: float):
            return gev_spectra.dnde_photon_pi0_phi(self, e, cme)

        def dnde_eta_phi(e, cme: float):
            return gev_spectra.dnde_photon_eta_phi(self, e, cme)

        def dnde_eta_omega(e, cme: float):
            return gev_spectra.dnde_photon_eta_omega(self, e, cme)

        def dnde_pi0_pi0_gamma(e, cme: float):
            return gev_spectra.dnde_photon_pi0_pi0_gamma(self, e, cme)

        def dnde_pi_pi_pi0(e, cme: float, *, npts: int = 1 << 14, nbins: int = 25):
            return gev_spectra.dnde_photon_pi_pi_pi0(
                self, e, cme, npts=npts, nbins=nbins
            )

        def dnde_pi_pi_eta(e, cme: float, *, nbins: int = 25):
            return gev_spectra.dnde_photon_pi_pi_eta(self, e, cme, nbins=nbins)

        def dnde_pi_pi_etap(e, cme: float, *, nbins: int = 25):
            return gev_spectra.dnde_photon_pi_pi_etap(self, e, cme, nbins=nbins)

        def dnde_pi_pi_omega(e, cme: float, *, nbins=25):
            return gev_spectra.dnde_photon_pi_pi_omega(self, e, cme, nbins=nbins)

        def dnde_pi0_pi0_omega(e, cme: float, *, nbins=25):
            return gev_spectra.dnde_photon_pi0_pi0_omega(self, e, cme, nbins=nbins)

        def dnde_pi0_k0_k0(e, cme: float, *, npts: int = 1 << 14, nbins=25):
            return gev_spectra.dnde_photon_pi0_k0_k0(
                self, e, cme, npts=npts, nbins=nbins
            )

        def dnde_pi0_k_k(e, cme: float, *, npts: int = 1 << 14, nbins=25):
            return gev_spectra.dnde_photon_pi0_k_k(self, e, cme, npts=npts, nbins=nbins)

        def dnde_pi_k_k0(e, cme: float, *, npts: int = 1 << 14, nbins=25):
            return gev_spectra.dnde_photon_pi_k_k0(self, e, cme, npts=npts, nbins=nbins)

        def dnde_pi_pi_pi_pi(e, cme: float, *, npts=1 << 14, nbins=25):
            return gev_spectra.dnde_photon_pi_pi_pi_pi(
                self, e, cme, npts=npts, nbins=nbins
            )

        def dnde_pi_pi_pi0_pi0(e, cme: float, *, npts: int = 1 << 15, nbins: int = 30):
            return gev_spectra.dnde_photon_pi_pi_pi0_pi0(
                self, e, cme, npts=npts, nbins=nbins
            )

        def dnde_zero(e, _: float):
            return np.zeros_like(e)

        return {
            "e e": dnde_e_e,
            "mu mu": dnde_mu_mu,
            "ve ve": dnde_zero,
            "vt vt": dnde_zero,
            "vm vm": dnde_zero,
            "pi pi": dnde_pi_pi,
            "k0 k0": dnde_k0_k0,
            "k k": dnde_k_k,
            "pi0 gamma": dnde_pi0_gamma,
            "eta gamma": dnde_eta_gamma,
            "pi0 phi": dnde_pi0_phi,
            "eta phi": dnde_eta_phi,
            "eta omega": dnde_eta_omega,
            "pi0 pi0 gamma": dnde_pi0_pi0_gamma,
            "pi pi pi0": dnde_pi_pi_pi0,
            "pi pi eta": dnde_pi_pi_eta,
            "pi pi etap": dnde_pi_pi_etap,
            "pi pi omega": dnde_pi_pi_omega,
            "pi0 pi0 omega": dnde_pi0_pi0_omega,
            "pi0 k0 k0": dnde_pi0_k0_k0,
            "pi0 k k": dnde_pi0_k_k,
            "pi k k0": dnde_pi_k_k0,
            "pi pi pi pi": dnde_pi_pi_pi_pi,
            "pi pi pi0 pi0": dnde_pi_pi_pi0_pi0,
        }

    def _gamma_ray_line_energies(self, e_cm) -> Dict[str, float]:
        def photon_energy(mass):
            return 0.5 * (e_cm - mass * (mass / e_cm))

        return {
            "pi0 gamma": photon_energy(_MPI0),
            "eta gamma": photon_energy(_META),
        }

    def _positron_spectrum_funcs(self) -> Dict[str, Callable]:
        raise NotImplementedError()

    def _positron_line_energies(self, e_cm) -> Dict[str, float]:
        return {
            "e e": e_cm / 2.0,
        }


class KineticMixingGeV(VectorMediatorGeV):
    r"""
    Create a ``VectorMediatorGeV`` object with kinetic mixing couplings.

    The couplings are defined as::

        gvuu = Qu qe eps
        gvdd = Qd qe eps
        gvss = Qd qe eps
        gvee = Qe qe eps
        gvmumu = Qe qe eps

    where Qu, Qd and Qe are the up-type quark, down-type quark and
    lepton electic charges in units of the electric charge, qe is the
    electric charge and eps is the kinetic mixing parameter.

    Parameters
    ----------
    mx : float
        Mass of the dark matter.
    mv : float
        Mass of the vector mediator.
    gvxx : float
        Coupling of vector mediator to dark matter.
    eps : float
        Kinetic mixing parameter.
    """

    def __init__(self, mx: float, mv: float, gvxx: float, eps: float) -> None:
        self._eps = eps

        super().__init__(
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            gvuu=-Qu * eps * qe,
            gvdd=-Qd * eps * qe,
            gvss=-Qd * eps * qe,
            gvee=-Qe * eps * qe,
            gvmumu=-Qe * eps * qe,
            gvveve=0.0,
            gvvmvm=0.0,
            gvvtvt=0.0,
        )

    def __repr__(self) -> str:
        return f"""KineticMixingGeV(
            mx={self.mx} [MeV],
            mv={self.mv} [MeV],
            gvxx={self.gvxx} [MeV],
            eps={self.eps} [MeV],
        )
        """

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, eps: float):
        self._eps = eps
        self.gvuu = -Qu * eps * qe
        self.gvdd = -Qd * eps * qe
        self.gvss = -Qd * eps * qe
        self.gvee = -Qe * eps * qe
        self.gvmumu = -Qe * eps * qe

    def __cannot_set_error(self, param: str) -> AttributeError:
        return AttributeError(
            f"""
        Cannot set {param}. Instead set 'eps' or use the 'VectorMediatorGeV'
        for more a general coupling structure."""
        )

    # Hide underlying properties' setters
    @VectorMediatorGeV.gvuu.setter
    def gvuu(self, _: float):
        raise self.__cannot_set_error("gvuu")

    @VectorMediatorGeV.gvdd.setter
    def gvdd(self, _: float):
        raise self.__cannot_set_error("gvdd")

    @VectorMediatorGeV.gvss.setter
    def gvss(self, _: float):
        raise self.__cannot_set_error("gvss")

    @VectorMediatorGeV.gvee.setter
    def gvee(self, _: float):
        raise self.__cannot_set_error("gvee")

    @VectorMediatorGeV.gvmumu.setter
    def gvmumu(self, _: float):
        raise self.__cannot_set_error("gvmumu")

    @VectorMediatorGeV.gvveve.setter
    def gvveve(self, _: float):
        raise self.__cannot_set_error("gvveve")

    @VectorMediatorGeV.gvvmvm.setter
    def gvvmvm(self, _: float):
        raise self.__cannot_set_error("gvvmvm")

    @VectorMediatorGeV.gvvtvt.setter
    def gvvtvt(self, _: float):
        raise self.__cannot_set_error("gvvtvt")


class BLGeV(VectorMediatorGeV):
    r"""
    Create a ``VectorMediatorGeV`` object with B-L couplings.

    The couplings are defined as::

        gvuu, gvdd, gvss = 1/3
        gvee, gvmumu, gvveve, gvvmvm, gvtvt = -1

    Parameters
    ----------
    mx : float
        Mass of the dark matter.
    mv : float
        Mass of the vector mediator.
    gvxx : float
        Coupling of vector mediator to dark matter.
    """

    def __init__(self, mx: float, mv: float, gvxx: float) -> None:
        gq = 1.0 / 3.0
        gl = -1.0

        super().__init__(
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            gvuu=gq,
            gvdd=gq,
            gvss=gq,
            gvee=gl,
            gvmumu=gl,
            gvveve=gl,
            gvvmvm=gl,
            gvvtvt=gl,
        )

    def __repr__(self) -> str:
        return f"""KineticMixingGeV(
            mx={self.mx} [MeV],
            mv={self.mv} [MeV],
            gvxx={self.gvxx} [MeV],
        )
        """

    def __cannot_set_error(self, param: str) -> AttributeError:
        return AttributeError(
            f"""
        Cannot set {param}. Instead use the 'VectorMediatorGeV'
        for more a general coupling structure."""
        )

    # Hide underlying properties' setters
    @VectorMediatorGeV.gvuu.setter
    def gvuu(self, _: float):
        raise self.__cannot_set_error("gvuu")

    @VectorMediatorGeV.gvdd.setter
    def gvdd(self, _: float):
        raise self.__cannot_set_error("gvdd")

    @VectorMediatorGeV.gvss.setter
    def gvss(self, _: float):
        raise self.__cannot_set_error("gvss")

    @VectorMediatorGeV.gvee.setter
    def gvee(self, _: float):
        raise self.__cannot_set_error("gvee")

    @VectorMediatorGeV.gvmumu.setter
    def gvmumu(self, _: float):
        raise self.__cannot_set_error("gvmumu")

    @VectorMediatorGeV.gvveve.setter
    def gvveve(self, _: float):
        raise self.__cannot_set_error("gvveve")

    @VectorMediatorGeV.gvvmvm.setter
    def gvvmvm(self, _: float):
        raise self.__cannot_set_error("gvvmvm")

    @VectorMediatorGeV.gvvtvt.setter
    def gvvtvt(self, _: float):
        raise self.__cannot_set_error("gvvtvt")
