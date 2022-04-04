from typing import Union, Dict, Callable, List

import numpy as np
import numpy.typing as npt

from hazma.parameters import Qd, Qe, Qu, qe
from hazma.theory import TheoryAnn

from hazma.vector_mediator.form_factors.utils import RealArray
from hazma.vector_mediator.form_factors.utils import ComplexArray
from hazma.vector_mediator.form_factors.phi_pi import FormFactorPhiPi0

from hazma.parameters import (
    charged_kaon_mass as _MK,
    charged_pion_mass as _MPI,
    eta_mass as _META,
    eta_prime_mass as _METAP,
    neutral_kaon_mass as _MK0,
    neutral_pion_mass as _MPI0,
    omega_mass as _MOMEGA,
    phi_mass as _MPHI,
    electron_mass as _ME,
    muon_mass as _MMU,
)

from hazma.utils import kallen_lambda


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
        from hazma.vector_mediator.form_factors.pipi import FormFactorPiPi
        from hazma.vector_mediator.form_factors.kk import FormFactorKK
        from hazma.vector_mediator.form_factors.pi_pi_pi0 import FormFactorPiPiPi0
        from hazma.vector_mediator.form_factors.pi_pi_eta import FormFactorPiPiEta
        from hazma.vector_mediator.form_factors.pi_pi_etap import FormFactorPiPiEtaP
        from hazma.vector_mediator.form_factors.pi_pi_omega import FormFactorPiPiOmega
        from hazma.vector_mediator.form_factors.pi_k_k import FormFactorPi0K0K0
        from hazma.vector_mediator.form_factors.pi_k_k import FormFactorPi0KpKm
        from hazma.vector_mediator.form_factors.pi_k_k import FormFactorPiKK0
        from hazma.vector_mediator.form_factors.four_pi import FormFactorPiPiPiPi

        # Compute and store the parameters needed to compute form factors.
        # self._ff_pipi_params = _compute_ff_params_pipi(2000)
        self._ff_pi_pi = FormFactorPiPi()
        # self._ff_kk_params = _compute_ff_params_kk(200)
        self._ff_k_k = FormFactorKK()
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

    # Import the form factors
    from hazma.vector_mediator.form_factors import (
        form_factor_eta_gamma as _form_factor_eta_gamma,
        # form_factor_kk as _form_factor_kk,
        form_factor_pi_gamma as _form_factor_pi_gamma,
        # form_factor_pipi as _form_factor_pipi,
        form_factor_omega_pi as _form_factor_pi_omega,
        form_factor_phi_pi as _form_factor_phi_pi,
        form_factor_eta_phi as _form_factor_eta_phi,
        form_factor_eta_omega as _form_factor_eta_omega,
    )

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
        if imode not in [0, 1]:
            raise ValueError(f"Invalid iso-spin {imode}")
        m = _MPI if imode == 1 else _MPI0

        if hasattr(q, "__len__"):
            if self.mv < 2 * m:
                return np.zeros_like(q)
            qq = 1e-3 * np.array(q, dtype=np.float64)
        else:
            if self.mv < 2 * m:
                return 0.0
            qq = 1e-3 * np.array([q], dtype=np.float64)

        ff = self._ff_pi_pi.form_factor(
            qq**2,
            self._gvuu,
            self._gvdd,
            imode=imode,
        )

        if len(ff) == 1:
            return ff[0]
        return ff

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
        if imode not in [0, 1]:
            raise ValueError(f"Invalid iso-spin {imode}")
        mk = _MK if imode == 1 else _MK0
        if hasattr(q, "__len__"):
            if self.mv < 2 * mk:
                return np.zeros_like(q)
            qq = 1e-3 * np.array(q)
        else:
            if self.mv < 2 * mk:
                return 0.0
            qq = 1e-3 * np.array([q])
        ff = self._ff_k_k.form_factor(
            qq**2,
            self._gvuu,
            self._gvdd,
            self._gvss,
            imode=imode,
        )
        if len(ff) == 1:
            return ff[0]
        return ff

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

    def _width_v_to_mm(
        self,
        mass: float,
        form_factor: Union[complex, ComplexArray],
        symmetry: float = 1.0,
    ) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        mesons.

        Parameters
        ----------
        mass: float
            Mass of the final state meson.
        form_factor: complex
            Vector form factor for the V-meson-meson vertex.
        symmetry: float
            Symmetry factor. If the final state mesons are identical, then this
            should be 1/2. Default is 1.0

        Returns
        -------
        gamma: float
            Partial width for the vector to decay into two mesons.
        """
        if self._mv < 2 * mass:
            return 0.0
        mu = mass / self.mv
        return (
            symmetry
            / 48.0
            / np.pi
            * self._mv
            * (1 - 4 * mu**2) ** 1.5
            * np.abs(form_factor) ** 2
        )

    @with_cache(cache_name="_width_cache", name="pi pi")
    def width_v_to_pi_pi(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        charged pions [𝜋⁺ 𝜋⁻].
        """
        mass = _MPI
        form_factor = self.form_factor_pi_pi(self.mv)
        return self._width_v_to_mm(mass, form_factor)

    @with_cache(cache_name="_width_cache", name="k0 k0")
    def width_v_to_k0_k0(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        neutral kaons [K⁰ K⁰].
        """
        form_factor = self.form_factor_k_k(self.mv, imode=0)
        return self._width_v_to_mm(_MK0, form_factor)

    @with_cache(cache_name="_width_cache", name="k k")
    def width_v_to_k_k(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        charged kaons [K⁺ K⁻].
        """
        form_factor = self.form_factor_k_k(self._mv, imode=1)
        return self._width_v_to_mm(_MK, form_factor)

    def _width_v_to_mg(self, mass: float, ff: complex) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a meson and photon.

        Parameters
        ----------
        mass: float
            Mass of the final state meson.
        ff: complex
            Vector form factor for the V-meson-meson vertex.

        Returns
        -------
        gamma: float
            Partial width for the vector to decay into a meson and photon.
        """
        if self._mv < mass:
            return 0.0
        # Note: form-factor has units of 1/GeV
        mu = mass / self.mv
        q = 0.5 * np.sqrt(kallen_lambda(1.0, mu**2, 0))
        return self.mv * q**3 * np.abs(ff) ** 2 / (12.0 * np.pi)

    @with_cache(cache_name="_width_cache", name="pi0 gamma")
    def width_v_to_pi0_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion [𝜋⁰] and photon [𝛾].
        """
        form_factor = self._form_factor_pi_gamma(self.mv)
        return self._width_v_to_mg(_MPI0, form_factor)  # type: ignore

    @with_cache(cache_name="_width_cache", name="eta gamma")
    def width_v_to_eta_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion [𝜋⁰] and photon [𝛾].
        """
        form_factor = self._form_factor_eta_gamma(self.mv)
        return self._width_v_to_mg(_META, form_factor)  # type: ignore

    def __width_v_to_v_s(self, ff: complex, mvector: float, mscalar: float) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a vector meson and a scalar meson.

        Parameters
        ----------
        ff: complex
            Form factor.
        mvector: float
            Mass of the vector meson.
        mscalar: float
            Mass of the scalar meson.
        """
        if self.mv < mvector + mscalar:
            return 0.0
        mv = self.mv
        q = 0.5 * np.sqrt(kallen_lambda(1.0, (mvector / mv) ** 2, (mscalar / mv) ** 2))
        return self.mv * q**3 * np.abs(ff) ** 2 / (12.0 * np.pi)

    @with_cache(cache_name="_width_cache", name="pi0 phi")
    def width_v_to_pi0_phi(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an phi [𝜙(1020)] and neutral pion [𝜋⁰].
        """
        if self.mv < _MPI0 + _MPHI:
            return 0.0

        ff = self._form_factor_phi_pi(self.mv)
        return self.__width_v_to_v_s(ff, _MPHI, _MPI0)  # type: ignore

    @with_cache(cache_name="_width_cache", name="eta phi")
    def width_v_to_eta_phi(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜂] and phi [𝜙(1020)].
        """
        ff = self._form_factor_eta_phi(self.mv)
        return self.__width_v_to_v_s(ff, _MPHI, _META)  # type: ignore

    @with_cache(cache_name="_width_cache", name="eta omega")
    def width_v_to_eta_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜂] and omega [𝜔(782)].
        """
        ff = self._form_factor_eta_omega(self.mv)
        return self.__width_v_to_v_s(ff, _MOMEGA, _META)  # type: ignore

    @with_cache(cache_name="_width_cache", name="pi0 pi0 gamma")
    def width_v_to_pi0_pi0_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two neutral pions and a photon [𝜋⁰ 𝜋⁰ 𝛾].
        """
        if self.mv < _MPI0 + _MOMEGA:
            return 0.0

        ff: complex = self._form_factor_pi_omega(self.mv)  # type: ignore
        # This channel comes from first going into pi0 + omega and then
        # having the omega decay into pi0 + gamma. Note, we ignore the
        # omega decaying into pi pi pi0 since this is taken into account
        # in the 4pion channel (which takes into account additional channels)
        pi0_omega = self.__width_v_to_v_s(ff, _MOMEGA, _MPI0)
        br_omega_to_pi0_gamma = 8.34e-2

        return br_omega_to_pi0_gamma * pi0_omega

    @with_cache(cache_name="_width_cache", name="pi pi pi0")
    def width_v_to_pi_pi_pi0(self, *, npts=10_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and a neutral pion.
        """
        if self.mv < 2 * _MPI + _MPI0:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi_pi_pi0.integrated_form_factor(
            mvgev**2, self.gvuu, self.gvdd, self.gvss, npts=npts
        )
        return integral[0] / (2 * mvgev) * 1e3

    @with_cache(cache_name="_width_cache", name="pi pi eta")
    def width_v_to_pi_pi_eta(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta.
        """
        if self.mv < 2 * _MPI + _META:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi_pi_eta.integrated_form_factor(
            mvgev, self.gvuu, self.gvdd
        )
        return integral / (2 * mvgev) * 1e3

    @with_cache(cache_name="_width_cache", name="pi pi etap")
    def width_v_to_pi_pi_etap(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        if self.mv < 2 * _MPI + _METAP:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi_pi_etap.integrated_form_factor(
            mvgev, self.gvuu, self.gvdd
        )
        return integral / (2 * mvgev) * 1e3

    @with_cache(cache_name="_width_cache", name="pi pi omega")
    def width_v_to_pi_pi_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        if self.mv < 2 * _MPI + _MOMEGA:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi_pi_omega.integrated_form_factor(
            mvgev, self.gvuu, self.gvdd, 1
        )
        return integral / (2 * mvgev) * 1e3

    @with_cache(cache_name="_width_cache", name="pi0 pi0 omega")
    def width_v_to_pi0_pi0_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        if self.mv < 2 * _MPI + _MOMEGA:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi_pi_omega.integrated_form_factor(
            mvgev, self.gvuu, self.gvdd, 0
        )
        return integral / (2 * mvgev) * 1e3

    @with_cache(cache_name="_width_cache", name="pi0 k0 k0")
    def width_v_to_pi0_k0_k0(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two neutral kaons.
        """
        if self.mv < _MPI0 + 2 * _MK0:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi0_k0_k0.integrated_form_factor(
            m=mvgev, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )
        return integral / (6.0 * mvgev) * 1e3

    @with_cache(cache_name="_width_cache", name="pi0 k k")
    def width_v_to_pi0_k_k(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two charged kaons.
        """
        if self.mv < _MPI0 + 2 * _MK:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi0_k_k.integrated_form_factor(
            m=mvgev, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )
        return integral / (6.0 * mvgev) * 1e3

    @with_cache(cache_name="_width_cache", name="pi k k0")
    def width_v_to_pi_k_k0(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two charged kaons.
        """
        if self.mv < _MPI + _MK + _MK0:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi_k_k0.integrated_form_factor(
            m=mvgev, gvuu=self.gvuu, gvdd=self.gvdd, gvss=self.gvss, npts=npts
        )
        return integral / (6.0 * mvgev) * 1e3

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

        mvgev = self.mv * 1e-3
        width, _ = self._ff_four_pi.decay_width(
            mv=mvgev, gvuu=self.gvuu, gvdd=self.gvdd, npts=npts, neutral=False
        )
        return 1e3 * width

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

        mvgev = self.mv * 1e-3
        width, _ = self._ff_four_pi.decay_width(
            mv=mvgev, gvuu=self.gvuu, gvdd=self.gvdd, npts=npts, neutral=True
        )
        return 1e3 * width

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
            "x x",
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

    def annihilation_cross_section_funcs(self) -> Dict[str, Callable[[float], float]]:
        raise NotImplementedError()

    def _sigma_xx_to_v_to_mm(
        self,
        e_cm: Union[float, RealArray],
        mass: float,
        form_factor: Union[complex, ComplexArray],
        symmetry: float = 1.0,
    ) -> Union[float, RealArray]:
        """
        Compute the dark matter annihilation cross section into mesons.

        Parameters
        ----------
        e_cm: Union[float, np.ndarray]
            Center-of-mass energy.
        mass: float
            Mass of the final state meson.
        form_factor: complex
            Vector form factor for the V-meson-meson vertex.
        symmetry: float
            Symmetry factor. If the final state mesons are identical, then this
            should be 1/2. Default is 1.0

        Returns
        -------
        sigma: Union[float, np.ndarray]
            Cross section for chi + chibar -> pi + pi.
        """
        gamv = self.width_v()
        mv = self.mv
        mx = self.mx
        if e_cm < 2.0 * mx or e_cm < 2.0 * mass:
            return 0.0

        s = e_cm**2
        num = (
            self.gvxx**2
            * np.abs(form_factor) ** 2
            * (s - 4 * mass**2) ** 1.5
            * (2.0 * mx**2 + s)
        )
        den = (
            48.0
            * np.pi
            * s
            * np.sqrt(s - 4.0 * mx**2)
            * (mv**2 * (gamv**2 - 2.0 * s) + mv**4 + s**2)
        )

        return symmetry * num / den

    def sigma_xx_to_v_to_pi_pi(
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
        mass = _MPI
        if hasattr(e_cm, "__len__"):
            ecm: RealArray = np.array(e_cm, dtype=np.float64)
        else:
            ecm: RealArray = np.array([e_cm], dtype=np.float64)

        # form_factor = self._form_factor_pipi(e_cm ** 2)
        ff = self._ff_pi_pi.form_factor(ecm**2, self._gvuu, self._gvdd, imode=0)
        return self._sigma_xx_to_v_to_mm(e_cm, mass, ff)

    def sigma_xx_to_v_to_k_k(
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
        mass = _MK
        form_factor = self.form_factor_k_k(e_cm**2)
        return self._sigma_xx_to_v_to_mm(e_cm, mass, form_factor)

    # ========================================================================
    # ---- Spectra -----------------------------------------------------------
    # ========================================================================

    def _spectrum_funcs(
        self,
    ) -> Dict[str, Callable[[Union[float, npt.NDArray[np.float64]], float], float]]:
        raise NotImplemented()

    def _gamma_ray_line_energies(self, e_cm) -> Dict[str, float]:
        raise NotImplemented()

    def _positron_spectrum_funcs(self) -> Dict[str, Callable]:
        raise NotImplemented()

    def _positron_line_energies(self, e_cm) -> Dict[str, float]:
        raise NotImplemented()


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
