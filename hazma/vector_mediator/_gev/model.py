"""
Py
"""

# pylint: disable=invalid-name,import-outside-toplevel
# pyright: basic, reportUnusedImport=false

import functools as ft
from typing import Callable, Dict, List, TypeVar, Union

import numpy as np
import numpy.typing as npt

import hazma.form_factors.vector as vff
from hazma.form_factors.vector import VectorFormFactorCouplings
from hazma.form_factors.vector._base import VectorFormFactor
from hazma.parameters import Qd, Qe, Qu
from hazma.parameters import charged_pion_mass as _MPI
from hazma.parameters import electron_mass as _ME
from hazma.parameters import eta_mass as _META
from hazma.parameters import muon_mass as _MMU
from hazma.parameters import neutral_pion_mass as _MPI0
from hazma.parameters import qe
from hazma.theory import TheoryAnn
from hazma.utils import RealOrRealArray

from . import neutrino as gev_neutrino_spectra
from . import positron as gev_positron_spectra
from . import spectra as gev_spectra
from . import types as gev_types

T = TypeVar("T", float, npt.NDArray[np.float64])


def with_cache(*, cache_name: str, name: str):
    """Simple decorator to cache values into a class variable.

    Parameters
    ----------
    cache_name: str
        Name of the class cache variable.
    name: str
        Name of the variable to cache.
    """

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

    from .cross_sections import (annihilation_cross_section_funcs,
                                 sigma_xx_to_e_e, sigma_xx_to_eta_gamma,
                                 sigma_xx_to_eta_omega, sigma_xx_to_eta_phi,
                                 sigma_xx_to_k0_k0, sigma_xx_to_k_k,
                                 sigma_xx_to_mu_mu, sigma_xx_to_pi0_gamma,
                                 sigma_xx_to_pi0_k0_k0, sigma_xx_to_pi0_k_k,
                                 sigma_xx_to_pi0_phi,
                                 sigma_xx_to_pi0_pi0_gamma,
                                 sigma_xx_to_pi0_pi0_omega,
                                 sigma_xx_to_pi_k_k0, sigma_xx_to_pi_pi,
                                 sigma_xx_to_pi_pi_eta, sigma_xx_to_pi_pi_etap,
                                 sigma_xx_to_pi_pi_omega,
                                 sigma_xx_to_pi_pi_pi0,
                                 sigma_xx_to_pi_pi_pi0_pi0,
                                 sigma_xx_to_pi_pi_pi_pi, sigma_xx_to_v_v,
                                 sigma_xx_to_ve_ve, sigma_xx_to_vm_vm,
                                 sigma_xx_to_vt_vt)
    from .thermal_cross_section import relic_density

    def __init__(  # pylint: disable=too-many-arguments
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

        # Compute and store the parameters needed to compute form factors.
        # pylint: disable=E1120
        self._ff_pi_pi = vff.VectorFormFactorPiPi()
        self._ff_pi0_pi0 = vff.VectorFormFactorPi0Pi0()
        self._ff_k_k = vff.VectorFormFactorKK()
        self._ff_k0_k0 = vff.VectorFormFactorK0K0()
        self._ff_eta_gamma = vff.VectorFormFactorEtaGamma()
        self._ff_eta_omega = vff.VectorFormFactorEtaOmega()
        self._ff_eta_phi = vff.VectorFormFactorEtaPhi()
        self._ff_pi0_gamma = vff.VectorFormFactorPi0Gamma()
        self._ff_pi0_omega = vff.VectorFormFactorPi0Omega()
        self._ff_pi0_phi = vff.VectorFormFactorPi0Phi()

        self._ff_pi_pi_pi0 = vff.VectorFormFactorPiPiPi0()
        self._ff_pi_pi_eta = vff.VectorFormFactorPiPiEta()
        self._ff_pi_pi_etap = vff.VectorFormFactorPiPiEtaPrime()
        self._ff_pi_pi_omega = vff.VectorFormFactorPiPiOmega()
        self._ff_pi0_pi0_omega = vff.VectorFormFactorPi0Pi0Omega()
        self._ff_pi0_k0_k0 = vff.VectorFormFactorPi0K0K0()
        self._ff_pi0_k_k = vff.VectorFormFactorPi0KpKm()
        self._ff_pi_k_k0 = vff.VectorFormFactorPiKK0()
        self._ff_pi_pi_pi_pi = vff.VectorFormFactorPiPiPiPi()
        self._ff_pi_pi_pi0_pi0 = vff.VectorFormFactorPiPiPi0Pi0()

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
        self._couplings = VectorFormFactorCouplings(self._gvuu, self._gvdd, self._gvss)

        final_states = self.list_annihilation_final_states()
        self._width_cache = {
            name: {"value": 0.0, "valid": False, "kwargs": {}} for name in final_states
        }
        self._width_cache["x x"] = {"value": 0.0, "valid": False, "kwargs": {}}

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

    def _call_width_cache(self, name, fn, **kwargs):
        if not self._width_cache[name]["valid"]:
            self._width_cache[name]["value"] = fn(self, **kwargs)

        return self._width_cache[name]["value"]

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
        return self._gvxx

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
        self._couplings = VectorFormFactorCouplings(self._gvuu, self._gvdd, self._gvss)
        self._invalidate_width_cache()

    @gvdd.setter
    def gvdd(self, val: float) -> None:
        self._gvdd = val
        self._couplings = VectorFormFactorCouplings(self._gvuu, self._gvdd, self._gvss)
        self._invalidate_width_cache()

    @gvss.setter
    def gvss(self, val: float) -> None:
        self._gvss = val
        self._couplings = VectorFormFactorCouplings(self._gvuu, self._gvdd, self._gvss)
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

    # ========================================================================
    # ---- Form Factors ------------------------------------------------------
    # ========================================================================

    def form_factor_pi_pi(self, q):
        r"""Compute the π⁺π⁻ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁺π⁻.
        """
        return self._ff_pi_pi.form_factor(q=q, couplings=self._couplings)

    def form_factor_pi0_pi0(self, q):
        r"""Compute the π⁰π⁰ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁰π⁰.
        """
        return self._ff_pi0_pi0.form_factor(q=q, couplings=self._couplings)

    def form_factor_k_k(self, q):
        r"""Compute the K⁺K⁻ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for K⁺K⁻.
        """
        return self._ff_k_k.form_factor(q=q, couplings=self._couplings)

    def form_factor_k0_k0(self, q):
        """
        Compute the K⁰K̅⁰ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for K⁰K̅⁰.
        """
        return self._ff_k0_k0.form_factor(q=q, couplings=self._couplings)

    def form_factor_pi0_gamma(self, q):
        r"""Compute the π⁰Ɣ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁰Ɣ.
        """
        return self._ff_pi0_gamma.form_factor(q=q, couplings=self._couplings)

    def form_factor_pi0_omega(self, q):
        r"""Compute the π⁰ω form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁰-ω.
        """
        return self._ff_pi0_omega.form_factor(q=q, couplings=self._couplings)

    def form_factor_pi0_phi(self, q):
        r"""Compute the π⁰ɸ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁰-ɸ.
        """
        return self._ff_pi0_phi.form_factor(q=q, couplings=self._couplings)

    def form_factor_eta_gamma(self, q):
        r"""Compute the ηƔ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for η-Ɣ.
        """
        return self._ff_eta_gamma.form_factor(q=q, couplings=self._couplings)

    def form_factor_eta_phi(self, q):
        r"""Compute the ηɸ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for ηɸ.
        """
        return self._ff_eta_phi.form_factor(q=q, couplings=self._couplings)

    def form_factor_eta_omega(self, q):
        r"""Compute the ηω form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for ηω.
        """
        return self._ff_eta_omega.form_factor(q=q, couplings=self._couplings)

    def form_factor_pi_pi_pi0(self, q, s, t):
        r"""Compute the π⁺π⁻π⁰ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        s: float or array-like
            Mandelstam variable s=(pπ⁺ + pπ⁻)^2.
        t: float or array-like
            Mandelstam variable t=(pπ⁺ + pπ⁰)^2.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁺π⁻π⁰.
        """
        return self._ff_pi_pi_pi0.form_factor(q, s, t, couplings=self._couplings)

    def form_factor_pi_pi_eta(self, q, s):
        r"""Compute the ηπ⁺π⁻ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        s: float or array-like
            Mandelstam variable s=(pη + pπ⁺)^2.
        t: float or array-like
            Mandelstam variable t=(pη + pπ⁻)^2.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for ηπ⁺π⁻.
        """
        return self._ff_pi_pi_eta.form_factor(q, s, couplings=self._couplings)

    def form_factor_pi_pi_etap(self, q, s):
        r"""Compute the η'π⁺π⁻ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        s: float or array-like
            Mandelstam variable s=(pη' + pπ⁺)^2.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for η'π⁺π⁻.
        """
        return self._ff_pi_pi_etap.form_factor(q, s, couplings=self._couplings)

    def form_factor_pi_pi_omega(self, q):
        r"""Compute the ωπ⁺π⁻ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for ωπ⁺π⁻.
        """
        return self._ff_pi_pi_omega.form_factor(q, couplings=self._couplings)

    def form_factor_pi0_pi0_omega(self, q):
        r"""Compute the ωπ⁰π⁰ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for ωπ⁰π⁰.
        """
        return self._ff_pi0_pi0_omega.form_factor(q, couplings=self._couplings)

    def form_factor_pi0_k0_k0(self, q, s, t):
        r"""Compute the π⁰K⁰K̅⁰ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        s: float or array-like
            Mandelstam variable s=(pπ⁰ + pK⁰)^2.
        t: float or array-like
            Mandelstam variable t=(pπ⁰ + pK̅⁰)^2.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁰K⁰K̅⁰.
        """
        return self._ff_pi0_k0_k0.form_factor(q, s, t, couplings=self._couplings)

    def form_factor_pi0_k_k(self, q, s, t):
        r"""Compute the π⁰K⁺K⁻ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        s: float or array-like
            Mandelstam variable s=(pπ⁰ + pK⁺)^2.
        t: float or array-like
            Mandelstam variable t=(pπ⁰ + pK⁻)^2.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁰K⁺K⁻.
        """
        return self._ff_pi0_k_k.form_factor(q, s, t, couplings=self._couplings)

    def form_factor_pi_k_k0(self, q, s, t):
        r"""Compute the π⁺K⁺K⁰ form factor.

        Parameters
        ----------
        q: float or array-like
            Center-of-mass energy in MeV.
        s: float or array-like
            Mandelstam variable s=(pπ⁰ + pK⁺)^2.
        t: float or array-like
            Mandelstam variable t=(pπ⁰ + pK⁰)^2.

        Returns
        -------
        ff: complex or ndarray[complex]
            Form factor for π⁺K⁺K⁰.
        """
        return self._ff_pi_k_k0.form_factor(q, s, t, couplings=self._couplings)

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
        return self.__width_v_to_f_f(0.0, self.gvveve) / 2.0

    @with_cache(cache_name="_width_cache", name="vm vm")
    def width_v_to_vm_vm(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        muon-neutrinos [νμ νμ].
        """
        return self.__width_v_to_f_f(0.0, self.gvvmvm) / 2.0

    @with_cache(cache_name="_width_cache", name="vt vt")
    def width_v_to_vt_vt(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        tau-neutrinos [ντ ντ].
        """
        return self.__width_v_to_f_f(0.0, self.gvvtvt) / 2.0

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
        return self._ff_pi_pi.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="pi0 pi0")
    def width_v_to_pi0_pi0(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        charged pions [𝜋⁺ 𝜋⁻].
        """
        return self._ff_pi0_pi0.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="k0 k0")
    def width_v_to_k0_k0(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        neutral kaons [K⁰ K⁰].
        """
        return self._ff_k0_k0.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="k k")
    def width_v_to_k_k(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator into two
        charged kaons [K⁺ K⁻].
        """
        return self._ff_k_k.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="pi0 gamma")
    def width_v_to_pi0_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion [𝜋⁰] and photon [𝛾].
        """
        mv: float = self.mv
        return self._ff_pi0_gamma.width(mv=mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="eta gamma")
    def width_v_to_eta_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion [𝜋⁰] and photon [𝛾].
        """
        return self._ff_eta_gamma.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="pi0 phi")
    def width_v_to_pi0_phi(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an phi [𝜙(1020)] and neutral pion [𝜋⁰].
        """
        return self._ff_pi0_phi.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="eta phi")
    def width_v_to_eta_phi(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜂] and phi [𝜙(1020)].
        """
        return self._ff_eta_phi.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="eta omega")
    def width_v_to_eta_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜂] and omega [𝜔(782)].
        """
        return self._ff_eta_omega.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="pi0 pi0 gamma")
    def width_v_to_pi0_pi0_gamma(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two neutral pions and a photon [𝜋⁰ 𝜋⁰ 𝛾].
        """
        pi0_omega = self._ff_pi0_omega.width(mv=self.mv, couplings=self._couplings)
        br_omega_to_pi0_gamma = 8.34e-2

        return br_omega_to_pi0_gamma * pi0_omega

    @with_cache(cache_name="_width_cache", name="pi pi pi0")
    def width_v_to_pi_pi_pi0(self, *, npts=10_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and a neutral pion.
        """
        return self._ff_pi_pi_pi0.width(
            mv=self.mv, couplings=self._couplings, npts=npts
        )

    @with_cache(cache_name="_width_cache", name="pi pi eta")
    def width_v_to_pi_pi_eta(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta.
        """
        mv: float = self.mv
        return self._ff_pi_pi_eta.width(mv=mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="pi pi etap")
    def width_v_to_pi_pi_etap(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        return self._ff_pi_pi_etap.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="pi pi omega")
    def width_v_to_pi_pi_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        return self._ff_pi_pi_omega.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="pi0 pi0 omega")
    def width_v_to_pi0_pi0_omega(self) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and an eta-prime.
        """
        return self._ff_pi0_pi0_omega.width(mv=self.mv, couplings=self._couplings)

    @with_cache(cache_name="_width_cache", name="pi0 k0 k0")
    def width_v_to_pi0_k0_k0(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two neutral kaons.
        """
        return self._ff_pi0_k0_k0.width(
            mv=self.mv, couplings=self._couplings, npts=npts
        )

    @with_cache(cache_name="_width_cache", name="pi0 k k")
    def width_v_to_pi0_k_k(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two charged kaons.
        """
        return self._ff_pi0_k_k.width(mv=self.mv, couplings=self._couplings, npts=npts)

    @with_cache(cache_name="_width_cache", name="pi k k0")
    def width_v_to_pi_k_k0(self, *, npts: int = 50_000) -> float:
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and two charged kaons.
        """
        return self._ff_pi_k_k0.width(mv=self.mv, couplings=self._couplings, npts=npts)

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
        return self._ff_pi_pi_pi_pi.width(
            mv=self.mv, couplings=self._couplings, npts=npts
        )

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
        return self._ff_pi_pi_pi0_pi0.width(
            mv=self.mv, couplings=self._couplings, npts=npts
        )

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

    @staticmethod
    def list_annihilation_final_states() -> List[str]:
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

    # ========================================================================
    # ---- Spectra -----------------------------------------------------------
    # ========================================================================

    def _spectrum_funcs(
        self,
    ) -> Dict[str, Callable[[Union[float, npt.NDArray[np.float64]], float], float]]:
        return gev_spectra.dnde_photon_spectrum_fns(self)

    def _gamma_ray_line_energies(self, e_cm) -> Dict[str, float]:
        def photon_energy(mass):
            return 0.5 * (e_cm - mass * (mass / e_cm))

        return {
            "pi0 gamma": photon_energy(_MPI0),
            "eta gamma": photon_energy(_META),
        }

    def _positron_spectrum_funcs(self) -> Dict[str, Callable]:
        return gev_positron_spectra.dnde_positron_spectrum_fns(self)

    def _positron_line_energies(self, e_cm) -> Dict[str, float]:
        return {
            "e e": e_cm / 2.0,
        }

    def _neutrino_spectrum_funcs(self) -> Dict[str, Callable]:
        return gev_neutrino_spectra.dnde_neutrino_spectrum_fns(self)

    def _neutrino_line_energies(self, e_cm, flavor: str) -> Dict[str, float]:
        if flavor == "e":
            return {"ve ve": e_cm / 2.0}
        if flavor == "mu":
            return {"vm vm": e_cm / 2.0}
        if flavor == "tau":
            return {"vt vt": e_cm / 2.0}
        raise ValueError(f"Invalid flavor {flavor}")

    def neutrino_lines(self, e_cm, flavor: str) -> Dict[str, Dict[str, float]]:
        """
        Returns
        -------
        lines : dict(str, dict(str, float))
            For each final state containing a monochromatic photon, gives the
            energy of the photon and branching fraction into that final state.
        """
        bfs = self.annihilation_branching_fractions(e_cm)
        lines = {}

        for fs, e in self._neutrino_line_energies(e_cm, flavor).items():
            lines[fs] = {"energy": e, "bf": bfs[fs]}

        return lines

    def neutrino_spectrum_funcs(self, flavor: str):
        """
        Wraps _positron_spectrum_functions so the function for a final state
        returns zero when annihilation into that state is not permitted.
        """
        sigma_ann_fns = self.annihilation_cross_section_funcs()
        spectrum_funcs = self._neutrino_spectrum_funcs()

        def make_dnde_wrapped(fs):
            def dnde_wrapped(energies, e_cm):
                if sigma_ann_fns[fs](e_cm) > 0:
                    return spectrum_funcs[fs](energies, e_cm, flavor=flavor)
                return np.zeros_like(energies)

            return dnde_wrapped

        return {fs: make_dnde_wrapped(fs) for fs in sigma_ann_fns}

    def neutrino_spectra(self, energies, e_cm, flavor: str):
        r"""
        Gets the contributions to the continuum neutrino annihilation spectrum
        for each final state.

        Parameters
        ---------
        energies: float or array
            Neutrino energy or energies at which to compute the spectrum.
        e_cm : float
            Center of mass energy for the annihilation in MeV.
        flavor: str, optional
            The flavor of neutrino. If None, results for each flavor are returned.

        Returns
        -------
        specs : dict(str, float or float numpy.array)
            Contribution to :math:`dN/dE_\gamma` at the given neutrino energies
            and center-of-mass energy for each relevant final state. More
            specifically, this is the spectrum for annihilation into each
            channel rescaled by the corresponding branching fraction into that
            channel.
        """
        bfs = self.annihilation_branching_fractions(e_cm)
        specs = {}

        for fs, dnde_fn in self.neutrino_spectrum_funcs(flavor).items():
            specs[fs] = bfs[fs] * dnde_fn(energies, e_cm)

        specs["total"] = sum(specs.values())

        return specs

    def total_neutrino_spectrum(self, energies, e_cm, flavor: str):
        r"""
        Computes the total positron ray spectrum.

        Parameters
        ----------
        e_ps : float or float numpy.array
            Positron energy or energies at which to compute the spectrum.
        e_cm : float
            Annihilation center of mass energy.

        Returns
        -------
        spec : float numpy.array
            Array containing the total annihilation positron spectrum.
        """

        if hasattr(energies, "__len__"):
            return self.neutrino_spectra(energies, e_cm, flavor=flavor)
        return self.neutrino_spectra(np.array([energies]), e_cm, flavor=flavor)


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
    lepton electric charges in units of the electric charge, qe is the
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
    g : float
        Coupling strength of the U(1) B-L interaction.
    qx : float
        Charge of DM under B-L.
    """

    def __init__(self, mx: float, mv: float, gvxx: float, g: float) -> None:
        gq = 1.0 / 3.0
        gl = -1.0
        self._g = g

        super().__init__(
            mx=mx,
            mv=mv,
            gvxx=gvxx,
            gvuu=g * gq,
            gvdd=g * gq,
            gvss=g * gq,
            gvee=g * gl,
            gvmumu=g * gl,
            gvveve=g * gl,
            gvvmvm=g * gl,
            gvvtvt=g * gl,
        )

    def __repr__(self) -> str:
        return f"""BLGeV(
            mx={self.mx} [MeV],
            mv={self.mv} [MeV],
            gvxx={self.gvxx},
            g={self.g},
        )
        """

    def __cannot_set_error(self, param: str) -> AttributeError:
        return AttributeError(
            f"""
        Cannot set {param}. Instead use the 'VectorMediatorGeV'
        for more a general coupling structure."""
        )

    def _update_charges(self) -> None:
        gq = 1.0 / 3.0
        gl = -1.0
        self._gvuu = self._g * gq
        self._gvdd = self._g * gq
        self._gvss = self._g * gq
        self._gvee = self._g * gl
        self._gvmumu = self._g * gl
        self._gvveve = self._g * gl
        self._gvvmvm = self._g * gl
        self._gvvtvt = self._g * gl

    @property
    def g(self) -> float:
        """Coupling strength of the U(1) B-L gauge group."""
        return self._g

    @g.setter
    def g(self, g: float) -> None:
        self._g = g
        self._update_charges()

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
