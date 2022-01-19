from typing import Union, List

import numpy as np

from hazma.parameters import Qd, Qe, Qu, qe
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
from hazma.theory import TheoryAnn
from hazma.vector_mediator._vector_mediator_cross_sections import (
    VectorMediatorCrossSections,
)
from hazma.vector_mediator._vector_mediator_fsr import VectorMediatorFSR
from hazma.vector_mediator._vector_mediator_positron_spectra import (
    VectorMediatorPositronSpectra,
)
from hazma.vector_mediator._vector_mediator_spectra import VectorMediatorSpectra
from hazma.vector_mediator._vector_mediator_widths import VectorMediatorWidths
from hazma.vector_mediator.form_factors.utils import RealArray
from hazma.vector_mediator.form_factors.utils import ComplexArray

from hazma.rambo import compute_decay_width


# Note that Theory must be inherited from AFTER all the other mixin classes,
# since they furnish definitions of the abstract methods in Theory.
class VectorMediator(
    VectorMediatorCrossSections,
    VectorMediatorFSR,
    VectorMediatorPositronSpectra,
    VectorMediatorSpectra,
    VectorMediatorWidths,
    TheoryAnn,
):
    r"""
    Create a VectorMediator object with generic couplings.

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
    ) -> None:
        self._mx: float = mx
        self._mv: float = mv
        self._gvxx: float = gvxx
        self._gvuu: float = gvuu
        self._gvdd: float = gvdd
        self._gvss: float = gvss
        self._gvee: float = gvee
        self._gvmumu: float = gvmumu
        self.width_v: float = 0.0
        self.compute_width_v()

    def __repr__(self) -> str:
        return (
            f"VectorMediator(\n"
            f"\tmx={self.mx} MeV,\n"
            f"\tmv={self.mv} MeV,\n"
            f"\tgvxx={self.gvxx},\n"
            f"\tgvuu={self.gvuu},\n"
            f"\tgvdd={self.gvdd},\n"
            f"\tgvss={self.gvss},\n"
            f"\tgvee={self.gvee},\n"
            f"\tgvmumu={self.gvmumu}\n"
            ")"
        )

    @property
    def mx(self) -> float:
        return self._mx

    @mx.setter
    def mx(self, mx: float) -> None:
        self._mx = mx
        self.compute_width_v()

    @property
    def mv(self) -> float:
        return self._mv

    @mv.setter
    def mv(self, mv: float) -> None:
        self._mv = mv
        self.compute_width_v()

    @property
    def gvxx(self) -> float:
        return self._gvxx

    @gvxx.setter
    def gvxx(self, gvxx: float) -> None:
        self._gvxx = gvxx
        self.compute_width_v()

    @property
    def gvuu(self) -> float:
        return self._gvuu

    @gvuu.setter
    def gvuu(self, gvuu: float) -> None:
        self._gvuu = gvuu
        self.compute_width_v()

    @property
    def gvdd(self) -> float:
        return self._gvdd

    @gvdd.setter
    def gvdd(self, gvdd: float) -> None:
        self._gvdd = gvdd
        self.compute_width_v()

    @property
    def gvss(self) -> float:
        return self._gvss

    @gvss.setter
    def gvss(self, gvss: float) -> None:
        self._gvss = gvss
        self.compute_width_v()

    @property
    def gvee(self) -> float:
        return self._gvee

    @gvee.setter
    def gvee(self, gvee: float) -> None:
        self._gvee = gvee
        self.compute_width_v()

    @property
    def gvmumu(self) -> float:
        return self._gvmumu

    @gvmumu.setter
    def gvmumu(self, gvmumu: float) -> None:
        self._gvmumu = gvmumu
        self.compute_width_v()

    def compute_width_v(self) -> None:
        """Recomputes the scalar's total width."""
        self.width_v = self.partial_widths()["total"]

    @staticmethod
    def list_annihilation_final_states() -> List[str]:
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ["mu mu", "e e", "pi pi", "pi0 g", "pi0 v", "v v"]

    def constraints(self):
        pass

    def constrain(self, p1, p1_vals, p2, p2_vals, ls_or_img="image"):
        pass


class KineticMixing(VectorMediator):
    r"""
    Create a ``VectorMediator`` object with kinetic mixing couplings.

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

        super(KineticMixing, self).__init__(
            mx,
            mv,
            gvxx,
            -Qu * eps * qe,
            -Qd * eps * qe,
            -Qd * eps * qe,
            -Qe * eps * qe,
            -Qe * eps * qe,
        )

    def __repr__(self) -> str:
        repr_ = "KineticMixing("
        repr_ += f"mx={self.mx} [MeV], "
        repr_ += f"mv={self.mv} [MeV], "
        repr_ += f"gvxx={self.gvxx}, "
        repr_ += f"eps={self.eps}"
        repr_ += ")"
        return repr_

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, eps: float):
        self._eps = eps
        self._gvuu = -Qu * eps * qe
        self._gvdd = -Qd * eps * qe
        self._gvss = -Qd * eps * qe
        self._gvee = -Qe * eps * qe
        self._gvmumu = -Qe * eps * qe
        self.compute_width_v()

    # Hide underlying properties' setters
    @VectorMediator.gvuu.setter
    def gvuu(self, _: float):
        raise AttributeError("Cannot set gvuu")

    @VectorMediator.gvdd.setter
    def gvdd(self, _: float):
        raise AttributeError("Cannot set gvdd")

    @VectorMediator.gvss.setter
    def gvss(self, _: float):
        raise AttributeError("Cannot set gvss")

    @VectorMediator.gvee.setter
    def gvee(self, _: float):
        raise AttributeError("Cannot set gvee")

    @VectorMediator.gvmumu.setter
    def gvmumu(self, _: float):
        raise AttributeError("Cannot set gvmumu")


class QuarksOnly(VectorMediator):
    r"""
    Create a VectorMediator object with only quark couplings.

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
    """

    def __init__(
        self, mx: float, mv: float, gvxx: float, gvuu: float, gvdd: float, gvss: float
    ) -> None:
        super(QuarksOnly, self).__init__(mx, mv, gvxx, gvuu, gvdd, gvss, 0.0, 0.0)

    def __repr__(self) -> str:
        repr_ = "QuarksOnly("
        repr_ += f"mx={self.mx} [MeV], "
        repr_ += f"mv={self.mv} [MeV], "
        repr_ += f"gvxx={self.gvxx}, "
        repr_ += f"gvuu={self.gvuu}"
        repr_ += f"gvdd={self.gvdd}"
        repr_ += f"gvss={self.gvss}"
        repr_ += ")"
        return repr_

    @staticmethod
    def list_annihilation_final_states() -> List[str]:
        return ["pi pi", "pi0 g", "pi0 v", "v v"]

    # Hide underlying properties' setters
    @VectorMediator.gvee.setter
    def gvee(self, _: float) -> AttributeError:
        raise AttributeError("Cannot set gvee")

    @VectorMediator.gvmumu.setter
    def gvmumu(self, _: float) -> AttributeError:
        raise AttributeError("Cannot set gvmumu")


class VectorMediatorGeV(VectorMediator):
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
        """
        from hazma.vector_mediator.form_factors.pipi import FormFactorPiPi
        from hazma.vector_mediator.form_factors.kk import FormFactorKK
        from hazma.vector_mediator.form_factors.pi_pi_pi0 import FormFactorPiPiPi0
        from hazma.vector_mediator.form_factors.pi_pi_eta import FormFactorPiPiEta
        from hazma.vector_mediator.form_factors.pi_pi_etap import FormFactorPiPiEtaP
        from hazma.vector_mediator.form_factors.pi_pi_omega import FormFactorPiPiOmega

        # Compute and store the parameters needed to compute form factors.
        # self._ff_pipi_params = _compute_ff_params_pipi(2000)
        self._ff_pipi = FormFactorPiPi()
        # self._ff_kk_params = _compute_ff_params_kk(200)
        self._ff_kk = FormFactorKK()
        self._ff_pi_pi_pi0 = FormFactorPiPiPi0()
        self._ff_pi_pi_eta = FormFactorPiPiEta()
        self._ff_pi_pi_etap = FormFactorPiPiEtaP()
        self._ff_pi_pi_omega = FormFactorPiPiOmega()

        super().__init__(mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu)

    # Import the form factors
    from hazma.vector_mediator.form_factors import (
        form_factor_eta_gamma as _form_factor_eta_gamma,
        # form_factor_kk as _form_factor_kk,
        form_factor_pi_gamma as _form_factor_pi_gamma,
        # form_factor_pipi as _form_factor_pipi,
        form_factor_omega_pi as _form_factor_omega_pi,
        form_factor_phi_pi as _form_factor_phi_pi,
        form_factor_eta_phi as _form_factor_eta_phi,
        form_factor_eta_omega as _form_factor_eta_omega,
    )

    @property
    def gvuu(self) -> float:
        """
        Coupling of vector mediator to the up quark.
        """
        return self._gvuu

    @gvuu.setter
    def gvuu(self, val: float) -> None:
        self._gvuu = val
        self._reset_state()

    @property
    def gvdd(self) -> float:
        """
        Coupling of vector mediator to the down quark.
        """
        return self._gvdd

    @gvdd.setter
    def gvdd(self, val: float) -> None:
        self._gvdd = val
        self._reset_state()

    @property
    def gvss(self) -> float:
        """
        Coupling of vector mediator to the down quark.
        """
        return self._gvss

    @gvss.setter
    def gvss(self, val: float) -> None:
        self._gvss = val
        self._reset_state()

    def _reset_state(self) -> None:
        """
        Function to reset the state of the derived quantities such as the
        vector width and form-factors.
        """
        pass

    def form_factor_pipi(
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

        ff = self._ff_pipi.form_factor(
            qq ** 2,
            self._gvuu,
            self._gvdd,
            imode=imode,
        )

        if len(ff) == 1:
            return ff[0]
        return ff

    def form_factor_kk(
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
        ff = self._ff_kk.form_factor(
            qq ** 2,
            self._gvuu,
            self._gvdd,
            self._gvss,
            imode=imode,
        )
        if len(ff) == 1:
            return ff[0]
        return ff

    def _width_v_to_mm(
        self,
        mass: float,
        form_factor: Union[complex, ComplexArray],
        symmetry: float = 1.0,
    ):
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
            * (1 - 4 * mu ** 2) ** 1.5
            * np.abs(form_factor) ** 2
        )

    def width_v_to_pipi(self):
        """
        Compute the partial width for the decay of the vector mediator into two
        charged pions [𝜋⁺ 𝜋⁻].
        """
        mass = _MPI
        form_factor = self.form_factor_pipi(self.mv)
        return self._width_v_to_mm(mass, form_factor)

    def width_v_to_k0k0(self):
        """
        Compute the partial width for the decay of the vector mediator into two
        neutral kaons [K⁰ K⁰].
        """
        mass = _MK0
        form_factor = self.form_factor_kk(self.mv, imode=0)
        return self._width_v_to_mm(mass, form_factor)

    def width_v_to_kk(self):
        """
        Compute the partial width for the decay of the vector mediator into two
        charged kaons [K⁺ K⁻].
        """
        mass = _MK
        form_factor = self.form_factor_kk(self._mv, imode=1)
        return self._width_v_to_mm(mass, form_factor)

    def _width_v_to_mg(self, mass: float, ff: complex):
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
        q = 0.5 * np.sqrt(kallen_lambda(1.0, mu ** 2, 0))
        return self.mv * q ** 3 * np.abs(ff) ** 2 / (12.0 * np.pi)

    def width_v_to_pi0g(self):
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion [𝜋⁰] and photon [𝛾].
        """
        mass = _MPI0
        form_factor = self._form_factor_pi_gamma(self.mv)
        return self._width_v_to_mg(mass, form_factor)

    def width_v_to_eta_gamma(self):
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion [𝜋⁰] and photon [𝛾].
        """
        mass = _META
        form_factor = self._form_factor_eta_gamma(self.mv)
        return self._width_v_to_mg(mass, form_factor)

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
        return self.mv * q ** 3 * np.abs(ff) ** 2 / (12.0 * np.pi)

    def width_v_to_omega_pi(self):
        """
        Compute the partial width for the decay of the vector mediator
        into an omega [𝜔(782)] and neutral pion [𝜋⁰].
        """
        ff = self._form_factor_omega_pi(self.mv)
        return self.__width_v_to_v_s(ff, _MOMEGA, _MPI0)

    def width_v_to_phi_pi(self):
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜙(1020)] and photon [𝜋⁰].
        """
        ff = self._form_factor_phi_pi(self.mv)
        return self.__width_v_to_v_s(ff, _MPHI, _MPI0)

    def width_v_to_eta_phi(self):
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜂] and phi [𝜙(1020)].
        """
        ff = self._form_factor_eta_phi(self.mv)
        return self.__width_v_to_v_s(ff, _MPHI, _META)

    def width_v_to_eta_omega(self):
        """
        Compute the partial width for the decay of the vector mediator
        into an eta [𝜂] and omega [𝜔(782)].
        """
        ff = self._form_factor_eta_omega(self.mv)
        return self.__width_v_to_v_s(ff, _MOMEGA, _META)

    def width_v_to_pi_pi_pi0(self):
        """
        Compute the partial width for the decay of the vector mediator
        into two charged pions and a neutral pion.
        """
        if self.mv < 2 * _MPI + _MPI0:
            return 0.0
        mvgev = self.mv * 1e-3
        integral = self._ff_pi_pi_pi0.integrated_form_factor(
            mvgev ** 2, self.gvuu, self.gvdd, self.gvss
        )
        return integral[0] / (2 * mvgev) * 1e3

    def width_v_to_pi_pi_eta(self):
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

    def width_v_to_pi_pi_etap(self):
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

    def width_v_to_pi_pi_omega(self):
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

    def width_v_to_pi0_pi0_omega(self):
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

    def width_v_to_xx(self):
        """
        Compute the partial width for the decay of the vector mediator
        into two dark matter particles.
        """
        mv = self.mv
        mx = self.mx
        if mv > 2.0 * mx:
            val = (
                (
                    self.gvxx ** 2
                    * np.sqrt(mv ** 2 - 4 * mx ** 2)
                    * (mv ** 2 + 2 * mx ** 2)
                )
                / (12.0 * mv ** 2 * np.pi)
            ).real

            assert val >= 0
            return val
        return 0.0

    def __width_v_to_ff(self, gvll, ml):
        mv = self.mv
        if mv > 2.0 * ml:
            val = (
                (gvll ** 2 * np.sqrt(-4 * ml ** 2 + mv ** 2) * (2 * ml ** 2 + mv ** 2))
                / (12.0 * mv ** 2 * np.pi)
            ).real
            assert val >= 0
            return val
        return 0.0

    def width_v_to_ee(self):
        """
        Compute the partial width for the decay of the vector mediator
        into two electrons.
        """
        return self.__width_v_to_ff(self.gvee, _ME)

    def width_v_to_mumu(self):
        """
        Compute the partial width for the decay of the vector mediator
        into two muons.
        """
        return self.__width_v_to_ff(self.gvmumu, _MMU)

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
        gamv = self.width_v
        mv = self.mv
        mx = self.mx
        if e_cm < 2.0 * mx or e_cm < 2.0 * mass:
            return 0.0

        s = e_cm ** 2
        num = (
            self.gvxx ** 2
            * np.abs(form_factor) ** 2
            * (s - 4 * mass ** 2) ** 1.5
            * (2.0 * mx ** 2 + s)
        )
        den = (
            48.0
            * np.pi
            * s
            * np.sqrt(s - 4.0 * mx ** 2)
            * (mv ** 2 * (gamv ** 2 - 2.0 * s) + mv ** 4 + s ** 2)
        )

        return symmetry * num / den

    def sigma_xx_to_v_to_pipi(
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
        ff = self._ff_pipi.form_factor(ecm ** 2, self._gvuu, self._gvdd, imode=0)
        return self._sigma_xx_to_v_to_mm(e_cm, mass, ff)

    def sigma_xx_to_v_to_kk(
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
        form_factor = self._form_factor_kk(e_cm ** 2)
        return self._sigma_xx_to_v_to_mm(e_cm, mass, form_factor)


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
            mx,
            mv,
            gvxx,
            -Qu * eps * qe,
            -Qd * eps * qe,
            -Qd * eps * qe,
            -Qe * eps * qe,
            -Qe * eps * qe,
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
        self._gvuu = -Qu * eps * qe
        self._gvdd = -Qd * eps * qe
        self._gvss = -Qd * eps * qe
        self._gvee = -Qe * eps * qe
        self._gvmumu = -Qe * eps * qe
        self.compute_width_v()

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
