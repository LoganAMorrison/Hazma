from typing import Union

import numpy as np

from hazma.parameters import Qd, Qe, Qu
from hazma.parameters import charged_kaon_mass as _MK
from hazma.parameters import charged_pion_mass as _MPI
from hazma.parameters import eta_mass as _META
from hazma.parameters import neutral_kaon_mass as _MK0
from hazma.parameters import neutral_pion_mass as _MPI0
from hazma.parameters import qe
from hazma.theory import TheoryAnn
from hazma.vector_mediator._vector_mediator_cross_sections import \
    VectorMediatorCrossSections
from hazma.vector_mediator._vector_mediator_fsr import VectorMediatorFSR
from hazma.vector_mediator._vector_mediator_positron_spectra import \
    VectorMediatorPositronSpectra
from hazma.vector_mediator._vector_mediator_spectra import \
    VectorMediatorSpectra
from hazma.vector_mediator._vector_mediator_widths import VectorMediatorWidths
from hazma.vector_mediator.form_factors.kk import \
    compute_kk_form_factor_parameters as __compute_ff_params_kk
from hazma.vector_mediator.form_factors.pipi import \
    compute_pipi_form_factor_parameters as __compute_ff_params_pipi

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

    def __init__(self, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu):
        self._mx = mx
        self._mv = mv
        self._gvxx = gvxx
        self._gvuu = gvuu
        self._gvdd = gvdd
        self._gvss = gvss
        self._gvee = gvee
        self._gvmumu = gvmumu
        self.compute_width_v()

    def __repr__(self):
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
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx
        self.compute_width_v()

    @property
    def mv(self):
        return self._mv

    @mv.setter
    def mv(self, mv):
        self._mv = mv
        self.compute_width_v()

    @property
    def gvxx(self):
        return self._gvxx

    @gvxx.setter
    def gvxx(self, gvxx):
        self._gvxx = gvxx
        self.compute_width_v()

    @property
    def gvuu(self):
        return self._gvuu

    @gvuu.setter
    def gvuu(self, gvuu):
        self._gvuu = gvuu
        self.compute_width_v()

    @property
    def gvdd(self):
        return self._gvdd

    @gvdd.setter
    def gvdd(self, gvdd):
        self._gvdd = gvdd
        self.compute_width_v()

    @property
    def gvss(self):
        return self._gvss

    @gvss.setter
    def gvss(self, gvss):
        self._gvss = gvss
        self.compute_width_v()

    @property
    def gvee(self):
        return self._gvee

    @gvee.setter
    def gvee(self, gvee):
        self._gvee = gvee
        self.compute_width_v()

    @property
    def gvmumu(self):
        return self._gvmumu

    @gvmumu.setter
    def gvmumu(self, gvmumu):
        self._gvmumu = gvmumu
        self.compute_width_v()

    def compute_width_v(self):
        """Recomputes the scalar's total width."""
        self.width_v = self.partial_widths()["total"]

    @staticmethod
    def list_annihilation_final_states():
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

    def __init__(self, mx, mv, gvxx, eps):
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

    def __repr__(self):
        return f"KineticMixing(mx={self.mx} MeV, mv={self.mv} MeV, gvxx={self.gvxx}, eps={self.eps})"

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, eps):
        self._eps = eps
        self._gvuu = -Qu * eps * qe
        self._gvdd = -Qd * eps * qe
        self._gvss = -Qd * eps * qe
        self._gvee = -Qe * eps * qe
        self._gvmumu = -Qe * eps * qe
        self.compute_width_v()

    # Hide underlying properties' setters
    @VectorMediator.gvuu.setter
    def gvuu(self, gvuu):
        raise AttributeError("Cannot set gvuu")

    @VectorMediator.gvdd.setter
    def gvdd(self, gvdd):
        raise AttributeError("Cannot set gvdd")

    @VectorMediator.gvss.setter
    def gvss(self, gvss):
        raise AttributeError("Cannot set gvss")

    @VectorMediator.gvee.setter
    def gvee(self, gvee):
        raise AttributeError("Cannot set gvee")

    @VectorMediator.gvmumu.setter
    def gvmumu(self, gvmumu):
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

    def __init__(self, mx, mv, gvxx, gvuu, gvdd, gvss):
        super(QuarksOnly, self).__init__(
            mx, mv, gvxx, gvuu, gvdd, gvss, 0.0, 0.0)

    def __repr__(self):
        return f"QuarksOnly(mx={self.mx} MeV, mv={self.mv} MeV, gvxx={self.gvxx}, gvuu={self.gvuu}, gvdd={self.gvdd}, gvss={self.gvss})"

    @staticmethod
    def list_annihilation_final_states():
        return ["pi pi", "pi0 g", "pi0 v", "v v"]

    # Hide underlying properties' setters
    @VectorMediator.gvee.setter
    def gvee(self, gvee):
        raise AttributeError("Cannot set gvee")

    @VectorMediator.gvmumu.setter
    def gvmumu(self, gvmumu):
        raise AttributeError("Cannot set gvmumu")


class VectorMediatorGeV(VectorMediator):
    """
    A generic dark matter model where interactions with the SM are mediated via
    an s-channel vector mediator. This model is valid for dark-matter masses
    up to 1 GeV.
    """

    def __init__(self, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu):
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

        # Compute and store the parameters needed to compute form factors.
        self._ff_pipi_params = __compute_ff_params_pipi(2000)
        self._ff_kk_params = __compute_ff_params_kk(200)

        super().__init__(mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu)

    # Import the form factors
    from hazma.vector_mediator.form_factors import (_form_factor_eta_gamma,
                                                    _form_factor_kk,
                                                    _form_factor_pi_gamma,
                                                    _form_factor_pipi)

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

    def _width_v_to_mm(
            self,
            mass: float,
            form_factor: complex,
            symmetry: float = 1.0
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
        return (
            symmetry
            / 48.0
            / np.pi
            * self._mv
            * (1 - 4 * mass ** 2 / self._mv ** 2) ** 1.5
            * abs(form_factor) ** 2
        )

    def width_v_to_pipi(self):
        """
        Compute the partial width for the decay of the vector mediator into two
        charged pions.
        """
        mass = _MPI
        form_factor = self._form_factor_pipi(self._mv**2)
        return self._width_v_to_mm(mass, form_factor)

    def width_v_to_k0k0(self):
        """
        Compute the partial width for the decay of the vector mediator into two
        neutral kaons.
        """
        mass = _MK0
        form_factor = self._form_factor_kk(self._mv**2, imode=0)
        return self._width_v_to_mm(mass, form_factor)

    def width_v_to_kk(self):
        """
        Compute the partial width for the decay of the vector mediator into two
        charged kaons.
        """
        mass = _MK
        form_factor = self._form_factor_kk(self._mv**2, imode=1)
        return self._width_v_to_mm(mass, form_factor)

    def _width_v_to_mg(self, mass, form_factor):
        """
        Compute the partial width for the decay of the vector mediator
        into a meson and photon.

        Parameters
        ----------
        mass: float
            Mass of the final state meson.
        form_factor: complex
            Vector form factor for the V-meson-meson vertex.

        Returns
        -------
        gamma: float
            Partial width for the vector to decay into a meson and photon.
        """
        return (
            self._mv
            * abs(form_factor) ** 2
            * (1.0 - (mass / self._mv) ** 2) ** 3
            / (6.0 * np.pi)
        )

    def width_v_to_pi0g(self):
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and photon.
        """
        mass = _MPI0
        form_factor = self._form_factor_pi_gamma(self._mv**2)
        return self._width_v_to_mg(mass, form_factor)

    def width_v_to_etag(self):
        """
        Compute the partial width for the decay of the vector mediator
        into an eta and photon.
        """
        mass = _META
        form_factor = self._form_factor_eta_gamma(self._mv**2)
        return self._width_v_to_mg(mass, form_factor)

    def _sigma_xx_to_v_to_mm(
            self,
            e_cm: Union[float, np.ndarray],
            mass: float,
            form_factor: Union[complex, np.ndarray],
            symmetry: float = 1.0
    ) -> Union[float, np.ndarray]:
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

        s = e_cm**2
        num = self.gvxx**2 * np.abs(form_factor)**2 * \
            (s - 4 * mass ** 2)**1.5 * (2.0 * mx**2 + s)
        den = 48.0 * np.pi * s * \
            np.sqrt(s - 4.0 * mx**2) * \
            (mv**2 * (gamv**2 - 2.0 * s) + mv**4 + s**2)

        return symmetry * num / den

    def sigma_xx_to_v_to_pipi(
            self,
            e_cm: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        form_factor = self._form_factor_pipi(e_cm**2)
        return self._sigma_xx_to_v_to_mm(e_cm, mass, form_factor)

    def sigma_xx_to_v_to_kk(
            self,
            e_cm: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        form_factor = self._form_factor_kk(e_cm**2)
        return self._sigma_xx_to_v_to_mm(e_cm, mass, form_factor)
