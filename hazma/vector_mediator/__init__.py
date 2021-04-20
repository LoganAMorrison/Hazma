from typing import Optional

import numpy as np

from hazma.parameters import Qd, Qe, Qu
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
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
# Functions for computing the V-pi-gamma form-factor
from hazma.vector_mediator.form_factors.pi_gamma import form_factor_pi_gamma
# Functions for computing the V-pi-pi form-factor
from hazma.vector_mediator.form_factors.pipi import (
    compute_pipi_form_factor_parameters, form_factor_pipi)


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

        self._form_factor_pipi_params = compute_pipi_form_factor_parameters(
            2000
        )

        super().__init__(mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu)

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

    def _form_factor_pipi(self, s: float, imode: Optional[int] = 1) -> complex:
        """
        Compute the pi-pi-V form factor.

        Parameters
        ----------
        s: float
            Square of the center-of-mass energy in MeV.
        imode: Optional[int]
            Iso-spin channel. Default is 1.

        Returns
        -------
        ff: complex
            Form factor from pi-pi-V.
        """
        sgev = s * 1e-6  # Convert to GeV
        ci1 = self._gvuu - self._gvdd
        return form_factor_pipi(sgev, self._form_factor_pipi_params, ci1)

    def _form_factor_pi0g(self, s):
        """
        Compute the pi-pi-V form factor for a give squared CME.

        Parameters
        ----------
        s: float
            Square of the center-of-mass energy in MeV.

        Returns
        -------
        ff: complex
            Form factor from pi0-gamma-V.
        """
        sgev = s * 1e-6  # Convert s to GeV
        return form_factor_pi_gamma(sgev, self._gvuu, self._gvdd, self._gvss)

    def width_v_to_pipi(self):
        """
        Compute the partial width for the decay of the vector mediator into two
        charged pions.
        """
        if self._mv < 2 * mpi0:
            return 0.0
        ff = self._form_factor_pipi(self._mv**2)
        return (
            1.0
            / 48.0
            / np.pi
            * self._mv
            * (1 - 4 * mpi0 ** 2 / self._mv ** 2) ** 1.5
            * abs(ff) ** 2
        )

    def width_v_to_pi0g(self):
        """
        Compute the partial width for the decay of the vector mediator
        into a neutral pion and photon.
        """
        if self._mv < mpi:
            return 0.0
        ff = self._form_factor_pi0g(self._mv**2)
        return (
            self._mv
            * abs(ff) ** 2
            * (1.0 - (mpi / self._mv) ** 2) ** 3
            / (6.0 * np.pi)
        )

    def sigma_xx_to_v_to_pipi(self, e_cm: float) -> float:
        """
        Compute the dark matter annihilation cross section into two charged
        pions.

        Parameters
        ----------
        e_cm: float
            Center-of-mass energy.

        Returns
        -------
        sigma: float
            Cross section for chi + chibar -> pi + pi.
        """
        gamv = self.width_v
        mv = self.mv
        mx = self.mx
        if e_cm < 2.0 * mx or e_cm < 2.0 * mpi:
            return 0.0

        s = e_cm**2
        # Compute
        ff = self._form_factor_pipi(s)
        num = self.gvxx**2 * abs(ff)**2 * (s - 4 * mpi **
                                           2)**1.5 * (2.0 * mx**2 + s)
        den = 48.0 * np.pi * s * \
            np.sqrt(s - 4.0 * mx**2) * \
            (mv**2 * (gamv**2 - 2.0 * s) + mv**4 + s**2)

        return num / den
