from hazma.theory import Theory

from hazma.vector_mediator._vector_mediator_cross_sections import (
    VectorMediatorCrossSections,
)
from hazma.vector_mediator._vector_mediator_fsr import VectorMediatorFSR
from hazma.vector_mediator._vector_mediator_positron_spectra import (
    VectorMediatorPositronSpectra,
)
from hazma.vector_mediator._vector_mediator_spectra import VectorMediatorSpectra
from hazma.vector_mediator._vector_mediator_widths import VectorMediatorWidths
from ..parameters import qe, Qu, Qd, Qe

from scipy.integrate import quad
from scipy.special import k1, kn


# Note that Theory must be inherited from AFTER all the other mixin classes,
# since they furnish definitions of the abstract methods in Theory.
class VectorMediator(
    VectorMediatorCrossSections,
    VectorMediatorFSR,
    VectorMediatorPositronSpectra,
    VectorMediatorSpectra,
    VectorMediatorWidths,
    Theory,
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

    def __thermal_cross_section_integrand(self, z, x):
        """
        Compute the integrand of the thermally average cross section for
        vector mediator model.

        Parameters
        ----------
        z: float
            Center of mass energy divided by DM mass.
        x: float
            Mass of the dark matter divided by its temperature.

        Returns
        -------
        integrand: float
            Integrand of the thermally-averaged cross-section.
        """
        sig = self.annihilation_cross_sections(self.mx * z)['total']
        kernal = z**2 * (z**2 - 4.0) * k1(x * z)
        return sig * kernal

    def thermal_cross_section(self, x):
        """
        Compute the thermally average cross section for vector mediator model.

        Parameters
        ----------
        x: float
            Mass of the dark matter divided by its temperature.
        model: dark matter model
            Dark matter model, i.e. `ScalarMediator`, `VectorMediator`
            or any model with a dark matter particle.

        Returns
        -------
        tcs: float
            Thermally average cross section.
        """

        # If x is really large, we will get divide by zero errors
        if x > 300:
            return 0.0

        pf = x / (2.0 * kn(2, x))**2

        # points at which integrand may have trouble are:
        #   1. endpoint
        #   2. when ss final state is accessible => z = 2 ms / mx
        #   3. when we hit mediator resonance => z = ms / mx
        return pf * quad(self.__thermal_cross_section_integrand, 2.0, 50.0 / x,
                         args=(x, ), points=[2.0, self.mv / self.mx,
                                             2.0 * self.mv / self.mx])[0]


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
