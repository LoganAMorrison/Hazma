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

    def description(self):
        """
        Returns a string giving the details of the model.
        """
        return """
        The vector mediator model appends to the Standard Model  a dark \n
        matter particle, `x` and a massive vector mediator, `v`. The \n
        Lagrangian at 1 GeV is: \n
        \t L = LSM + Lkin + gvxx xbar.GA[mu].x V[mu] \n
        \t   + gvuu ubar.GA[mu].u V[mu] + gvdd dbar.GA[mu].d V[mu] \n
        \t   + gvss sbar.GA[mu].s V[mu] + gvee ebar.GA[mu].e V[mu] \n
        \t   + gvmumu mubar.GA[mu].mu V[mu] \n \n


        Parameters \n
        ---------- \n
        mx : float \n
        \t Mass of the initial dark matter. \n
        mv : float \n
        \t Mass of the vector mediator. \n
        gvxx : float \n
        \t Coupling of vector mediator to the dark matter. \n
        gvuu : float \n
        \t Coupling of vector mediator to standard model up quark. \n
        gvdd : float \n
        \t Coupling of vector mediator to standard model down quark. \n
        gvss : float \n
        \t Coupling of vector mediator to standard model strange quark. \n
        gvee : float \n
        \t Coupling of vector mediator to standard model electron. \n
        gvmumu : float \n
        \t Coupling of vector mediator to standard model muon. \n

        Methods \n
        ------- \n
        list_final_states : \n
        \t Return a list of the available final states. \n
        cross_sections : \n
        \t Computes the all the cross sections of the theory and returns \n
        \t a dictionary containing the cross sections. \n
        branching_fractions : \n
        \t Computes the all the branching fractions of the theory and \n
        \t returns a dictionary containing the branching fractions. \n
        spectra : \n
        \t Computes all spectra of the theory for a pair of initial \n
        \t state fermions annihilating into each available final state \n
        \t and returns a dictionary of arrays containing the spectra. \n
        spectrum_functions :
        \t Returns a dictionary of all the avaiable spectrum functions for \n
        \t a pair of dark matter particles with mass `mx` \n
        \t annihilating into each available final state. \n
        partial_widths : \n
        \t Returns a dictionary for the partial decay widths of the scalar \n
        \t mediator. \n
        """

    @classmethod
    def list_annihilation_final_states(cls):
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ["mu mu", "e e", "pi pi", "pi0 g", "v v"]

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

    def description(self):
        """
        Returns a string giving the details of the model.
        """
        return """
        The Kinetic Mixing model is specific implementation of the vector \n
        mediator model in which the vector mediator mixes with the Standard \n
        Model photon with a mixing parameter `eps`. This causes the \n
        couplings of the vector mediator the charged Standard Model fermions \n
        to be  proportional to the fermion charge times `eps`. The \n
        Lagrangian is: \n
        \t L = LSM + Lkin + gvxx xbar.GA[mu].x V[mu] \n
        \t   + 2/3 e epsubar.GA[mu].u V[mu] - 1/3 e eps dbar.GA[mu].d V[mu] \n
        \t   - 1/3 e eps sbar.GA[mu].s V[mu] - e eps ebar.GA[mu].e V[mu] \n
        \t   - e eps mubar.GA[mu].mu V[mu] \n \n

        Parameters \n
        ---------- \n
        mx : float \n
        \t Mass of the initial dark matter. \n
        mv : float \n
        \t Mass of the vector mediator. \n
        gvxx : float \n
        \t Coupling of vector mediator to the dark matter. \n
        eps : float \n
        \t Kinetic mixing parameter. \n \n

        Methods \n
        ------- \n
        list_final_states : \n
        \t Return a list of the available final states. \n
        cross_sections : \n
        \t Computes the all the cross sections of the theory and returns \n
        \t a dictionary containing the cross sections. \n
        branching_fractions : \n
        \t Computes the all the branching fractions of the theory and \n
        \t returns a dictionary containing the branching fractions. \n
        spectra : \n
        \t Computes all spectra of the theory for a pair of initial \n
        \t state fermions annihilating into each available final state \n
        \t and returns a dictionary of arrays containing the spectra. \n
        spectrum_functions :
        \t Returns a dictionary of all the avaiable spectrum functions for \n
        \t a pair of dark matter particles with mass `mx` \n
        \t annihilating into each available final state. \n
        partial_widths : \n
        \t Returns a dictionary for the partial decay widths of the scalar \n
        \t mediator. \n
        """


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
        super(QuarksOnly, self).__init__(mx, mv, gvxx, gvuu, gvdd, gvss, 0.0, 0.0)

    # Hide underlying properties' setters
    @VectorMediator.gvee.setter
    def gvee(self, gvee):
        raise AttributeError("Cannot set gvee")

    @VectorMediator.gvmumu.setter
    def gvmumu(self, gvmumu):
        raise AttributeError("Cannot set gvmumu")

    def description(self):
        """
        Returns a string giving the details of the model.
        """
        return """
        This model is identical to the vector mediator model except the only \n
        Standard Model particles that the vector mediator couples to are the \n
        quarks. The Lagrangian is:\n
        \t L = LSM + Lkin + gvxx xbar.GA[mu].x V[mu] \n
        \t   + 2/3 e epsubar.GA[mu].u V[mu] - 1/3 e eps dbar.GA[mu].d V[mu] \n
        \t   - 1/3 e eps sbar.GA[mu].s V[mu] \n \n

        Parameters \n
        ---------- \n
        mx : float \n
        \t Mass of the initial dark matter. \n
        mv : float \n
        \t Mass of the vector mediator. \n
        gvxx : float \n
        \t Coupling of vector mediator to the dark matter. \n
        gvuu : float \n
        \t Coupling of vector mediator to standard model up quark. \n
        gvdd : float \n
        \t Coupling of vector mediator to standard model down quark. \n
        gvss : float \n
        \t Coupling of vector mediator to standard model strange quark. \n \n

        Methods \n
        ------- \n
        list_final_states : \n
        \t Return a list of the available final states. \n
        cross_sections : \n
        \t Computes the all the cross sections of the theory and returns \n
        \t a dictionary containing the cross sections. \n
        branching_fractions : \n
        \t Computes the all the branching fractions of the theory and \n
        \t returns a dictionary containing the branching fractions. \n
        spectra : \n
        \t Computes all spectra of the theory for a pair of initial \n
        \t state fermions annihilating into each available final state \n
        \t and returns a dictionary of arrays containing the spectra. \n
        spectrum_functions :
        \t Returns a dictionary of all the avaiable spectrum functions for \n
        \t a pair of dark matter particles with mass `mx` \n
        \t annihilating into each available final state. \n
        partial_widths : \n
        \t Returns a dictionary for the partial decay widths of the scalar \n
        \t mediator. \n
        """
