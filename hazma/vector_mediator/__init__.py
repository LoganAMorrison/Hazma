from ..theory import Theory

from _vector_mediator_cross_sections import VectorMediatorCrossSections
from _vector_mediator_fsr import VectorMediatorFSR
from _vector_mediator_positron_spectra import VectorMediatorPositronSpectra
from _vector_mediator_spectra import VectorMediatorSpectra
from _vector_mediator_widths import VectorMediatorWidths


import warnings
from ..hazma_errors import PreAlphaWarning


class VectorMediator(VectorMediatorCrossSections,
                     VectorMediatorFSR,
                     VectorMediatorPositronSpectra,
                     VectorMediatorSpectra,
                     VectorMediatorWidths,
                     Theory):
    r"""
    Create a vector mediator model object.
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
        warnings.warn("", PreAlphaWarning)
        pass

    @classmethod
    def list_final_states(cls):
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ['mu mu', 'e e', 'pi pi', 'pi0 g', 'v v']

    def constraints(self):
        pass

    def constrain(self, p1, p1_vals, p2, p2_vals, ls_or_img="image"):
        pass
