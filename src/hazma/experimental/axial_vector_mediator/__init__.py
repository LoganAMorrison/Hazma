from hazma.theory import Theory

import warnings
from hazma.hazma_errors import PreAlphaWarning

from hazma.axial_vector_mediator._axial_vector_mediator_cross_sections import (
    AxialVectorMediatorCrossSections,
)
from hazma.axial_vector_mediator._axial_vector_mediator_positron_spectra import (
    AxialVectorMediatorPositronSpectra,
)
from hazma.axial_vector_mediator._axial_vector_mediator_spectra import (
    AxialVectorMediatorSpectra,
)
from hazma.axial_vector_mediator._axial_vector_mediator_widths import (
    AxialVectorMediatorWidths,
)


class AxialVectorMediator(
    AxialVectorMediatorCrossSections,
    AxialVectorMediatorPositronSpectra,
    AxialVectorMediatorSpectra,
    AxialVectorMediatorWidths,
    Theory,
):
    r"""
    Create an axial vector mediator model object.
    """

    def __init__(self, mx, ma, gaxx, gauu, gadd, gass, gaee, gamumu):
        self._mx = mx
        self._ma = ma
        self._gaxx = gaxx
        self._gauu = gauu
        self._gadd = gadd
        self._gass = gass
        self._gaee = gaee
        self._gamumu = gamumu
        self.compute_width_a()

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx
        self.compute_width_a()

    @property
    def ma(self):
        return self._ma

    @ma.setter
    def ma(self, ma):
        self._ma = ma
        self.compute_width_a()

    @property
    def gaxx(self):
        return self._gaxx

    @gaxx.setter
    def gaxx(self, gaxx):
        self._gaxx = gaxx
        self.compute_width_a()

    @property
    def gauu(self):
        return self._gauu

    @gauu.setter
    def gauu(self, gauu):
        self._gauu = gauu
        self.compute_width_a()

    @property
    def gadd(self):
        return self._gadd

    @gadd.setter
    def gadd(self, gadd):
        self._gadd = gadd
        self.compute_width_a()

    @property
    def gass(self):
        return self._gass

    @gass.setter
    def gass(self, gass):
        self._gass = gass
        self.compute_width_a()

    @property
    def gaee(self):
        return self._gaee

    @gaee.setter
    def gaee(self, gaee):
        self._gaee = gaee
        self.compute_width_a()

    @property
    def gamumu(self):
        return self._gamumu

    @gamumu.setter
    def gamumu(self, gamumu):
        self._gamumu = gamumu
        self.compute_width_a()

    def description(self):
        warnings.warn("", PreAlphaWarning)
        pass

    @classmethod
    def list_annihilation_final_states(cls):
        """
        Return a list of the available final states.

        Returns
        -------
        fs : array-like
            Array of the available final states.
        """
        return ["pi0 pi pi"]

    def constraints(self):
        pass

    def constrain(self, p1, p1_vals, p2, p2_vals, ls_or_img="image"):
        pass
