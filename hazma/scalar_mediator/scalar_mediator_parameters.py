import numpy as np

from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq
from ..parameters import fpi, b0, vh
from scalar_mediator_widths import partial_widths


class ScalarMediatorParameters(object):
    def __init__(self, mx, ms, gsxx, gsff, gsGG, gsFF):
        self._mx = mx
        self._ms = ms
        self._gsxx = gsxx
        self._gsff = gsff
        self._gsGG = gsGG
        self._gsFF = gsFF
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def ms(self):
        return self._ms

    @ms.setter
    def ms(self, ms):
        self._ms = ms
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsxx(self):
        return self._gsxx

    @gsxx.setter
    def gsxx(self, gsxx):
        self._gsxx = gsxx
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsff(self):
        return self._gsff

    @gsff.setter
    def gsff(self, gsff):
        self._gsff = gsff
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsGG(self):
        return self._gsGG

    @gsGG.setter
    def gsGG(self, gsGG):
        self._gsGG = gsGG
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    @property
    def gsFF(self):
        return self._gsFF

    @gsFF.setter
    def gsFF(self, gsFF):
        self._gsFF = gsFF
        self.compute_vs()
        self.compute_width_s()  # vs MUST be computed first

    def compute_vs(self):
        """Updates and returns the value of the scalar vev.
        """
        if 3*self.gsff + 2*self.gsGG == 0:
            self.vs = 0.
        else:
            trM = muq + mdq + msq

            ms = self.ms
            gsff = self.gsff
            gsGG = self.gsGG

            self.vs = (-3*ms*vh +
                       np.sqrt(4*b0*fpi**2*(3*gsff + 2*gsGG)**2*trM +
                               9*ms**2*vh**2)) / (6*gsff*ms + 4*gsGG*ms)

            return self.vs

    def compute_width_s(self):
        """Updates the scalar's total width.
        """
        self.width_s = partial_widths(self)["total"]

    # #################### #
    """ HELPER FUNCTIONS """
    # #################### #

    def fpiT(self, vs):
        """Returns the Lagrangian parameter fpiT.
        """
        return fpi / np.sqrt(1. + 4.*self._gsGG*vs / (9.*vh))

    def b0T(self, vs, fpiT):
        """Returns the Lagrangian parameter b0T.
        """
        return b0 * (fpi/fpiT)**2 / (1. +
                                     vs/vh * (2.*self._gsGG/3. + self._gsff))

    def msT(self, fpiT, b0T):
        """Returns the Lagrangian parameter msT.
        """
        trM = muq + mdq + msq

        return np.sqrt(self._ms**2 -
                       16.*self._gsGG*b0T*fpiT**2 / (81.*vh**2) *
                       (2.*self._gsGG - 9.*self._gsff) * trM)
