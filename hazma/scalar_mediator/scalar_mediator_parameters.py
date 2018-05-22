from math import sqrt

from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq
from ..parameters import fpi, b0, vh
from scalar_mediator_widths import partial_widths
trM = muq + mdq + msq


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
        """
        Returns the value of the scalar vev.
        """
        vs_roots = self.__vs_roots()
        fpiTs = [self.fpiT(vs) for vs in vs_roots]
        kappas = [self.__kappa(fpiT) for fpiT in fpiTs]
        BTs = [self.BT(kappa) for kappa in kappas]
        msTs = [self.msT(fpiT, BT) for (fpiT, BT) in zip(fpiTs, BTs)]
        alphas = [self.__alpha(fpiT, BT, msT)
                  for (fpiT, BT, msT) in zip(fpiTs, BTs, msTs)]
        betas = [self.__beta(fpiT, BT, msT)
                 for (fpiT, BT, msT) in zip(fpiTs, BTs, msTs)]

        potvals = [- alpha * vs + 0.5 * beta * vs **
                   2 for (alpha, beta, vs) in zip(alphas, betas, vs_roots)]

        if potvals[0] < potvals[1]:
            self.vs = vs_roots[0]
        else:
            self.vs = vs_roots[1]

    def compute_width_s(self):
        """Recomputes the scalar's total width."""
        self.width_s = partial_widths(self)["total"]

    # #################### #
    """ HELPER FUNCTIONS """
    # #################### #

    def fpiT(self, vs):
        """
        Returns the unphysical value of fpi.
        """
        return fpi / sqrt(1.0 + 4. * self._gsGG * vs / 9. / vh)

    def BT(self, kappa):
        """
        Returns the unphysical value of B.
        """
        if self.__is_vev_zero():
            return b0
        else:
            return b0 * (1 + kappa) /  \
                (1 + 6. * kappa * (1. + 3. * self._gsff / (2.*self._gsGG)))

    def msT(self, fpiT, BT):
        """
        Returns the unphysical mass of the scalar mediator.
        """
        gamma = BT * fpiT * trM / vh
        return sqrt(self._ms**2 +
                    16. * gamma * self._gsff * self._gsGG / 9. / vh -
                    32. * gamma * self._gsGG**2 / 81. / vh)

    def __alpha(self, fT, BT, msT):
        """
        Returns coefficent of linear term in the scalar potential before adding
        scalar vev.
        """
        return (BT * fT**2 * (self._gsff + (2 * self._gsGG) / 3.) * trM) / vh

    def __beta(self, fT, BT, msT):
        """
        Returns curvature of the scalar potential.
        """
        return msT**2 - (16 * BT * fT**2 * self._gsff * self._gsGG * trM) / \
            (9. * vh**2) + (32 * BT * fT**2 *
                            self._gsGG**2 * trM) / (81. * vh**2)

    def __vs_roots(self):
        """
        Returns the two possible values of the scalar potential.
        """
        if self.__is_vev_zero():
            return 0., 0.
        else:
            root1 = (-3 * self._ms * sqrt(trM) * vh +
                     sqrt(4 * b0 * fpi**2 *
                          (3 * self._gsff + 2 * self._gsGG)**2 +
                          9 * self._ms**2 * trM * vh**2)) / \
                (2. * (3 * self._gsff + 2 * self._gsGG) * self._ms * sqrt(trM))
            root2 = (-3 * self._ms * sqrt(trM) * vh -
                     sqrt(4 * b0 * fpi**2 *
                          (3 * self._gsff + 2 * self._gsGG)**2 +
                          9 * self._ms**2 * trM * vh**2)) / \
                (2. * (3 * self._gsff + 2 * self._gsGG) * self._ms * sqrt(trM))

            return root1, root2

    def __kappa(self, fpiT):
        return fpi**2 / fpiT**2 - 1.

    def __is_vev_zero(self):
        """Checks whether the scalar's vev is zero

        Returns
        -------
        True if 2 gsGG + 3 gsff == 0, False otherwise.
        """
        return 2. * self._gsGG + 3. * self._gsff == 0.
