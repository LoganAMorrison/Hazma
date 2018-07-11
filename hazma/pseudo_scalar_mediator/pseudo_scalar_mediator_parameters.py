from ..parameters import vh, b0, fpi
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import neutral_pion_mass as mpi0
import numpy as np


class PseudoScalarMediatorParameters(object):

    def __init__(self, mx, mp, gpxx, gpuu, gpdd, gpss, gpee, gpmumu, gpGG,
                 gpFF):
        self._mx = mx
        self._mp = mp
        self._gpxx = gpxx
        self._gpuu = gpuu
        self._gpdd = gpdd
        self._gpss = gpss
        self._gpee = gpee
        self._gpmumu = gpmumu
        self._gpGG = gpGG
        self._gpFF = gpFF

        self.determine_mixing()

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx

    @property
    def mp(self):
        return self._mp

    @mp.setter
    def mp(self, mp):
        self._mp = mp

    @property
    def gpxx(self):
        return self._gpxx

    @gpxx.setter
    def gpxx(self, gpxx):
        self._gpxx = gpxx

    @property
    def gpff(self):
        return self._gpff

    @gpff.setter
    def gpff(self, gpff):
        self._gpff = gpff

    def determine_mixing(self):
        eps = b0*fpi*(self.gpuu - self.gpdd + (muq - mdq) / vh * self.gpGG)

        # Mixing angle between pi0 and p. Here I have assumed that the pi0 mass
        # is given by leading order chiPT.
        self._beta = eps / (self.mp**2 - mpi0**2)

        # Shifted mass of neutral pion
        self.mpi0 = np.sqrt(mpi0**2 + eps**2 / (mpi0**2 - self.mp**2))

        if abs(self.mpi0 - mpi0) > 10.:
            print "Warning: your choice of mp and/or couplings produced a " + \
                    "10 MeV or larger shift in m_pi0."
