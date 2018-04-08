class PseudoScalarMediatorParameters(object):

    def __init__(self, mx, mp, gpxx, gpff):
        self._mx = mx
        self._mp = mp
        self._gpxx = gpxx
        self._gpff = gpff

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
    def ms(self, mp):
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
