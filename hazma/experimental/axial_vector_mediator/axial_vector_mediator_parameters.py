class AxialVectorMediatorParameters:
    def __init__(self, mx, ma, gaxx, gaff):
        self._mx = mx
        self._ma = ma
        self._gaxx = gaxx
        self._gaff = gaff

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx

    @property
    def ma(self):
        return self._ma

    @ma.setter
    def ms(self, ma):
        self._ma = ma

    @property
    def gaxx(self):
        return self._gaxx

    @gaxx.setter
    def gaxx(self, gaxx):
        self._gaxx = gaxx

    @property
    def gaff(self):
        return self._gaff

    @gaff.setter
    def gaff(self, gaff):
        self._gaff = gaff
