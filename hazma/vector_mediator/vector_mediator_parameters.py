class VectorMediatorParameters(object):

    def __init__(self, mx, mv, gvxx, gvff):
        self._mx = mx
        self._mv = mv
        self._gvxx = gvxx
        self._gvff = gvff

    @property
    def mx(self):
        return self._mx

    @mx.setter
    def mx(self, mx):
        self._mx = mx

    @property
    def mv(self):
        return self._mv

    @mv.setter
    def ms(self, mv):
        self._mv = mv

    @property
    def gvxx(self):
        return self._gvxx

    @gvxx.setter
    def gvxx(self, gvxx):
        self._gvxx = gvxx

    @property
    def gvff(self):
        return self._gvff

    @gvff.setter
    def gvff(self, gvff):
        self._gvff = gvff
