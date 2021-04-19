import numpy as np


class FormFactorPiGamma:
    def __init__(self, gvuu, gvdd, gvss):

        self._ci0 = 3.0 * (gvuu + gvdd)
        self._ci1 = gvuu - gvdd
        self._cs = -3.0 * gvss

        self._res_masses = [0.77526, 0.78284, 1.01952, 1.45, 1.70]
        self._res_widths = [0.1491, 0.00868, 0.00421, 0.40, 0.30]
        self._amp = [0.0426, 0.0434, 0.00303, 0.00523, 0.0]
        self._phase = [-12.7, 0.0, 158.0, 180.0, 0.0]
        self._mpi = 0.13957061

        self._c_rho = 1.0
        self._c_omega = 1.0
        self._c_phi = 1.0
        self._c_rho_om_phi = [
            self._ci1,
            self._ci0,
            self._cs,
            self._ci0,
            self._ci0
        ]

    def _widths(self, q2, ix):
        if ix == 0:
            widths = (
                self._res_widths[0]
                * self._res_masses[0] ** 2
                / q2
                * (
                    (q2 - 4.0 * self._mpi ** 2)
                    / (self._res_masses[0] ** 2 - 4.0 * self._mpi ** 2)
                )**1.5
            )
        else:
            widths = self._res_widths[ix]
        return widths

    def __call__(self, q2):
        Q = np.sqrt(q2)
        ii = 0.0 + 1.0j
        form = 0.0
        for i in range(0, len(self._res_masses)):
            Di = self._res_masses[i] ** 2 - q2 - ii * Q * self._widths(q2, i)
            form += (
                self._c_rho_om_phi[i]
                * self._amp[i]
                * self._res_masses[i] ** 2
                * np.exp(ii * np.radians(self._phase[i]))
                / Di
            )
        return form
