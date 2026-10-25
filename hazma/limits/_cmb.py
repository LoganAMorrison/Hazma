import importlib.resources as importlib_resources

from scipy.interpolate import interp1d
import numpy as np

from hazma.parameters import temp_cmb_formation

from ._abstract import AbstractLimit

"""
Functions required for computing CMB limits and related quantities.
"""

# Get paths to files inside the module
_f_eff_ep_ref = importlib_resources.files("hazma.limits.data") / "f_eff_ep.dat"
_f_eff_g_ref = importlib_resources.files("hazma.limits.data") / "f_eff_g.dat"

# Load f_eff^{e+ e-}
with importlib_resources.as_file(_f_eff_ep_ref) as path:
    f_eff_ep_data = np.loadtxt(path, delimiter=",").T
    f_eff_ep = interp1d(f_eff_ep_data[0] / 1.0e6, f_eff_ep_data[1])  # eV -> MeV

# Load f_eff^{e+ e-}
with importlib_resources.as_file(_f_eff_g_ref) as path:
    f_eff_g_data = np.loadtxt(path, delimiter=",").T
    f_eff_g = interp1d(f_eff_g_data[0] / 1.0e6, f_eff_g_data[1])  # eV -> MeV


#: Planck 2018 95% upper limit on p_ann from temperature + polarization
#: measurements, in cm^3 s^-1 MeV^-1
p_ann_planck_temp_pol = 3.5e-31  # temperature + polarization
#: Planck 2018 95% upper limit on p_ann from temperature + polarization +
#: lensing measurements, in cm^3 s^-1 MeV^-1
p_ann_planck_temp_pol_lensing = 3.3e-31  # temp + pol + lensing
#: Planck 2018 95% upper limit on p_ann from temperature + polarization +
#: lensing + BAO measurements, in cm^3 s^-1 MeV^-1
p_ann_planck_temp_pol_lensing_bao = 3.2e-31  # temp + pol + lensing + BAO


class CMBLimit(AbstractLimit):
    def __init__(self, x_kd=1e-4, p_ann=p_ann_planck_temp_pol):
        super().__init__()
        self._x_kd = x_kd
        self._p_ann = p_ann

    @property
    def x_kd(self):
        return self._x_kd

    @property
    def p_ann(self):
        return self._p_ann

    @property
    def description(self):
        return "[purple] CMB"

    @property
    def name(self):
        return "cmb"

    @staticmethod
    def vx_cmb(mx, x_kd):
        """Computes the DM relative velocity at CMB using eq. 28 from `this
        reference <https://arxiv.org/abs/1309.4091>`_.

        Parameters
        ----------
        mx : float
            Dark matter mass in MeV.
        x_kd: float
            T_kd / m_x, where T_kd is the dark matter's kinetic decoupling
            temperature.

        Returns
        -------
        v_x : float
            The DM relative velocity at the time of CMB formation.
        """
        return 2.0e-4 * 10e6 * temp_cmb_formation / mx * np.sqrt(1.0e-4 / x_kd)

    def _constrain(self, model):
        r"""
        Computes the CMB limit on <sigma v> with the model's ``cmb_limit``.

        :meth:`hazma.theory.TheoryCMB.cmb_limit` states the condition and the
        population factor it assumes.

        Parameters
        ----------
        model : TheoryAnn
            Dark matter model to constrain.

        Returns
        -------
        <sigma v> : float
            Upper bound on <sigma v>, in cm^3 s^-1.
        """
        return model.cmb_limit(x_kd=self.x_kd, p_ann=self.p_ann)

    def f_eff_g(self, model):
        return model.f_eff_g(self.x_kd)

    def f_eff_ep(self, model):
        return model.f_eff_ep(self.x_kd)

    def f_eff(self, model):
        r"""
        Computes :math:`f_{\mathrm{eff}}` the efficiency with which dark matter
        annihilations around recombination inject energy into the thermal
        plasma.
        """
        return model.f_eff(self.x_kd)
