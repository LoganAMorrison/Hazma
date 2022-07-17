from scipy.interpolate import interp1d
import numpy as np
from pkg_resources import resource_filename
from hazma.parameters import temp_cmb_formation

"""
Functions required for computing CMB limits and related quantities.
"""

# Get paths to files inside the module
f_eff_ep_rf = resource_filename(__name__, "cmb_data/f_eff_ep.dat")
f_eff_g_rf = resource_filename(__name__, "cmb_data/f_eff_g.dat")

# Load f_eff^{e+ e-}
f_eff_ep_data = np.loadtxt(f_eff_ep_rf, delimiter=",").T
f_eff_ep = interp1d(f_eff_ep_data[0] / 1.0e6, f_eff_ep_data[1])  # eV -> MeV

# Load f_eff^{e+ e-}
f_eff_g_data = np.loadtxt(f_eff_g_rf, delimiter=",").T
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
