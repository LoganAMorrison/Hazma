"""
Functions required for computing CMB limits and related quantities.
"""
from scipy.interpolate import interp1d
import numpy as np
from pkg_resources import resource_filename

# Get paths to files inside the module
f_eff_ep_rf = resource_filename(__name__, "cmb_data/f_eff_ep.dat")
f_eff_g_rf = resource_filename(__name__, "cmb_data/f_eff_g.dat")

# Load f_eff^{e+ e-}
f_eff_ep_data = np.loadtxt(f_eff_ep_rf, delimiter=",").T
f_eff_ep = interp1d(f_eff_ep_data[0] / 1.0e6, f_eff_ep_data[1])  # eV -> MeV

# Load f_eff^{e+ e-}
f_eff_g_data = np.loadtxt(f_eff_g_rf, delimiter=",").T
f_eff_g = interp1d(f_eff_g_data[0] / 1.0e6, f_eff_g_data[1])  # eV -> MeV


def vx_cmb(mx, x_kd):
    """Computes the DM relative velocity at CMB.

    Notes
    -----
    This is equation 28 from arXiv:1506.03811 [hep-ph].

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
    temp_cmb = 0.235  # eV

    return 2.0e-4 * temp_cmb / mx * np.sqrt(1.0e-4 / x_kd)
