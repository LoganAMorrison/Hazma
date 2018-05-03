"""
Parameters relevant to computing constraints from gamma ray experiments
"""
from collections import namedtuple
import numpy as np
from pkg_resources import resource_filename
from scipy.interpolate import interp1d


# Get paths to files inside the module
A_eff_e_astrogam_rf = resource_filename(__name__,
                                        "e-astrogam_effective_area.dat")


def dPhi_dEdOmega_B_default(e):
    """Simple model for the gamma ray background, d^2Phi / dE dOmega.

    Notes
    -----
    This is the background model from arXiv:1504.04024, eq. 14. It was derived
    by performing a simple power law fit to COMPTEL data from 0.8 - 30 MeV and
    EGRET data from 30 MeV - 10 GeV.

    Parameters
    ----------
    e : float
        Photon energy in MeV. Must be between 0.8 MeV and 10 GeV.

    Returns
    -------
    d^2Phi / dE dOmega : float
        The gamma ray background.
    """
    if 0.8 <= e and e <= 10e3:
        return 2.74e-3 / e**2
    else:
        raise ValueError("The gamma ray background model is not applicable " +
                         "for energy %f MeV." % e)

# Effective areas in cm^2
# e-ASTROGAM
e_gams_e_a, A_effs_e_a = np.transpose(np.loadtxt(A_eff_e_astrogam_rf,
                                                 delimiter=","))
A_eff_e_astrogam = interp1d(e_gams_e_a, A_effs_e_a, bounds_error=False,
                            fill_value=0.0)

# Approximate observing time for experiments in seconds
# e-ASTROGAM
T_obs_e_astrogam = 365. * 24. * 60.**2


# Angular sizes (in sr) and J factors (in MeV^2 cm^-5) for various objects
TargetParams = namedtuple("gamma_ray_target_params",
                          ["J_factor", "dOmega"])
# Dwarf with high J factor
draco_params = TargetParams(1.0e29, 1.0e-3)
# EGRET diffuse target region
egret_diffuse_region_params = TargetParams(3.79e27, 6.585)
# Fermi-LAT diffuse region
fermi_diffuse_region_params = TargetParams(3.48e27, 8.269)
