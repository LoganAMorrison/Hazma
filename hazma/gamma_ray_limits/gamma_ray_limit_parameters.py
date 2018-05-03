"""
Parameters relevant to computing constraints from gamma ray experiments
"""
from collections import namedtuple
import numpy as np
from pkg_resources import resource_filename
from scipy.interpolate import interp1d


# Get paths to files inside the module
bg_rf = resource_filename(__name__, "background_1703-02546.dat")
A_eff_e_astrogam_rf = resource_filename(__name__,
                                        "e-astrogam_effective_area.dat")

# Background model from arxiv:1703.02546
e_Bs, dPhi_dEdOmega_Bs = np.transpose(np.loadtxt(bg_rf, delimiter=","))
dPhi_dEdOmega_Bs = 1.0e3 * dPhi_dEdOmega_Bs / e_Bs**2  # convert GeV^2 -> MeV^2
dPhi_dEdOmega_B_default = interp1d(e_Bs, dPhi_dEdOmega_Bs)

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
