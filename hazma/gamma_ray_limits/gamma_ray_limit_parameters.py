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
A_eff_fermi_lat_rf = resource_filename(__name__,
                                       "fermi_lat_effective_area.dat")
A_eff_egret_rf = resource_filename(__name__, "egret_effective_area.dat")
A_eff_comptel_rf = resource_filename(__name__, "comptel_effective_area.dat")

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

# Fermi-LAT
e_gams_fl, A_effs_fl = np.transpose(np.loadtxt(A_eff_fermi_lat_rf,
                                               delimiter=","))
A_eff_fermi_lat = interp1d(e_gams_fl, A_effs_fl, bounds_error=False,
                           fill_value=0.0)

# EGRET
e_gams_e, A_effs_e = np.transpose(np.loadtxt(A_eff_egret_rf, delimiter=","))
A_eff_egret = interp1d(e_gams_e, A_effs_e, bounds_error=False, fill_value=0.0)

# Fermi-LAT
e_gams_c, A_effs_c = np.transpose(np.loadtxt(A_eff_comptel_rf, delimiter=","))
A_eff_comptel = interp1d(e_gams_c, A_effs_c, bounds_error=False,
                         fill_value=0.0)

# Approximate observing time for experiments in seconds
# e-ASTROGAM
T_obs_e_astrogam = 365. * 24. * 60.**2
# EGRET
T_obs_egret = 1.0e8
# COMPTEL
T_obs_comptel = 1.0e8
# Fermi-LAT
T_obs_fermi_lat = 1.0e8


# Angular sizes (in sr) and J factors (in MeV^2 cm^-5) for various objects
dOmega_dSph = 1.0e-3
J_factor_draco = 1.0e29
# EGRET diffuse target region
dOmega_egret_diffuse = 6.585
J_factor_egret_diffuse = 3.79e27
# Fermi-LAT diffuse region
dOmega_fermi_diffuse = 8.269
J_factor_fermi_diffuse = 3.48e27

# Object to source information
TargetParams = namedtuple("gamma_ray_target_params",
                          ["J_factor", "dOmega"])
# Generic dSph parameters
dSph_params = TargetParams(J_factor_draco, dOmega_dSph)
