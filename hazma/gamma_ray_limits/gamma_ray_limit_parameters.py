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


# Reference angular size of dSph galaxy
delta_Omega_dSph = 1.0e-3  # sr
# J factors for various objects, all in MeV^2 cm^-5
J_factor_draco = 1.0e29

# Object to source information
TargetParams = namedtuple("gamma_ray_target_params",
                          ["dPhi_dEdOmega_B", "J_factor", "delta_Omega"])
# Generic dSph parameters
dSph_params = TargetParams(dPhi_dEdOmega_B_default, J_factor_draco,
                           delta_Omega_dSph)
