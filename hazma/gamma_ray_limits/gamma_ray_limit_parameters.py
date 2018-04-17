"""
Parameters relevant to computing constraints from gamma ray experiments
"""
from collections import namedtuple
import numpy as np
from scipy.interpolate import interp1d


# Background model from arxiv:1703.02546
e_Bs, dPhi_dEdOmega_Bs = np.transpose(np.loadtxt("background_1703-02546.csv",
                                                 delimiter=","))
dPhi_dEdOmega_Bs = 1.0e3 * dPhi_dEdOmega_Bs / e_Bs**2  # convert GeV^2 -> MeV^2
dPhi_dEdOmega_B_default = interp1d(e_Bs, dPhi_dEdOmega_Bs)


# e-ASTROGAM's effective area in cm^2 (see arxiv:1711.01265)
e_gams_eA, A_effs_eA = np.transpose(np.loadtxt("e-astrogam_effective_area.csv",
                                               delimiter=","))
A_eff_e_ASTROGAM = interp1d(e_gams_eA, A_effs_eA, fill_value=0.0)


# Reference angular size of dSph galaxy
delta_Omega_dSph = 1.0e-3  # sr
# Approximate observing time for e-ASTROGAM
T_obs_e_ASTROGAM = 365. * 24. * 60.**2  # s
# J factors for various objects, all in MeV^2 cm^-5
J_factor_draco = 1.0e29


# Object to hold gamma ray parameters
ExperimentParams = namedtuple("gamma_ray_exp_params", ["A_eff", "T_obs"])
# e-ASTROGAM's parameters
eASTROGAM_params = ExperimentParams(A_eff_e_ASTROGAM, T_obs_e_ASTROGAM)


# Object to source information
TargetParams = namedtuple("gamma_ray_target_params",
                          ["dPhi_dEdOmega_B", "J_factor", "delta_Omega"])
# Generic dSph parameters
dSph_params = TargetParams(dPhi_dEdOmega_B_default, J_factor_draco,
                           delta_Omega_dSph)
