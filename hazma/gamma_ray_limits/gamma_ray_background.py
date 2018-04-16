"""
Functions for computing gamma ray background.
"""
import numpy as np
from scipy.interpolate import interp1d


# Background model from arxiv:1703.02546
e_Bs, dPhi_dE_dOmega_Bs = np.transpose(np.loadtxt("background_1703-02546.csv",
                                                  delimiter=","))
# 10^3 required to convert GeV -> MeV
dPhi_dE_dOmega_Bs = 1.0e3 * dPhi_dE_dOmega_Bs / e_Bs**2

dPhi_dE_dOmega_B_default = interp1d(e_Bs, dPhi_dE_dOmega_Bs)
