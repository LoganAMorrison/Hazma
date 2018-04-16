"""
Functions for computing gamma ray background.
"""
import numpy as np
from scipy.interpolate import interp1d

# Background model from arxiv:1703.02546
e_Bs, dPhi_dEdOmega_Bs = np.transpose(np.loadtxt("background_1703-02546.csv",
                                                 delimiter=","))
# 10^3 required to convert GeV -> MeV
dPhi_dEdOmega_Bs = 1.0e3 * dPhi_dEdOmega_Bs / e_Bs**2

dPhi_dEdOmega_B_default = interp1d(e_Bs, dPhi_dEdOmega_Bs)
