from hazma import muon
import numpy as np


def test_muon_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    muon.fsr(eng_gams, 1000.)
