from hazma import muon
import numpy as np

eng_gams = np.logspace(-3.0, 3.0, num=150)

muon.decay_spectra(eng_gams, 1000.)
