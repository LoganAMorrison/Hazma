from hazma.decay import muon
from hazma.decay import charged_pion
from hazma.decay import neutral_pion
from hazma.decay import charged_kaon
from hazma.decay import long_kaon
from hazma.decay import short_kaon
import numpy as np


def test_muon_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    muon(eng_gams, 1000.)


def test_charged_pion_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    charged_pion(eng_gams, 1000.)


def test_neutral_pion_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    neutral_pion(eng_gams, 1000.)


def test_charged_kaon_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    charged_kaon(eng_gams, 1000.)


def test_long_kaon_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    long_kaon(eng_gams, 1000.)


def test_short_kaon_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    short_kaon(eng_gams, 1000.)
