from hazma import muon
from hazma import electron
from hazma import charged_pion
from hazma import neutral_pion
from hazma import charged_kaon
from hazma import long_kaon
from hazma import short_kaon
import numpy as np


def test_muon_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    muon.decay_spectra(eng_gams, 1000.)


def test_electron_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    electron.decay_spectra(eng_gams, 1000.)


def test_charged_pion_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    charged_pion.decay_spectra(eng_gams, 1000.)


def test_neutral_pion_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    neutral_pion.decay_spectra(eng_gams, 1000.)


def test_charged_kaon_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    charged_kaon.decay_spectra(eng_gams, 1000.)


def test_long_kaon_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    long_kaon.decay_spectra(eng_gams, 1000.)


def test_short_kaon_decay_spectra():
    eng_gams = np.logspace(-3., 3., num=100)
    short_kaon.decay_spectra(eng_gams, 1000.)
