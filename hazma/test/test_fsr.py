from hazma import muon
from hazma import electron
from hazma import charged_pion
from hazma import neutral_pion
from hazma import charged_kaon
from hazma import long_kaon
from hazma import short_kaon
import numpy as np


def test_muon_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    muon.fsr(eng_gams, 1000.)


def test_electron_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    electron.fsr(eng_gams, 1000.)


def test_charged_pion_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    charged_pion.fsr(eng_gams, 1000.)


def test_neutral_pion_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    neutral_pion.fsr(eng_gams, 1000.)


def test_charged_kaon_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    charged_kaon.fsr(eng_gams, 1000.)


def test_long_kaon_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    long_kaon.fsr(eng_gams, 1000.)


def test_short_kaon_fsr():
    eng_gams = np.logspace(-3., 3., num=100)
    short_kaon.fsr(eng_gams, 1000.)
