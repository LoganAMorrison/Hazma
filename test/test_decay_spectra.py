from hazma.decay import muon
from hazma.decay import charged_pion
from hazma.decay import neutral_pion
from hazma.decay import charged_kaon
from hazma.decay import long_kaon
from hazma.decay import short_kaon
import numpy as np

import unittest


class TestDecay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.eng_gams = np.logspace(-3., 3., num=100)

    def tearDown(self):
        pass

    def test_muon_decay(self):
        muon(self.eng_gams, 1000.)

    def test_muon_decay_spectra(self):
        muon(self.eng_gams, 1000.)

    def test_charged_pion_decay_spectra(self):
        charged_pion(self.eng_gams, 1000.)

    def test_neutral_pion_decay_spectra(self):
        neutral_pion(self.eng_gams, 1000.)

    def test_charged_kaon_decay_spectra(self):
        charged_kaon(self.eng_gams, 1000.)

    def test_long_kaon_decay_spectra(self):
        long_kaon(self.eng_gams, 1000.)

    def test_short_kaon_decay_spectra(self):
        short_kaon(self.eng_gams, 1000.)


if __name__ == '__main__':
    unittest.main()
