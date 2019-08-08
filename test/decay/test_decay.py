from hazma.decay import muon
from hazma.decay import neutral_pion
from hazma.decay import charged_pion
import numpy as np
import unittest

mu_dir = "test/decay/muon_data/"
np_dir = "test/decay/npion_data/"
cp_dir = "test/decay/cpion_data/"


class TestDecay(unittest.TestCase):
    def setUp(self):
        self.load_muon_data()
        self.load_npion_data()
        self.load_cpion_data()

    def load_muon_data(self):
        self.emu1 = np.load(mu_dir + "emu1.npy")
        self.emu2 = np.load(mu_dir + "emu2.npy")
        self.emu3 = np.load(mu_dir + "emu3.npy")

        self.eng_gams = np.load(mu_dir + "egams.npy")

        self.muspec1 = np.load(mu_dir + "spec1.npy")
        self.muspec2 = np.load(mu_dir + "spec2.npy")
        self.muspec3 = np.load(mu_dir + "spec3.npy")

    def load_npion_data(self):
        self.enp1 = np.load(np_dir + "enp1.npy")
        self.enp2 = np.load(np_dir + "enp2.npy")
        self.enp3 = np.load(np_dir + "enp3.npy")

        self.eng_gams = np.load(np_dir + "egams.npy")

        self.npspec1 = np.load(np_dir + "spec1.npy")
        self.npspec2 = np.load(np_dir + "spec2.npy")
        self.npspec3 = np.load(np_dir + "spec3.npy")

    def load_cpion_data(self):
        self.ecp1 = np.load(cp_dir + "ecp1.npy")
        self.ecp2 = np.load(cp_dir + "ecp2.npy")
        self.ecp3 = np.load(cp_dir + "ecp3.npy")

        self.eng_gams = np.load(cp_dir + "egams.npy")

        self.cpspec1 = np.load(cp_dir + "spec1.npy")
        self.cpspec2 = np.load(cp_dir + "spec2.npy")
        self.cpspec3 = np.load(cp_dir + "spec3.npy")

    def test_muon_spec(self):
        spec1 = muon(self.eng_gams, self.emu1)
        spec2 = muon(self.eng_gams, self.emu2)
        spec3 = muon(self.eng_gams, self.emu3)

        for (val1, val2) in zip(self.muspec1, spec1):
            self.assertAlmostEqual(val1, val2, places=3)
        for (val1, val2) in zip(self.muspec2, spec2):
            self.assertAlmostEqual(val1, val2, places=3)
        for (val1, val2) in zip(self.muspec3, spec3):
            self.assertAlmostEqual(val1, val2, places=3)

    def test_npion_spec(self):
        spec1 = neutral_pion(self.eng_gams, self.enp1)
        spec2 = neutral_pion(self.eng_gams, self.enp2)
        spec3 = neutral_pion(self.eng_gams, self.enp3)

        for (val1, val2) in zip(self.npspec1, spec1):
            self.assertAlmostEqual(val1, val2, places=3)
        for (val1, val2) in zip(self.npspec2, spec2):
            self.assertAlmostEqual(val1, val2, places=3)
        for (val1, val2) in zip(self.npspec3, spec3):
            self.assertAlmostEqual(val1, val2, places=3)

    def test_cpion_spec(self):
        spec1 = charged_pion(self.eng_gams, self.ecp1)
        spec2 = charged_pion(self.eng_gams, self.ecp2)
        spec3 = charged_pion(self.eng_gams, self.ecp3)

        for (val1, val2) in zip(self.cpspec1, spec1):
            self.assertAlmostEqual(val1, val2, places=3)
        for (val1, val2) in zip(self.cpspec2, spec2):
            self.assertAlmostEqual(val1, val2, places=3)
        for (val1, val2) in zip(self.cpspec3, spec3):
            self.assertAlmostEqual(val1, val2, places=3)
