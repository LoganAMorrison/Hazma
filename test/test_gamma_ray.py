from hazma import gamma_ray
import warnings
import numpy as np

import unittest

warnings.filterwarnings("ignore")


class TestGammaRay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.cme = 5000.0
        self.eng_gams = np.logspace(0.0, np.log10(5000), num=10, dtype=np.float64)

    def tearDown(self):
        pass

    def test_mu_mu_mu(self):
        particles = np.array(["muon", "muon", "muon"])
        gamma_ray.gamma_ray_decay(particles, self.cme, self.eng_gams)

    def test_ck_mu_mu(self):
        particles = np.array(["charged_kaon", "muon", "muon"])
        gamma_ray.gamma_ray_decay(particles, self.cme, self.eng_gams)

    def test_sk_mu_mu(self):
        particles = np.array(["short_kaon", "muon", "muon"])
        gamma_ray.gamma_ray_decay(particles, self.cme, self.eng_gams)

    def test_lk_mu_mu(self):
        particles = np.array(["long_kaon", "muon", "muon"])
        gamma_ray.gamma_ray_decay(particles, self.cme, self.eng_gams)

    def test_cp_mu_mu(self):
        particles = np.array(["charged_pion", "muon", "muon"])
        gamma_ray.gamma_ray_decay(particles, self.cme, self.eng_gams)

    def test_np_mu_mu(self):
        particles = np.array(["neutral_pion", "muon", "muon"])
        gamma_ray.gamma_ray_decay(particles, self.cme, self.eng_gams)
