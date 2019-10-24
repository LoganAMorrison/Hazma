from hazma.relic_density import relic_density
from hazma.parameters import omega_h2_cdm
import warnings

import unittest

warnings.filterwarnings("ignore")


class ToyModel(object):
    def __init__(self, mx, sigmav):
        self.mx = mx
        self.sigmav = sigmav

    def annihilation_cross_sections(self, ecm):
        """
        Compute the dark matter annihilation cross section for a given
        center of mass energy.

        Parameters
        ----------
        ecm: float
            Center of mass energy.

        Returns
        -------
        sigmav: float
            Dark matter annihilation cross section.
        """
        return {'total': self.sigmav}


class TestRelicDensity(unittest.TestCase):

    def setUp(self):
        mx1, sigmav1 = 10.313897683787216e3, 1.966877938634266e-3
        mx2, sigmav2 = 104.74522360006331e3, 1.7597967261428258e-3
        mx3, sigmav3 = 1063.764854316313e3, 1.837766552668581e-3
        mx4, sigmav4 = 10000.0e3, 1.8795945459427076e-3

        self.model1 = ToyModel(mx1, sigmav1)
        self.model2 = ToyModel(mx2, sigmav2)
        self.model3 = ToyModel(mx3, sigmav3)
        self.model4 = ToyModel(mx4, sigmav4)

    def test_relic_density(self):
        rd1 = relic_density(self.model1)
        rd2 = relic_density(self.model1)
        rd3 = relic_density(self.model1)
        rd4 = relic_density(self.model1)
        self.assertAlmostEqual(rd1, omega_h2_cdm, places=3)
        self.assertAlmostEqual(rd2, omega_h2_cdm, places=3)
        self.assertAlmostEqual(rd3, omega_h2_cdm, places=3)
        self.assertAlmostEqual(rd4, omega_h2_cdm, places=3)
