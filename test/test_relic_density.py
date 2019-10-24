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
        self.model1 = ToyModel(10.313897683787216, 1.966877938634266e-9)
        self.model2 = ToyModel(04.74522360006331, 1.7597967261428258e-9)
        self.model3 = ToyModel(1063.764854316313, 1.837766552668581e-9)
        self.model4 = ToyModel(10000.0, 1.8795945459427076e-9)

    def test_relic_density(self):
        rd1 = relic_density(self.model1)
        rd2 = relic_density(self.model1)
        rd3 = relic_density(self.model1)
        rd4 = relic_density(self.model1)
        self.assertAlmostEqual(rd1, omega_h2_cdm, places=3)
        self.assertAlmostEqual(rd2, omega_h2_cdm, places=3)
        self.assertAlmostEqual(rd3, omega_h2_cdm, places=3)
        self.assertAlmostEqual(rd4, omega_h2_cdm, places=3)
