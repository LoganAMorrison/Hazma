from hazma.relic_density import relic_density
from hazma.parameters import omega_h2_cdm
import warnings

import unittest

warnings.filterwarnings("ignore")


class ToyModel(object):
    def __init__(self, mx, sigmav):
        self.mx = mx
        self.sigmav = sigmav

    def thermal_cross_section(self, x):
        """
        Compute the dark matter thermal cross section for a given
        ratio of mass to temperature.

        Parameters
        ----------
        x: float
            DM mass over temperature.

        Returns
        -------
        sigmav: float
            Dark matter thermmal cross section.
        """
        return self.sigmav


class TestRelicDensity(unittest.TestCase):

    def setUp(self):
        mx1, sigmav1 = 10.313897683787216e3, 1.966877938634266e-15
        mx2, sigmav2 = 104.74522360006331e3, 1.7597967261428258e-15
        mx3, sigmav3 = 1063.764854316313e3, 1.837766552668581e-15
        mx4, sigmav4 = 10000.0e3, 1.8795945459427076e-15

        self.models = [ToyModel(mx1, sigmav1), ToyModel(mx2, sigmav2),
                       ToyModel(mx3, sigmav3), ToyModel(mx4, sigmav4)]

    def test_relic_density(self):
        for model in self.models:
            rd = relic_density(model)
            fractional_diff = abs(rd - omega_h2_cdm) / omega_h2_cdm
            # check that our result is within 1% of correct relic density
            self.assertLessEqual(fractional_diff, 1e-2)
