import unittest

# import pytest
import numpy as np

from hazma.parameters import standard_model_masses as sm_masses
from hazma import rh_neutrino

ME = sm_masses["e"]
MMU = sm_masses["mu"]
MPI0 = sm_masses["pi0"]
MPI = sm_masses["pi"]
MRHO = sm_masses["rho"]
MOMEGA = sm_masses["omega"]
MPHI = sm_masses["phi"]


class TestWidths(unittest.TestCase):
    def setUp(self) -> None:
        masses = {
            "e": [
                0.5109989461,
                53.59568566915,
                120.57308672305,
                137.52909447305,
                175.69906897304998,
                243.18746897305,
                277.09948447305,
                386.66438947305005,
                521.02499947305,
                661.5609999999999,
                775.51549947305,
                779.2154994730499,
                901.0605,
                1398.4159994730499,
            ],
            "mu": [
                0.5109989461,
                53.59568566915,
                120.57308672305,
                173.1467745,
                228.27275674999999,
                262.18477225000004,
                329.67317225,
                464.03378225,
                573.59868725,
                687.2976872500001,
                778.96,
                831.7891872499999,
                950.18968725,
                1450.9896872499999,
            ],
            "tau": [
                0.5109989461,
                67.9993989461,
                173.1467745,
                245.2287645,
                413.50139,
                661.5609999999999,
                778.96,
                901.0605,
                1398.4159994730499,
            ],
        }
        self.models = {
            key: [rh_neutrino.RHNeutrino(mass, 1e-3, key) for mass in ms]
            for key, ms in masses.items()
        }

        self.thresholds = {}
        for key, models in self.models.items():
            self.thresholds[key] = {}
            for states in models[0].list_decay_final_states():
                self.thresholds[key][states] = sum(
                    map(lambda s: sm_masses[s], states.split(" "))
                )

    def test_decay_widths(self):
        """Test RHNeutrino.decay_widths"""

        for key, models in self.models.items():
            for model in models:
                widths = model._decay_widths()
                for states, width in widths.items():
                    if self.thresholds[key][states] > model.mx:
                        assert width == 0.0

    def test_dnde_photon(self):
        """Test RHNeutrino.decay_widths"""

        for _, models in self.models.items():
            for model in models:
                es = np.geomspace(1e-2, 1, 5) * model.mx
                model.spectra(es)
