import os
import unittest
from os import path

import numpy as np
from numpy.testing import assert_allclose

from hazma.decay import charged_pion, muon, neutral_pion


class TestDecay(unittest.TestCase):
    def setUp(self):
        self.base_dir = path.dirname(__file__)

    def load_data(self, data_dir):
        """Loads test data.

        Arguments
        ---------
        data_dir : str
            Directory containing test data relative to this file.

        Returns
        -------
        spectra : dict(float, (np.array, np.array))
            Reference data. The keys are the decaying particle's energies and
            the values are a tuple of photon energies and spectrum values,
            assumed to be sorted by photon energy.
        """
        # Count number of tests. There are two files for each test (one
        # containing the particle's energy, the other with the spectrum values)
        # and one file containing the photon energies.
        data_dir = path.join(self.base_dir, data_dir)
        n_tests = (len(os.listdir(data_dir)) - 1) // 2

        # Load energies, spectrum values and photon energies
        e_gams = np.load(path.join(data_dir, "e_gams.npy"))

        spectra = {}

        for i in range(1, n_tests + 1):
            e = float(np.load(path.join(data_dir, "e_{}.npy".format(i))))
            spectra[e] = e_gams, np.load(path.join(data_dir, "dnde_{}.npy".format(i)))

        return spectra

    def compare_spectra(self, data_dir, dnde_func):
        """Compares recomputed spectra with reference data.
        """
        spectra = self.load_data(data_dir)

        for e, (e_gams, dnde_ref) in spectra.items():
            # Compute spectrum
            dnde = dnde_func(e_gams, e)
            # Compare
            for e_gam, val, val_ref in zip(e_gams, dnde, dnde_ref):
                assert_allclose(
                    val,
                    val_ref,
                    rtol=1e-5,
                    err_msg="reference spectrum from {} does not match recomputed value at e_gam = {}".format(
                        data_dir, e_gam
                    ),
                )

    def test_dnde_muon(self):
        self.compare_spectra("mu_data", muon)

    def test_dnde_neutral_pion(self):
        self.compare_spectra("pi0_data", neutral_pion)

    def test_dnde_charged_pion(self):
        self.compare_spectra("pi_data", charged_pion)
