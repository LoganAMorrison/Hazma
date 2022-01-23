import os
import unittest
from os import path
from typing import Callable, Union, Optional, List

import numpy as np
from numpy.testing import assert_allclose

from hazma.decay import muon, neutral_pion
from hazma.decay import charged_pion, charged_pion_decay_modes
from hazma.decay import charged_kaon, charged_kaon_decay_modes
from hazma.decay import short_kaon, short_kaon_decay_modes
from hazma.decay import long_kaon, long_kaon_decay_modes
from hazma.parameters import muon_mass as mmu
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import charged_kaon_mass as mk
from hazma.parameters import neutral_kaon_mass as mk0
from hazma.utils import RealArray

CallableDecay = Union[
    Callable[[RealArray, float], RealArray],
    Callable[[RealArray, float, Optional[List[str]]], RealArray],
]


class TestDecayIntegration(unittest.TestCase):
    def setUp(self):
        self.xs = np.geomspace(1e-4, 1.0, 10)

    def threshold(self, f: CallableDecay, mass: float, *args) -> None:
        """
        Test that computing spectrum at threshold works and results in positive values.

        Parameters
        ----------
        f: Callable[[, float], ]
            Function to compute dN/dE.
        mass: float
            Mass of the particle in MeV.
        """
        egams = self.xs * mass
        dnde = f(egams, mass, *args)

        for val in dnde:
            self.assertGreaterEqual(val, 0.0)

    def boosted(self, f: CallableDecay, mass: float, *args) -> None:
        es = mass * np.array([1.1, 2.0, 3.0])

        for e in es:
            egams = self.xs * e
            dnde = f(egams, e, *args)
            for val in dnde:
                self.assertGreaterEqual(val, 0.0)

    def test_dnde_muon_threshold(self):
        self.threshold(muon, mmu)

    def test_dnde_muon_boosted(self):
        self.boosted(muon, mmu)

    def test_dnde_neutral_pion_threshold(self):
        self.threshold(neutral_pion, mpi0)

    def test_dnde_neutral_pion_boosted(self):
        self.boosted(neutral_pion, mpi0)

    def test_dnde_charged_pion_threshold(self):
        f, m, modes = charged_pion, mpi, charged_pion_decay_modes()
        self.threshold(f, m)
        for mode in modes:
            self.threshold(f, m, [mode])

    def test_dnde_charged_pion_boosted(self):
        f, m, modes = charged_pion, mpi, charged_pion_decay_modes()
        self.boosted(f, m)
        for mode in modes:
            self.boosted(f, m, [mode])

    def test_dnde_charged_kaon_threshold(self):
        f, m, modes = charged_kaon, mk, charged_kaon_decay_modes()
        self.threshold(f, m)
        for mode in modes:
            self.threshold(f, m, [mode])

    def test_dnde_charged_kaon_boosted(self):
        f, m, modes = charged_kaon, mk, charged_kaon_decay_modes()
        self.boosted(f, m)
        for mode in modes:
            self.boosted(f, m, [mode])

    def test_dnde_short_kaon_threshold(self):
        f, m, modes = short_kaon, mk0, short_kaon_decay_modes()
        self.threshold(f, m)
        for mode in modes:
            self.threshold(f, m, [mode])

    def test_dnde_short_kaon_boosted(self):
        f, m, modes = short_kaon, mk0, short_kaon_decay_modes()
        self.boosted(f, m)
        for mode in modes:
            self.boosted(f, m, [mode])

    def test_dnde_long_kaon_threshold(self):
        f, m, modes = long_kaon, mk0, long_kaon_decay_modes()
        self.threshold(f, m)
        for mode in modes:
            self.threshold(f, m, [mode])

    def test_dnde_long_kaon_boosted(self):
        f, m, modes = long_kaon, mk0, long_kaon_decay_modes()
        self.boosted(f, m)
        for mode in modes:
            self.boosted(f, m, [mode])


class TestDecayChange(unittest.TestCase):
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
        """Compares recomputed spectra with reference data."""
        spectra = self.load_data(data_dir)

        def make_error_msg(e_gam, e):
            msg = f"reference spectrum from {data_dir} does not match "
            msg = msg + f"recomputed value at e_gam = {e_gam}, e = {e}."
            return msg

        for e, (e_gams, dnde_ref) in spectra.items():
            # Compute spectrum
            dnde = dnde_func(e_gams, e)
            # Compare
            for e_gam, val, val_ref in zip(e_gams, dnde, dnde_ref):
                assert_allclose(
                    val, val_ref, rtol=1e-5, err_msg=make_error_msg(e_gam, e)
                )

    def test_dnde_muon_change(self):
        self.compare_spectra("mu_data", muon)

    def test_dnde_neutral_pion_change(self):
        self.compare_spectra("pi0_data", neutral_pion)

    def test_dnde_charged_pion_change(self):
        self.compare_spectra("pi_data", charged_pion)
