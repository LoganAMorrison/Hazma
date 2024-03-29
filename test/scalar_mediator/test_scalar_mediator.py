"""
Test module for the scalar mediator model.
"""

import unittest

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hazma.scalar_mediator import HeavyQuark, ScalarMediator

sm1_dir = "test/scalar_mediator/data/sm_1/"
sm2_dir = "test/scalar_mediator/data/sm_2/"


@pytest.mark.skip(reason="Needs to be updated")
class TestScalarMediator(unittest.TestCase):
    def setUp(self):
        self.load_sm1_data()
        self.load_sm2_data()

        self.sm1 = ScalarMediator(**self.params1)
        self.sm2 = ScalarMediator(**self.params2)

    def tearDown(self):
        pass

    def load_sm1_data(self):
        """
        Loads in the data for the first scalar mediator object.

        Notes
        -----
        This is a Higgs portal-like model with:
            mx         = 250 MeV
            ms         = 550 Mev
            gsxx       = 1
            sin(theta) = 10^-3
            Lambda     = 246 GeV
        We also set the e_cm to:
            e_cm = 2 mx (1 + 0.5 v_rel^2)
        with v_rel = 10^-3.

        """
        self.params1 = np.load(sm1_dir + "params.npy", allow_pickle=True)[()]
        self.e_cm1 = np.load(sm1_dir + "e_cm.npy", allow_pickle=True)
        self.cs1_old = np.load(sm1_dir + "ann_cross_sections.npy", allow_pickle=True)[
            ()
        ]
        self.bf1_old = np.load(
            sm1_dir + "ann_branching_fractions.npy", allow_pickle=True
        )[()]
        self.vs1 = np.load(sm1_dir + "vs.npy", allow_pickle=True)
        self.spec1_old = np.load(sm1_dir + "spectra.npy", allow_pickle=True)[()]
        self.egams1 = np.load(sm1_dir + "e_gams.npy", allow_pickle=True)
        self.ps1_old = np.load(sm1_dir + "partial_widths.npy", allow_pickle=True)[()]
        self.eng_ps1 = np.load(sm1_dir + "e_ps.npy", allow_pickle=True)
        self.pspec1_old = np.load(sm1_dir + "positron_spectra.npy", allow_pickle=True)[
            ()
        ]
        self.lns1_old = np.load(sm1_dir + "positron_lines.npy", allow_pickle=True)[()]

    def load_sm2_data(self):
        """
        Loads in the data for the second scalar mediator object.

        Notes
        -----
        This is a Higgs portal-like model with:
            mx         = 250 MeV
            ms         = 200 Mev
            gsxx       = 1
            sin(theta) = 10^-3
            Lambda     = 246 GeV
        We also set the e_cm to:
            e_cm = 2 mx (1 + 0.5 v_rel^2)
        with v_rel = 10^-3.

        """
        self.params2 = np.load(sm1_dir + "params.npy", allow_pickle=True)[()]
        self.e_cm2 = np.load(sm1_dir + "e_cm.npy", allow_pickle=True)
        self.cs2_old = np.load(sm1_dir + "ann_cross_sections.npy", allow_pickle=True)[
            ()
        ]
        self.bf2_old = np.load(
            sm1_dir + "ann_branching_fractions.npy", allow_pickle=True
        )[()]
        self.vs2 = np.load(sm1_dir + "vs.npy", allow_pickle=True)
        self.spec2_old = np.load(sm1_dir + "spectra.npy", allow_pickle=True)[()]
        self.egams2 = np.load(sm1_dir + "e_gams.npy", allow_pickle=True)
        self.ps2_old = np.load(sm1_dir + "partial_widths.npy", allow_pickle=True)[()]
        self.eng_ps2 = np.load(sm1_dir + "e_ps.npy", allow_pickle=True)
        self.pspec2_old = np.load(sm1_dir + "positron_spectra.npy", allow_pickle=True)[
            ()
        ]
        self.lns2_old = np.load(sm1_dir + "positron_lines.npy", allow_pickle=True)[()]

    def test_list_final_states(self):
        """
        Test that the scalar mediator final state are equal to:
            'mu mu', 'e e', 'g g', 'pi0 pi0', 'pi pi', 's s'
        """
        list_fs = ["mu mu", "e e", "g g", "pi0 pi0", "pi pi", "s s"]
        hq_fss = ["g g", "pi0 pi0", "pi pi", "s s"]

        self.assertEqual(self.sm1.list_annihilation_final_states(), list_fs)
        self.assertEqual(self.sm2.list_annihilation_final_states(), list_fs)

        self.assertEqual(ScalarMediator.list_annihilation_final_states(), list_fs)
        self.assertEqual(HeavyQuark.list_annihilation_final_states(), hq_fss)

    def test_cross_sections(self):
        """
        Test the scalar mediator cross sections for:
            'g g', 'e e', 'pi0 pi0', 'total', 's s', 'mu mu', 'pi pi'
        """
        cs1_new = self.sm1.annihilation_cross_sections(self.e_cm1)
        cs2_new = self.sm2.annihilation_cross_sections(self.e_cm2)

        print(
            self.cs1_old["pi pi"],
            cs1_new["pi pi"],
            round(self.cs1_old["pi pi"] - cs1_new["pi pi"], 5),
        )
        print(
            self.cs2_old["pi pi"],
            cs2_new["pi pi"],
            round(self.cs2_old["pi pi"] - cs2_new["pi pi"], 5),
        )

        print("SM1")
        for key in self.cs1_old.keys():
            print(key, self.cs1_old[key], cs1_new[key])
            assert_allclose(self.cs1_old[key], cs1_new[key], rtol=1e-4, err_msg=key)

        print("SM2")
        for key in self.cs2_old.keys():
            print(key, self.cs2_old[key], cs2_new[key])
            assert_allclose(self.cs2_old[key], cs2_new[key], rtol=1e-4, err_msg=key)

    def test_branching_fractions(self):
        """
        Test the scalar mediator branching fractions for:
            'g g', 'e e', 'pi0 pi0', 'total', 's s', 'mu mu', 'pi pi'
        """
        bf1_new = self.sm1.annihilation_branching_fractions(self.e_cm1)
        bf2_new = self.sm2.annihilation_branching_fractions(self.e_cm2)

        for key in self.bf1_old.keys():
            val1, val2 = self.bf1_old[key], bf1_new[key]
            assert_allclose(val1, val2, rtol=1e-4, err_msg=key)

        for key in self.bf2_old.keys():
            val1, val2 = self.bf2_old[key], bf2_new[key]
            assert_allclose(val1, val2, rtol=1e-4, err_msg=key)

    def test_compute_vs(self):
        """
        Test that the scalar mediator vev is correct.
        """
        self.assertEqual(self.sm1.compute_vs(), self.vs1)
        self.assertEqual(self.sm2.compute_vs(), self.vs2)

    def test_spectra(self):
        """
        Test the scalar mediator spectra for:
            'total', 'e e', 'pi0 pi0', 's s', 'mu mu', 'pi pi'
        """
        spec1_new = self.sm1.spectra(self.egams1, self.e_cm1)
        spec2_new = self.sm2.spectra(self.egams2, self.e_cm2)

        for key in self.spec1_old.keys():
            for (val1, val2) in zip(self.spec1_old[key], spec1_new[key]):
                assert_allclose(val1, val2, rtol=1e-4, err_msg=key)

        for key in self.spec2_old.keys():
            for (val1, val2) in zip(self.spec2_old[key], spec2_new[key]):
                assert_allclose(val1, val2, rtol=1e-4, err_msg=key)

    def test_spectrum_funcs(self):
        self.sm1.spectrum_funcs()
        self.sm2.spectrum_funcs()

    def test_partial_widths(self):
        """
        Test the scalar mediator partial widths for:
            'x x', 'e e', 'pi0 pi0', 'total', 'g g', 'mu mu', 'pi pi'
        """
        ps1_new = self.sm1.partial_widths()
        ps2_new = self.sm2.partial_widths()

        for key in self.ps1_old.keys():
            val1, val2 = self.ps1_old[key], ps1_new[key]
            assert_allclose(val1, val2, rtol=1e-4, err_msg=key)

        for key in self.ps2_old.keys():
            val1, val2 = self.ps2_old[key], ps2_new[key]
            assert_allclose(val1, val2, rtol=1e-4, err_msg=key)

    def test_positron_spectra(self):
        """
        Test the positron spectra for:
            'total', 'mu mu', 'pi pi'
        """
        pspec1_new = self.sm1.positron_spectra(self.eng_ps1, self.e_cm1)
        pspec2_new = self.sm2.positron_spectra(self.eng_ps2, self.e_cm2)

        for key in self.pspec1_old.keys():
            for (val1, val2) in zip(self.pspec1_old[key], pspec1_new[key]):
                assert_allclose(val1, val2, rtol=1e-4, err_msg=key)

        for key in self.pspec2_old.keys():
            for (val1, val2) in zip(self.pspec2_old[key], pspec2_new[key]):
                assert_allclose(val1, val2, rtol=1e-4, err_msg=key)

    def test_positron_lines(self):
        """
        Test the positron lines for 'e e'.
        """
        lns1_new = self.sm1.positron_lines(self.e_cm1)
        lns2_new = self.sm2.positron_lines(self.e_cm2)

        assert_allclose(lns1_new["e e"]["bf"], self.lns1_old["e e"]["bf"], rtol=1e-4)

        assert_allclose(
            lns1_new["e e"]["energy"], self.lns1_old["e e"]["energy"], rtol=1e-4
        )

        assert_allclose(lns2_new["e e"]["bf"], self.lns2_old["e e"]["bf"], rtol=1e-4)

        assert_allclose(
            lns2_new["e e"]["energy"], self.lns2_old["e e"]["energy"], rtol=1e-4
        )
