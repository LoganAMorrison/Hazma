"""
Test module for the scalar mediator model.
"""

from hazma.vector_mediator import VectorMediator, KineticMixing, QuarksOnly
import numpy as np
import unittest

vm1_dir = "test/vector_mediator/data/vm_1/"
vm2_dir = "test/vector_mediator/data/vm_2/"
vm3_dir = "test/vector_mediator/data/vm_3/"
vm4_dir = "test/vector_mediator/data/vm_4/"
vm5_dir = "test/vector_mediator/data/vm_5/"
vm6_dir = "test/vector_mediator/data/vm_6/"


class TestVectorMediator(unittest.TestCase):
    def setUp(self):
        self.load_vm1_data()
        self.load_vm2_data()
        self.load_vm3_data()
        self.load_vm4_data()
        self.load_vm5_data()
        self.load_vm6_data()

        self.vm1 = KineticMixing(**self.vm1_par)
        self.vm2 = KineticMixing(**self.vm2_par)
        self.vm3 = VectorMediator(**self.vm3_par)
        self.vm4 = VectorMediator(**self.vm4_par)
        self.vm5 = VectorMediator(**self.vm5_par)
        self.vm6 = VectorMediator(**self.vm6_par)

    def tearDown(self):
        pass

    def load_vm1_data(self):
        """
        Loads in the data for the first vector mediator object.

        Notes
        -----
        This is a kinetic mixing-like model with:
            mx   = 250.
            gvxx = 1.
            eps  = 0.1
        """
        self.cme1 = np.load(vm1_dir + "e_cm.npy", allow_pickle=True)
        self.eng_gams1 = np.load(vm1_dir + "e_gams.npy", allow_pickle=True)
        self.eng_ps1 = np.load(vm1_dir + "e_ps.npy", allow_pickle=True)

        self.vm1_bfs = np.load(
            vm1_dir + "ann_branching_fractions.npy", allow_pickle=True
        )[()]
        self.vm1_css = np.load(vm1_dir + "ann_cross_sections.npy", allow_pickle=True)[
            ()
        ]
        self.vm1_par = np.load(vm1_dir + "params.npy", allow_pickle=True)[()]
        self.vm1_pws = np.load(vm1_dir + "partial_widths.npy", allow_pickle=True)[()]
        self.vm1_pos_spec = np.load(
            vm1_dir + "positron_spectra.npy", allow_pickle=True
        )[()]
        self.vm1_pos_line = np.load(vm1_dir + "positron_lines.npy", allow_pickle=True)[
            ()
        ]
        self.vm1_gam_spec = np.load(vm1_dir + "spectra.npy", allow_pickle=True)[()]

    def load_vm2_data(self):
        """
        Loads in the data for the first vector mediator object.

        Notes
        -----
        This is a kinetic mixing-like model with:
            mx   = 250.
            gvxx = 1.
            eps  = 0.1
        """
        self.cme2 = np.load(vm2_dir + "e_cm.npy", allow_pickle=True)
        self.eng_gams2 = np.load(vm2_dir + "e_gams.npy", allow_pickle=True)
        self.eng_ps2 = np.load(vm2_dir + "e_ps.npy", allow_pickle=True)

        self.vm2_bfs = np.load(
            vm2_dir + "ann_branching_fractions.npy", allow_pickle=True
        )[()]
        self.vm2_css = np.load(vm2_dir + "ann_cross_sections.npy", allow_pickle=True)[
            ()
        ]
        self.vm2_par = np.load(vm2_dir + "params.npy", allow_pickle=True)[()]
        self.vm2_pws = np.load(vm2_dir + "partial_widths.npy", allow_pickle=True)[()]
        self.vm2_pos_spec = np.load(
            vm2_dir + "positron_spectra.npy", allow_pickle=True
        )[()]
        self.vm2_pos_line = np.load(vm2_dir + "positron_lines.npy", allow_pickle=True)[
            ()
        ]
        self.vm2_gam_spec = np.load(vm2_dir + "spectra.npy", allow_pickle=True)[()]

    def load_vm3_data(self):
        """
        Loads in the data for the first vector mediator object.

        Notes
        -----
        This is a kinetic mixing-like model with:
            mx   = 250.
            gvxx = 1.
            eps  = 0.1
        """
        self.cme3 = np.load(vm3_dir + "e_cm.npy", allow_pickle=True)
        self.eng_gams3 = np.load(vm3_dir + "e_gams.npy", allow_pickle=True)
        self.eng_ps3 = np.load(vm3_dir + "e_ps.npy", allow_pickle=True)

        self.vm3_bfs = np.load(
            vm3_dir + "ann_branching_fractions.npy", allow_pickle=True
        )[()]
        self.vm3_css = np.load(vm3_dir + "ann_cross_sections.npy", allow_pickle=True)[
            ()
        ]
        self.vm3_par = np.load(vm3_dir + "params.npy", allow_pickle=True)[()]
        self.vm3_pws = np.load(vm3_dir + "partial_widths.npy", allow_pickle=True)[()]
        self.vm3_pos_spec = np.load(
            vm3_dir + "positron_spectra.npy", allow_pickle=True
        )[()]
        self.vm3_pos_line = np.load(vm3_dir + "positron_lines.npy", allow_pickle=True)[
            ()
        ]
        self.vm3_gam_spec = np.load(vm3_dir + "spectra.npy", allow_pickle=True)[()]

    def load_vm4_data(self):
        """
        Loads in the data for the first vector mediator object.

        Notes
        -----
        This is a kinetic mixing-like model with:
            mx   = 250.
            gvxx = 1.
            eps  = 0.1
        """
        self.cme4 = np.load(vm4_dir + "e_cm.npy", allow_pickle=True)
        self.eng_gams4 = np.load(vm4_dir + "e_gams.npy", allow_pickle=True)
        self.eng_ps4 = np.load(vm4_dir + "e_ps.npy", allow_pickle=True)

        self.vm4_bfs = np.load(
            vm4_dir + "ann_branching_fractions.npy", allow_pickle=True
        )[()]
        self.vm4_css = np.load(vm4_dir + "ann_cross_sections.npy", allow_pickle=True)[
            ()
        ]
        self.vm4_par = np.load(vm4_dir + "params.npy", allow_pickle=True)[()]
        self.vm4_pws = np.load(vm4_dir + "partial_widths.npy", allow_pickle=True)[()]
        self.vm4_pos_spec = np.load(
            vm4_dir + "positron_spectra.npy", allow_pickle=True
        )[()]
        self.vm4_pos_line = np.load(vm4_dir + "positron_lines.npy", allow_pickle=True)[
            ()
        ]
        self.vm4_gam_spec = np.load(vm4_dir + "spectra.npy", allow_pickle=True)[()]

    def load_vm5_data(self):
        """
        Loads in the data for the first vector mediator object.

        Notes
        -----
        This is a kinetic mixing-like model with:
            mx   = 250.
            gvxx = 1.
            eps  = 0.1
        """
        self.cme5 = np.load(vm5_dir + "e_cm.npy", allow_pickle=True)
        self.eng_gams5 = np.load(vm5_dir + "e_gams.npy", allow_pickle=True)
        self.eng_ps5 = np.load(vm5_dir + "e_ps.npy", allow_pickle=True)

        self.vm5_bfs = np.load(
            vm5_dir + "ann_branching_fractions.npy", allow_pickle=True
        )[()]
        self.vm5_css = np.load(vm5_dir + "ann_cross_sections.npy", allow_pickle=True)[
            ()
        ]
        self.vm5_par = np.load(vm5_dir + "params.npy", allow_pickle=True)[()]
        self.vm5_pws = np.load(vm5_dir + "partial_widths.npy", allow_pickle=True)[()]
        self.vm5_pos_spec = np.load(
            vm5_dir + "positron_spectra.npy", allow_pickle=True
        )[()]
        self.vm5_pos_line = np.load(vm5_dir + "positron_lines.npy", allow_pickle=True)[
            ()
        ]
        self.vm5_gam_spec = np.load(vm5_dir + "spectra.npy", allow_pickle=True)[()]

    def load_vm6_data(self):
        """
        Loads in the data for the first vector mediator object.

        Notes
        -----
        This is a kinetic mixing-like model with:
            mx   = 250.
            gvxx = 1.
            eps  = 0.1
        """
        self.cme6 = np.load(vm6_dir + "e_cm.npy", allow_pickle=True)
        self.eng_gams6 = np.load(vm6_dir + "e_gams.npy", allow_pickle=True)
        self.eng_ps6 = np.load(vm6_dir + "e_ps.npy", allow_pickle=True)

        self.vm6_bfs = np.load(
            vm6_dir + "ann_branching_fractions.npy", allow_pickle=True
        )[()]
        self.vm6_css = np.load(vm6_dir + "ann_cross_sections.npy", allow_pickle=True)[
            ()
        ]
        self.vm6_par = np.load(vm6_dir + "params.npy", allow_pickle=True)[()]
        self.vm6_pws = np.load(vm6_dir + "partial_widths.npy", allow_pickle=True)[()]
        self.vm6_pos_spec = np.load(
            vm6_dir + "positron_spectra.npy", allow_pickle=True
        )[()]
        self.vm6_pos_line = np.load(vm6_dir + "positron_lines.npy", allow_pickle=True)[
            ()
        ]
        self.vm6_gam_spec = np.load(vm6_dir + "spectra.npy", allow_pickle=True)[()]

    def test_list_final_states(self):
        """
        Test that the scalar mediator final state are equal to:
            'mu mu', 'e e', 'pi pi', 'pi0 g', 'v v'
        """
        list_fs = ["mu mu", "e e", "pi pi", "pi0 g", "pi0 v", "v v"]
        quark_only_fss = ["pi pi", "pi0 g", "pi0 v", "v v"]

        self.assertEqual(self.vm1.list_annihilation_final_states(), list_fs)
        self.assertEqual(self.vm2.list_annihilation_final_states(), list_fs)
        self.assertEqual(self.vm3.list_annihilation_final_states(), list_fs)
        self.assertEqual(self.vm4.list_annihilation_final_states(), list_fs)
        self.assertEqual(self.vm5.list_annihilation_final_states(), list_fs)
        self.assertEqual(self.vm6.list_annihilation_final_states(), list_fs)

        self.assertEqual(VectorMediator.list_annihilation_final_states(), list_fs)
        self.assertEqual(KineticMixing.list_annihilation_final_states(), list_fs)
        self.assertEqual(QuarksOnly.list_annihilation_final_states(), quark_only_fss)

    def test_cross_sections(self):
        """
        Test the scalar mediator cross sections for:
            'g g', 'e e', 'pi0 pi0', 'total', 's s', 'mu mu', 'pi pi'
        """
        css1 = self.vm1.annihilation_cross_sections(self.cme1)
        css2 = self.vm2.annihilation_cross_sections(self.cme2)
        css3 = self.vm3.annihilation_cross_sections(self.cme3)
        css4 = self.vm4.annihilation_cross_sections(self.cme4)
        css5 = self.vm5.annihilation_cross_sections(self.cme5)
        css6 = self.vm6.annihilation_cross_sections(self.cme6)

        for key in self.vm1_css.keys():
            self.assertAlmostEqual(self.vm1_css[key], css1[key], places=3)
        for key in self.vm2_css.keys():
            self.assertAlmostEqual(self.vm2_css[key], css2[key], places=3)
        for key in self.vm3_css.keys():
            self.assertAlmostEqual(self.vm3_css[key], css3[key], places=3)
        for key in self.vm4_css.keys():
            self.assertAlmostEqual(self.vm4_css[key], css4[key], places=3)
        for key in self.vm5_css.keys():
            self.assertAlmostEqual(self.vm5_css[key], css5[key], places=3)
        for key in self.vm6_css.keys():
            self.assertAlmostEqual(self.vm6_css[key], css6[key], places=3)

    def test_branching_fractions(self):
        """
        Test the scalar mediator branching fractions for:
            'g g', 'e e', 'pi0 pi0', 'total', 's s', 'mu mu', 'pi pi'
        """
        bfs1 = self.vm1.annihilation_branching_fractions(self.cme1)
        bfs2 = self.vm2.annihilation_branching_fractions(self.cme2)
        bfs3 = self.vm3.annihilation_branching_fractions(self.cme3)
        bfs4 = self.vm4.annihilation_branching_fractions(self.cme4)
        bfs5 = self.vm5.annihilation_branching_fractions(self.cme5)
        bfs6 = self.vm6.annihilation_branching_fractions(self.cme6)

        for key in self.vm1_bfs.keys():
            val1, val2 = self.vm1_bfs[key], bfs1[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm2_bfs.keys():
            val1, val2 = self.vm2_bfs[key], bfs2[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm3_bfs.keys():
            val1, val2 = self.vm3_bfs[key], bfs3[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm4_bfs.keys():
            val1, val2 = self.vm4_bfs[key], bfs4[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm5_bfs.keys():
            val1, val2 = self.vm5_bfs[key], bfs5[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm6_bfs.keys():
            val1, val2 = self.vm6_bfs[key], bfs6[key]
            self.assertAlmostEqual(val1, val2, places=3)

    def test_spectra(self):
        """
        Test the scalar mediator spectra for:
            'total', 'e e', 'pi0 pi0', 's s', 'mu mu', 'pi pi'
        """
        spec1 = self.vm1.spectra(self.eng_gams1, self.cme1)
        spec2 = self.vm2.spectra(self.eng_gams2, self.cme2)
        spec3 = self.vm3.spectra(self.eng_gams3, self.cme3)
        spec4 = self.vm4.spectra(self.eng_gams4, self.cme4)
        spec5 = self.vm5.spectra(self.eng_gams5, self.cme5)
        spec6 = self.vm6.spectra(self.eng_gams6, self.cme6)

        for key in self.vm1_gam_spec.keys():
            for (val1, val2) in zip(self.vm1_gam_spec[key], spec1[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm2_gam_spec.keys():
            for (val1, val2) in zip(self.vm2_gam_spec[key], spec2[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm3_gam_spec.keys():
            for (val1, val2) in zip(self.vm3_gam_spec[key], spec3[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm4_gam_spec.keys():
            for (val1, val2) in zip(self.vm4_gam_spec[key], spec4[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm5_gam_spec.keys():
            for (val1, val2) in zip(self.vm5_gam_spec[key], spec5[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm6_gam_spec.keys():
            for (val1, val2) in zip(self.vm6_gam_spec[key], spec6[key]):
                self.assertAlmostEqual(val1, val2, places=3)

    def test_spectrum_funcs(self):
        self.vm1.spectrum_funcs()
        self.vm2.spectrum_funcs()
        self.vm3.spectrum_funcs()
        self.vm4.spectrum_funcs()
        self.vm5.spectrum_funcs()
        self.vm6.spectrum_funcs()

    def test_partial_widths(self):
        """
        Test the scalar mediator partial widths for:
        """
        pws1 = self.vm1.partial_widths()
        pws2 = self.vm2.partial_widths()
        pws3 = self.vm3.partial_widths()
        pws4 = self.vm4.partial_widths()
        pws5 = self.vm5.partial_widths()
        pws6 = self.vm6.partial_widths()

        for key in self.vm1_pws.keys():
            val1, val2 = self.vm1_pws[key], pws1[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm2_pws.keys():
            val1, val2 = self.vm2_pws[key], pws2[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm3_pws.keys():
            val1, val2 = self.vm3_pws[key], pws3[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm4_pws.keys():
            val1, val2 = self.vm4_pws[key], pws4[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm5_pws.keys():
            val1, val2 = self.vm5_pws[key], pws5[key]
            self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm6_pws.keys():
            val1, val2 = self.vm6_pws[key], pws6[key]
            self.assertAlmostEqual(val1, val2, places=3)

    def test_positron_spectra(self):
        """
        Test the positron spectra for:
            'total', 'mu mu', 'pi pi'
        """
        pspec1 = self.vm1.positron_spectra(self.eng_ps1, self.cme1)
        pspec2 = self.vm2.positron_spectra(self.eng_ps2, self.cme2)
        pspec3 = self.vm3.positron_spectra(self.eng_ps3, self.cme3)
        pspec4 = self.vm4.positron_spectra(self.eng_ps4, self.cme4)
        pspec5 = self.vm5.positron_spectra(self.eng_ps5, self.cme5)
        pspec6 = self.vm6.positron_spectra(self.eng_ps6, self.cme6)

        for key in self.vm1_pos_spec.keys():
            for (val1, val2) in zip(self.vm1_pos_spec[key], pspec1[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm2_pos_spec.keys():
            for (val1, val2) in zip(self.vm2_pos_spec[key], pspec2[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm3_pos_spec.keys():
            for (val1, val2) in zip(self.vm3_pos_spec[key], pspec3[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm4_pos_spec.keys():
            for (val1, val2) in zip(self.vm4_pos_spec[key], pspec4[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm5_pos_spec.keys():
            for (val1, val2) in zip(self.vm5_pos_spec[key], pspec5[key]):
                self.assertAlmostEqual(val1, val2, places=3)
        for key in self.vm6_pos_spec.keys():
            for (val1, val2) in zip(self.vm6_pos_spec[key], pspec6[key]):
                self.assertAlmostEqual(val1, val2, places=3)

    def test_positron_lines(self):
        """
        Test the positron lines for 'e e'.
        """
        ln1 = self.vm1.positron_lines(self.cme1)
        ln2 = self.vm2.positron_lines(self.cme2)
        ln3 = self.vm3.positron_lines(self.cme3)
        ln4 = self.vm4.positron_lines(self.cme4)
        ln5 = self.vm5.positron_lines(self.cme5)
        ln6 = self.vm6.positron_lines(self.cme6)

        for key in self.vm1_pos_line["e e"]:
            self.assertAlmostEqual(
                ln1["e e"][key], self.vm1_pos_line["e e"][key], places=3
            )
        for key in self.vm2_pos_line["e e"]:
            self.assertAlmostEqual(
                ln2["e e"][key], self.vm2_pos_line["e e"][key], places=3
            )
        for key in self.vm3_pos_line["e e"]:
            self.assertAlmostEqual(
                ln3["e e"][key], self.vm3_pos_line["e e"][key], places=3
            )
        for key in self.vm4_pos_line["e e"]:
            self.assertAlmostEqual(
                ln4["e e"][key], self.vm4_pos_line["e e"][key], places=3
            )
        for key in self.vm5_pos_line["e e"]:
            self.assertAlmostEqual(
                ln5["e e"][key], self.vm5_pos_line["e e"][key], places=3
            )
        for key in self.vm6_pos_line["e e"]:
            self.assertAlmostEqual(
                ln6["e e"][key], self.vm6_pos_line["e e"][key], places=3
            )
