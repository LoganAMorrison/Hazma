"""
Test module for the scalar mediator model.
"""

from hazma.vector_mediator import VectorMediator, KineticMixing
import numpy as np
import unittest

shr_dir = 'Hazma/test/vector_mediator/shared_data/'
vm1_dir = 'Hazma/test/vector_mediator/vm1_data/'
vm2_dir = 'Hazma/test/vector_mediator/vm2_data/'
vm3_dir = 'Hazma/test/vector_mediator/vm3_data/'
vm4_dir = 'Hazma/test/vector_mediator/vm4_data/'
vm5_dir = 'Hazma/test/vector_mediator/vm5_data/'
vm6_dir = 'Hazma/test/vector_mediator/vm6_data/'


class TestVectorMediator(unittest.TestCase):

    def setUp(self):
        self.load_shared_data()
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

    def load_shared_data(self):
        """
        Load in the shared data.
        """

        self.cme = np.load(shr_dir + 'cme.npy')
        self.eng_gams = np.load(shr_dir + 'spectra_egams.npy')
        self.eng_ps = np.load(shr_dir + 'eng_ps.npy')

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
        self.vm1_bfs = np.load(vm1_dir + 'ann_branching_fractions.npy')[()]
        self.vm1_css = np.load(vm1_dir + 'ann_cross_sections.npy')[()]
        self.vm1_par = np.load(vm1_dir + 'params.npy')[()]
        self.vm1_pws = np.load(vm1_dir + 'partial_widths.npy')[()]
        self.vm1_pos_spec = np.load(vm1_dir + 'positron_spectra.npy')[()]
        self.vm1_pos_line = np.load(vm1_dir + 'ps_lines.npy')[()]
        self.vm1_gam_spec = np.load(vm1_dir + 'spectra.npy')[()]

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
        self.vm2_bfs = np.load(vm2_dir + 'ann_branching_fractions.npy')[()]
        self.vm2_css = np.load(vm2_dir + 'ann_cross_sections.npy')[()]
        self.vm2_par = np.load(vm2_dir + 'params.npy')[()]
        self.vm2_pws = np.load(vm2_dir + 'partial_widths.npy')[()]
        self.vm2_pos_spec = np.load(vm2_dir + 'positron_spectra.npy')[()]
        self.vm2_pos_line = np.load(vm2_dir + 'ps_lines.npy')[()]
        self.vm2_gam_spec = np.load(vm2_dir + 'spectra.npy')[()]

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
        self.vm3_bfs = np.load(vm3_dir + 'ann_branching_fractions.npy')[()]
        self.vm3_css = np.load(vm3_dir + 'ann_cross_sections.npy')[()]
        self.vm3_par = np.load(vm3_dir + 'params.npy')[()]
        self.vm3_pws = np.load(vm3_dir + 'partial_widths.npy')[()]
        self.vm3_pos_spec = np.load(vm3_dir + 'positron_spectra.npy')[()]
        self.vm3_pos_line = np.load(vm3_dir + 'ps_lines.npy')[()]
        self.vm3_gam_spec = np.load(vm3_dir + 'spectra.npy')[()]

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
        self.vm4_bfs = np.load(vm4_dir + 'ann_branching_fractions.npy')[()]
        self.vm4_css = np.load(vm4_dir + 'ann_cross_sections.npy')[()]
        self.vm4_par = np.load(vm4_dir + 'params.npy')[()]
        self.vm4_pws = np.load(vm4_dir + 'partial_widths.npy')[()]
        self.vm4_pos_spec = np.load(vm4_dir + 'positron_spectra.npy')[()]
        self.vm4_pos_line = np.load(vm4_dir + 'ps_lines.npy')[()]
        self.vm4_gam_spec = np.load(vm4_dir + 'spectra.npy')[()]

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
        self.vm5_bfs = np.load(vm5_dir + 'ann_branching_fractions.npy')[()]
        self.vm5_css = np.load(vm5_dir + 'ann_cross_sections.npy')[()]
        self.vm5_par = np.load(vm5_dir + 'params.npy')[()]
        self.vm5_pws = np.load(vm5_dir + 'partial_widths.npy')[()]
        self.vm5_pos_spec = np.load(vm5_dir + 'positron_spectra.npy')[()]
        self.vm5_pos_line = np.load(vm5_dir + 'ps_lines.npy')[()]
        self.vm5_gam_spec = np.load(vm5_dir + 'spectra.npy')[()]

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
        self.vm6_bfs = np.load(vm6_dir + 'ann_branching_fractions.npy')[()]
        self.vm6_css = np.load(vm6_dir + 'ann_cross_sections.npy')[()]
        self.vm6_par = np.load(vm6_dir + 'params.npy')[()]
        self.vm6_pws = np.load(vm6_dir + 'partial_widths.npy')[()]
        self.vm6_pos_spec = np.load(vm6_dir + 'positron_spectra.npy')[()]
        self.vm6_pos_line = np.load(vm6_dir + 'ps_lines.npy')[()]
        self.vm6_gam_spec = np.load(vm6_dir + 'spectra.npy')[()]

    def test_description(self):
        """
        Test that the scalar mediator class desciption is working.
        """
        self.vm1.description()
        self.vm2.description()
        self.vm3.description()
        self.vm4.description()
        self.vm5.description()
        self.vm6.description()

    def test_list_final_states(self):
        """
        Test that the scalar mediator final state are equal to:
            'mu mu', 'e e', 'pi pi', 'pi0 g', 'v v'
        """
        list_fs = ['mu mu', 'e e', 'pi pi', 'pi0 g', 'v v']

        assert self.vm1.list_annihilation_final_states() == list_fs
        assert self.vm2.list_annihilation_final_states() == list_fs
        assert self.vm3.list_annihilation_final_states() == list_fs
        assert self.vm4.list_annihilation_final_states() == list_fs
        assert self.vm5.list_annihilation_final_states() == list_fs
        assert self.vm6.list_annihilation_final_states() == list_fs

    def test_cross_sections(self):
        """
        Test the scalar mediator cross sections for:
            'g g', 'e e', 'pi0 pi0', 'total', 's s', 'mu mu', 'pi pi'
        """
        css1 = self.vm1.annihilation_cross_sections(self.cme)
        css2 = self.vm2.annihilation_cross_sections(self.cme)
        css3 = self.vm3.annihilation_cross_sections(self.cme)
        css4 = self.vm4.annihilation_cross_sections(self.cme)
        css5 = self.vm5.annihilation_cross_sections(self.cme)
        css6 = self.vm6.annihilation_cross_sections(self.cme)

        for key in self.vm1_css.keys():
            assert np.isclose(self.vm1_css[key], css1[key],
                              atol=0.0, rtol=1e-3)
        for key in self.vm2_css.keys():
            assert np.isclose(self.vm2_css[key], css2[key],
                              atol=0.0, rtol=1e-3)
        for key in self.vm3_css.keys():
            assert np.isclose(self.vm3_css[key], css3[key],
                              atol=0.0, rtol=1e-3)
        for key in self.vm4_css.keys():
            assert np.isclose(self.vm4_css[key], css4[key],
                              atol=0.0, rtol=1e-3)
        for key in self.vm5_css.keys():
            assert np.isclose(self.vm5_css[key], css5[key],
                              atol=0.0, rtol=1e-3)
        for key in self.vm6_css.keys():
            assert np.isclose(self.vm6_css[key], css6[key],
                              atol=0.0, rtol=1e-3)

    def test_branching_fractions(self):
        """
        Test the scalar mediator branching fractions for:
            'g g', 'e e', 'pi0 pi0', 'total', 's s', 'mu mu', 'pi pi'
        """
        bfs1 = self.vm1.annihilation_branching_fractions(self.cme)
        bfs2 = self.vm2.annihilation_branching_fractions(self.cme)
        bfs3 = self.vm3.annihilation_branching_fractions(self.cme)
        bfs4 = self.vm4.annihilation_branching_fractions(self.cme)
        bfs5 = self.vm5.annihilation_branching_fractions(self.cme)
        bfs6 = self.vm6.annihilation_branching_fractions(self.cme)

        for key in self.vm1_bfs.keys():
            val1, val2 = self.vm1_bfs[key], bfs1[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm2_bfs.keys():
            val1, val2 = self.vm2_bfs[key], bfs2[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm3_bfs.keys():
            val1, val2 = self.vm3_bfs[key], bfs3[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm4_bfs.keys():
            val1, val2 = self.vm4_bfs[key], bfs4[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm5_bfs.keys():
            val1, val2 = self.vm5_bfs[key], bfs5[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm6_bfs.keys():
            val1, val2 = self.vm6_bfs[key], bfs6[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)

    def test_spectra(self):
        """
        Test the scalar mediator spectra for:
            'total', 'e e', 'pi0 pi0', 's s', 'mu mu', 'pi pi'
        """
        spec1 = self.vm1.spectra(self.eng_gams, self.cme)
        spec2 = self.vm2.spectra(self.eng_gams, self.cme)
        spec3 = self.vm3.spectra(self.eng_gams, self.cme)
        spec4 = self.vm4.spectra(self.eng_gams, self.cme)
        spec5 = self.vm5.spectra(self.eng_gams, self.cme)
        spec6 = self.vm6.spectra(self.eng_gams, self.cme)

        for key in self.vm1_gam_spec.keys():
            for (val1, val2) in zip(self.vm1_gam_spec[key], spec1[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm2_gam_spec.keys():
            for (val1, val2) in zip(self.vm2_gam_spec[key], spec2[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm3_gam_spec.keys():
            for (val1, val2) in zip(self.vm3_gam_spec[key], spec3[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm4_gam_spec.keys():
            for (val1, val2) in zip(self.vm4_gam_spec[key], spec4[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm5_gam_spec.keys():
            for (val1, val2) in zip(self.vm5_gam_spec[key], spec5[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm6_gam_spec.keys():
            for (val1, val2) in zip(self.vm6_gam_spec[key], spec6[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)

    def test_spectrum_functions(self):
        self.vm1.spectrum_functions()
        self.vm2.spectrum_functions()
        self.vm3.spectrum_functions()
        self.vm4.spectrum_functions()
        self.vm5.spectrum_functions()
        self.vm6.spectrum_functions()

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
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm2_pws.keys():
            val1, val2 = self.vm2_pws[key], pws2[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm3_pws.keys():
            val1, val2 = self.vm3_pws[key], pws3[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm4_pws.keys():
            val1, val2 = self.vm4_pws[key], pws4[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm5_pws.keys():
            val1, val2 = self.vm5_pws[key], pws5[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm6_pws.keys():
            val1, val2 = self.vm6_pws[key], pws6[key]
            assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)

    def test_positron_spectra(self):
        """
        Test the positron spectra for:
            'total', 'mu mu', 'pi pi'
        """
        pspec1 = self.vm1.positron_spectra(self.eng_ps, self.cme)
        pspec2 = self.vm2.positron_spectra(self.eng_ps, self.cme)
        pspec3 = self.vm3.positron_spectra(self.eng_ps, self.cme)
        pspec4 = self.vm4.positron_spectra(self.eng_ps, self.cme)
        pspec5 = self.vm5.positron_spectra(self.eng_ps, self.cme)
        pspec6 = self.vm6.positron_spectra(self.eng_ps, self.cme)

        for key in self.vm1_pos_spec.keys():
            for (val1, val2) in zip(self.vm1_pos_spec[key], pspec1[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm2_pos_spec.keys():
            for (val1, val2) in zip(self.vm2_pos_spec[key], pspec2[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm3_pos_spec.keys():
            for (val1, val2) in zip(self.vm3_pos_spec[key], pspec3[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm4_pos_spec.keys():
            for (val1, val2) in zip(self.vm4_pos_spec[key], pspec4[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm5_pos_spec.keys():
            for (val1, val2) in zip(self.vm5_pos_spec[key], pspec5[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)
        for key in self.vm6_pos_spec.keys():
            for (val1, val2) in zip(self.vm6_pos_spec[key], pspec6[key]):
                assert np.isclose(val1, val2, atol=0.0, rtol=1e-3)

    def test_positron_lines(self):
        """
        Test the positron lines for 'e e'.
        """
        ln1 = self.vm1.positron_lines(self.cme)
        ln2 = self.vm2.positron_lines(self.cme)
        ln3 = self.vm3.positron_lines(self.cme)
        ln4 = self.vm4.positron_lines(self.cme)
        ln5 = self.vm5.positron_lines(self.cme)
        ln6 = self.vm6.positron_lines(self.cme)

        for key in self.vm1_pos_line['e e']:
            assert np.isclose(
                ln1['e e'][key],  self.vm1_pos_line['e e'][key],
                atol=0.0, rtol=1e-3)
        for key in self.vm2_pos_line['e e']:
            assert np.isclose(
                ln2['e e'][key],  self.vm2_pos_line['e e'][key],
                atol=0.0, rtol=1e-3)
        for key in self.vm3_pos_line['e e']:
            assert np.isclose(
                ln3['e e'][key],  self.vm3_pos_line['e e'][key],
                atol=0.0, rtol=1e-3)
        for key in self.vm4_pos_line['e e']:
            assert np.isclose(
                ln4['e e'][key],  self.vm4_pos_line['e e'][key],
                atol=0.0, rtol=1e-3)
        for key in self.vm5_pos_line['e e']:
            assert np.isclose(
                ln5['e e'][key],  self.vm5_pos_line['e e'][key],
                atol=0.0, rtol=1e-3)
        for key in self.vm6_pos_line['e e']:
            assert np.isclose(
                ln6['e e'][key],  self.vm6_pos_line['e e'][key],
                atol=0.0, rtol=1e-3)
