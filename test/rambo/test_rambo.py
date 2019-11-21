import unittest

import numpy as np
from numpy.testing import assert_allclose

from hazma.field_theory_helper_functions.common_functions import minkowski_dot as MDot
from hazma.parameters import GF, alpha_em
from hazma.parameters import electron_mass as me
from hazma.parameters import muon_mass as mmu
from hazma.parameters import qe
from hazma.rambo import compute_annihilation_cross_section, compute_decay_width

mw = 80.385 * 10 ** 3  # W-mass
mz = 91.1876 * 10 ** 3  # Z-Mass


class TestRambo(unittest.TestCase):
    def setUp(self):
        pass

    def test_compute_annihilation_cross_section(self):
        def msqrd_ee_to_mumu(momenta):
            p3 = momenta[0]
            p4 = momenta[1]

            P = sum(momenta)
            Q = P[0]
            pi_mag = np.sqrt(Q ** 2 / 4.0 - me ** 2)

            p1 = np.array([Q / 2.0, 0.0, 0.0, pi_mag])
            p2 = np.array([Q / 2.0, 0.0, 0.0, -pi_mag])

            return (
                2.0
                * qe ** 4
                * (
                    MDot(p1, p4) * MDot(p2, p3)
                    + MDot(p1, p3) * MDot(p2, p4)
                    + MDot(p1, p2) * me ** 2
                    + MDot(p3, p4) * mmu ** 2
                    + 2.0 * me ** 2 * mmu ** 2
                )
            ) / (MDot(p3, p4) + mmu ** 2) ** 2

        isp_masses = np.array([me, me])
        fsp_masses = np.array([mmu, mmu])
        cme = 1000.0

        rambo = compute_annihilation_cross_section(
            isp_masses, fsp_masses, cme, num_ps_pts=5000, mat_elem_sqrd=msqrd_ee_to_mumu
        )

        analytic = 4.0 * np.pi * alpha_em ** 2 / (3.0 * cme ** 2)

        assert_allclose(rambo[0], analytic, rtol=5e-3)

    def test_compute_decay_width_muon(self):
        """
        Test rambo decay width function on mu -> e nu nu.
        """

        def msqrd_mu_to_enunu(momenta):
            """
            Matrix element squared for mu -> e nu nu.
            """
            pe = momenta[0]
            pve = momenta[1]
            pvmu = momenta[2]

            pmu = sum(momenta)

            return 64.0 * GF ** 2 * MDot(pe, pvmu) * MDot(pmu, pve)

        fsp_masses = np.array([me, 0.0, 0.0])

        rambo = compute_decay_width(
            fsp_masses, mmu, num_ps_pts=50000, mat_elem_sqrd=msqrd_mu_to_enunu
        )
        r = me ** 2 / mmu ** 2
        corr_fac = 1.0 - 8.0 * r + 8 * r ** 3 - r ** 4 - 12.0 * r ** 2 * np.log(r)
        analytic = GF ** 2 * mmu ** 5 / (192.0 * np.pi ** 3) * corr_fac

        assert_allclose(rambo[0], analytic, rtol=5e-3)

    def test_compute_decay_width_Zee(self):
        """
        Test rambo decay width function on Z -> e e.
        """
        sw = np.sqrt(0.2223)
        cw = np.sqrt(1.0 - sw ** 2)
        fsp_masses = np.array([me, me])

        def msqd_Z_to_ee(momenta):

            p1 = momenta[0]
            p2 = momenta[1]

            return (
                qe ** 2
                * (
                    2 * (1 - 4 * sw ** 2 + 8 * sw ** 4) * MDot(p1, p2) ** 2
                    + 2 * (1 - 4 * sw ** 2 + 8 * sw ** 4) * me ** 4
                    + 12 * sw ** 2 * (-1 + 2 * sw ** 2) * me ** 2 * mz ** 2
                    + (1 - 4 * sw ** 2 + 8 * sw ** 4)
                    * MDot(p1, p2)
                    * (4 * me ** 2 + mz ** 2)
                )
            ) / (6.0 * cw ** 2 * sw ** 2 * mz ** 2)

        rambo = compute_decay_width(
            fsp_masses, mz, num_ps_pts=10, mat_elem_sqrd=msqd_Z_to_ee
        )

        num = qe ** 2 * (8.0 * sw ** 4 - 4.0 * sw ** 2 + 1) * mz
        den = 96.0 * np.pi * cw ** 2 * sw ** 2
        analytic = num / den

        assert_allclose(rambo[0], analytic, rtol=5e-3)
