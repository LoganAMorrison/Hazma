import unittest

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hazma import parameters
from hazma.parameters import GF, alpha_em
from hazma.parameters import electron_mass as me
from hazma.parameters import muon_mass as mmu
from hazma.parameters import qe
from hazma.phase_space import Rambo
from hazma.utils import ldot

mw = parameters.wboson_mass
mz = parameters.zboson_mass


class TestRamboProperties(unittest.TestCase):
    def setUp(self):
        self.masses = [1.0, 2.0, 3.0, 4.0]
        self.cme = sum(self.masses) * 2
        self.phase_space = Rambo(cme=self.cme, masses=self.masses)

    def test_momenta(self):
        """Test that total momenta sum to (cme, 0, 0, 0)."""
        momenta, _ = self.phase_space.generate(100, seed=1)
        # Note: shape is (4, #fsp, #batch-size)
        # np.sum -> (4, #batch-size)
        # np.transpose -> (#batch-size, 4)
        total_momenta = np.transpose(np.sum(momenta, axis=1), (1, 0))

        for tot in total_momenta:
            assert tot[0] == pytest.approx(self.cme)
            assert tot[1] == pytest.approx(0.0, abs=1e-8)
            assert tot[2] == pytest.approx(0.0, abs=1e-8)
            assert tot[3] == pytest.approx(0.0, abs=1e-8)


class TestRambo(unittest.TestCase):
    def setUp(self):
        pass

    def test_compute_annihilation_cross_section(self):
        def msqrd_ee_to_mumu(momenta):
            p3 = momenta[:, 0]
            p4 = momenta[:, 1]

            Q = np.sum(momenta, axis=1)[0]
            pi_mag = np.sqrt(Q**2 / 4.0 - me**2)
            p1 = np.zeros_like(p3)
            p2 = np.zeros_like(p3)

            p1[0] = 0.5 * Q
            p2[0] = 0.5 * Q
            p1[3] = pi_mag
            p2[3] = -pi_mag

            return (
                2.0
                * qe**4
                * (
                    ldot(p1, p4) * ldot(p2, p3)
                    + ldot(p1, p3) * ldot(p2, p4)
                    + ldot(p1, p2) * me**2
                    + ldot(p3, p4) * mmu**2
                    + 2.0 * me**2 * mmu**2
                )
            ) / (ldot(p3, p4) + mmu**2) ** 2

        fsp_masses = np.array([mmu, mmu])
        cme = 1000.0

        rambo = Rambo(cme, fsp_masses, msqrd=msqrd_ee_to_mumu).cross_section(
            m1=me, m2=me, n=5000
        )
        analytic = 4.0 * np.pi * alpha_em**2 / (3.0 * cme**2)
        assert_allclose(rambo[0], analytic, rtol=5e-3)

    def test_compute_decay_width_muon(self):
        """
        Test rambo decay width function on mu -> e nu nu.
        """

        def msqrd_mu_to_enunu(momenta):
            """
            Matrix element squared for mu -> e nu nu.
            """
            pe = momenta[:, 0]
            pve = momenta[:, 1]
            pvmu = momenta[:, 2]

            pmu = np.sum(momenta, axis=1)

            return 64.0 * GF**2 * ldot(pe, pvmu) * ldot(pmu, pve)

        fsp_masses = np.array([me, 0.0, 0.0])
        rambo = Rambo(mmu, fsp_masses, msqrd=msqrd_mu_to_enunu).decay_width(n=50_000)

        r = me**2 / mmu**2
        corr_fac = 1.0 - 8.0 * r + 8 * r**3 - r**4 - 12.0 * r**2 * np.log(r)
        analytic = GF**2 * mmu**5 / (192.0 * np.pi**3) * corr_fac

        assert_allclose(rambo[0], analytic, rtol=5e-3)

    def test_compute_decay_width_Zee(self):
        """
        Test rambo decay width function on Z -> e e.
        """
        sw = parameters.sin_theta_weak
        cw = np.sqrt(1.0 - sw**2)
        fsp_masses = np.array([me, me])

        def msqrd_z_to_ee(momenta):
            p1 = momenta[:, 0]
            p2 = momenta[:, 1]

            return (
                qe**2
                * (
                    2 * (1 - 4 * sw**2 + 8 * sw**4) * ldot(p1, p2) ** 2
                    + 2 * (1 - 4 * sw**2 + 8 * sw**4) * me**4
                    + 12 * sw**2 * (-1 + 2 * sw**2) * me**2 * mz**2
                    + (1 - 4 * sw**2 + 8 * sw**4)
                    * ldot(p1, p2)
                    * (4 * me**2 + mz**2)
                )
            ) / (6.0 * cw**2 * sw**2 * mz**2)

        rambo = Rambo(mmu, fsp_masses, msqrd=msqrd_z_to_ee).decay_width(n=10)

        num = qe**2 * (8.0 * sw**4 - 4.0 * sw**2 + 1) * mz
        den = 96.0 * np.pi * cw**2 * sw**2
        analytic = num / den

        assert_allclose(rambo[0], analytic, rtol=5e-3)
