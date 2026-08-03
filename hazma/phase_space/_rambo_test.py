"""Tests for the RAMBO phase space generator."""

# pylint: disable=invalid-name

import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hazma import parameters
from hazma.phase_space import Rambo
from hazma.utils import ldot, lnorm_sqr

MW = parameters.wboson_mass
MZ = parameters.zboson_mass
MB = parameters.bottom_quark_mass
MC = parameters.charm_quark_mass
ME = parameters.electron_mass
MMU = parameters.muon_mass

CW = parameters.cos_theta_weak
SW = parameters.sin_theta_weak
EL = parameters.qe
GF = parameters.GF
alpha_em = parameters.alpha_em

SEED = 1234


class TestRamboProperties(unittest.TestCase):
    """Tests for properties of the outputs of Rambo."""

    def setUp(self):
        self.masses = [1.0, 2.0, 3.0, 4.0]
        self.cme = sum(self.masses) * 2
        self.phase_space = Rambo(cme=self.cme, masses=self.masses)
        self.nevents = 100
        self.momenta, self.weights = self.phase_space.generate(self.nevents, seed=SEED)

    def test_momenta(self):
        """Test that total momenta sum to (cme, 0, 0, 0)."""
        # Momenta have shape (4, nfsp, nevents). Summing over axis=1 gives the
        # sum of all final-state particle momenta.
        momentum_sum = np.sum(self.momenta, axis=1)

        # Energies should be equal to center-of-mass energy
        self.assertTrue(np.allclose(momentum_sum[0], self.cme))
        # 3-momenta should be zero
        self.assertTrue(np.allclose(momentum_sum[1], 0.0))
        self.assertTrue(np.allclose(momentum_sum[2], 0.0))
        self.assertTrue(np.allclose(momentum_sum[3], 0.0))

    def test_masses(self):
        """Test the masses of the final state particle momenta."""
        shape = self.momenta.shape

        for i in range(shape[1]):
            momentum = self.momenta[:, i]
            msqr = lnorm_sqr(momentum)
            self.assertTrue(np.allclose(msqr, self.masses[i] ** 2))


class TestRamboCrossSections(unittest.TestCase):
    """Tests for cross sections."""

    def setUp(self):
        pass

    def test_compute_annihilation_cross_section(self):
        """Test the cross section for e^+ e^- => mu^+ + mu^-."""

        def msqrd_ee_to_mumu(momenta):
            p3 = momenta[:, 0]
            p4 = momenta[:, 1]

            Q = np.sum(momenta, axis=1)[0]
            pi_mag = np.sqrt(Q**2 / 4.0 - ME**2)
            p1 = np.zeros_like(p3)
            p2 = np.zeros_like(p3)

            p1[0] = 0.5 * Q
            p2[0] = 0.5 * Q
            p1[3] = pi_mag
            p2[3] = -pi_mag

            return (
                2.0
                * EL**4
                * (
                    ldot(p1, p4) * ldot(p2, p3)
                    + ldot(p1, p3) * ldot(p2, p4)
                    + ldot(p1, p2) * ME**2
                    + ldot(p3, p4) * MMU**2
                    + 2.0 * ME**2 * MMU**2
                )
            ) / (ldot(p3, p4) + MMU**2) ** 2

        fsp_masses = np.array([MMU, MMU])
        cme = 1000.0

        rambo = Rambo(cme, fsp_masses, msqrd=msqrd_ee_to_mumu).cross_section(
            m1=ME,
            m2=ME,
            n=5000,
            seed=SEED,
        )
        analytic = 4.0 * np.pi * alpha_em**2 / (3.0 * cme**2)
        assert_allclose(rambo[0], analytic, rtol=5e-3)


class TestRamboDecayWidths(unittest.TestCase):
    """Tests for decay widths."""

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

        fsp_masses = np.array([ME, 0.0, 0.0])
        rambo = Rambo(MMU, fsp_masses, msqrd=msqrd_mu_to_enunu).decay_width(
            n=50_000, seed=SEED
        )

        r = ME**2 / MMU**2
        corr_fac = 1.0 - 8.0 * r + 8 * r**3 - r**4 - 12.0 * r**2 * np.log(r)
        analytic = GF**2 * MMU**5 / (192.0 * np.pi**3) * corr_fac

        assert_allclose(rambo[0], analytic, rtol=5e-3)


@pytest.mark.parametrize(
    "mf,ncf,t3f,qf",
    [
        (ME, 1.0, -0.5, -1.0),
        (MB, 3.0, -0.5, -1.0 / 3.0),
        (MC, 3.0, 0.5, 2.0 / 3.0),
    ],
)
def test_width_z_to_f_f(mf, ncf, t3f, qf):
    """Compute the decay width of the Z into two fermions."""
    pre = EL / (SW * CW)
    gl = pre * (t3f - qf * SW**2)
    gr = -pre * qf * SW**2

    def msqrd_z_to_f_f(_):
        return (
            ncf * 2 / 3 * (6 * gl * gr * mf**2 + (gl**2 + gr**2) * (-(mf**2) + MZ**2))
        )

    mr2 = (mf / MZ) ** 2
    af = t3f - qf * SW**2
    bf = -qf * SW**2

    analytic = (
        ncf
        * EL**2
        * MZ
        / (24 * np.pi * CW**2 * SW**2)
        * np.sqrt(1 - 4 * mr2)
        * ((af**2 + bf**2) * (1 - mr2) + 6 * af * bf * mr2)
    )

    phase_space = Rambo(MZ, [mf, mf], msqrd=msqrd_z_to_f_f)
    width = phase_space.decay_width(10_000, seed=SEED)[0]

    assert width == pytest.approx(analytic)
