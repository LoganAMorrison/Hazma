import unittest
import functools as ft

from pytest import approx

import numpy as np
from numpy import testing as np_testing

from hazma.parameters import standard_model_masses as sm_masses
from hazma import spectra
from hazma.utils import lnorm_sqr


def dnde_zero_factory(neutrino: bool):
    shape = (3,) if neutrino else tuple()

    def dnde_zero(energies, _):
        scalar = np.isscalar(energies)
        es = np.atleast_1d(energies).astype(float)
        zs = np.zeros(shape + es.shape, dtype=es.dtype)

        if scalar:
            return np.take(zs, 0, axis=-1)
        return zs

    return dnde_zero


photon_spectra_dict = {
    "mu": spectra.dnde_photon_muon,
    "pi0": spectra.dnde_photon_neutral_pion,
    "pi": spectra.dnde_photon_charged_pion,
    "eta": spectra.dnde_photon_eta,
    "etap": spectra.dnde_photon_eta_prime,
    "k": spectra.dnde_photon_charged_kaon,
    "kl": spectra.dnde_photon_long_kaon,
    "ks": spectra.dnde_photon_short_kaon,
    "rho": spectra.dnde_photon_charged_rho,
    "rho0": spectra.dnde_photon_neutral_rho,
    "omega": spectra.dnde_photon_omega,
    "phi": spectra.dnde_photon_phi,
}

fsr_dict = {
    "e": ft.partial(spectra.dnde_photon_ap_fermion, mass=sm_masses["e"], charge=-1.0),
    "mu": ft.partial(spectra.dnde_photon_ap_fermion, mass=sm_masses["mu"], charge=-1.0),
    "pi": ft.partial(spectra.dnde_photon_ap_scalar, mass=sm_masses["pi"], charge=1.0),
    "k": ft.partial(spectra.dnde_photon_ap_scalar, mass=sm_masses["k"], charge=1.0),
}

positron_spectra_dict = {
    "k": spectra.dnde_positron_charged_kaon,
    "pi": spectra.dnde_positron_charged_pion,
    "rho": spectra.dnde_positron_charged_rho,
    "eta": spectra.dnde_positron_eta,
    "etap": spectra.dnde_positron_eta_prime,
    "kl": spectra.dnde_positron_long_kaon,
    "mu": spectra.dnde_positron_muon,
    "rho0": spectra.dnde_positron_neutral_rho,
    "rho": spectra.dnde_positron_charged_rho,
    "omega": spectra.dnde_positron_omega,
    "phi": spectra.dnde_positron_phi,
    "ks": spectra.dnde_positron_short_kaon,
}

neutrino_spectra_dict = {
    "k": spectra.dnde_neutrino_charged_kaon,
    "pi": spectra.dnde_neutrino_charged_pion,
    "rho": spectra.dnde_neutrino_charged_rho,
    "eta": spectra.dnde_neutrino_eta,
    "etap": spectra.dnde_neutrino_eta_prime,
    "kl": spectra.dnde_neutrino_long_kaon,
    "mu": spectra.dnde_neutrino_muon,
    "rho0": spectra.dnde_neutrino_neutral_rho,
    "rho": spectra.dnde_neutrino_charged_rho,
    "omega": spectra.dnde_neutrino_omega,
    "phi": spectra.dnde_neutrino_phi,
    "ks": spectra.dnde_neutrino_short_kaon,
}

spectra_dict = {
    "photon": photon_spectra_dict,
    "positron": positron_spectra_dict,
    "neutrino": neutrino_spectra_dict,
}


class TestIndividualSpectra(unittest.TestCase):
    def test_single_arg(self):
        """Test spectra with single argument."""
        ratio = 2.0

        for product, dnde_dict in spectra_dict.items():
            if product in ["photon", "neutrino"]:
                eprod = 0.1
            else:
                eprod = sm_masses["e"] * 1.1

            for state, dnde_fn in dnde_dict.items():
                mass = sm_masses[state]
                estate = ratio * mass
                dnde = dnde_fn(eprod, estate)

                if product == "neutrino":
                    assert not np.isscalar(dnde)
                    assert np.isscalar(dnde[0])
                    assert np.isscalar(dnde[1]), f"{state}"
                    assert np.isscalar(dnde[2]), f"{state}"
                else:
                    assert np.isscalar(dnde)

    def test_single_array_arg(self):
        """Test spectra with length 1 array argument (should retain array)."""
        ratio = 2.0

        for product, dnde_dict in spectra_dict.items():
            if product in ["photon", "neutrino"]:
                eprod = 0.1
            else:
                eprod = sm_masses["e"] * 1.1

            for state, dnde_fn in dnde_dict.items():
                mass = sm_masses[state]
                estate = ratio * mass
                dnde = dnde_fn(np.array([eprod]), estate)

                if product == "neutrino":
                    assert not np.isscalar(dnde)
                    assert not np.isscalar(dnde[0])
                    assert not np.isscalar(dnde[1])
                    assert not np.isscalar(dnde[2])
                else:
                    assert not np.isscalar(dnde)

    def test_supthreshold(self):
        """Test spectra below the mass threshold of decaying particle."""
        ratio = 2.0

        for dnde_dict in spectra_dict.values():
            for state, dnde_fn in dnde_dict.items():
                mass = sm_masses[state]
                energies = np.geomspace(mass / 10.0, mass * 10, 10)
                estate = ratio * mass
                dnde_fn(energies, estate)

    def test_subthreshold(self):
        """Test spectra below the mass threshold of decaying particle."""

        def err_msg(state, product, energy):
            return f"Non-zero for {state} -> {product}, {state}-energy={energy}"

        for product, dnde_dict in spectra_dict.items():
            for state, dnde_fn in dnde_dict.items():
                mass = sm_masses[state]
                energies = np.geomspace(mass / 10.0, mass * 10, 10)
                estate = mass / 2.0
                dnde = dnde_fn(energies, estate)
                np_testing.assert_allclose(
                    dnde, np.zeros_like(dnde), err_msg=err_msg(state, product, estate)
                )


class TestNBodySpectra(unittest.TestCase):
    r"""Tests for hazma.spectra.dnde_photon, hazma.spectra.dnde_positron,
    hazma.spectra.dnde_neutrino.
    """

    def setUp(self):
        self.dnde_dict = {
            "photon": spectra.dnde_photon,
            "positron": spectra.dnde_positron,
            "neutrino": spectra.dnde_neutrino,
        }

        self.states = {
            "k",
            "pi",
            "rho",
            "eta",
            "etap",
            "kl",
            "mu",
            "rho0",
            "rho",
            "omega",
            "phi",
            "ks",
        }

    def test_single_state(self):
        """Test that passing a single state works."""

        for state in self.states:
            for product, dnde_fn in self.dnde_dict.items():
                mass = sm_masses[state]
                nbody = dnde_fn(1.0, mass, state)
                single = spectra_dict[product][state](1.0, mass)
                if product == "neutrino":
                    for i in range(3):
                        assert nbody[i] == approx(single[i], rel=1e-2, abs=1e-2)
                else:
                    assert nbody == approx(single, rel=1e-2, abs=1e-2)

    def test_two_body_final_state(self):
        """Test that passing two states works."""

        for s1 in self.states:
            for s2 in self.states:
                for product, dnde_fn in self.dnde_dict.items():
                    sd = spectra_dict[product]
                    m1 = sm_masses[s1]
                    m2 = sm_masses[s2]
                    cme = (m1 + m2) * 1.1
                    eprod = 1.0
                    if product == "photon":
                        nbody = dnde_fn(eprod, cme, (s1, s2), include_fsr=False)
                    else:
                        nbody = dnde_fn(eprod, cme, (s1, s2))

                    e1 = (cme**2 + m1**2 - m2**2) / (2 * cme)
                    e2 = (cme**2 - m1**2 + m2**2) / (2 * cme)
                    single = sd[s1](eprod, e1) + sd[s2](eprod, e2)

                    if product == "neutrino":
                        for i in range(3):
                            assert nbody[i] == approx(single[i], rel=1e-2, abs=1e-2)
                    else:
                        assert nbody == approx(single, rel=1e-2, abs=1e-2)

    def test_n_body_final_state_no_msqrd(self):
        """Just check that n-body final state work."""

        final_states = [
            ("omega", "phi", "rho"),
            ("pi0", "k", "pi", "e"),
            ("mu", "mu", "e", "e", "eta"),
        ]

        for fs in final_states:
            for _, dnde_fn in self.dnde_dict.items():
                masses = [sm_masses[s] for s in fs]
                cme = 2.0 * sum(masses)
                eprod = 2.0
                dnde_fn(eprod, cme, fs)

    def test_n_body_final_state_msqrd(self):
        """Just check that n-body final states work (n=3,4,5)."""

        final_states = [
            ("omega", "phi", "rho"),
            ("pi0", "k", "pi", "e"),
            ("mu", "mu", "e", "e", "eta"),
        ]

        def msqrd3(s, t):
            return s * t

        # Test matrix elements
        def msqrd(momenta):
            m2 = np.ones_like(momenta[0, 0, ...])
            for i in range(momenta.shape[1]):
                for j in range(momenta.shape[1]):
                    m2 *= lnorm_sqr(momenta[:, i] + momenta[:, j])
            return m2

        for fs in final_states:
            for _, dnde_fn in self.dnde_dict.items():
                masses = [sm_masses[s] for s in fs]
                cme = 2.0 * sum(masses)
                eprod = 2.0
                if len(fs) == 3:
                    dnde_fn(eprod, cme, fs, msqrd=msqrd3)
                else:
                    dnde_fn(eprod, cme, fs, msqrd=msqrd)
