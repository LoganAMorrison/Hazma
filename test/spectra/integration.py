import unittest

from pytest import approx

import numpy as np
from numpy import testing as np_testing

from hazma.parameters import standard_model_masses as sm_masses
from hazma import spectra

"""

    neutrino_energies: npt.ArrayLike,
    cme: float,
    final_states: Union[str, Sequence[str]],
    msqrd: Optional[MSqrd] = None,
    *,
    three_body_integrator: str = "quad",
    npts: int = 1 << 14,
    nbins: int = 25,
    flavor: Optional[str] = None,

"""


photon_spectra_dict = {
    "k": spectra.dnde_photon_charged_kaon,
    "pi": spectra.dnde_photon_charged_pion,
    "rho": spectra.dnde_photon_charged_rho,
    "eta": spectra.dnde_photon_eta,
    "etap": spectra.dnde_photon_eta_prime,
    "kl": spectra.dnde_photon_long_kaon,
    "mu": spectra.dnde_photon_muon,
    "pi0": spectra.dnde_photon_neutral_pion,
    "rho0": spectra.dnde_photon_neutral_rho,
    "omega": spectra.dnde_photon_omega,
    "phi": spectra.dnde_photon_phi,
    "ks": spectra.dnde_photon_short_kaon,
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
        """Test the passing a single state works."""

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
