from os import path

import numpy as np

from hazma.decay import charged_pion, muon, neutral_pion
from hazma.parameters import charged_pion_mass as mpi
from hazma.parameters import muon_mass as mmu
from hazma.parameters import neutral_pion_mass as mpi0


def save_data(data_dir, es, dnde_func, e_gams):
    # Get this file's directory
    data_dir = path.join(path.dirname(__file__), data_dir)
    np.save(path.join(data_dir, "e_gams.npy"), e_gams)

    for i, e in enumerate(es):
        # Save energy of decaying particle
        np.save(path.join(data_dir, "e_{}.npy".format(i + 1)), e)

        # Save spectrum
        dnde = dnde_func(e_gams, e)
        np.save(path.join(data_dir, "dnde_{}.npy".format(i + 1)), dnde)


def generate_test_data():
    # Use the same photon energies for all particles
    e_gams = np.geomspace(1.0, 1e3, num=500)

    # Use the same energy-to-mass ratios for all particles
    e_over_m = np.array([1.0, 1.1, 5.0])

    save_data("mu_data", e_over_m * mmu, muon, e_gams)
    save_data("pi0_data", e_over_m * mpi0, neutral_pion, e_gams)
    save_data("pi_data", e_over_m * mpi, charged_pion, e_gams)


if __name__ == "__main__":
    generate_test_data()
