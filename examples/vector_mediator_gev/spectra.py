"""Script to generate the photon spectrum from dark matter annihilations."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from hazma.vector_mediator import KineticMixingGeV


def photon_spectra(dm_mass: float, vm_mass: float, n: int):
    """Generate the photon spectrum."""

    model = KineticMixingGeV(
        mx=dm_mass,
        mv=vm_mass,
        gvxx=1.0,
        eps=1e-3,
    )

    dnde_fns = model.spectrum_funcs()
    xs = np.geomspace(1e-4, 1.0, n)

    cme = model.mx * 3.0
    photon_energies = 0.5 * cme * xs

    dnde_dict = {
        state: dnde_fn(photon_energies, cme) for state, dnde_fn in dnde_fns.items()
    }
    dndx_dict = {state: dnde * cme / 2.0 for state, dnde in dnde_dict.items()}

    plt.figure(dpi=150)

    for state, dndx in dndx_dict.items():
        plt.plot(xs, xs**2 * dndx, label=state)

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e-5, 10)
    plt.show()


def positron_spectra(dm_mass: float, vm_mass: float, n: int):
    """Generate the photon spectrum."""

    model = KineticMixingGeV(
        mx=dm_mass,
        mv=vm_mass,
        gvxx=1.0,
        eps=1e-3,
    )

    dnde_fns = model.positron_spectrum_funcs()
    xs = np.geomspace(1e-4, 1.0, n)

    cme = model.mx * 3.0
    photon_energies = 0.5 * cme * xs

    dnde_dict = {
        state: dnde_fn(photon_energies, cme) for state, dnde_fn in dnde_fns.items()
    }
    dndx_dict = {state: dnde * cme / 2.0 for state, dnde in dnde_dict.items()}

    plt.figure(dpi=150)

    for state, dndx in dndx_dict.items():
        plt.plot(xs, xs**2 * dndx, label=state)

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e-5, 10)
    plt.show()


def neutrino_spectra(dm_mass: float, vm_mass: float, n: int, flavor: str):
    """Generate the neutrino spectrum."""

    model = KineticMixingGeV(
        mx=dm_mass,
        mv=vm_mass,
        gvxx=1.0,
        eps=1e-3,
    )

    dnde_fns = model.neutrino_spectrum_funcs(flavor)
    xs = np.geomspace(1e-4, 1.0, n)

    cme = model.mx * 3.0
    energies = 0.5 * cme * xs

    dnde_dict = {state: dnde_fn(energies, cme) for state, dnde_fn in dnde_fns.items()}
    dndx_dict = {state: dnde * cme / 2.0 for state, dnde in dnde_dict.items()}

    plt.figure(dpi=150)

    for state, dndx in dndx_dict.items():
        plt.plot(xs, xs**2 * dndx, label=state)

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e-5, 10)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spectrum",
        type=str,
        help="Final state",
        choices=["photon", "positron", "nu-e", "nu-mu", "nu-tau"],
    )
    parser.add_argument(
        "--dm-mass",
        type=float,
        help="Dark matter mass in MeV",
        default=1e3,
    )
    parser.add_argument(
        "--vm-mass",
        type=float,
        help="Vector mediator mass in MeV",
        default=3e3,
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of spectrum points",
        default=200,
    )
    args = parser.parse_args()

    if args.spectrum == "photon":
        photon_spectra(
            dm_mass=args.dm_mass,
            vm_mass=args.vm_mass,
            n=args.n,
        )
    elif args.spectrum == "positron":
        positron_spectra(
            dm_mass=args.dm_mass,
            vm_mass=args.vm_mass,
            n=args.n,
        )
    elif args.spectrum == "nu-e":
        neutrino_spectra(
            dm_mass=args.dm_mass,
            vm_mass=args.vm_mass,
            n=args.n,
            flavor="e",
        )
    else:
        neutrino_spectra(
            dm_mass=args.dm_mass,
            vm_mass=args.vm_mass,
            n=args.n,
            flavor="tau",
        )
