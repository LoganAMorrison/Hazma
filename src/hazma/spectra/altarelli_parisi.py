from typing import Any, Callable

import numpy as np

from hazma import parameters


def _scalar_splitting(x):
    return 2.0 * (1.0 - x) / x


def _fermion_splitting(x):
    return (1.0 + (1.0 - x) ** 2) / x


def _dndx_photon_fsr(x, s: float, m: float, split: Callable[[Any], Any], q=1.0):
    pre = q**2 * parameters.alpha_em / (2.0 * np.pi)
    xm = 1.0 - x
    kernel = np.zeros_like(x)

    mask = s * xm / m**2 > np.e
    kernel[mask] = split(x[mask]) * (np.log(s * xm[mask] / m**2) - 1.0)

    return pre * kernel


def _dnde_photon_fsr(e, s: float, m: float, split: Callable[[Any], Any], q=1.0):
    e_to_x = 2.0 / np.sqrt(s)
    x = e * e_to_x
    return _dndx_photon_fsr(x, s, m, split, q) * e_to_x


def dndx_photon_ap_fermion(x, s: float, mass: float, charge=1.0):
    """
    Compute dndx from the FSR off a charged fermion f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    x: array_like
        Scaled energies of the fermion, x = 2E/sqrt(s).
    s: float
        Squared center-of-mass energy flowing through of the radiating
        fermion and Y in the process X -> (f + Y) + Z.
    mass: float
        Mass of the radiating fermion.
    charge: float
        Charge of the radiating fermion. Default is 1.
    """
    return _dndx_photon_fsr(x, s, mass, _fermion_splitting, q=charge)


def dndx_photon_ap_scalar(x, s, mass, charge=1.0):
    """
    Compute dndx from the FSR off a charged scalar f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    x: array_like
        Scaled energies of the fermion, x = 2E/sqrt(s).
    s: float
        Squared center-of-mass energy flowing through of the radiating
        scalar and Y in the process X -> (f + Y) + Z.
    mass: float
        Mass of the radiating scalar.
    charge: float
        Charge of the radiating scalar. Default is 1.
    """
    return _dndx_photon_fsr(x, s, mass, _scalar_splitting, q=charge)


def dnde_photon_ap_fermion(e, s, mass, charge=1.0):
    """
    Compute dN/dE from the FSR off a charged fermion f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    e: array_like
        Photon energies.
    s: float
        Squared center-of-mass energy flowing through of the radiating
        fermion and Y in the process X -> (f + Y) + Z.
    mass: float
        Mass of the radiating fermion.
    charge: float
        Charge of the radiating fermion. Default is 1.
    """
    return _dnde_photon_fsr(e, s, mass, _fermion_splitting, q=charge)


def dnde_photon_ap_scalar(e, s, mass, charge=1.0):
    """
    Compute dN/dE from the FSR off a charged scalar f from a
    process of the form X -> (f + Y + gamma) + Z.

    Parameters
    ----------
    e: array_like
        Photon energy.
    s: float
        Squared center-of-mass energy flowing through of the radiating
        scalar and Y in the process X -> (f + Y) + Z.
    mass: float
        Mass of the radiating scalar.
    charge: float
        Charge of the radiating scalar. Default is 1.
    """
    return _dnde_photon_fsr(e, s, mass, _scalar_splitting, q=charge)
