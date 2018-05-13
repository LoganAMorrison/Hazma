"""
Module for computing positron spectra.

@author: Logan Morrison and Adam Coogan
@date: May 2018

"""

from .parameters import muon_mass as mmu
from .parameters import electron_mass as me
from .parameters import charged_pion_mass as mpi

import numpy as np

from scipy.integrate import quad


def __dnde_muon(ee):
    """
    Returns the positron spectrum from a muon.

    Parameters
    ----------
    ee : float
        Energy of the positron.

    Returns
    -------
    dnde : float
        The value of the spectrum given a positron energy `ee`.
    """
    r = me / mmu
    s = me**2 - 2 * ee * mmu + mmu**2
    smax = mmu**2 * (1. - r)**2
    smin = 0.
    if s < smin or smax < s:
        return 0.0
    dnds = (2 * (mmu**4 * (-1 + r**2)**2 + mmu**2 *
                 (1 + r**2) * s - 2 * s**2) *
            np.sqrt(mmu**4 * (-1 + r**2)**2 -
                    2 * mmu**2 * (1 + r**2) * s + s**2)) / mmu**8
    return 2 * mmu * dnds


def dnde_muon(ee):
    """
    Returns the positron spectrum from a muon.

    Parameters
    ----------
    ee : float or array-like
        Energy of the positron.

    Returns
    -------
    dnde : float or array-like
        The value of the spectrum given a positron energy `ee`.
    """
    if hasattr(ee, "__len__"):
        return np.array([__dnde_muon(e) for e in ee])
    return __dnde_muon(ee)


def __dnde_cpion(ee):
    """
    Returns the positron spectrum from a charged pion.

    Parameters
    ----------
    ee : float or array-like
        Energy of the positron.

    Returns
    -------
    dnde : float or array-like
        The value of the spectrum given a positron energy `ee`.
    """
    emu = (mmu**2 / mpi + mpi) / 2.
    gamma = emu / mmu
    beta = np.sqrt(1. - 1. / gamma**2)

    def integrand(c2):
        ee1 = gamma * ee * (1. - beta * c2)
        return dnde_muon(ee1) / 2. / gamma / abs(1. - beta * c2)

    return quad(integrand, -1., 1.)[0]


def dnde_cpion(ee):
    """
    Returns the positron spectrum from a muon.

    Parameters
    ----------
    ee : float or array-like
        Energy of the positron.

    Returns
    -------
    dnde : float or array-like
        The value of the spectrum given a positron energy `ee`.
    """
    if hasattr(ee, "__len__"):
        return np.array([__dnde_cpion(e) for e in ee])
    return __dnde_cpion(ee)
