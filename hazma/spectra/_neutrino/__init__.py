"""
Module for computing neutrino spectra.
@author: Logan Morrison and Adam Coogan
"""

from typing import overload, Optional, Tuple, Union

from hazma.spectra._neutrino import _muon
from hazma.spectra._neutrino import _pion
from hazma.utils import RealArray, RealOrRealArray


RealOrTupleOrRealArray = Union[float, Tuple[float, float, float], RealArray]
RealOrTuple = Union[float, Tuple[float, float, float]]


@overload
def dnde_neutrino_muon(
    neutrino_energies: float, muon_energy: float, flavor: Optional[str] = ...
) -> RealOrTuple:
    ...


@overload
def dnde_neutrino_muon(
    neutrino_energies: RealArray, muon_energy: float, flavor: Optional[str] = ...
) -> RealArray:
    ...


def dnde_neutrino_muon(
    neutrino_energies: RealOrRealArray, muon_energy: float, flavor: Optional[str] = None
) -> RealOrTupleOrRealArray:
    """
    Returns the neutrino spectrum from muon decay.

    Parameters
    ----------
    neutrino_energies : float or numpy.ndarray
        Energy(ies) of the neutrinos.
    muon_energy : float
        Energy of the muon.
    flavor : str, optional
        Flavor of neutrino to compute spectrum for. If None, spectrum for all
        flavors are return. Options are "e", "mu" or "tau".

    Returns
    -------
    dnde : float or numpy.array
        The neutrino spectrum. If flavor is None, the result has the shape
        (3, len(neutrino_energies)). Otherwise, has the shape (len(neutrino_energies),).
    """
    dnde = _muon.dnde_neutrino_muon(neutrino_energies, muon_energy)
    if flavor is None:
        return dnde
    elif flavor == "e":
        return dnde[0]
    elif flavor == "mu":
        return dnde[1]
    elif flavor == "tau":
        return dnde[2]
    else:
        raise ValueError(f"Invalid flavor {flavor}. Use 'e', 'mu' or 'tau'.")


@overload
def dnde_neutrino_charged_pion(
    neutrino_energies: float, pion_energy: float, flavor: Optional[str] = ...
) -> RealOrTuple:
    ...


@overload
def dnde_neutrino_charged_pion(
    neutrino_energies: RealArray, pion_energy: float, flavor: Optional[str] = ...
) -> RealArray:
    ...


def dnde_neutrino_charged_pion(
    neutrino_energies: RealOrRealArray, pion_energy: float, flavor: Optional[str] = None
) -> RealOrTupleOrRealArray:
    """
    Returns the neutrino spectrum from the decay of a charged pion.

    Parameters
    ----------
    neutrino_energies : float or numpy.ndarray
        Energy(ies) of the neutrinos.
    pion_energy : float
        Energy of the charged pion.
    flavor : str, optional
        Flavor of neutrino to compute spectrum for. If None, spectrum for all
        flavors are return. Options are "e", "mu" or "tau".

    Returns
    -------
    dnde : float or numpy.array
        The neutrino spectrum. If flavor is None, the result has the shape
        (3, len(neutrino_energies)). Otherwise, has the shape (len(neutrino_energies),).
    """
    dnde = _pion.dnde_neutrino_charged_pion(neutrino_energies, pion_energy)
    if flavor is None:
        return dnde
    elif flavor == "e":
        return dnde[0]
    elif flavor == "mu":
        return dnde[1]
    elif flavor == "tau":
        return dnde[2]
    else:
        raise ValueError(f"Invalid flavor {flavor}. Use 'e', 'mu' or 'tau'.")
