"""
Module for computing neutrino spectra.
@author: Logan Morrison and Adam Coogan
"""

from typing import overload, Optional, Union  # , List, Dict, Callable, NamedTuple

import numpy as np

from hazma import parameters
from hazma.spectra._neutrino import _muon
from hazma.spectra._neutrino import _pion
from hazma.utils import RealArray, RealOrRealArray

from ._utils import (
    load_interp as _load_interp,
    dnde_neutrino as _dnde_neutrino,
)


_eta_interp_e = _load_interp("eta_neutrino_e.csv")
_eta_interp_mu = _load_interp("eta_neutrino_mu.csv")
_charged_kaon_integrand_interp_e = _load_interp("charged_kaon_neutrino_e.csv")
_charged_kaon_integrand_interp_mu = _load_interp("charged_kaon_neutrino_mu.csv")
_long_kaon_integrand_interp_e = _load_interp("long_kaon_neutrino_e.csv")
_long_kaon_integrand_interp_mu = _load_interp("long_kaon_neutrino_mu.csv")
_short_kaon_integrand_interp_e = _load_interp("short_kaon_neutrino_e.csv")
_short_kaon_integrand_interp_mu = _load_interp("short_kaon_neutrino_mu.csv")
_omega_integrand_interp_e = _load_interp("omega_neutrino_e.csv")
_omega_integrand_interp_mu = _load_interp("omega_neutrino_mu.csv")
_eta_prime_integrand_interp_e = _load_interp("eta_prime_neutrino_e.csv")
_eta_prime_integrand_interp_mu = _load_interp("eta_prime_neutrino_mu.csv")
_phi_integrand_interp_e = _load_interp("phi_neutrino_e.csv")
_phi_integrand_interp_mu = _load_interp("phi_neutrino_mu.csv")
_charged_rho_integrand_interp_e = _load_interp("charged_rho_neutrino_e.csv")
_charged_rho_integrand_interp_mu = _load_interp("charged_rho_neutrino_mu.csv")
_neutral_rho_integrand_interp_e = _load_interp("neutral_rho_neutrino_e.csv")
_neutral_rho_integrand_interp_mu = _load_interp("neutral_rho_neutrino_mu.csv")


@overload
def dnde_neutrino_muon(
    neutrino_energies: float, muon_energy: float, flavor: Optional[str] = ...
) -> float: ...


@overload
def dnde_neutrino_muon(
    neutrino_energies: RealArray, muon_energy: float, flavor: Optional[str] = ...
) -> RealArray: ...


def dnde_neutrino_muon(
    neutrino_energies: RealOrRealArray, muon_energy: float, flavor: Optional[str] = None
) -> RealOrRealArray:
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
) -> float: ...


@overload
def dnde_neutrino_charged_pion(
    neutrino_energies: RealArray, pion_energy: float, flavor: Optional[str] = ...
) -> RealArray: ...


def dnde_neutrino_charged_pion(
    neutrino_energies: RealOrRealArray, pion_energy: float, flavor: Optional[str] = None
) -> RealOrRealArray:
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


@overload
def dnde_neutrino_charged_kaon(
    neutrino_energies: float, kaon_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_charged_kaon(
    neutrino_energies: RealArray, kaon_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_charged_kaon(
    neutrino_energies: RealOrRealArray, kaon_energy: float, flavor: Optional[str] = None
) -> RealOrRealArray:
    """
    Returns the neutrino spectrum from the decay of a charged kaon.

    Parameters
    ----------
    neutrino_energies : float or numpy.array
        Energy(ies) of the neutrino/electron.
    kaon_energy : float or numpy.array
        Energy of the charged kaon.
    flavor : str, optional
        Flavor of neutrino to compute spectrum for. If None, spectrum for all
        flavors are return. Options are "e", "mu" or "tau".

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a neutrino energy(ies)
        ``neutrino_energies`` and charged kaon energy ``pion_energy``.
    """
    interp_e = _charged_kaon_integrand_interp_e
    interp_mu = _charged_kaon_integrand_interp_mu
    parent_mass = parameters.charged_kaon_mass
    parent_energy = kaon_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energies,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


@overload
def dnde_neutrino_long_kaon(
    neutrino_energies: float, kaon_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_long_kaon(
    neutrino_energies: RealArray, kaon_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_long_kaon(
    neutrino_energies: RealOrRealArray, kaon_energy: float, flavor: Optional[str] = None
) -> RealOrRealArray:
    """
    Returns the neutrino spectrum from the decay of a long kaon.

    Parameters
    ----------
    neutrino_energies : float or numpy.array
        Energy(ies) of the neutrino/electron.
    kaon_energy : float or numpy.array
        Energy of the long kaon.
    flavor : str, optional
        Flavor of neutrino to compute spectrum for. If None, spectrum for all
        flavors are return. Options are "e", "mu" or "tau".

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a neutrino energy(ies)
        ``neutrino_energies`` and long kaon energy ``kaon_energy``.
    """
    interp_e = _long_kaon_integrand_interp_e
    interp_mu = _long_kaon_integrand_interp_mu
    parent_mass = parameters.neutral_kaon_mass
    parent_energy = kaon_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energies,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


@overload
def dnde_neutrino_short_kaon(
    neutrino_energies: float, kaon_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_short_kaon(
    neutrino_energies: RealArray, kaon_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_short_kaon(
    neutrino_energies: RealOrRealArray, kaon_energy: float, flavor: Optional[str] = None
) -> RealOrRealArray:
    """
    Returns the neutrino spectrum from the decay of a short kaon.

    Parameters
    ----------
    neutrino_energies : float or numpy.array
        Energy(ies) of the neutrino/electron.
    kaon_energy : float or numpy.array
        Energy of the short kaon.
    flavor : str, optional
        Flavor of neutrino to compute spectrum for. If None, spectrum for all
        flavors are return. Options are "e", "mu" or "tau".

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a neutrino energy(ies)
        ``neutrino_energies`` and short kaon energy ``kaon_energy``.
    """
    interp_e = _short_kaon_integrand_interp_e
    interp_mu = _short_kaon_integrand_interp_mu
    parent_mass = parameters.neutral_kaon_mass
    parent_energy = kaon_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energies,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


@overload
def dnde_neutrino_eta(
    neutrino_energy: float, eta_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_eta(
    neutrino_energy: RealArray, eta_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_eta(
    neutrino_energy: Union[RealArray, float],
    eta_energy: float,
    flavor: Optional[str] = None,
) -> Union[RealArray, float]:
    interp_e = _eta_interp_e
    interp_mu = _eta_interp_mu
    parent_mass = parameters.eta_mass
    parent_energy = eta_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


@overload
def dnde_neutrino_omega(
    neutrino_energy: float, omega_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_omega(
    neutrino_energy: RealArray, omega_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_omega(
    neutrino_energy: Union[RealArray, float],
    omega_energy: float,
    flavor: Optional[str] = None,
) -> Union[RealArray, float]:
    interp_e = _omega_integrand_interp_e
    interp_mu = _omega_integrand_interp_mu
    parent_mass = parameters.omega_mass
    parent_energy = omega_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


@overload
def dnde_neutrino_neutral_rho(
    neutrino_energy: float, rho_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_neutral_rho(
    neutrino_energy: RealArray, rho_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_neutral_rho(
    neutrino_energy: Union[RealArray, float],
    rho_energy: float,
    flavor: Optional[str] = None,
) -> Union[RealArray, float]:
    interp_e = _neutral_rho_integrand_interp_e
    interp_mu = _neutral_rho_integrand_interp_mu
    parent_mass = parameters.rho_mass
    parent_energy = rho_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


@overload
def dnde_neutrino_charged_rho(
    neutrino_energy: float, rho_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_charged_rho(
    neutrino_energy: RealArray, rho_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_charged_rho(
    neutrino_energy: Union[RealArray, float],
    rho_energy: float,
    flavor: Optional[str] = None,
) -> Union[RealArray, float]:
    interp_e = _charged_rho_integrand_interp_e
    interp_mu = _charged_rho_integrand_interp_mu
    parent_mass = parameters.rho_mass
    parent_energy = rho_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


@overload
def dnde_neutrino_eta_prime(
    neutrino_energy: float, eta_prime_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_eta_prime(
    neutrino_energy: RealArray, eta_prime_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_eta_prime(
    neutrino_energy: Union[RealArray, float],
    eta_prime_energy: float,
    flavor: Optional[str] = None,
) -> Union[RealArray, float]:
    interp_e = _eta_prime_integrand_interp_e
    interp_mu = _eta_prime_integrand_interp_mu
    parent_mass = parameters.eta_prime_mass
    parent_energy = eta_prime_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


@overload
def dnde_neutrino_phi(
    neutrino_energy: float, phi_energy: float, flavor: Optional[str] = None
) -> float: ...


@overload
def dnde_neutrino_phi(
    neutrino_energy: RealArray, phi_energy: float, flavor: Optional[str] = None
) -> RealArray: ...


def dnde_neutrino_phi(
    neutrino_energy: Union[RealArray, float],
    phi_energy: float,
    flavor: Optional[str] = None,
) -> Union[RealArray, float]:
    interp_e = _phi_integrand_interp_e
    interp_mu = _phi_integrand_interp_mu
    parent_mass = parameters.phi_mass
    parent_energy = phi_energy

    return _dnde_neutrino(
        neutrino_energy=neutrino_energy,
        parent_energy=parent_energy,
        parent_mass=parent_mass,
        interp_e=interp_e,
        interp_mu=interp_mu,
        flavor=flavor,
    )


# _DecayFn = Callable[[RealOrRealArray, float, Optional[str]], RealOrRealArray]


# class _NuDecayFn(NamedTuple):
#     dnde: _DecayFn
#     mass: float


# _dnde_neutrino_dict: Dict[str, _NuDecayFn] = {
#     "pi": _NuDecayFn(dnde_neutrino_charged_pion, parameters.charged_pion_mass),
#     "mu": _NuDecayFn(dnde_neutrino_muon, parameters.muon_mass),
#     "k": _NuDecayFn(dnde_neutrino_charged_kaon, parameters.charged_kaon_mass),
#     "kl": _NuDecayFn(dnde_neutrino_long_kaon, parameters.long_kaon_mass),
#     "ks": _NuDecayFn(dnde_neutrino_short_kaon, parameters.short_kaon_mass),
#     "eta": _NuDecayFn(dnde_neutrino_eta, parameters.eta_mass),
#     "omega": _NuDecayFn(dnde_neutrino_omega, parameters.omega_mass),
#     "etap": _NuDecayFn(dnde_neutrino_eta_prime, parameters.eta_prime_mass),
#     "rho": _NuDecayFn(dnde_neutrino_charged_rho, parameters.rho_mass),
# }


# @overload
# def dnde_neutrino(
#     neutrino_energy: float,
#     phi_energy: float,
#     particles: List[str],
#     flavor: Optional[str] = None,
# ) -> Dict[str, float]:
#     ...


# @overload
# def dnde_neutrino(
#     neutrino_energy: RealArray,
#     phi_energy: float,
#     particles: List[str],
#     flavor: Optional[str] = None,
# ) -> Dict[str, RealArray]:
#     ...


# def dnde_neutrino(
#     neutrino_energy: Union[RealArray, float],
#     cme: float,
#     particles: List[str],
#     flavor: Optional[str] = None,
# ) -> Union[Dict[str, float], Dict[str, RealArray]]:
#     def dnde(p):
#         return _dnde_neutrino_dict[p].dnde(
#             neutrino_energy,
#             phi_energy,
#             flavor,
#         )

#     return {p: dnde(p) for p in particles}  # type: ignore
