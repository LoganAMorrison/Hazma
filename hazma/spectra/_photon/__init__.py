"""
Module for computing decay spectra from a muon and light mesons.

@author: Logan Morrison and Adam Coogan
"""
from typing import List, Optional, overload
from warnings import warn

import numpy as np

from hazma.spectra._photon import _muon, _pion, _rho, _kaon, _eta, _omega, _eta_prime
from hazma.utils import RealArray, RealOrRealArray


def __mode_deprecation_warning():
    msg = "Using a single item from `modes` is deprecated."
    msg += " Use a list of modes instead. E.g. modes=[mode1,mode2,...]"
    warn(msg)


def __mode_deprecation_convert(modes, availible: List[str]) -> List[str]:
    if modes is None:
        return availible
    elif isinstance(modes, str):
        # For backwards compatiblity
        __mode_deprecation_warning()
        if modes == "total":
            return availible
        else:
            return [modes]
    else:
        return modes


def __check_modes(modes, availible):
    for mode in modes:
        assert (
            mode in availible
        ), f"Invalid mode {mode} specified. The availible modes are: {availible}"


@overload
def dnde_photon_muon(photon_energies: float, muon_energy: float) -> float:
    ...


@overload
def dnde_photon_muon(photon_energies: RealArray, muon_energy: float) -> RealArray:
    ...


def dnde_photon_muon(
    photon_energies: RealOrRealArray, muon_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray decay spectrum from the muon decay
    :math:`\mu^{\pm} \to e^{\pm} \nu_{e} \nu_{\mu}`.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    muon_energy : double
        Muon energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energies`` given muon energy ``eng_mu``.

    Examples
    --------

    Calculate spectrum for single gamma ray energy::

        from hazma import decay
        photon_energy, muon_energy = 200., 1000.
        decay.muon(photon_energy, muon_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import decay
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        muon_energy = 1000.
        decay.muon(photon_energies, muon_energy)
    """
    return _muon.dnde_photon(photon_energies, muon_energy)


@overload
def dnde_photon_neutral_pion(photon_energies: float, pion_energy: float) -> float:
    ...


@overload
def dnde_photon_neutral_pion(
    photon_energies: RealArray, pion_energy: float
) -> RealArray:
    ...


def dnde_photon_neutral_pion(
    photon_energies: RealOrRealArray, pion_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from the neutral pion decay
    :math:`\pi^{0} \to \gamma \gamma`.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    pion_energy : float
        Neutral pion energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        `photon_energies` given neutral pion energy `pion_energy`.

    Examples
    --------

    Calculate spectrum for single gamma ray energy::

        from hazma import decay
        photon_energies, pion_energy = 200., 1000.
        decay.neutral_pion(photon_energies, pion_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import decay
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        pion_energy = 1000.
        decay.neutral_pion(photon_energies, pion_energy)
    """
    return _pion.dnde_photon_neutral_pion(photon_energies, pion_energy)


@overload
def dnde_photon_charged_pion(photon_energy: float, pion_energy: float) -> float:
    ...


@overload
def dnde_photon_charged_pion(photon_energy: RealArray, pion_energy: float) -> RealArray:
    ...


def dnde_photon_charged_pion(
    photon_energy: RealOrRealArray, pion_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from the charged pion decay :math:`\pi^{\pm}
    \to \mu^{\pm} \nu_{\mu} \to e^{\pm} \nu_{e} \nu_{\mu} \gamma`.

    Parameters
    ----------
    photon_energy : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    pion_energy : double
        Charged pion energy in laboratory frame.

    Returns
    -------
    spec : Union[float, np.ndarray]
        List of gamma ray spectrum values, :math:`dN/dE`, evaluated at
        `photon_energies` given charged pion energy `eng_pi`.

    Examples
    --------
    Calculate spectrum for single gamma ray energy::

        from hazma import decay
        photon_energies, pion_energy = 200., 1000.
        decay.charged_pion(photon_energies, pion_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import decay
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        pion_energy = 1000.
        decay.charged_pion(photon_energies, pion_energy)
    """
    return _pion.dnde_photon_charged_pion(photon_energy, pion_energy)


@overload
def dnde_photon_charged_kaon(photon_energy: float, kaon_energy: float) -> float:
    ...


@overload
def dnde_photon_charged_kaon(photon_energy: RealArray, kaon_energy: float) -> RealArray:
    ...


def dnde_photon_charged_kaon(
    photon_energy: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from charged kaon decay into various final states.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Charged kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energies`` given Kaon energy ``kaon_energy``.

    Notes
    -----
    The decay modes implemented are

    .. math:: K^{\pm} \to \mu^{\pm} \nu_{\mu}

    .. math:: K^{\pm} \to \pi^{\pm} \pi^{0}

    .. math:: K^{\pm} \to \pi^{\pm} \pi^{\mp} + \pi^{\pm}

    .. math:: K^{\pm} \to e^{\pm} \nu_{e}

    .. math:: K^{\pm} \to \mu^{\pm} \nu_{\mu} \pi^{0}

    .. math:: K^{\pm} \to \pi^{\pm} \pi^{0} \pi^{0}

    Examples
    --------

    Calculate spectrum for single gamma ray energy::

        from hazma import decay
        photon_energies, kaon_energy = 200., 1000.
        decay.charged_kaon(photon_energies, kaon_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import decay
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        kaon_energy = 1000.
        decay.charged_kaon(photon_energies, kaon_energy)
    """
    return _kaon.dnde_photon_charged_kaon(photon_energy, kaon_energy)


@overload
def dnde_photon_short_kaon(photon_energy: float, kaon_energy: float) -> float:
    ...


@overload
def dnde_photon_short_kaon(photon_energy: RealArray, kaon_energy: float) -> RealArray:
    ...


def dnde_photon_short_kaon(
    photon_energy: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from short kaon decay into various final states.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Charged kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energies`` given muon energy ``eng_mu``.

    Notes
    -----
    The decay modes implemented are

    .. math:: K_{S} \to \pi^{+}  + \pi^{-}

    .. math:: K_{S} \to \pi^{0} + \pi^{0}

    Examples
    --------

    Calculate spectrum for single gamma ray energy::

        from hazma import decay
        photon_energies, kaon_energy = 200., 1000.
        decay.short_kaon(photon_energies, kaon_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import decay
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        kaon_energy = 1000.
        decay.short_kaon(photon_energies, kaon_energy)
    """
    return _kaon.dnde_photon_short_kaon(photon_energy, kaon_energy)


@overload
def dnde_photon_long_kaon(photon_energy: float, kaon_energy: float) -> float:
    ...


@overload
def dnde_photon_long_kaon(photon_energy: RealArray, kaon_energy: float) -> RealArray:
    ...


def dnde_photon_long_kaon(
    photon_energy: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from long kaon decay into various final
    states.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Charged kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `photon_energies`
        given muon energy `kaon_energy`.

    Examples
    --------
    Calculate spectrum for single gamma ray energy::

        from hazma import decay
        photon_energies, kaon_energy = 200., 1000.
        decay.long_kaon(photon_energies, kaon_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import decay
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        kaon_energy = 1000.
        decay.long_kaon(photon_energies, kaon_energy)

    Notes
    -----
    The decay modes implemented are

    .. math:: K_{L} \to \pi^{\pm} e^{\pm} \nu_{e}

    .. math:: K_{L} \to \pi^{\pm} \mu^{\mp} \nu_{\mu}

    .. math:: K_{L} \to \pi^{0} \pi^{0} \pi^{0}

    .. math:: K_{L} \to \pi^{+} \pi^{-} \pi^{0}

    """
    return _kaon.dnde_photon_long_kaon(photon_energy, kaon_energy)


def electron(photon_energies, _: float):
    r"""Compute gamma-ray spectrum from electron decay (returns zero).

    The purpose of this function is so we can use the electron as a final
    state in `hazma.gamma_ray`.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    electron_energy : double
        Electron energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        An array of zeros.
    """
    if hasattr(photon_energies, "__len__"):
        return np.array([0.0 for _ in photon_energies])
    return 0.0


@overload
def dnde_photon_neutral_rho(photon_energies: float, rho_energy: float) -> float:
    ...


@overload
def dnde_photon_neutral_rho(photon_energies: RealArray, rho_energy: float) -> RealArray:
    ...


def dnde_photon_neutral_rho(
    photon_energies: RealOrRealArray, rho_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray decay spectrum from the charged rho decay
    :math:`\rho \to \pi^{\pm} + \pi^{\mp}`.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    rho_energy : double
        Rho energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energies`` given rho energy ``rho_energy``.
    """
    return _rho.dnde_photon_neutral_rho(photon_energies, rho_energy)


@overload
def dnde_photon_charged_rho(photon_energies: float, rho_energy: float) -> float:
    ...


@overload
def dnde_photon_charged_rho(photon_energies: RealArray, rho_energy: float) -> RealArray:
    ...


def dnde_photon_charged_rho(
    photon_energies: RealOrRealArray, rho_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray decay spectrum from the rho decay
    :math:`\rho^{\pm} \to \pi^{\pm} + \pi^{0}`.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    rho_energy : double
        Rho energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energies`` given rho energy ``rho_energy``.
    """
    return _rho.dnde_photon_charged_rho(photon_energies, rho_energy)


@overload
def dnde_photon_eta(photon_energy: float, eta_energy: float) -> float:
    ...


@overload
def dnde_photon_eta(photon_energy: RealArray, eta_energy: float) -> RealArray:
    ...


def dnde_photon_eta(
    photon_energy: RealOrRealArray, eta_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray decay spectrum from the decay of the eta

    Parameters
    ----------
    photon_energy : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    eta_energy : double
        Eta energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        Array of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energy`` given eta energy ``eta_energy``.
    """
    return _eta.dnde_photon_eta(photon_energy, eta_energy)


@overload
def dnde_photon_omega(photon_energy: float, omega_energy: float) -> float:
    ...


@overload
def dnde_photon_omega(photon_energy: RealArray, omega_energy: float) -> RealArray:
    ...


def dnde_photon_omega(
    photon_energy: RealOrRealArray, omega_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray decay spectrum from the decay of the omega.

    Parameters
    ----------
    photon_energy : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    omega_energy : double
        Omega energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        Array of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energy`` given omega energy ``omega_energy``.
    """
    return _omega.dnde_photon_omega(photon_energy, omega_energy)


@overload
def dnde_photon_eta_prime(photon_energy: float, eta_prime_energy: float) -> float:
    ...


@overload
def dnde_photon_eta_prime(
    photon_energy: RealArray, eta_prime_energy: float
) -> RealArray:
    ...


def dnde_photon_eta_prime(
    photon_energy: RealOrRealArray, eta_prime_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray decay spectrum from the decay of the omega.

    Parameters
    ----------
    photon_energy : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    eta_prime_energy : double
        Eta' energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        Array of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energy`` given omega energy ``omega_energy``.
    """
    return _eta_prime.dnde_photon_eta_prime(photon_energy, eta_prime_energy)
