"""
Module for computing decay spectra from a muon and light mesons.

@author: Logan Morrison and Adam Coogan
"""

from typing import overload

from hazma._core import photon as _core_photon
from hazma.spectra._photon import _rho
from hazma.utils import RealArray, RealOrRealArray


@overload
def dnde_photon_muon(photon_energies: float, muon_energy: float) -> float: ...


@overload
def dnde_photon_muon(photon_energies: RealArray, muon_energy: float) -> RealArray: ...


def dnde_photon_muon(
    photon_energies: RealOrRealArray, muon_energy: float
) -> RealOrRealArray:
    r"""Compute the gamma-ray decay spectrum from muon decay.

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
        Units are MeV^-1; ``photon_energies`` and ``muon_energy`` are both
        in MeV.

    Examples
    --------
    Calculate spectrum for single gamma ray energy::

        from hazma import spectra
        photon_energy, muon_energy = 200., 1000.
        spectra.dnde_photon_muon(photon_energy, muon_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import spectra
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        muon_energy = 1000.
        spectra.dnde_photon_muon(photon_energies, muon_energy)
    """
    return _core_photon.dnde_photon_muon(photon_energies, muon_energy)


@overload
def dnde_photon_neutral_pion(photon_energies: float, pion_energy: float) -> float: ...


@overload
def dnde_photon_neutral_pion(
    photon_energies: RealArray, pion_energy: float
) -> RealArray: ...


def dnde_photon_neutral_pion(
    photon_energies: RealOrRealArray, pion_energy: float
) -> RealOrRealArray:
    r"""Compute the gamma-ray spectrum from neutral pion decay.

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

        from hazma import spectra
        photon_energies, pion_energy = 200., 1000.
        spectra.dnde_photon_neutral_pion(photon_energies, pion_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import spectra
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        pion_energy = 1000.
        spectra.dnde_photon_neutral_pion(photon_energies, pion_energy)
    """
    return _core_photon.dnde_photon_neutral_pion(photon_energies, pion_energy)


@overload
def dnde_photon_charged_pion(photon_energy: float, pion_energy: float) -> float: ...


@overload
def dnde_photon_charged_pion(
    photon_energy: RealArray, pion_energy: float
) -> RealArray: ...


def dnde_photon_charged_pion(
    photon_energy: RealOrRealArray, pion_energy: float
) -> RealOrRealArray:
    r"""Compute the gamma-ray spectrum from charged pion decay.

    :math:`\pi^{\pm} \to \mu^{\pm} \nu_{\mu}
    \to e^{\pm} \nu_{e} \nu_{\mu} \gamma`.

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

        from hazma import spectra
        photon_energies, pion_energy = 200., 1000.
        spectra.dnde_photon_charged_pion(photon_energies, pion_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import spectra
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        pion_energy = 1000.
        spectra.dnde_photon_charged_pion(photon_energies, pion_energy)
    """
    return _core_photon.dnde_photon_charged_pion(photon_energy, pion_energy)


@overload
def dnde_photon_charged_kaon(photon_energy: float, kaon_energy: float) -> float: ...


@overload
def dnde_photon_charged_kaon(
    photon_energy: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_photon_charged_kaon(
    photon_energy: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from charged kaon decay into various final states.

    Parameters
    ----------
    photon_energy : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Charged kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energy`` given Kaon energy ``kaon_energy``.
        Units are MeV^-1; ``photon_energy`` and ``kaon_energy`` are both in MeV.

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

        from hazma import spectra
        photon_energies, kaon_energy = 200., 1000.
        spectra.dnde_photon_charged_kaon(photon_energies, kaon_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import spectra
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        kaon_energy = 1000.
        spectra.dnde_photon_charged_kaon(photon_energies, kaon_energy)
    """
    return _core_photon.dnde_photon_charged_kaon(photon_energy, kaon_energy)


@overload
def dnde_photon_short_kaon(photon_energy: float, kaon_energy: float) -> float: ...


@overload
def dnde_photon_short_kaon(
    photon_energy: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_photon_short_kaon(
    photon_energy: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from short kaon decay into various final states.

    Parameters
    ----------
    photon_energy : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Short kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energy`` given kaon energy ``kaon_energy``.
        Units are MeV^-1; ``photon_energy`` and ``kaon_energy`` are both in MeV.

    Notes
    -----
    The decay modes implemented are

    .. math:: K_{S} \to \pi^{+}  + \pi^{-}

    .. math:: K_{S} \to \pi^{0} + \pi^{0}

    Examples
    --------
    Calculate spectrum for single gamma ray energy::

        from hazma import spectra
        photon_energies, kaon_energy = 200., 1000.
        spectra.dnde_photon_short_kaon(photon_energies, kaon_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import spectra
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        kaon_energy = 1000.
        spectra.dnde_photon_short_kaon(photon_energies, kaon_energy)
    """
    return _core_photon.dnde_photon_short_kaon(photon_energy, kaon_energy)


@overload
def dnde_photon_long_kaon(photon_energy: float, kaon_energy: float) -> float: ...


@overload
def dnde_photon_long_kaon(
    photon_energy: RealArray, kaon_energy: float
) -> RealArray: ...


def dnde_photon_long_kaon(
    photon_energy: RealOrRealArray, kaon_energy: float
) -> RealOrRealArray:
    r"""Compute the gamma-ray spectrum from long kaon decay.

    Sums over the various final states.

    Parameters
    ----------
    photon_energy : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Long kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energy`` given kaon energy ``kaon_energy``.
        Units are MeV^-1; ``photon_energy`` and ``kaon_energy`` are both in MeV.

    Examples
    --------
    Calculate spectrum for single gamma ray energy::

        from hazma import spectra
        photon_energies, kaon_energy = 200., 1000.
        spectra.dnde_photon_long_kaon(photon_energies, kaon_energy)

    Calculate spectrum for array of gamma ray energies::

        from hazma import spectra
        import numpy as np
        photon_energies = np.logspace(0.0, 3.0, num=200, dtype=float)
        kaon_energy = 1000.
        spectra.dnde_photon_long_kaon(photon_energies, kaon_energy)

    Notes
    -----
    The decay modes implemented are

    .. math:: K_{L} \to \pi^{\pm} e^{\pm} \nu_{e}

    .. math:: K_{L} \to \pi^{\pm} \mu^{\mp} \nu_{\mu}

    .. math:: K_{L} \to \pi^{0} \pi^{0} \pi^{0}

    .. math:: K_{L} \to \pi^{+} \pi^{-} \pi^{0}

    """
    return _core_photon.dnde_photon_long_kaon(photon_energy, kaon_energy)


@overload
def dnde_photon_neutral_rho(photon_energies: float, rho_energy: float) -> float: ...


@overload
def dnde_photon_neutral_rho(
    photon_energies: RealArray, rho_energy: float
) -> RealArray: ...


def dnde_photon_neutral_rho(
    photon_energies: RealOrRealArray, rho_energy: float
) -> RealOrRealArray:
    r"""Compute the gamma-ray decay spectrum from neutral rho decay.

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
def dnde_photon_charged_rho(photon_energies: float, rho_energy: float) -> float: ...


@overload
def dnde_photon_charged_rho(
    photon_energies: RealArray, rho_energy: float
) -> RealArray: ...


def dnde_photon_charged_rho(
    photon_energies: RealOrRealArray, rho_energy: float
) -> RealOrRealArray:
    r"""Compute the gamma-ray decay spectrum from charged rho decay.

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
def dnde_photon_eta(photon_energy: float, eta_energy: float) -> float: ...


@overload
def dnde_photon_eta(photon_energy: RealArray, eta_energy: float) -> RealArray: ...


def dnde_photon_eta(
    photon_energy: RealOrRealArray, eta_energy: float
) -> RealOrRealArray:
    r"""Compute the gamma-ray decay spectrum from eta decay.

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
        Units are MeV^-1; ``photon_energy`` and ``eta_energy`` are both in MeV.
    """
    return _core_photon.dnde_photon_eta(photon_energy, eta_energy)


@overload
def dnde_photon_omega(photon_energy: float, omega_energy: float) -> float: ...


@overload
def dnde_photon_omega(photon_energy: RealArray, omega_energy: float) -> RealArray: ...


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
        Units are MeV^-1; ``photon_energy`` and ``omega_energy`` are both in MeV.
    """
    return _core_photon.dnde_photon_omega(photon_energy, omega_energy)


@overload
def dnde_photon_eta_prime(photon_energy: float, eta_prime_energy: float) -> float: ...


@overload
def dnde_photon_eta_prime(
    photon_energy: RealArray, eta_prime_energy: float
) -> RealArray: ...


def dnde_photon_eta_prime(
    photon_energy: RealOrRealArray, eta_prime_energy: float
) -> RealOrRealArray:
    r"""Compute the gamma-ray decay spectrum from eta-prime decay.

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
        ``photon_energy`` given eta' energy ``eta_prime_energy``.
        Units are MeV^-1; ``photon_energy`` and ``eta_prime_energy`` are both in MeV.
    """
    return _core_photon.dnde_photon_eta_prime(photon_energy, eta_prime_energy)


@overload
def dnde_photon_phi(photon_energy: float, phi_energy: float) -> float: ...


@overload
def dnde_photon_phi(photon_energy: RealArray, phi_energy: float) -> RealArray: ...


def dnde_photon_phi(
    photon_energy: RealOrRealArray, phi_energy: float
) -> RealOrRealArray:
    r"""Compute gamma-ray decay spectrum from the decay of the phi(1020).

    Parameters
    ----------
    photon_energy : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    phi_energy : double
        Phi energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        Array of gamma ray spectrum values, dNdE, evaluated at
        ``photon_energy`` given phi energy ``phi_energy``.
        Units are MeV^-1; ``photon_energy`` and ``phi_energy`` are both in MeV.
    """
    return _core_photon.dnde_photon_phi(photon_energy, phi_energy)
