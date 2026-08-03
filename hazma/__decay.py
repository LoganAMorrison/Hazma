"""
Module for computing decay spectra from a muon and light mesons.

@author: Logan Morrison and Adam Coogan
"""

from typing import List, Optional, overload
from warnings import warn

warn(
    "'hazma.decay' is deprecated. Use 'hazma.spectra' instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np

from hazma._decay import (
    decay_charged_kaon,
    decay_charged_pion,
    decay_long_kaon,
    decay_muon,
    decay_neutral_pion,
    decay_rho,
    decay_short_kaon,
)
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
def muon(photon_energies: float, muon_energy: float) -> float: ...


@overload
def muon(photon_energies: RealArray, muon_energy: float) -> RealArray: ...


def muon(photon_energies: RealOrRealArray, muon_energy: float) -> RealOrRealArray:
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
    return decay_muon.muon_decay_spectrum(photon_energies, muon_energy)


@overload
def neutral_pion(photon_energies: float, pion_energy: float) -> float: ...


@overload
def neutral_pion(photon_energies: RealArray, pion_energy: float) -> RealArray: ...


def neutral_pion(
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
    return decay_neutral_pion.neutral_pion_decay_spectrum(photon_energies, pion_energy)


def charged_pion_decay_modes() -> List[str]:
    """
    Return a list of implemented charged pion radiative decay modes.
    """
    return ["munu", "munug", "enug"]


@overload
def charged_pion(
    photon_energies: float, pion_energy: float, modes: Optional[List[str]] = None
) -> float: ...


@overload
def charged_pion(
    photon_energies: RealArray, pion_energy: float, modes: Optional[List[str]] = None
) -> RealArray: ...


def charged_pion(
    photon_energies: RealOrRealArray,
    pion_energy: float,
    modes: Optional[List[str]] = None,
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from the charged pion decay :math:`\pi^{\pm}
    \to \mu^{\pm} \nu_{\mu} \to e^{\pm} \nu_{e} \nu_{\mu} \gamma`.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    pion_energy : double
        Charged pion energy in laboratory frame.
    modes : List[str], optional
        A list modes the user would like include. The availible entries are:
        "munu", "munug" and "enug". Default is all of these.

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
    availible = charged_pion_decay_modes()
    modes_ = __mode_deprecation_convert(modes, availible)
    __check_modes(modes_, availible)

    return decay_charged_pion.charged_pion_decay_spectrum(
        photon_energies, pion_energy, modes_
    )


def charged_kaon_decay_modes() -> List[str]:
    """
    Return a list of implemented charged kaon radiative decay modes.
    """
    return ["0enu", "0munu", "00p", "mmug", "munu", "p0", "p0g", "ppm"]


@overload
def charged_kaon(
    photon_energies: float, kaon_energy: float, modes: Optional[List[str]]
) -> float: ...


@overload
def charged_kaon(
    photon_energies: RealArray, kaon_energy: float, modes: Optional[List[str]]
) -> RealArray: ...


def charged_kaon(
    photon_energies: RealOrRealArray,
    kaon_energy: float,
    modes: Optional[List[str]] = None,
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from charged kaon decay into various final states.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Charged kaon energy in laboratory frame.
    modes : List[str], optional
        A list modes the user would like to have included. The availible entries are:
        "0enu", "0munu", "00p", "mmug", "munu", "p0", "p0g" and "ppm". Here
        "p" stands for pi plus, "m" stands for pi minus and "0" stands pi 0. The default
        is all of the modes.

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
    availible = charged_kaon_decay_modes()
    modes_ = __mode_deprecation_convert(modes, availible)
    __check_modes(modes_, availible)

    return decay_charged_kaon.charged_kaon_decay_spectrum(
        photon_energies, kaon_energy, modes_
    )


def short_kaon_decay_modes() -> List[str]:
    """
    Return a list of implemented short kaon radiative decay modes.
    """
    return ["00", "pm", "pmg"]


@overload
def short_kaon(
    photon_energies: float, kaon_energy: float, modes: Optional[List[str]]
) -> float: ...


@overload
def short_kaon(
    photon_energies: RealArray, kaon_energy: float, modes: Optional[List[str]]
) -> RealArray: ...


def short_kaon(
    photon_energies: RealOrRealArray,
    kaon_energy: float,
    modes: Optional[List[str]] = None,
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from short kaon decay into various final states.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Charged kaon energy in laboratory frame.
    mode : str
        The mode the user would like to have returned. The options are "total",
        "00", "pm" or "pmg". Here "p" stands for pi plus, "m" stands for pi
        minus and "0" stands pi 0.

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
    availible = short_kaon_decay_modes()
    modes_ = __mode_deprecation_convert(modes, availible)
    __check_modes(modes_, availible)

    return decay_short_kaon.short_kaon_decay_spectrum(
        photon_energies, kaon_energy, modes_
    )


def long_kaon_decay_modes() -> List[str]:
    """
    Return a list of implemented long kaon radiative decay modes.
    """
    return ["000", "penu", "penug", "pm0", "pm0g", "pmunu", "pmunug"]


@overload
def long_kaon(
    photon_energies: float, kaon_energy: float, modes: Optional[List[str]]
) -> float: ...


@overload
def long_kaon(
    photon_energies: RealArray, kaon_energy: float, modes: Optional[List[str]]
) -> RealArray: ...


def long_kaon(
    photon_energies: RealOrRealArray,
    kaon_energy: float,
    modes: Optional[List[str]] = None,
) -> RealOrRealArray:
    r"""Compute gamma-ray spectrum from long kaon decay into various final
    states.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Charged kaon energy in laboratory frame.
    mode : str
        The mode the user would like to have returned. The options are "total",
        "000", "penu", "penug", "pm0", "pm0g", "pmunu" or "pmunug". Here "p"
        stands for pi plus, "m" stands for pi minus and "0" stands pi 0.

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
    availible = long_kaon_decay_modes()
    modes_ = __mode_deprecation_convert(modes, availible)
    __check_modes(modes_, availible)

    return decay_long_kaon.long_kaon_decay_spectrum(
        photon_energies, kaon_energy, modes_
    )


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
def neutral_rho(photon_energies: float, rho_energy: float) -> float: ...


@overload
def neutral_rho(photon_energies: RealArray, rho_energy: float) -> RealArray: ...


def neutral_rho(photon_energies: RealOrRealArray, rho_energy: float) -> RealOrRealArray:
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
    return decay_rho.neutral_rho_decay_spectrum(photon_energies, rho_energy)


@overload
def charged_rho(photon_energies: float, rho_energy: float) -> float: ...


@overload
def charged_rho(photon_energies: RealArray, rho_energy: float) -> RealArray: ...


def charged_rho(photon_energies: RealOrRealArray, rho_energy: float) -> RealOrRealArray:
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
    return decay_rho.charged_rho_decay_spectrum(photon_energies, rho_energy)
