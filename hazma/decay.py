"""
Module for computing decay spectra from a muon and light mesons.

@author: Logan Morrison and Adam Coogan
"""
from warnings import warn
import numpy as np
from hazma.decay_helper_functions import decay_long_kaon
from hazma.decay_helper_functions import decay_charged_pion
from hazma.decay_helper_functions import decay_charged_kaon
from hazma.decay_helper_functions import decay_muon
from hazma.decay_helper_functions import decay_neutral_pion
from hazma.decay_helper_functions import decay_short_kaon


def __mode_deprecation_warning():
    msg = "Using a single item from `modes` is deprecated."
    msg += " Use a list of modes instead. E.g. modes=[mode1,mode2,...]"
    warn(msg)


def __mode_deprecation_convert(modes, availible):
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


def muon(photon_energies, muon_energy):
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


def neutral_pion(photon_energies, pion_energy):
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


__CHG_PI_MODES = ["munu", "munug", "enug"]


def charged_pion(photon_energies, pion_energy, modes=None):
    r"""Compute gamma-ray spectrum from the charged pion decay :math:`\pi^{\pm}
    \to \mu^{\pm} \nu_{\mu} \to e^{\pm} \nu_{e} \nu_{\mu} \gamma`.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    pion_energy : double
        Charged pion energy in laboratory frame.
    modes : List[str], optional
        A list modes the user would like include. The availible entries
        are: "munu", "munug" and "enug". Default is all of these.

    Returns
    -------
    spec : double np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `photon_energies`
        given charged pion energy `eng_pi`.

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
    modes_ = __mode_deprecation_convert(modes, __CHG_PI_MODES)
    __check_modes(modes_, __CHG_PI_MODES)

    return decay_charged_pion.charged_pion_decay_spectrum(
        photon_energies, pion_energy, modes_
    )


__CHG_K_MODES = ["0enu", "0munu", "00p", "mmug", "munu", "p0", "p0g", "ppm"]


def charged_kaon(photon_energies, kaon_energy, modes=None):
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
    modes_ = __mode_deprecation_convert(modes, __CHG_K_MODES)
    __check_modes(modes_, __CHG_K_MODES)

    return decay_charged_kaon.charged_kaon_decay_spectrum(
        photon_energies, kaon_energy, modes_
    )


__SHORT_K_MODES = ["00", "pm", "pmg"]


def short_kaon(photon_energies, kaon_energy, modes=None):
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
    modes_ = __mode_deprecation_convert(modes, __SHORT_K_MODES)
    __check_modes(modes_, __SHORT_K_MODES)

    return decay_short_kaon.short_kaon_decay_spectrum(
        photon_energies, kaon_energy, modes_
    )


__LONG_K_MODES = ["000", "penu", "penug", "pm0", "pm0g", "pmunu", "pmunug"]


def long_kaon(photon_energies, kaon_energy, modes=None):
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
    modes_ = __mode_deprecation_convert(modes, __LONG_K_MODES)
    __check_modes(modes_, __LONG_K_MODES)

    return decay_long_kaon.long_kaon_decay_spectrum(
        photon_energies, kaon_energy, modes_
    )


def electron(photon_energies, _):
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
