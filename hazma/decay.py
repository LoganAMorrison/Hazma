"""
Module for computing decay spectra from a muon and light mesons.

@author: Logan Morrison and Adam Coogan
@date: January 2018

"""
import numpy as np
from hazma.decay_helper_functions import decay_long_kaon
from hazma.decay_helper_functions import decay_charged_pion
from hazma.decay_helper_functions import decay_charged_kaon
from hazma.decay_helper_functions import decay_muon
from hazma.decay_helper_functions import decay_neutral_pion
from hazma.decay_helper_functions import decay_short_kaon


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
    # mu = decay_muon.Muon()
    if hasattr(photon_energies, "__len__"):
        return decay_muon.Spectrum(photon_energies, muon_energy)
    return decay_muon.SpectrumPoint(photon_energies, muon_energy)


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
    if hasattr(photon_energies, "__len__"):
        return decay_neutral_pion.Spectrum(photon_energies, pion_energy)
    return decay_neutral_pion.SpectrumPoint(photon_energies, pion_energy)


def charged_pion(photon_energies, pion_energy, mode="total"):
    r"""Compute gamma-ray spectrum from the charged pion decay :math:`\pi^{\pm}
    \to \mu^{\pm} \nu_{\mu} \to e^{\pm} \nu_{e} \nu_{\mu} \gamma`.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    pion_energy : double
        Charged pion energy in laboratory frame.
    mode : str {"total"}
        The mode the user would like to have returned. The options are "total",
        "munu", "munug" and "enug".

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

    if mode != "total" and mode != "munu" and mode != "munug" and mode != "enug":
        val_err_mess = (
            "mode '{}'' is not availible. Please use 'total'"
            + "'munu', 'munug' or 'enug'.".format(mode)
        )
        raise ValueError(val_err_mess)

    if hasattr(photon_energies, "__len__"):
        return decay_charged_pion.Spectrum(photon_energies, pion_energy, mode)
    return decay_charged_pion.SpectrumPoint(photon_energies, pion_energy, mode)


def charged_kaon(photon_energies, kaon_energy, mode="total"):
    r"""Compute gamma-ray spectrum from charged kaon decay into various final states.

    Parameters
    ----------
    photon_energies : float or numpy.ndarray
        Photon energy(ies) in laboratory frame.
    kaon_energy : float
        Charged kaon energy in laboratory frame.
    mode : str {"total"}
        The mode the user would like to have returned. The options are "total",
        "0enu", "0munu", "00p", "mmug", "munu", "p0", "p0g" and "ppm". Here
        "p" stands for pi plus, "m" stands for pi minus and "0" stands pi 0.

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

    if (
        mode != "total"
        and mode != "0enu"
        and mode != "0munu"
        and mode != "00p"
        and mode != "mmug"
        and mode != "munu"
        and mode != "p0"
        and mode != "p0g"
        and mode != "ppm"
    ):
        val_err_mess = (
            "mode '{}'' is not availible. Please use 'total'"
            + "'0enu', '0munu', '00p', 'mmug', 'munu', 'p0',"
            + " 'p0g' or 'ppm'.".format(mode)
        )
        raise ValueError(val_err_mess)

    if hasattr(photon_energies, "__len__"):
        return decay_charged_kaon.Spectrum(photon_energies, kaon_energy, mode)
    return decay_charged_kaon.SpectrumPoint(photon_energies, kaon_energy, mode)


def short_kaon(photon_energies, kaon_energy, mode="total"):
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
    if mode != "total" and mode != "00" and mode != "pm" and mode != "pmg":
        val_err_mess = (
            "mode '{}'' is not available. Please use 'total'"
            + "'00', 'pm' or 'pmg'.".format(mode)
        )
        raise ValueError(val_err_mess)

    if hasattr(photon_energies, "__len__"):
        return decay_short_kaon.Spectrum(photon_energies, kaon_energy, mode)
    return decay_short_kaon.SpectrumPoint(photon_energies, kaon_energy, mode)


def long_kaon(photon_energies, kaon_energy, mode="total"):
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
    if (
        mode != "total"
        and mode != "000"
        and mode != "penu"
        and mode != "penug"
        and mode != "pm0"
        and mode != "pm0g"
        and mode != "pmunu"
        and mode != "pmunug"
    ):
        val_err_mess = (
            "mode '{}'' is not availible. Please use 'total'"
            + "'000', 'penu', 'penug', 'pm0', 'pm0g',"
            + "'pmunu' or 'pmunug'.".format(mode)
        )
        raise ValueError(val_err_mess)

    if hasattr(photon_energies, "__len__"):
        return decay_long_kaon.Spectrum(photon_energies, kaon_energy, mode)
    return decay_long_kaon.SpectrumPoint(photon_energies, kaon_energy, mode)


def electron(photon_energies, electron_energy):
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
