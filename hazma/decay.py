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


def muon(eng_gam, eng_mu):
    r"""Compute dNdE from muon decay.

    Compute dNdE from decay :math:`\mu^{\pm} \to e^{\pm} + \nu_{e} +\nu_{\mu}
    + \gamma` in the laborartory frame given a gamma ray engergy of ``eng_gam``
    and muon energy of ``eng_mu``.

    Parameters
    ----------
    eng_gam : numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_mu : double
        Muon energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at ``eng_gam`` given
        muon energy ``eng_mu``.

    Examples
    --------
    Calculate spectrum for single gamma ray energy

    >>> from hazma import decay
    >>> eng_gam, eng_mu = 200., 1000.
    >>> spec = decay.muon(eng_gam, eng_mu)

    Calculate spectrum for array of gamma ray energies

    >>> from hazma import decay
    >>> import numpy as np
    >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
    >>> eng_mu = 1000.
    >>> spec = decay.muon(eng_gams, eng_mu)

    """
    # mu = decay_muon.Muon()
    if hasattr(eng_gam, "__len__"):
        return decay_muon.Spectrum(eng_gam, eng_mu)
    return decay_muon.SpectrumPoint(eng_gam, eng_mu)


def neutral_pion(eng_gam, eng_pi):
    r"""Compute dNdE from neutral pion decay.

    Compute dNdE from decay :math:`\pi^{0} \to \gamma + \gamma` in the
    laborartory frame given a gamma ray engergy of ``eng_gam`` and neutral pion
    energy of ``eng_pi``.

    Parameters
    ----------
    eng_gam : double or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_pi : float
        Neutral pion energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at
        `eng_gams` given neutral pion energy `eng_pi`.

    Examples
    --------
    Calculate spectrum for single gamma ray energy

    >>> from hazma import decay
    >>> eng_gam, eng_pi = 200., 1000.
    >>> spec = decay.neutral_pion(eng_gam, eng_pi)

    Calculate spectrum for array of gamma ray energies

    >>> from hazma import decay
    >>> import numpy as np
    >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
    >>> eng_pi = 1000.
    >>> spec = decay.neutral_pion(eng_gams, eng_pi)

    """
    if hasattr(eng_gam, '__len__'):
        return decay_neutral_pion.Spectrum(eng_gam, eng_pi)
    return decay_neutral_pion.SpectrumPoint(eng_gam, eng_pi)


def charged_pion(eng_gam, eng_pi, mode="total"):
    r"""Compute dNdE from charged pion decay.

    Compute dNdE from decay :math:`\pi^{\pm} \to \mu^{\pm} + \nu_{\mu} \to
    e^{\pm} + \nu_{e} + \nu_{\mu} + \gamma` in the laborartory frame given a
    gamma ray engergy of ``eng_gam`` and muon energy of ``eng_pi``.

    Parameters
    ----------
    eng_gam : double or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_pi : double
        Charged pion energy in laboratory frame.
    mode : str {"total"}
        The mode the user would like to have returned. The options are "total",
        "munu", "munug" and "enug".

    Returns
    -------
    spec : double np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams` given
        charged pion energy `eng_pi`.

    Examples
    --------
    Calculate spectrum for single gamma ray energy

    >>> from hazma import decay
    >>> eng_gam, eng_pi = 200., 1000.
    >>> spec = decay.charged_pion(eng_gam, eng_pi)

    Calculate spectrum for array of gamma ray energies

    >>> from hazma import decay
    >>> import numpy as np
    >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
    >>> eng_pi = 1000.
    >>> spec = decay.charged_pion(eng_gams, eng_pi)

    """

    if mode != "total" and mode != "munu" and mode != "munug" and \
            mode != "enug":
        val_err_mess = "mode '{}'' is not availible. Please use 'total'" +\
            "'munu', 'munug' or 'enug'.".format(mode)
        raise ValueError(val_err_mess)

    if hasattr(eng_gam, "__len__"):
        return decay_charged_pion.Spectrum(eng_gam, eng_pi, mode)
    return decay_charged_pion.SpectrumPoint(eng_gam, eng_pi, mode)


def charged_kaon(eng_gam, eng_k, mode="total"):
    r"""Compute dNdE from charged kaon decay.

    Compute dNdE from decay of charged kaon through :math:`K\to X` in the
    laboratory frame given a gamma ray engergy of ``eng_gam`` and charged
    kaon energy of ``eng_k``.

    Parameters
    ----------
    eng_gam : float or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_k : float
        Charged kaon energy in laboratory frame.
    mode : str {"total"}
        The mode the user would like to have returned. The options are "total",
        "0enu", "0munu", "00p", "mmug", "munu", "p0", "p0g" and "ppm". Here
        "p" stands for pi plus, "m" stands for pi minus and "0" stands pi 0.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at ``eng_gams``
        given muon energy ``eng_mu``.

    Notes
    -----
    The decay modes impemendted are

    .. math:: K^{\pm} \to \mu^{\pm}  + \nu_{\mu}

    .. math:: K^{\pm} \to \pi^{\pm}  + \pi^{0}

    .. math:: K^{\pm} \to \pi^{\pm} + \pi^{\mp} + \pi^{\pm}

    .. math:: K^{\pm} \to e^{\pm}  + \nu_{e}

    .. math:: K^{\pm} \to \mu^{\pm}  + \nu_{\mu} + \pi^{0}

    .. math:: K^{\pm} \to \pi^{\pm} + \pi^{0} + \pi^{0}

    Examples
    --------
    Calculate spectrum for single gamma ray energy

    >>> from hazma import decay
    >>> eng_gam, eng_k = 200., 1000.
    >>> spec = decay.charged_kaon(eng_gam, eng_k)

    Calculate spectrum for array of gamma ray energies

    >>> from hazma import decay
    >>> import numpy as np
    >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
    >>> eng_k = 1000.
    >>> spec = decay.charged_kaon(eng_gams, eng_k)

    """

    if mode != "total" and mode != "0enu" and mode != "0munu" and \
            mode != "00p" and mode != "mmug" and mode != "munu" and \
            mode != "p0" and mode != "p0g" and mode != "ppm":
        val_err_mess = "mode '{}'' is not availible. Please use 'total'" +\
            "'0enu', '0munu', '00p', 'mmug', 'munu', 'p0'," +\
            " 'p0g' or 'ppm'.".format(mode)
        raise ValueError(val_err_mess)

    if hasattr(eng_gam, "__len__"):
        return decay_charged_kaon.Spectrum(eng_gam, eng_k, mode)
    return decay_charged_kaon.SpectrumPoint(eng_gam, eng_k, mode)


def short_kaon(eng_gam, eng_k, mode="total"):
    r"""Compute dNdE from short kaon decay.

    Compute dNdE from decay of short kaon through :math:`K\to X` in the
    laboratory frame given a gamma ray engergy of ``eng_gam`` and short kaon
    energy of ``eng_k``.

    Parameters
    ----------
    eng_gam : double or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_k : float
        Charged kaon energy in laboratory frame.
    mode : str
        The mode the user would like to have returned. The options are "total",
        "00", "pm" or "pmg". Here "p" stands for pi plus, "m" stands for pi
        minus and "0" stands pi 0.
    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at ``eng_gams``
        given muon energy ``eng_mu``.

    Notes
    -----
    The decay modes impemendted are

    .. math:: K_{S} \to \pi^{+}  + \pi^{-}

    .. math:: K_{S} \to \pi^{0} + \pi^{0}

    Examples
    --------
    Calculate spectrum for single gamma ray energy

    >>> from hazma import decay
    >>> eng_gam, eng_ks = 200., 1000.
    >>> spec = decay.short_kaon(eng_gam, eng_ks)

    Calculate spectrum for array of gamma ray energies

    >>> from hazma import decay
    >>> import numpy as np
    >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
    >>> eng_ks = 1000.
    >>> spec = decay.short_kaon(eng_gams, eng_ks)

    """
    if mode != "total" and mode != "00" and mode != "pm" and \
            mode != "pmg":
        val_err_mess = "mode '{}'' is not availible. Please use 'total'" +\
            "'00', 'pm' or 'pmg'.".format(mode)
        raise ValueError(val_err_mess)

    if hasattr(eng_gam, "__len__"):
        return decay_short_kaon.Spectrum(eng_gam, eng_k, mode)
    return decay_short_kaon.SpectrumPoint(eng_gam, eng_k, mode)


def long_kaon(eng_gam, eng_k, mode="total"):
    r"""Compute dNdE from long kaon decay.

    Compute dNdE from decay of charged kaon through :math:`K\to X` in the
    laboratory frame given a gamma ray engergy of ``eng_gam`` and long kaon
    energy of ``eng_k``.

    Parameters
    ----------
    eng_gam : float or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_k : float
        Charged kaon energy in laboratory frame.
    mode : str
        The mode the user would like to have returned. The options are "total",
        "000", "penu", "penug", "pm0", "pm0g", "pmunu" or "pmunug". Here "p"
        stands for pi plus, "m" stands for pi minus and "0" stands pi 0.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
        given muon energy `eng_mu`.

    Examples
    --------
    Calculate spectrum for single gamma ray energy

    >>> from hazma import decay
    >>> eng_gam, eng_kl = 200., 1000.
    >>> spec = decay.long_kaon(eng_gam, eng_kl)

    Calculate spectrum for array of gamma ray energies

    >>> from hazma import decay
    >>> import numpy as np
    >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
    >>> eng_kl = 1000.
    >>> spec = decay.long_kaon(eng_gams, eng_kl)

    Notes
    -----
    The decay modes impemendted are

    .. math:: K_{L} \to \pi^{\pm} + e^{\pm} + \nu_{e}

    .. math:: K_{L} \to \pi^{\pm} + \mu^{\mp} + \nu_{\mu}

    .. math:: K_{L} \to \pi^{0} + \pi^{0} + \pi^{0}

    .. math:: K_{L} \to \pi^{\pm} + \pi^{\mp} + \pi^{0}

    """
    if mode != "total" and mode != "000" and mode != "penu" and \
            mode != "penug" and mode != "pm0" and mode != "pm0g" and \
            mode != "pmunu" and mode != "pmunug":
        val_err_mess = "mode '{}'' is not availible. Please use 'total'" +\
            "'000', 'penu', 'penug', 'pm0', 'pm0g'," +\
            "'pmunu' or 'pmunug'.".format(mode)
        raise ValueError(val_err_mess)

    if hasattr(eng_gam, "__len__"):
        return decay_long_kaon.Spectrum(eng_gam, eng_k, mode)
    return decay_long_kaon.SpectrumPoint(eng_gam, eng_k, mode)


def electron(eng_gam, eng_e):
    r"""Compute dNdE from electron decay (returns zero).


    Parameters
    ----------
    eng_gam : numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_mu : double
        Electron energy in laboratory frame.

    Returns
    -------
    spec : numpy.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at ``eng_gam`` given
        muon energy ``eng_e``.
    """
    # mu = decay_muon.Muon()
    if hasattr(eng_gam, "__len__"):
        return np.array([0.0 for _ in eng_gam])
    return 0.0
