
from decay_helper_functions import decay_long_kaon
from decay_helper_functions import decay_charged_pion
from decay_helper_functions import decay_charged_kaon
from decay_helper_functions import decay_muon
from decay_helper_functions import decay_neutral_pion
from decay_helper_functions import decay_short_kaon


def muon(eng_gam, eng_mu):
    """Compute dNdE from muon decay.

    Compute dNdE from decay mu -> e nu nu gamma in the laborartory frame given
    a gamma ray engergy of ``eng_gam`` and muon energy of ``eng_mu``.

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

    >>> from hazma import muon
    >>> eng_gam, eng_mu = 200., 1000.
    >>> spec = muon.decay_spectra(eng_gam, eng_mu)

    Calculate spectrum for array of gamma ray energies

    >>> from hazma import muon
    >>> import numpy as np
    >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
    >>> eng_mu = 1000.
    >>> spec = muon.decay_spectra(eng_gams, eng_mu)
    """
    # mu = decay_muon.Muon()
    if hasattr(eng_gam, "__len__"):
        return decay_muon.Spectrum(eng_gam, eng_mu)
    return decay_muon.SpectrumPoint(eng_gam, eng_mu)


def neutral_pion(eng_gam, eng_pi):
    """Compute dNdE from neutral pion decay.

    Compute dNdE from decay pi0 -> gamma gamma in the laborartory frame given
    a gamma ray engergy of `eng_gam` and neutral pion energy of `eng_pi`.

    Paramaters
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
    """
    if hasattr(eng_gam, '__len__'):
        return decay_neutral_pion.Spectrum(eng_gam, eng_pi)
    return decay_neutral_pion.SpectrumPoint(eng_gam, eng_pi)


def charged_pion(eng_gam, eng_pi):
    """Compute dNdE from charged pion decay.

    Compute dNdE from decay pi -> mu nu -> e nu nu g in the laborartory frame
    given a gamma ray engergy of `eng_gam` and muon energy of `eng_pi`.

    Paramaters
    ----------
    eng_gam : double or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_pi : double
        Charged pion energy in laboratory frame.

    Returns
    -------
    spec : double np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams` given
        charged pion energy `eng_pi`.

    Examples
    --------
    Calculate spectrum for single gamma ray energy

    >>> from hazma import charged_pion
    >>> eng_gam, eng_pi = 200., 1000.
    >>> spec = charged_pion.decay_spectra(eng_gam, eng_pi)

    Calculate spectrum for array of gamma ray energies

    >>> from hazma import charged_pion
    >>> import numpy as np
    >>> eng_gams = np.logspace(0.0, 3.0, num=200, dtype=float)
    >>> eng_pi = 1000.
    >>> spec = charged_pion.decay_spectra(eng_gams, eng_pi)
    """
    # charged_pi = decay_charged_pion.ChargedPion()
    if hasattr(eng_gam, "__len__"):
        return decay_charged_pion.Spectrum(eng_gam, eng_pi)
    return decay_charged_pion.SpectrumPoint(eng_gam, eng_pi)


def charged_kaon(eng_gam, eng_k):
    """Compute dNdE from charged kaon decay.

    Compute dNdE from decay of charged kaon through K -> X in the
    laboratory frame given a gamma ray engergy of `eng_gam` and charged
    kaon energy of `eng_k`. The decay modes impemendted are
    * k -> mu  + nu
    * k -> pi  + pi0
    * k -> pi  + pi  + pi
    * k -> pi0 + e   + nu
    * k -> pi0 + mu  + nu
    * k -> pi  + pi0 + pi0

    Paramaters
    ----------
    eng_gam : float or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_k : float
        Charged kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
        given muon energy `eng_mu`.
    """
    if hasattr(eng_gam, "__len__"):
        return decay_charged_kaon.Spectrum(eng_gam, eng_k)
    return decay_charged_kaon.SpectrumPoint(eng_gam, eng_k)


def short_kaon(eng_gam, eng_k):
    """Compute dNdE from short kaon decay.

    Compute dNdE from decay of short kaon through K -> X in the laboratory
    frame given a gamma ray engergy of `eng_gam` and short kaon energy of
    `eng_k`. The decay modes impemendted are
    * ks    -> pi  + pi
    * ks    -> pi0 + pi0

    Paramaters
    ----------
    eng_gam : double or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_k : float
        Charged kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
        given muon energy `eng_mu`.
    """
    if hasattr(eng_gam, "__len__"):
        return decay_short_kaon.Spectrum(eng_gam, eng_k)
    return decay_short_kaon.SpectrumPoint(eng_gam, eng_k)


def long_kaon(eng_gam, eng_k):
    """Compute dNdE from long kaon decay.

    Compute dNdE from decay of charged kaon through K -> X in the laboratory
    frame given a gamma ray engergy of `eng_gam` and long kaon energy of
    `eng_k`. The decay modes impemendted are
    * kl -> pi  + e   + nu
    * kl -> pi  + mu  + nu
    * kl -> pi0 + pi0  + pi0
    * kl -> pi  + pi  + pi0

    Paramaters
    ----------
    eng_gam : float or numpy.ndarray
        Gamma ray energy(ies) in laboratory frame.
    eng_k : float
        Charged kaon energy in laboratory frame.

    Returns
    -------
    spec : np.ndarray
        List of gamma ray spectrum values, dNdE, evaluated at `eng_gams`
        given muon energy `eng_mu`.
    """
    if hasattr(eng_gam, "__len__"):
        return decay_long_kaon.Spectrum(eng_gam, eng_k)
    return decay_long_kaon.SpectrumPoint(eng_gam, eng_k)
