"""
Module for computing positron spectra.

@author: Logan Morrison and Adam Coogan
@date: May 2018

"""

from hazma.positron_helper_functions import positron_muon
from hazma.positron_helper_functions import positron_charged_pion
from hazma.positron_helper_functions.positron_rambo\
    import positron, positron_point

import numpy as np


def muon(eng_p, eng_mu):
    """
    Returns the positron spectrum from a muon.

    Parameters
    ----------
    eng_p : float or numpy.array
        Energy of the positron.
    eng_mu : float or array-like
        Energy of the muon.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy `ee`
        and muon energy `emu`
    """
    if hasattr(eng_p, "__len__"):
        return positron_muon.Spectrum(eng_p, eng_mu)
    return positron_muon.SpectrumPoint(eng_p, eng_mu)


def charged_pion(eng_p, eng_pi):
    """
    Returns the positron spectrum from a muon.

    Parameters
    ----------
    eng_p : float or numpy.array
        Energy of the positron.
    eng_pi : float or numpy.array
        Energy of the charged pion.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies) `eng_p`
        and charged pion energy `eng_pi`
    """
    if hasattr(eng_p, "__len__"):
        return positron_charged_pion.Spectrum(eng_p, eng_pi)
    return positron_charged_pion.SpectrumPoint(eng_p, eng_pi)


def positron_rambo(particles, cme, eng_ps,
                   mat_elem_sqrd=lambda k_list: 1.0,
                   num_ps_pts=1000, num_bins=25, verbose=False):
    r"""Returns total gamma ray spectrum from a set of particles.

    Blah and blah.

    Parameters
    ----------
    isp_masses : np.ndarray[double, ndim=1]
        Array of masses of the initial state particles.
    particles : array_like
        List of particle names. Availible particles are 'muon', 'electron'
        'charged_pion', 'neutral pion', 'charged_kaon', 'long_kaon',
        'short_kaon'.
    cme : double
        Center of mass energy of the final state in MeV.
    eng_gams : np.ndarray[double, ndim=1]
        List of gamma ray energies in MeV to evaluate spectra at.
    mat_elem_sqrd : double(*func)(np.ndarray, )
        Function for the matrix element squared of the proccess. Must be a
        function taking in a list of four momenta of size (num_fsp, 4).
        Default value is a flat matrix element.
    num_ps_pts : int {1000}, optional
        Number of phase space points to use.
    num_bins : int {25}, optional
        Number of bins to use.

    Returns
    -------
    spec : np.ndarray
        Total gamma ray spectrum from all final state particles.

    Notes
    -----
    The total spectrum is computed using

    .. math::
        \frac{dN}{dE}(E_{\gamma})
        =\sum_{i,j}P_{i}(E_{j})\frac{dN_i}{dE}(E_{\gamma}, E_{j})

    where :math:`i` runs over the final state particles, :math:`j` runs over
    energies sampled from probability distributions. :math:`P_{i}(E_{j})` is
    the probability that particle :math:`i` has energy :math:`E_{j}`. The
    probabilities are computed using `hazma.phase_space_generator.rambo`. The
    total number of energies used is `num_bins`.

    Examples
    --------
    Example of generating a spectrum from a muon, charged kaon and long kaon
    with total energy of 5000 MeV.

    >>> from hazma.gamma_ray import gamma_ray
    >>> import numpy as np
    >>>
    >>> particles = np.array(['muon', 'charged_kaon', 'long_kaon'])
    >>> cme = 5000.
    >>> eng_gams = np.logspace(0., np.log10(cme), num=200, dtype=np.float64)
    >>>
    >>> spec = gamma_ray(particles, cme, eng_gams)

    """
    if type(particles) == str:
        particles = [particles]

    particles = np.array(particles)

    if hasattr(eng_ps, '__len__'):
        return positron(particles, cme, eng_ps,
                        mat_elem_sqrd=mat_elem_sqrd,
                        num_ps_pts=num_ps_pts,
                        num_bins=num_bins,
                        verbose=verbose)
    return positron_point(particles, cme, eng_ps, mat_elem_sqrd,
                          num_ps_pts, num_bins)
