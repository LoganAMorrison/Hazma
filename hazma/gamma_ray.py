"""
Module for computing gamma ray spectra from a many-particle final state.

* author : Logan Morrison and Adam Coogan
* date : December 2017
"""

from .gamma_ray_helper_functions.gamma_ray_generator import gamma, gamma_point


def gamma_ray(particles, cme, eng_gams, mat_elem_sqrd=lambda k_list: 1.0,
              num_ps_pts=1000, num_bins=25):
    """Returns total gamma ray spectrum from a set of particles.

    Blah Blah

    Parameters
    ----------

    particles : np.ndarray
        List of particle names. Availible particles are 'muon', 'electron'
        'charged_pion', 'neutral pion', 'charged_kaon', 'long_kaon',
        'short_kaon'
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
        \frac{dN}{dE}(E_{\gamma}) =
        \sum_{i,j}P_{i}(E_{j})\frac{dN_i}{dE}(E_{\gamma}, E_{j})

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

    if hasattr(eng_gams, '__len__'):
        return gamma(particles, cme, eng_gams, mat_elem_sqrd,
                     num_ps_pts, num_bins)
    return gamma_point(particles, cme, eng_gams, mat_elem_sqrd,
                       num_ps_pts, num_bins)
