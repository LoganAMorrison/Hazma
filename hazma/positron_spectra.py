"""
Module for computing positron spectra.

@author: Logan Morrison and Adam Coogan
@date: May 2018
"""

import numpy as np

from hazma.positron_helper_functions import positron_charged_pion, positron_muon
from hazma.positron_helper_functions.positron_decay import positron, positron_point


def muon(positron_energies, muon_energy):
    """
    Returns the positron spectrum from muon decay.

    Parameters
    ----------
    positron_energies : float or numpy.array
        Energy(ies) of the positron/electron.
    muon_energy : float or array-like
        Energy of the muon.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies)
        ``positron_energies`` and muon energy ``muon_energy``.
    """
    if hasattr(positron_energies, "__len__"):
        return positron_muon.Spectrum(positron_energies, muon_energy)
    return positron_muon.SpectrumPoint(positron_energies, muon_energy)


def charged_pion(positron_energies, pion_energy):
    """
    Returns the positron spectrum from the decay of a charged pion.

    Parameters
    ----------
    positron_energies : float or numpy.array
        Energy(ies) of the positron/electron.
    pion_energy : float or numpy.array
        Energy of the charged pion.

    Returns
    -------
    dnde : float or numpy.array
        The value of the spectrum given a positron energy(ies)
        ``positron_energies`` and charged pion energy ``pion_energy``.
    """
    if hasattr(positron_energies, "__len__"):
        return positron_charged_pion.Spectrum(positron_energies, pion_energy)
    return positron_charged_pion.SpectrumPoint(positron_energies, pion_energy)


def positron_decay(
    particles,
    cme,
    positron_energies,
    mat_elem_sqrd=lambda k_list: 1.0,
    num_ps_pts=1000,
    num_bins=25,
    verbose=False,
):
    r"""Returns total gamma ray spectrum from a set of particles.

    Parameters
    ----------
    particles : array_like
        List of particle names. Available particles are 'muon', and
        'charged_pion'.
    cme : double
        Center of mass energy of the final state in MeV.
    positron_energies : np.ndarray[double, ndim=1]
        List of positron energies in MeV to evaluate spectra at.
    mat_elem_sqrd : double(\*func)(np.ndarray)
        Function for the matrix element squared of the process. Must be a
        function taking in a list of four momenta of size (num_fsp, 4).
        Default value is a flat matrix element returning 1..
    num_ps_pts : int {1000}, optional
        Number of phase space points to use.
    num_bins : int {25}, optional
        Number of bins to use.
    verbose: Bool
        If True, then addition information about the runtime progress and state
        are displayed.

    Returns
    -------
    spec : np.ndarray
        Total gamma ray spectrum from all final state particles.

    Notes
    -----
    The total spectrum is computed using

    .. math::
        \frac{dN}{dE}(E_{e^{\pm}}) = \sum_{i,j}P_{i}(E_{j})
        \frac{dN_i}{dE}(E_{e^{\pm}}, E_{j})

    where :math:`i` runs over the final state particles, :math:`j` runs over
    energies sampled from probability distributions. :math:`P_{i}(E_{j})` is
    the probability that particle :math:`i` has energy :math:`E_{j}`. The
    probabilities are computed using ``hazma.phase_space_generator.rambo``. The
    total number of energies used is ``num_bins``.

    Examples
    --------

    Generate spectrum from a muon, and two charged pions
    with total energy of 5 GeV::

        from hazma.positron_spectra import positron_decay
        from hazma.parameters import electron_mass as me
        import numpy as np
        particles = np.array(['muon', 'charged_pion', 'charged_pion'])
        cme = 5000.
        positron_energies = np.logspace(np.log10(me), np.log10(cme),
                                        num=200, dtype=np.float64)
        positron_decay(particles, cme, positron_energies)

    """
    if type(particles) == str:
        particles = [particles]

    particles = np.array(particles)

    if hasattr(positron_energies, "__len__"):
        return positron(
            particles,
            cme,
            positron_energies,
            mat_elem_sqrd=mat_elem_sqrd,
            num_ps_pts=num_ps_pts,
            num_bins=num_bins,
            verbose=verbose,
        )
    return positron_point(
        particles, cme, positron_energies, mat_elem_sqrd, num_ps_pts, num_bins
    )
