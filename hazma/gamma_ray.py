"""Module for computing gamma ray spectra from a many-particle final state.

"""
# author : Logan Morrison and Adam Coogan
# date : December 2017

from hazma import rambo
from hazma.gamma_ray_helper_functions.gamma_ray_generator import gamma, gamma_point
from hazma.field_theory_helper_functions.common_functions import cross_section_prefactor
import numpy as np


def gamma_ray_decay(
    particles,
    cme,
    photon_energies,
    mat_elem_sqrd=lambda k_list: 1.0,
    num_ps_pts=1000,
    num_bins=25,
    verbose=False,
):
    r"""Returns gamma ray spectrum from the decay of a set of particles.

    This function works by running the Monte-Carlo algorithm RAMBO on the
    final state particles to obtain energies distributions for each final state
    particle. The energies distributions are then convolved with the decay
    spectra in order to obtain the gamma-ray spectrum.

    Parameters
    ----------
    particles : `array_like`
        List of particle names. Available particles are 'muon', 'electron'
        'charged_pion', 'neutral pion', 'charged_kaon', 'long_kaon', 'short_kaon'.
    cme : double
        Center of mass energy of the final state in MeV.
    photon_energies : np.ndarray[double, ndim=1]
        List of photon energies in MeV to evaluate spectra at.
    mat_elem_sqrd : double(\*func)(np.ndarray, )
        Function for the matrix element squared of the process. Must be a
        function taking in a list of four momenta of shape (len(particles), 4).
        Default value is a flat matrix element with a value of 1.
    num_ps_pts : int {1000}, optional
        Number of phase space points to use.
    num_bins : int {25}, optional
        Number of bins to use.
    verbose: Bool
        If true, additional output is displayed while the function is computing
        spectra.

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
    probabilities are computed using ``hazma.phase_space_generator.rambo``. The
    total number of energies used is ``num_bins``.

    Examples
    --------

    Example of generating a spectrum from a muon, charged kaon and long kaon
    with total energy of 5000 MeV::

        from hazma.gamma_ray import gamma_ray_decay
        import numpy as np
        particles = np.array(['muon', 'charged_kaon', 'long_kaon'])
        cme = 5000.
        photon_energies = np.logspace(0., np.log10(cme), num=200, dtype=np.float64)
        gamma_ray_decay(particles, cme, photon_energies)

    """
    if type(particles) == str:
        particles = [particles]

    particles = np.array(particles)

    if hasattr(photon_energies, "__len__"):
        return gamma(
            particles,
            cme,
            photon_energies,
            mat_elem_sqrd=mat_elem_sqrd,
            num_ps_pts=num_ps_pts,
            num_bins=num_bins,
            verbose=verbose,
        )
    return gamma_point(
        particles, cme, photon_energies, mat_elem_sqrd, num_ps_pts, num_bins
    )


def gamma_ray_fsr(
    isp_masses,
    fsp_masses,
    cme,
    mat_elem_sqrd_tree=lambda k_list: 1.0,
    mat_elem_sqrd_rad=lambda k_list: 1.0,
    num_ps_pts=1000,
    num_bins=25,
):
    r"""Returns the FSR spectrum for a user-specified particle physics process.

    This is done by first running a Monte Carlo to compute the non-radiative
    cross cross_section, :math:`\sigma(I \to F)` from the non-radiative squared
    matrix element: :math:`|M(I \to F)|^2`. The differential radiative cross
    section, :math:`\frac{d\sigma(I \to F + \gamma)}{dE_{\gamma}}`, is then
    computed using a Monte Carlo using the radiative squared matrix element
    :math:`|M(I \to F + \gamma)|^2`. The spectrum is then

    .. math:: \frac{dN}{dE_{\gamma}} = \frac{1}{\sigma(I \to F)}
        \frac{d\sigma(I \to F + \gamma)}{dE_{\gamma}}

    Parameters
    ----------
    isp_masses : np.ndarray[double, ndim=1]
        Array of masses of the initial state particles.
    fsp_masses : np.ndarray[double, ndim=1]
        Array of masses of the final state particles. Note that the photon
        should be the last mass.
    cme : double
        Center of mass energy of the process.
    mat_elem_sqrd_tree : double(*)(np.ndarray[double, ndim=1])
        Tree level squared matrix element.
    mat_elem_sqrd_rad : double(*)(np.ndarray[double, ndim=1])
        Radiative squared matrix element.
    num_ps_pts : int
        Number of Monte Carlo events to generate.
    num_bins : int
        Number of gamma ray energies to use.

    Returns
    -------
    photon_energies : np.ndarray[double, ndim=1]
        Array of the gamma ray energies.
    spectrum : np.ndarray[double, ndim=1]
        Array of the spectrum values evaluated at the gamma ray energies.

    Examples
    --------

    Compute spectrum from dark matter annihilating into a pair of charged
    pion through via an effective operator:

        .. math:: \frac{c_1}{\Lambda}\chi\bar{\chi}\pi^{+}\pi^{-}

    Step up a class to compute the tree and radiative matrix elements. We could
    just specify the functions, but we would have to use globals to specify
    parameters. We find a class easier::


        from hazma.gamma_ray import gamma_ray_fsr
        from hazma.parameters import charged_pion_mass as mpi
        from hazma.parameters import qe # electric charge
        from hazma.field_theory_helper_functions.common_functions \
            import minkowski_dot as MDot
        class Msqrds:
            def __init__(self, mx, c1, lam):
                self.mx = mx
                self.c1 = c1
                self.lam = lam
            def tree(self, momenta):
                ppi1 = momenta[0] # first charged pion four-momentum
                ppi2 = momenta[1] # second charged pion four-momentum
                cme = ppi1[0] + ppi2[0] # center-of-mass energy
                return c1**2 * (cme**2 - 4 * self.mx**2) / (2 * self.lam**2)
            def radiative(self, momenta):
                ppi1 = momenta[0] # first charged pion four-momentum
                ppi2 = momenta[1] # second charged pion four-momentum
                k = momenta[2] # photon four-momentum
                Q = ppi1[0] + ppi2[0] + k[0] # center-of-mass energy
                mux = self.mx / Q
                mupi = mpi / Q
                s = MDot(ppi1 + ppi2, ppi1 + ppi2)
                t = MDot(ppi1 + k, ppi1 + k)
                u = MDot(ppi2 + k, ppi2 + k)
                return ((2 * self.c1**2 * (-1 + 4*mux**2) * Q**2 * qe**2 *
                        (s * (-(mupi**2 * Q**2) + t) * (mupi**2 * Q**2 - u) +
                        (-2 * mupi**3 * Q**3 + mupi * Q * (t + u))**2)) /
                        (self.lam**2 * (-(mupi**2 * Q**2) + t)**2 *
                        (-(mupi**2 * Q**2) + u)**2)

    Now we instantiate an object for the matrix elements, specify the
    parameters to pass to ``gamma_ray_fsr`` and compute spectra::

        msqrds = Msqrds(mx=200.0, c1=1.0, lam=1e6)
        num_ps_pts = 10**6
        isp_masses = np.array([msqrds.mx, msqrds.mx])
        fsp_masses = np.array([mpi, mpi, 0.0])
        cme = 2.0 * msqrds.mx * (1 + 0.5 * 1e-6)
        num_bins = 150
        gamma_ray_fsr(isp_masses, fsp_masses, cme, msqrds.tree,
                      msqrds.radiative, num_ps_pts, num_bins)

    """
    if len(isp_masses) == 1:
        cross_section = rambo.compute_decay_width(
            num_ps_pts, fsp_masses[0:-1], cme, mat_elem_sqrd=mat_elem_sqrd_tree
        )[0]

        pre_factor = 1.0 / (2.0 * cme)
    else:
        cross_section = rambo.compute_annihilation_cross_section(
            isp_masses,
            fsp_masses[0:-1],
            cme,
            num_ps_pts=num_ps_pts,
            mat_elem_sqrd=mat_elem_sqrd_tree,
        )[0]

        m1 = isp_masses[0]
        m2 = isp_masses[1]
        pre_factor = cross_section_prefactor(m1, m2, cme)

    eng_hists = rambo.generate_energy_histogram(
        fsp_masses,
        cme,
        num_ps_pts=num_ps_pts,
        mat_elem_sqrd=mat_elem_sqrd_rad,
        num_bins=num_bins,
    )[0]

    photon_energies = eng_hists[-1, 0]
    spectrum = eng_hists[-1, 1] * pre_factor / cross_section

    return photon_energies, spectrum
