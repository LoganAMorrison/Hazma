RAMBO
=====

Overview
--------

In ``hazma``, the ``rambo`` module can be used to perform various tasks. We
include a generic functions for generating phase-space points called
``generate_phase_space_point`` and ``generate_phase_space``, which will
compute a single or many phase-space points. The ``generate_phase_space``
function additionally allows for the user to specify a matrix element.
We include a function called ``integrate_over_phase_space`` which will
perform the integral:

.. math::

     \int \left(\prod_{i=1}^{N}\dfrac{d^{3}\vec{p}_{i}}{(2\pi)^3}\dfrac{1}{2E_{i}}\right)(2\pi)^4\delta^{4}\left(P-\sum_{i=1}^{N}p_{i}\right)\left|\mathcal{M}\right|^2

where :math:`P^{\mu}` is the total four-momentum, :math:`p_{i}^{\mu}` are
the individual four-momenta for each of the :math:`N` final-state particles
and :math:`\mathcal{M}` is the matrix element. ``rambo.py`` also contains
functions for computing cross-sections (decay widths) for :math:`2\to N`
(:math:`1\to N`) processes called ``compute_annihilation_cross_section``
(``compute_decay_width``).  For example, if we would like to compute the
partial decay width of :math:`\mu\to e\nu\nu`, one could use the following.
First, we declare a function for the matrix element. The function for the
matrix element must take in a list of the four-momenta and return the matrix
element:

.. code-block:: python

    # Import the fermi constant
    from hazma.parameters import GF
    # Import a helper function for scalar products of four-vectors
    from hazma.field_theory_helper_functions.common_functions import \
        minkowski_dot as MDot
    # Declare the matrix element
    def msqrd_mu_to_enunu(momenta):
        pe = momenta[0] # electron four-momentum
        pve = momenta[1] # electron-neutrino four-momentum
        pvmu = momenta[2] # muon-neutrino four-momentum
        pmu = sum(momenta) # muon four-momentum
        # Return matrix element
        return 64. * GF**2 * MDot(pe, pvmu) * MDot(pmu, pve)

Then, the partial decay width can be computed using:

.. code-block:: python

    # Import function to compute decay width
    from hazma.rambo import compute_decay_width
    # import masses of muon and electron
    from hazma.parameters import muon_mass as mmu
    from hazma.parameters import electron_mass as me
    # Specify the masses of the electron and neutrinos
    fsp_masses = np.array([me, 0., 0.])
    # compute the partial width
    partial_width = compute_decay_width(fsp_masses, mmu, num_ps_pts=50000,
                                            mat_elem_sqrd=msqrd_mu_to_enunu)

Using 50000 phase-space points, we are able to within :math:`5\%` of the
analytical result. In addition, ``rambo`` includes a function for
performing partial integrations over all variables except the energy of
one of the final-state particles called ``generate_energy_histogram``.
This function returns a multi-dimensional array with the first index
labeling the final-state particles and zeroth component of the second
index given the energies and the 1 component of the second index giving
the probability that the final-state particle has the particular energy.
This function can be used via:

.. code-block:: python

    from hazma.rambo import generate_energy_histogram
    import numpy as np
    num_ps_pts = 100000 #number of phase-space points to use
    # masses of final-state particles
    masses = np.array([100., 100., 0.0, 0.0])
    cme = 1000. # center-of-mass energy
    num_bins = 100 # number of energy bins to use
    # computing energy histograms
    eng_hist = generate_energy_histogram(num_ps_pts, masses, cme,
                                         num_bins=num_bins)
    # plot the results
    import matplotlib as plt
    for i in range(len(masses)):
        # pts[i, 0] are the energies of particle i
        # pts[i, 1] are the probabilities
        plt.loglog(pts[i, 0], pts[i, 1])

Functions
---------

.. autofunction:: hazma.rambo.generate_phase_space_point

.. autofunction:: hazma.rambo.generate_phase_space

.. autofunction:: hazma.rambo.generate_energy_histogram

.. autofunction:: hazma.rambo.integrate_over_phase_space

.. autofunction:: hazma.rambo.compute_annihilation_cross_section

.. autofunction:: hazma.rambo.compute_decay_width
