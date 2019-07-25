Gamma ray spectra
=================

Overview
--------

This page discusses how to compute the gamma-ray spectrum for a particular
particle physics process. Since computing gamma-ray spectra is a
model-dependent process, we include in ``hazma`` tools for computing gamma-ray
spectra from *both* FSR and the decay of final state-particles.

The ``gamma_ray`` module contains two
functions called ``gamma_ray_decay`` and ``gamma_ray_fsr``. The
``gamma_ray_decay`` function accepts a list of the final-state particles,
the center-of-mass energy, the gamma-ray energies to compute the spectrum
at and optionally the matrix element. Currently, the final-state particles
can be :math:`\pi^{0}`, :math:`\pi^{\pm}`, :math:`\mu^{\pm}`,
:math:`e^{\pm}`, :math:`K^{\pm}`, :math:`K_{L}` and :math:`K_{S}` where
:math:`K` stands for kaon.

We caution that when including many final-state mesons, one needs to take
care to supply the properly unitarized matrix element. The ``gamma_ray_decay``
functions works by first computing the energies distributions of all the
final-state particles and convolving the energy distributions with the decay
spectra of the final-state particles. The ``gamma_ray_decay`` function can be
used as follows:

.. code-block:: python

    >>> from hazma.gamma_ray_decay import gamma_ray_decay
    >>> import numpy as np
    # specify the final-state particles
    >>> particles = np.array(['muon', 'charged_kaon', 'long_kaon'])
    # choose the center of mass energy
    >>> cme = 5000.
    # choose list of the gamma-ray energies to compute spectra at
    >>> eng_gams = np.logspace(0., np.log10(cme), num=200, dtype=np.float64)
    # compute the gamma-ray spectra assuming a constant matrix element
    >>> spec = gamma_ray_decay(particles, cme, eng_gams)

The ``gamma_ray_fsr`` function computes the gamma-ray spectrum from
:math:`X\to Y\gamma`, i.e.:

.. math::

    \dfrac{dN(X\to Y\gamma)}{dE_{\gamma}} = \dfrac{1}{\sigma(X\to Y)}\dfrac{d\sigma(X\to Y\gamma)}{dE_{\gamma}}

where :math:`X` and :math:`Y` are any particles excluding the photon. This
function takes in as input a list of the initial state particle masses
(either 1 or 2 particles), the final state particle masses, the
center-of-mass energy, a function for the tree-level matrix element
(for :math:`X\to Y`) and a function for the radiative matrix element
(:math:`X\to Y\gamma`). The functions for the matrix elements must take is
a single argument which is a list of the four-momenta for the final state
particles. As an example, we consider the process of two dark-matter
particles annihilating into charged pions,
:math:`\bar{\chi}\chi\to \pi^{+}\pi^{-}(\gamma)` using the model from
:ref:`advanced_usage`. In :ref:`advanced_usage`, we gave the
analytic expressions for the gamma-ray spectra. The tree-level and
radiative matrix elements for this process are:

.. math::

    |\mathcal{M}(\bar{\chi}\chi\to \pi^{+}\pi^{-})|^2 &= \frac{c_1^2 \left(s-4 m_{\chi}^2\right)}{2 \Lambda^2}\\
    |\mathcal{M}(\bar{\chi}\chi\to \pi^{+}\pi^{-}\gamma)|^2 &= \dfrac{2 c_1^2 \left(4 \mu_{\chi}^2-1\right) Q^2 e^2}{\Lambda^2 \left(t-\mu_{\pi}^2 Q^2\right)^2
    \left(u-\mu_{\pi}^2 Q^2\right)^2}\\
    &\qquad \times \left(\left(\mu_{\pi} Q (t+u)-2 \mu_{\pi}^3 Q^3\right)^2+s
    \left(t-\mu_{\pi}^2 Q^2\right) \left(\mu_{\pi}^2
    Q^2-u\right)\right)

where :math:`Q` is the center-of-mass energy, :math:`e` is the
electromagnetic coupling, :math:`\mu_{\pi,\chi} = m_{\pi,\chi}/Q` and

.. math::

    s = (p_{\pi,1} + p_{\pi,2})^2, t = (p_{\pi,1} + k)^2,  u = (p_{\pi,2} + k)^2

with :math:`p_{\pi,1,2}` are the four-momenta of the two final-state pions
and :math:`k` is the four-momenta of the final-state photon. Below, we
create a class to implement functions for the tree and radiative matrix
elements. Note that these functions take in an array of four-momenta.

.. code-block:: python

    from hazma.field_theory_helper_functions.common_functions import \
    minkowski_dot as MDot

    class Msqrd(object):
        def __init__(self, mx, c1, lam):
            self.mx = mx # DM mass
            self.c1 = c1 # effective coupling of DM to charged pions
            self.lam = lam # cut off scale for effective theory

        def tree(self, momenta):
            ppi1 = momenta[0] # first charged pion four-momentum
            ppi2 = momenta[1] # second charged pion four-momentum
            Q = ppi1[0] + ppi2[0] # center-of-mass energy
            return -((self.c1**2 * (4 * self.mx**2-Q**2)) / (2 * self.lam**2))

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

            return ((2*self.c1**2*(-1 + 4*mux**2)*Q**2*qe**2 *
                     (s*(-(mupi**2*Q**2) + t)*(mupi**2*Q**2 - u) +
                      (-2*mupi**3*Q**3 + mupi*Q*(t + u))**2)) /
                    (self.lam**2*(-(mupi**2*Q**2) + t)**2*
                    (-(mupi**2*Q**2) + u)**2))

Next, we can compute the gamma-ray spectrum for
:math:`\bar{\chi}\chi\to\pi^{+}\pi^{-}\gamma` using:

.. code-block:: python

    >>> from hazma.gamma_ray import gamma_ray_fsr
    # specify the parameters of the model
    >>> params = {'mx': 200.0, 'c1':1.0, 'lam':1e4}
    # create instance of our Msqrd class
    >>> msqrds = Msqrd(**params)
    # specify the initial and final state masses
    >>> isp_masses = np.array([msqrds.mx, msqrds.mx])
    >>> fsp_masses = np.array([mpi, mpi, 0.0])
    # choose the center-of-mass energy
    >>> cme = 4.0 * msqrds.mx
    # compute the gamma-ray spectrum
    >>> spec = gamma_ray_fsr(isp_masses, fsp_masses, cme, msqrds.tree,
                             msqrds.radiative, num_ps_pts=500000, num_bins=50)
    # plot the spectrum
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(dpi=100)
    >>> plt.plot(spec[0], spec[1])
    >>> plt.yscale('log')
    >>> plt.xscale('log')
    >>> plt.ylabel(r'$dN/dE_{\gamma} \ (\mathrm{MeV}^{-1})$', fontsize=16)
    >>> plt.xlabel(r'$E_{\gamma} \ (\mathrm{MeV})$', fontsize=16)

Functions
---------

.. autofunction:: hazma.gamma_ray.gamma_ray_decay

.. autofunction:: hazma.gamma_ray.gamma_ray_fsr
