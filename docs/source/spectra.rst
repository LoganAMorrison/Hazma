.. currentmodule:: hazma.spectra

Spectra (:mod:`hazma.spectra`)
==============================

Overview
--------

The :mod:`hazma.spectra` module contains functions for:

* Computing photon,positron and neutrino spectra from the decays of individual SM particles,
* Computing spectra from an :math:`N`-body final state,
* Boosting spectra into new frames,
* Computing FSR spectra using Altarelli-Parisi splitting functions.

Below, we demonstrate how each of these actions can be easily performed in ``hazma``.

Computing spectra (General API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute the spectra for an arbitrary final state, use
:meth:`hazma.spectra.dnde_photon`, :meth:`hazma.spectra.dnde_positron` or
:meth:`hazma.spectra.dnde_neutrino`. These functions takes the energies to compute the
spectra, the center-of-mass energy and the final states and returns the
spectrum.

.. plot::
   :include-source: True

   import numpy as np
   from hazma import spectra

   cme = 1500.0 # 1.5 GeV
   energies = np.geomspace(1e-3, 1.0, 100) * cme

   # Compute the photon spectrum from a muon, a long kaon and a charged kaon
   dnde = spectra.dnde_photon(energies, cme, ["mu", "kl", "k"])

   plt.plot(energies, energies **2 * dnde)
   plt.yscale("log")
   plt.xscale("log")
   plt.xlabel(r"$E_{\gamma} \ [\mathrm{MeV}]$", fontsize=16)
   plt.ylabel(r"$E_{\gamma}^2\dv*{N}{E_{\gamma}} \ [\mathrm{MeV}]$", fontsize=16)
   plt.tight_layout()
   plt.ylim(1e-1, 2e2)
   plt.xlim(np.min(energies), np.max(energies))
   plt.show()


Under the hood, ``hazma`` computes the spectra from an :math:`N`-body final state
by computing the expectation values of the decay spectra from each particle
w.r.t. each particles energy distribution. In the case of photon spectra, we
include both the spectra from the decays of final state particles as well as the
final state radiation from charged states. The final spectrum is:

.. math::

    \frac{dN}{dE} =
    \sum_{f}\frac{dN_{\mathrm{dec.}}}{dE} +
    \sum_{f}\frac{dN_{\mathrm{FSR}}}{dE}

where the **decay** and **FSR** components are:

.. math::

    \frac{dN_{\mathrm{dec}}}{dE} &= \sum_{f}\int{d\epsilon_f}
    P(\epsilon_f) \frac{dN_{f}}{dE}(E,\epsilon_f)\\
    \frac{dN}{dE} &= \sum_{i\neq j}\frac{1}{S_iS_j}\int{ds_{ij}}
    P(s_{ij}) \left(
    \frac{dN_{i,\mathrm{FSR}}}{dE}(E,s_{ij}) +
    \frac{dN_{j,\mathrm{FSR}}}{dE}(E,s_{ij})\right)

where :math:`P(\epsilon_f)` is the probability distribution of particle
:math:`f` having energy :math:`\epsilon_f`. The sum over `f` in the decay
expression is over all final-state particles. For the FSR expression,
:math:`s_{ij}` is the squared invariant-mass of particles :math:`i` and
:math:`j`. The factors :math:`S_{i}` and :math:`S_{j}` are symmetry factors
included to avoid overcounting.

We can see what the distributions look like using the :mod:`hazma.phase_space`
module. For three body final state consisting of a muon, long kaon and charged
pion, we can compute the energy distributions and invariant mass distributions
in a couple of ways. The first option is using Monte-Carlo integration using *Rambo*:

.. code:: python

          from hazma import phase_space
          from hazma.parameters import standard_model_masses as sm_masses

          cme = 1500.0
          states = ["mu", "kl", "k"]
          masses = tuple(sm_masses[s] for s in states)
          dists = phase_space.Rambo(cme, masses).energy_distributions(n=1<<14, nbins=30)

The second option is to use the specialized :class:`~hazma.phase_space.ThreeBody` class:

.. code:: python

          from hazma import phase_space
          from hazma.parameters import standard_model_masses as sm_masses

          cme = 1500.0
          states = ["mu", "kl", "k"]
          masses = tuple(sm_masses[s] for s in states)
          dists = phase_space.ThreeBody(cme, masses).energy_distributions(nbins=30)


.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  from hazma import phase_space
  from hazma.parameters import standard_model_masses as sm_masses

  cme = 1500.0
  states = ["mu", "kl", "k"]
  masses = tuple(sm_masses[s] for s in states)
  dists_r = phase_space.Rambo(cme, masses).energy_distributions(n=1<<14, nbins=30)
  dists_q = phase_space.ThreeBody(cme, masses).energy_distributions(nbins=30)

  plt.figure(dpi=150)
  colors = ["steelblue", "firebrick", "goldenrod"]
  labels = [r"$\mu^{\pm}$", r"$K_{L}$", r"$K^{\pm}$"]
  for i in range(3):
    plt.plot(dists_r[i].bin_centers, dists_r[i].probabilities, ls="--", c=colors[i], label=labels[i] + " (rambo)")
    plt.plot(dists_q[i].bin_centers, dists_q[i].probabilities, ls="-", c=colors[i], label=labels[i] + " (quad)")

  plt.xlabel(r"$\epsilon \ [\mathrm{MeV}]$", fontsize=16)
  plt.ylabel(r"$P(\epsilon) \ [\mathrm{MeV}^{-1}]$", fontsize=16)
  plt.tight_layout()
  plt.ylim(1e-3, 7e-3)
  # plt.xlim(np.min(energies), np.max(energies))
  plt.legend(ncol=3)
  plt.show()


For positron and neutrino spectra, the calculation is the same except that we omit the FSR.


Individual Particle Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hazma`` provides access to the individual spectra for all supported particles.
All the photon, positron and neutrino spectra follow the naming convention
``dnde_photon_*``, ``dnde_positron_*`` and ``dnde_neutrino_*``. Each spectrum function
takes in the energies of the photon/positron/neutrino and the energy of the
decaying particle are returns the spectrum. As an example, let's generate all
the photon spectra for the particles available in ``hazma``:

.. plot::
   :include-source: True

   import numpy as np
   import matplotlib.pyplot as plt
   from hazma import spectra

   cme = 1500.0
   es = np.geomspace(1e-4, 1.0, 100) * cme

   # Make a dictionary of all the available decay spectrum functions
   dnde_fns = {
      "mu":    spectra.dnde_photon_muon,
      "pi":    spectra.dnde_photon_charged_pion,
      "pi0":   spectra.dnde_photon_neutral_pion,
      "k":     spectra.dnde_photon_charged_kaon,
      "kl":    spectra.dnde_photon_long_kaon,
      "ks":    spectra.dnde_photon_short_kaon,
      "eta":   spectra.dnde_photon_eta,
      "etap":  spectra.dnde_photon_eta_prime,
      "rho":   spectra.dnde_photon_charged_rho,
      "rho0":  spectra.dnde_photon_neutral_rho,
      "omega": spectra.dnde_photon_omega,
      "phi":   spectra.dnde_photon_phi,
   }

   # Compute the spectra. Each function takes the energy where the spectrum
   # is evaluated as the 1st argument and the energy of the decaying
   # particle as the 2nd
   dndes = {key: fn(es, cme) for key, fn in dnde_fns.items()}

   plt.figure(dpi=150)
   for key, dnde in dndes.items():
      plt.plot(es, es**2 * dnde, label=key)
   plt.yscale("log")
   plt.xscale("log")
   plt.ylim(1e-2, 1e4)
   plt.xlim(np.min(es), np.max(es))
   plt.ylabel(r"$E_{\gamma}^2\dv{N}{dE} \ [\mathrm{MeV}]$", fontdict=dict(size=16))
   plt.xlabel(r"$E_{\gamma} \ [\mathrm{MeV}]$", fontdict=dict(size=16))
   plt.legend()
   plt.tight_layout()


You can also produce the same results as above using :py:meth:`dnde_photon`. To
demonstrate this, let's repeat the above with the positron and neutrino spectra:

.. plot::
   :include-source: True

   import numpy as np
   import matplotlib.pyplot as plt
   from hazma import spectra

   cme = 1500.0
   es = np.geomspace(1e-3, 1.0, 100) * cme

   # `dnde_photon`, `dnde_positron` and `dnde_neutrino` all have an
   # `availible_final_states` attribute that returns a list of strings for all
   # the available particles
   e_states = spectra.dnde_positron.availible_final_states
   nu_states = spectra.dnde_neutrino.availible_final_states

   dndes_e = {key: spectra.dnde_positron(es, cme, key) for key in e_states}
   dndes_nu = {key: spectra.dnde_neutrino(es, cme, key) for key in nu_states}

   plt.figure(dpi=150, figsize=(12,4))
   axs = [plt.subplot(1,3,i+1) for i in range(3)]
   for key, dnde in dndes_e.items():
       axs[0].plot(es, es**2 * dnde, label=key)

   for key, dnde in dndes_nu.items():
       axs[1].plot(es, es**2 * dnde[0], label=key)
       axs[2].plot(es, es**2 * dnde[1], label=key)

   titles = ["positron", "electron-neutrino", "muon-neutrino"]
   for i, ax in enumerate(axs):
       ax.set_yscale("log")
       ax.set_xscale("log")
       ax.set_xlim(np.min(es), np.max(es))
       ax.set_ylim(1e-2, 1e3)
       ax.set_xlim(np.min(es), np.max(es))
       ax.set_title(titles[i])
       ax.set_ylabel(r"$E^2\dv{N}{E} \ [\mathrm{MeV}]$", fontdict=dict(size=16))
       ax.set_xlabel(r"$E \ [\mathrm{MeV}]$", fontdict=dict(size=16))
       ax.legend()

   plt.tight_layout()


Including the Matrix Element
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three of the functions, ``dnde_photon``, ``dnde_positron`` and
``dnde_neutrino`` allow the user to supply as squared matrix element. The matrix
element is then used to correctly determine the energy distributions of the
final state particles. If no matrix element is supplied, it is taken to be
unity.

To demonstrate the use of a matrix element, we consider the decay of a muon into
an electron and two neutrinos. We will compute the photon , the positron and the
neutrino spectra. We compute the photon spectrum using only the FSR (no initial
state radiation or internal bremstralung from the W.)

.. plot::
   :include-source: True

   import numpy as np
   import matplotlib.pyplot as plt
   from hazma import spectra
   from hazma.parameters import GF
   from hazma.parameters import muon_mass as MMU

   def msqrd(s, t):
      return 16.0 * GF**2 * t * (MMU**2 - t)

   es = np.geomspace(1e-4, 1.0, 100) * MMU
   final_states = ["e", "ve", "vm"]

   dndes = {
      1: {
         # Photon
         r"$\gamma$ approx": spectra.dnde_photon(es, MMU, final_states, msqrd=msqrd),
         r"$\gamma$ analytic": spectra.dnde_photon_muon(es, MMU),
      },
      2: {
         # Positron
         r"$e^{+}$ approx": spectra.dnde_positron(es, MMU, final_states, msqrd=msqrd),
         r"$e^{+}$ analytic": spectra.dnde_positron_muon(es, MMU),
      },
      3: {
         # Electron and muon neutrino
         r"$\nu_e$ approx": spectra.dnde_neutrino(es, MMU, final_states, msqrd=msqrd, flavor="e"),
         r"$\nu_e$ analytic": spectra.dnde_neutrino_muon(es, MMU, flavor="e"),
         r"$\nu_{\mu}$ approx": spectra.dnde_neutrino(es, MMU, final_states, msqrd=msqrd, flavor="mu"),
         r"$\nu_{\mu}$ analytic": spectra.dnde_neutrino_muon(es, MMU, flavor="mu"),
      },
   }


   plt.figure(dpi=150, figsize=(12, 4))
   for i, d in dndes.items():
      ax = plt.subplot(1, 3, i)
      lss = ["-", "--", "-", "--"]
      for j, (key, dnde) in enumerate(d.items()):
         ax.plot(es, dnde, ls=lss[j], label=key)
      ax.set_yscale("log")
      ax.set_xscale("log")
      ax.set_xlim(1e-1, 1e2)
      ax.set_ylim(1e-5, 1)
      if i == 1:
         ax.set_ylabel(r"$\dv{N}{E} \ [\mathrm{MeV}]$", fontdict=dict(size=16))
      ax.set_xlabel(r"$E \ [\mathrm{MeV}]$", fontdict=dict(size=16))
      ax.legend()

   plt.tight_layout()


We can see that the results match the analytic results in large portions of the
energy range. For the positron and neutrino spectra, the disagreements for low
energies is due to the energy binning (we a linear binning procedure.) If we set
``nbins`` to a larger value, we can cover more of the energy range.


Boosting Spectra
~~~~~~~~~~~~~~~~

It's common that one is interested in a spectrum in a boosted frame. `hazma`
provides facilities to boost a spectrum from one frame to another. See
:ref:`<table boost>` for the available functions. As an example, let's take the
photon spectrum from a muon decay and boost it from the muon rest frame to a new
frame. There are multiple ways to do this. The first way would be to compute the
spectrum in the rest frame and boost the array. This is done using:

.. plot::
   :include-source: True

   import matplotlib.pyplot as plt
   import numpy as np
   from hazma import spectra
   from hazma.parameters import muon_mass as MMU

   beta = 0.3
   gamma = 1.0 / np.sqrt(1.0 - beta**2)
   emu = gamma * MMU

   es = np.geomspace(1e-4, 1.0, 100) * MMU
   # Rest frame spectrum (emu = muon mass)
   dnde_rf = spectra.dnde_photon_muon(es, MMU)
   # Analytic boosted spectrum
   dnde_boosted_analytic = spectra.dnde_photon_muon(es, emu)
   # Approximate boosted spectrum
   dnde_boosted = spectra.dnde_boost_array(dnde_rf, es, beta=beta)

   plt.figure(dpi=150)
   plt.plot(es, dnde_boosted, label="dnde_boost_array")
   plt.plot(es, dnde_boosted_analytic, label="analytic", ls="--")
   plt.yscale("log")
   plt.xscale("log")
   plt.ylabel(r"$\dv{N}{E} \ [\mathrm{MeV}^{-1}]$", fontdict=dict(size=16))
   plt.xlabel(r"$E \ [\mathrm{MeV}]$", fontdict=dict(size=16))
   plt.legend()
   plt.show()


This method will always have issues near the minimum energy, as we have no
information about the spectrum below the minimum energy.  A second way of
performing the calculation is to use :py:meth:`make_boost_function`. This method
take in the spectrum in the original frame and returns a new function which is
able to compute the boost integral.

.. plot::
   :include-source: True

   import matplotlib.pyplot as plt
   import numpy as np
   from hazma import spectra
   from hazma.parameters import muon_mass as MMU

   beta = 0.3
   gamma = 1.0 / np.sqrt(1.0 - beta**2)
   emu = gamma * MMU

   es = np.geomspace(1e-4, 1.0, 100) * emu

   # Rest frame function:
   dnde_rf_fn = lambda e: spectra.dnde_photon_muon(e, MMU)
   # New boosted spectrum function:
   dnde_boosted_fn = spectra.make_boost_function(dnde_rf_fn)

   # Analytic boosted spectrum
   dnde_boosted_analytic = spectra.dnde_photon_muon(es, emu)
   # Approximate boosted spectrum
   dnde_boosted = dnde_boosted_fn(es, beta=beta)

   plt.figure(dpi=150)
   plt.plot(es, dnde_boosted, label="make_boost_function")
   plt.plot(es, dnde_boosted_analytic, label="analytic", ls="--")
   plt.yscale("log")
   plt.xscale("log")
   plt.ylabel(r"$\dv{N}{E} \ [\mathrm{MeV}^{-1}]$", fontdict=dict(size=16))
   plt.xlabel(r"$E \ [\mathrm{MeV}]$", fontdict=dict(size=16))
   plt.legend()
   plt.show()

The last method is to use :py:meth:`dnde_boost`, which similar to
:py:meth:`make_boost_function` but it returns the boosted spectrum rather than a
function. To use it, do the following:

.. code-block::

   import numpy as np
   from hazma import spectra
   from hazma.parameters import muon_mass as MMU

   beta = 0.3
   gamma = 1.0 / np.sqrt(1.0 - beta**2)
   emu = gamma * MMU

   es = np.geomspace(1e-4, 1.0, 100) * emu

   # Analytic boosted spectrum
   dnde_boosted_analytic = spectra.dnde_photon_muon(es, emu)
   # Approximate boosted spectrum
   dnde_boosted = spectra.dnde_boost(
      lambda e: spectra.dnde_photon_muon(e, MMU),
      es,
      beta=beta
   )


Approximate FSR Using the Altarelli-Parisi Splitting Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the limit that the center-of-mass energy of a process is much larger than the
mass of a charged state, the final state radiation is approximately equal to the
it's splitting function (with additional kinematic factors.) The splitting
functions for scalar and fermionic states are:

.. math::

   P_{S}(x) &= \frac{2(1-x)}{x},\\
   P_{F}(x) &= \frac{1 + (1-x)^2}{x}.

In these expressions, :math:`x = 2 E / \sqrt{s}`, with :math:`E` being the
particle's energy and :math:`\sqrt{s}` the center-of-mass energy. Given some
process :math:`I \to A + B + \cdots + s` or :math:`I \to A + B + \cdots + f` (with :math:`s` and :math:`f` being a scalar and fermionic state), the approximate FSR spectra are:

.. math::

   \dv{N_{S}}{E} &= \frac{Q_{S}^2\alpha_{\mathrm{EM}}}{\sqrt{s}\pi}
   P_{S}(x)\qty(\log(\frac{s(1-x)}{m_{S}^{2}}) - 1)\\
   \dv{N_{F}}{E} &= \frac{Q_{F}^2\alpha_{\mathrm{EM}}}{\sqrt{s}\pi}
   P_{F}(x)\qty(\log(\frac{s(1-x)}{m_{F}^{2}}) - 1)

where :math:`Q_{S,F}` and :math:`m_{S,F}` are the charges and masses of the
scalar and fermion. To compute these spectra in `hazma`, we provide the functions
:py:meth:`dnde_photon_ap_scalar` and :py:meth:`dnde_photon_ap_fermion`. Below we
demonstrate how to use these:

.. plot::
   :include-source: True

   import matplotlib.pyplot as plt
   import numpy as np
   from hazma import spectra
   from hazma.parameters import muon_mass as MMU
   from hazma.parameters import charged_pion_mass as MPI

   cme = 500.0 # 500 MeV
   es = np.geomspace(1e-4, 1.0, 100) * cme

   # These functions take s = cme^2 as second argument and mass of the state as
   # the third.
   dnde_fsr_mu = spectra.dnde_photon_ap_fermion(es, cme**2, MMU)
   dnde_fsr_pi = spectra.dnde_photon_ap_scalar(es, cme**2, MPI)

   plt.figure(dpi=150)
   plt.plot(es, dnde_fsr_mu, label=r"$\mu$")
   plt.plot(es, dnde_fsr_pi, label=r"$\pi$", ls="--")
   plt.yscale("log")
   plt.xscale("log")
   plt.ylabel(r"$\dv{N}{E} \ [\mathrm{MeV}^{-1}]$", fontdict=dict(size=16))
   plt.xlabel(r"$E \ [\mathrm{MeV}]$", fontdict=dict(size=16))
   plt.legend()
   plt.show()

API Reference
-------------


Altarelli-Parisi
~~~~~~~~~~~~~~~~

Functions for compute FSR spectra using the Altarelli-Parisi splitting functions:

.. list-table ::
    :header-rows: 1

    * - Function
      - Description

    * - :meth:`~hazma.spectra.dnde_photon_ap_fermion`
      - Approximate FSR from fermion.
    * - :meth:`~hazma.spectra.dnde_photon_ap_scalar`
      - Approximate FSR from scalar.

.. autofunction:: dnde_photon_ap_fermion
.. autofunction:: dnde_photon_ap_scalar


Boost
~~~~~

.. _table boost:

.. list-table ::
    :header-rows: 1


    * - Function
      - Description

    * - :meth:`~hazma.spectra.boost_delta_function`
      - Boost a delta-function.
    * - :meth:`~hazma.spectra.double_boost_delta_function`
      - Perform two boosts of a delta-function.
    * - :meth:`~hazma.spectra.dnde_boost_array`
      - Boost spectrum specified as an array.
    * - :meth:`~hazma.spectra.dnde_boost`
      - Boost spectrum specified as function.
    * - :meth:`~hazma.spectra.make_boost_function`
      - Construct a boost function.


.. autofunction:: boost_delta_function
.. autofunction:: double_boost_delta_function
.. autofunction:: dnde_boost_array
.. autofunction:: dnde_boost
.. autofunction:: make_boost_function


Photon Spectra
~~~~~~~~~~~~~~

`hazma` includes decay spectra from several unstable particles:


.. list-table ::
    :header-rows: 1

    * - Function
      - Description

    * - :meth:`~hazma.spectra.dnde_photon`
      - Photon spectrum from decay/FSR of n-body final states.
    * - :meth:`~hazma.spectra.dnde_photon_muon`
      - Photon spectrum from decay of :math:`\mu^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_photon_neutral_pion`
      - Photon spectrum from decay of :math:`\pi^{0}`.
    * - :meth:`~hazma.spectra.dnde_photon_charged_pion`
      - Photon spectrum from decay of :math:`\pi^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_photon_charged_kaon`
      - Photon spectrum from decay of :math:`K^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_photon_long_kaon`
      - Photon spectrum from decay of :math:`K_{L}`.
    * - :meth:`~hazma.spectra.dnde_photon_short_kaon`
      - Photon spectrum from decay of :math:`K_{S}`.
    * - :meth:`~hazma.spectra.dnde_photon_eta`
      - Photon spectrum from decay of :math:`\eta`.
    * - :meth:`~hazma.spectra.dnde_photon_eta_prime`
      - Photon spectrum from decay of :math:`\eta'`.
    * - :meth:`~hazma.spectra.dnde_photon_charged_rho`
      - Photon spectrum from decay of :math:`\rho^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_photon_neutral_rho`
      - Photon spectrum from decay of :math:`\rho^{0}`.
    * - :meth:`~hazma.spectra.dnde_photon_omega`
      - Photon spectrum from decay of :math:`\omega`.
    * - :meth:`~hazma.spectra.dnde_photon_phi`
      - Photon spectrum from decay of :math:`\phi`.

.. autofunction:: dnde_photon
.. autofunction:: dnde_photon_muon
.. autofunction:: dnde_photon_neutral_pion
.. autofunction:: dnde_photon_charged_pion
.. autofunction:: dnde_photon_charged_kaon
.. autofunction:: dnde_photon_long_kaon
.. autofunction:: dnde_photon_short_kaon
.. autofunction:: dnde_photon_eta
.. autofunction:: dnde_photon_eta_prime
.. autofunction:: dnde_photon_charged_rho
.. autofunction:: dnde_photon_neutral_rho
.. autofunction:: dnde_photon_omega
.. autofunction:: dnde_photon_phi




Positron Spectra
~~~~~~~~~~~~~~~~

.. list-table ::
    :header-rows: 1

    * - Function
      - Description

    * - :meth:`~hazma.spectra.dnde_positron`
      - Positron spectrum from decay of n-body final states.
    * - :meth:`~hazma.spectra.dnde_positron_muon`
      - Positron spectrum from decay of :math:`\mu^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_positron_charged_pion`
      - Positron spectrum from decay of :math:`\pi^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_positron_charged_kaon`
      - Positron spectrum from decay of :math:`K^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_positron_long_kaon`
      - Positron spectrum from decay of :math:`K_{L}`.
    * - :meth:`~hazma.spectra.dnde_positron_short_kaon`
      - Positron spectrum from decay of :math:`K_{S}`.
    * - :meth:`~hazma.spectra.dnde_positron_eta`
      - Positron spectrum from decay of :math:`\eta`.
    * - :meth:`~hazma.spectra.dnde_positron_eta_prime`
      - Positron spectrum from decay of :math:`\eta'`.
    * - :meth:`~hazma.spectra.dnde_positron_charged_rho`
      - Positron spectrum from decay of :math:`\rho^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_positron_neutral_rho`
      - Positron spectrum from decay of :math:`\rho^{0}`.
    * - :meth:`~hazma.spectra.dnde_positron_omega`
      - Positron spectrum from decay of :math:`\omega`.
    * - :meth:`~hazma.spectra.dnde_positron_phi`
      - Positron spectrum from decay of :math:`\phi`.

.. autofunction:: dnde_positron
.. autofunction:: dnde_positron_muon
.. autofunction:: dnde_positron_neutral_pion
.. autofunction:: dnde_positron_charged_pion
.. autofunction:: dnde_positron_charged_kaon
.. autofunction:: dnde_positron_long_kaon
.. autofunction:: dnde_positron_short_kaon
.. autofunction:: dnde_positron_eta
.. autofunction:: dnde_positron_eta_prime
.. autofunction:: dnde_positron_charged_rho
.. autofunction:: dnde_positron_neutral_rho
.. autofunction:: dnde_positron_omega
.. autofunction:: dnde_positron_phi


Neutrino Spectra
~~~~~~~~~~~~~~~~

.. list-table ::
    :header-rows: 1

    * - Function
      - Description

    * - :meth:`~hazma.spectra.dnde_neutrino`
      - Neutrino spectrum from decay of n-body final states.
    * - :meth:`~hazma.spectra.dnde_neutrino_muon`
      - Neutrino spectrum from decay of :math:`\mu^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_neutrino_charged_pion`
      - Neutrino spectrum from decay of :math:`\pi^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_neutrino_charged_kaon`
      - Neutrino spectrum from decay of :math:`K^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_neutrino_long_kaon`
      - Neutrino spectrum from decay of :math:`K_{L}`.
    * - :meth:`~hazma.spectra.dnde_neutrino_short_kaon`
      - Neutrino spectrum from decay of :math:`K_{S}`.
    * - :meth:`~hazma.spectra.dnde_neutrino_eta`
      - Neutrino spectrum from decay of :math:`\eta`.
    * - :meth:`~hazma.spectra.dnde_neutrino_eta_prime`
      - Neutrino spectrum from decay of :math:`\eta'`.
    * - :meth:`~hazma.spectra.dnde_neutrino_charged_rho`
      - Neutrino spectrum from decay of :math:`\rho^{\pm}`.
    * - :meth:`~hazma.spectra.dnde_neutrino_omega`
      - Neutrino spectrum from decay of :math:`\omega`.
    * - :meth:`~hazma.spectra.dnde_neutrino_phi`
      - Neutrino spectrum from decay of :math:`\phi`.


.. autofunction:: dnde_neutrino
.. autofunction:: dnde_neutrino_muon
.. autofunction:: dnde_neutrino_neutral_pion
.. autofunction:: dnde_neutrino_charged_pion
.. autofunction:: dnde_neutrino_charged_kaon
.. autofunction:: dnde_neutrino_long_kaon
.. autofunction:: dnde_neutrino_short_kaon
.. autofunction:: dnde_neutrino_eta
.. autofunction:: dnde_neutrino_eta_prime
.. autofunction:: dnde_neutrino_charged_rho
.. autofunction:: dnde_neutrino_neutral_rho
.. autofunction:: dnde_neutrino_omega
.. autofunction:: dnde_neutrino_phi

..  LocalWords:  fermion dnde muon pion kaon
