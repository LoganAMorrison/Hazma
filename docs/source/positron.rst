Positron spectra
================

Overview
--------

In order to compute the limits on a given model from energy injections
into the CMB spectra, one needs to know the gamma-ray and electron/positron
spectra for the model. ``hazma`` contains a dedicated module,
``positron_spectra``, for computing the electron/positron spectra from
decays of :math:`\pi^{\pm}` and :math:`\mu^{\pm}`. As in the ``decay``
module, the ``positron_spectra`` module allows users to compute the
electron/positron spectra for arbitrary energies of the parent-particle.
The procedure for computing the spectra for arbitrary parent-particle
energies is identical to the procedure used for ``decay``. In addition
to the decay spectra of muons and pions, ``hazma`` contains a function to
compute the electron/positron from a matrix element called ``positron_decay``.

Functions
---------

.. autofunction:: hazma.positron_spectra.muon

.. autofunction:: hazma.positron_spectra.charged_pion

.. autofunction:: hazma.positron_spectra.positron_decay
