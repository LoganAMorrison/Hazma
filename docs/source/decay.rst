Decay spectra
=============

Overview
--------

The ``decay.py`` module contains high-performance functions for computing
the gamma-ray spectra from :math:`\pi^{\pm}`, math::`\pi^{0}` and
:math:`\mu^{\pm}` decays.

The functions in this module allow the user to compute the decay spectra
for arbitrary parent-particle energy. In order to obtain spectra for
arbitrary parent-particle energy, we compute the decay spectra in the
rest-frame of the parent-particle and perform a Lorentz boost, which
amounts to doing a change-of-variables along with a "convolution" integral.
To achieve higher computational performance, we perform all integrations
in ``c`` using ``cython`` and build extension modules to interface with
python.

Functions
---------

.. autofunction:: hazma.decay.muon

.. autofunction:: hazma.decay.neutral_pion

.. autofunction:: hazma.decay.charged_pion

.. autofunction:: hazma.decay.charged_kaon

.. autofunction:: hazma.decay.short_kaon

.. autofunction:: hazma.decay.long_kaon
