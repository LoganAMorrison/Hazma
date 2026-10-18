Axial-vector mediator
=====================

Overview
--------

The ``axial_vector_mediator`` module contains a model in which dark matter
couples to a spin-1 mediator :math:`Z'` through vector and axial currents,

.. math::

    \mathcal{L} \supset Z'_\mu \bar{\chi}\gamma^\mu
    \left(g_{V}^{\chi} + g_{A}^{\chi}\gamma_5\right)\chi,

while the Standard Model couples through the axial light-quark current of an
:class:`~hazma.form_factors.axial.AxialCurrent` and through the lepton and
neutrino couplings of :class:`~hazma.axial_vector_mediator.LeptonCouplings`.
Below 1 GeV the quark current is evaluated with the hadronic theory chosen by
the required ``theory`` argument. Only ``"chpt"``, the U(3) leading-order chiral
Lagrangian with the Wess-Zumino-Witten term, is implemented; ``"rcht"`` raises
``NotImplementedError``. A model instance uses one theory and never sums over
theories.

The model parameters are:

1. :math:`m_{\chi}`: dark matter mass, in MeV,
2. :math:`M_{Z'}`: mediator mass, in MeV,
3. :math:`g_{V}^{\chi}`, :math:`g_{A}^{\chi}`: vector and axial couplings to
   dark matter,
4. ``quarks``: the axial light-quark current,
5. ``leptons``: the charged-lepton and neutrino couplings,
6. ``gacc``: the charm axial coupling, which enters only the two-photon
   triangle,
7. ``width``: an optional total width that replaces the sum of partial widths.

Rates
-----

Each final state returns :math:`\sigma v_{\mathrm{Mol}}` and the cross section
:math:`\sigma = \sigma v_{\mathrm{Mol}} / v_{\mathrm{Mol}}` in
:math:`\mathrm{MeV}^{-2}` per dark matter pair, with
:math:`v_{\mathrm{Mol}} = 2\sqrt{1 - 4 m_\chi^2/s}`. Neither includes the
dark matter population factor or the number of identical particles in the
final state; the thermal-average, gamma-ray and CMB code apply those. The
mediator width dresses only the transverse pole, and the source-longitudinal
density contributes a non-resonant contact term.

A rate raises instead of returning zero where the model has no provider: past
the declared range of a channel (600 MeV for three pions, 1600 MeV for two
photons), where the first-order meson mixing treatment fails, and, for totals
and widths, while the mediator-pair channel or an :math:`\eta\pi\pi` channel is
open. A channel whose couplings make its rate identically zero returns zero
without a range check and never closes a thermal-average window. The first-order
mixing treatment is checked only when the axial dark matter coupling is
nonzero, and a rate at an exact pole of zero width raises. The ``pi pi pi0``
and ``pi0 pi0 pi0`` channels open at three times the neutral pion mass, because
the leading-order action has one common pion mass; the physical charged-pion
threshold needs the later hadronic work. The :math:`\eta\pi\pi` guard covers
both ``eta pi pi`` and ``eta pi0 pi0``. The continuum photon and positron
spectra are not yet implemented.

Classes
-------

.. autoclass:: hazma.axial_vector_mediator.AxialVectorMediator
    :members:

.. autoclass:: hazma.axial_vector_mediator.LeptonCouplings
