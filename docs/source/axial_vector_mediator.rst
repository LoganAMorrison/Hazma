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
without a range check and never closes a thermal-average window. ``g g`` and
``pi0 pi0 pi0`` have no transverse density, so they vanish without an axial
dark matter coupling, and the meson poles are then not listed as resonances.
The first-order mixing treatment is checked only when the axial dark matter
coupling is nonzero, and a rate at an exact pole of zero width raises. The
``pi pi pi0`` and ``pi0 pi0 pi0`` channels open at three times the neutral pion
mass, because the leading-order action has one common pion mass, so the
physical charged-pion threshold is not modeled. The :math:`\eta\pi\pi` guard
covers both ``eta pi pi`` and ``eta pi0 pi0``.

Yields
------

``rest_frame_yields``, ``rest_frame_counts`` and ``rest_frame_lines`` return
the stable-particle yields of each open final state per annihilation into
it, in the center-of-mass frame: ``product="photon"``, ``"positron"`` for
e⁺ alone, ``"electron_positron"`` for the charge-summed e± injection, and
``"neutrino"`` for ν + ν̄ per flavor with shape ``(3, N)``. Densities are in
:math:`\mathrm{MeV}^{-1}`, counts are particles per bin between consecutive
edges, and a line is an ``"energy"`` in MeV with a ``"multiplicity"``, the
particles per annihilation. ``spectra()`` and ``positron_spectra()`` weight
the continua by the branching fractions.

The muons of ``mu mu`` decay with the effective polarization their chiral
couplings give them, and the muon of :math:`\pi \to \mu\nu` fully polarized,
so the e⁺ and neutrino spectra carry the spin's angular correlation. The
three-pion final states are produced at the common pion mass of the
leading-order action; a charged pion then decays with its physical mass at
its production Lorentz factor. Photons come from the unpolarized radiative
muon and pion kernels and from the :math:`\pi^0 \to \gamma\gamma` box. The
:math:`\pi^0` Dalitz pairs are not included, so ``pi0 pi0 pi0`` has no e±
yield, and ``"tau tau"`` yields raise ``ValueError`` while that state is
open.

Polarized cascades
------------------

``mediator_decay_yields`` and ``mediator_decay_counts`` return the same
products in the lab for one mediator of lab energy ``mediator_energy``,
above its mass, per decay into each open final state. The mediator's spin
state is a 3×3 helicity ``density`` in the basis (+1, 0, −1) along its
flight direction, Hermitian, positive semidefinite and of unit trace;
``numpy.eye(3) / 3`` is unpolarized, and a pair produced at threshold is
transverse, ``numpy.diag([0.5, 0.0, 0.5])``. Only the diagonal, the helicity
fractions :math:`F_\lambda`, reaches these yields, and each daughter's
angular density about the flight direction is
:math:`[1 + g(F_+ - F_-)P_1 + h(3F_0 - 1)P_2]/2` with the analyzing powers
of its decay.

The fermion pairs take their analyzing powers and spins from the tree-level
helicity amplitudes, and each muon decays with its lab helicity, which the
boost mixes with its spin in the decay plane. ``pi pi pi0`` comes from the
transverse leading-order current alone, which aligns the :math:`\pi^0` with
the mediator's polarization and the charged pions through their angle to
it. ``mediator_pair_yields`` and ``mediator_pair_counts`` give the inclusive
yield :math:`2\sum_f B_f K_f` of a mediator pair sharing one density, with
the branching fractions of ``partial_widths``; the probability that both
mediators decay into :math:`f` is :math:`B_f^2`, which is not that yield.

Resolved photons
----------------

``rest_frame_resolved_photons``, ``rest_frame_resolved_photon_counts``,
``mediator_decay_resolved_photons`` and
``mediator_decay_resolved_photon_counts`` return the prompt real-emission
photons at :math:`O(\alpha)` per Born annihilation or decay into each open
final state, above an infrared ``cut`` in MeV on the photon energy in the
centre-of-mass or mediator rest frame. They add to the photons of the yields
above, which hold no prompt radiation, and are zero for the states without
charged particles.

``pi pi pi0`` is exact at tree level within the leading-order current: in
sigma-model coordinates the photon couples only to the external charged
pions, so the amplitude is the Born current at :math:`\tilde s = (q - p_0)^2`
times the eikonal factor, for the transverse and source-longitudinal densities
alike. A photon from a moving mediator keeps its
:math:`(3F_0 - 1)P_2` alignment. ``e e`` and ``mu mu`` take the
Altarelli-Parisi leading-log spectrum of each lepton, an approximation that
needs :math:`\sqrt{s} \gg 2m` and whose photons follow their leptons' angular
density. Each final state takes exactly
one of the two, so nothing is counted twice. The virtual corrections that
cancel the logarithm of the cut, and the :math:`O(\alpha)` recoil and virtual
corrections to the decay yields, are not included. With the physical pion
masses of the mediator note, the exact emission reproduces its restricted LO
fractions above 1 MeV, :math:`9.1\times10^{-4}` at 450 MeV and
:math:`3.1\times10^{-3}` at 500 MeV; the model's common pion mass gives
larger fractions near threshold.

Gamma-ray limits and the CMB
----------------------------

The photon ``spectra`` add each final state's resolved prompt photons to its
continuum. In the centre-of-mass frame the resolved density at a photon energy
does not depend on a cut below that energy, so these spectra need no cut, and
``total_spectrum``, ``total_conv_spectrum_fn``, ``binned_limit``,
``unbinned_limit``, ``f_eff`` and ``cmb_limit`` use them as they use any
``TheoryAnn`` model's. The ``e e`` photons are those of
``SingleChannelAnn("e e")``.

``mediator_pair_f_eff`` is the CMB deposition efficiency of annihilation into
a mediator pair alone. It folds the photons and charge-summed e± of
``mediator_pair_yields``, for two mediators of half the centre-of-mass energy
``f_eff`` uses, with the deposition curves of arXiv:1506.03811, reads the
tables as ``f_eff`` does, and normalizes to :math:`2m_\chi`. Like the pair
yields, it holds no prompt radiation. A transverse pair decaying into
``pi pi pi0`` deposits 0.4021864 of its energy at
:math:`m_\chi/M = 600/450` MeV and 0.3750662 at 900/500 MeV.

Classes
-------

.. autoclass:: hazma.axial_vector_mediator.AxialVectorMediator
    :members:

.. autoclass:: hazma.axial_vector_mediator.LeptonCouplings
