Scalar mediator
===============

Overview
--------

The ``scalar_mediator`` module contains three models for which dark
matter interacts with the Standard Model through a scalar mediator. For
energies :math:`\mu\gg 1~\mathrm{GeV}`, the interaction Lagrangian can for
this theory is given by:

.. math::

    \mathcal{L}_{\mathrm{Int}(S)} & = -S \left( g_{S\chi} + g_{Sf} \sum_f \frac{y_f}{\sqrt{2}} \bar{f} f \right) \\
                    & \hspace{2cm} + \frac{S}{\Lambda} \left( g_{SG} \frac{\alpha_\mathrm{EM}}{4\pi} F_{\mu\nu} F^{\mu\nu} + g_{SF} \frac{\alpha_s}{4\pi} G_{\mu\nu}^a G^{a \mu\nu} \right)

where the :math:`y_{f}`'s are the Yukawa couplings for the Standard Model
fermions and :math:`F_{\mu\nu}` and :math:`G^{a}_{\mu\nu}` are the field
strength tensors for the photon and gluons (we will describe the remaining
parameters below.) For energies :math:`\mu<1~\mathrm{GeV}`, the quarks and
gluons confine into mesons and baryons. In order to describe the
interactions of the scalar mediator to the mesons, we use Chiral
Perturbation theory. The interaction Lagrangian becomes:

.. math::

    \mathcal{L}_{\mathrm{Int}(S)} & = \frac{2 g_{SG}}{9 \Lambda} S \left[ (\partial_\mu \pi^0) (\partial^\mu \pi^0) + 2 (\partial_\mu \pi^+) (\partial^\mu \pi^-) \right]\\
                         & \hspace{1cm} + \frac{4 i e g_{SG}}{9 \Lambda} S A^\mu \left[ \pi^- (\partial_\mu \pi^+) - \pi^+ (\partial_\mu \pi^-) \right]\\
                         & \hspace{1cm} - \frac{B (m_u + m_d)}{6} \left( \frac{3 g_{Sf}}{v_h} + \frac{2 g_{SG}}{3 \Lambda} \right) S \left[ (\pi^0)^2 + 2 \pi^+ \pi^- \right]\\
                         & \hspace{1cm} + \frac{B (m_u + m_d) g_{SG}}{81 \Lambda} \left( \frac{2 g_{SG}}{\Lambda} - \frac{9 g_{Sf}}{v_h} \right) S^2 \left[ (\pi^0)^2 + 2 \pi^+ \pi^- \right]\\
                         & \hspace{1cm} + \frac{4 e^2 g_{SF}}{9\Lambda} S \pi^+ \pi^- A_\mu A^\mu\\
                         & \hspace{1cm} - g_{S \chi} S \bar{\chi} \chi - g_{Sf} S \sum_{\ell=e,\mu} \frac{y_\ell}{\sqrt{2}} \bar{\ell} \ell.

where :math:`B\approx 2800~\mathrm{MeV}`, :math:`v_{h} = 246~\mathrm{GeV}`
and the model parameters are:

1. :math:`m_{\chi}`: dark matter mass,
2. :math:`m_{S}`: scalar mediator mass,
3. :math:`g_{S\chi}`: coupling of scalar mediator to dark matter,
4. :math:`g_{Sf}`: coupling of scalar mediator to standard model fermions,
5. :math:`g_{SG}`: effective coupling of scalar mediator to gluons,
6. :math:`g_{SF}`: effective coupling of scalar mediator to photons and
7. :math:`\Lambda`: cut-off scale for the effective interactions.

In addition to the generic scalar mediator mode, ``hazma`` also contains
specialized models for realizations of the Higgs-portal and Heavy-quark
theories. In the Higgs-portal model, we assume that the scalar mediator
doesn't directly interact with the standard model particles aside from the
Higgs. We assume the the scalar mixes with the Higgs and inherits all its
interactions to the Standard model through the mixing. The Higgs-portal
model contains the following parameters:

1. :math:`m_{\chi}`: dark matter mass,
2. :math:`m_{S}`: scalar mediator mass,
3. :math:`g_{S\chi}`: coupling of scalar mediator to dark matter,
4. :math:`\sin\theta`: mixing angle between the scalar mediator and Higgs.

The generic couplings are obtained from these parameters through the
following relationships:

.. math::

    g_{Sf} = \sin\theta,  g_{SG} = 3\sin\theta, g_{SF} = -\frac{5}{6}\sin\theta,  \Lambda = v_{h}.

The Heavy-quark model assumes that there exists a new heavy quark and that
the scalar mediator only couples the the heavy quark. The parameters of this
model are:

1. :math:`m_{\chi}`: dark matter mass,
2. :math:`m_{S}`: scalar mediator mass,
3. :math:`g_{S\chi}`: coupling of scalar mediator to dark matter,
4. :math:`g_{SQ}`: coupling of scalar mediator to the heavy quark,
5. :math:`Q_{Q}`: charge of the heavy quark,
6. :math:`m_{Q}`: mass of the heavy quark.

The relationships between these parameters and the generic parameters are:

.. math::

    g_{SG} = g_{SQ},  g_{SF} = 2Q_{Q}^2g_{SQ},  \Lambda = m_{Q}.

For details on how to uses these classes, see :ref:`basic_usage`.

Classes
-------

.. autoclass:: hazma.scalar_mediator.ScalarMediator

.. autoclass:: hazma.scalar_mediator.HiggsPortal

.. autoclass:: hazma.scalar_mediator.HeavyQuark
