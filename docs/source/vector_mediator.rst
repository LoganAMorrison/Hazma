Vector mediator
===============

Overview
--------

The ``vector_mediator`` module contains two models for which dark
matter interacts with the Standard Model through a vector mediator. For
energies :math:`\mu\gg 1~\mathrm{GeV}`, the interaction Lagrangian can for
this theory is given by:

.. math::

    \mathcal{L}_{\mathrm{Int}(V)}= V_\mu \left( g_{V\chi} \bar{\chi} \gamma^\mu \chi + \sum_f g_{Vf} \bar{f} \gamma^\mu f \right) - \frac{\epsilon}{2} V^{\mu\nu} F_{\mu\nu}.

where the :math:`\epsilon` is the kinetic mixing couplings for the
Standard Model fermions and :math:`F_{\mu\nu}` is the field
strength tensor for the photon (we will describe the remaining
parameters below.) For energies :math:`\mu<1~\mathrm{GeV}`, the quarks and
gluons confine into mesons and baryons. In order to describe the
interactions of the vector mediator to the mesons, we use Chiral
Perturbation theory. The interaction Lagrangian becomes:

.. math::

    \mathcal{L}_{\mathrm{Int}(V)} & = -i (g_{Vu} - g_{Vd}) V^\mu \left( \pi^+ \partial_\mu \pi^- - \pi^- \partial_\mu \pi^+ \right)\\
                         & \hspace{1cm} + ( g_{Vu} - g_{Vd} )^2 V_\mu V^\mu \pi^+ \pi^-\\
                         & \hspace{1cm} + 2 e (Q_u - Q_d) (g_{Vu} - g_{Vd}) A_\mu V^\mu \pi^+ \pi^-\\
                         & \hspace{1cm} + \frac{1}{8\pi^2 f_\pi} \epsilon^{\mu\nu\rho\sigma} (\partial_\mu \pi^0)\\
                         & \hspace{2cm} \times \left\{ e (2 g_{Vu} + g_{Vd}) \left[ (\partial_\nu A_\rho) V_\sigma + (\partial_\nu V_\rho) A_\sigma \right] \right. \\
                         & \hspace{4cm} \left. + 3 (g_{Vu}^2 - g_{Vd}^2) (\partial_\nu V_\rho) V_\sigma \right\}\\
                         & \hspace{1cm} + V_{\mu}\left(g_{Ve}\bar{e}\gamma^{\mu}e + g_{V\mu}\bar{\mu}\gamma^{\mu}\mu\right)

where :math:`f_{\pi}\approx 93~\mathrm{MeV}` and the model parameters are:

1. :math:`m_{\chi}`: dark matter mass,
2. :math:`m_{V}`: vector mediator mass,
3. :math:`g_{V\chi}`: coupling of vector mediator to dark matter,
4. :math:`g_{Vq}`: (:math:`q=u,d,s`) coupling of vector mediator to standard model quarks,
5. :math:`g_{V\ell}`: (:math:`\ell=e,\mu`) coupling of vector mediator to standard model leptons,

In addition to the generic vector mediator model, ``hazma`` also contains
specialized models for realization for a theory in which the vector mediator
mixes with the Standard model photon. In the kinetic-mixing model, we
assume that the vector mediator doesn't directly interact with the Standard
model particles aside from the mixing with the photon. The vector then
inherits its coupling to the charged Standard model fermions from the photon.
The kinetic-mixing model contains the following parameters:

1. :math:`m_{\chi}`: dark matter mass,
2. :math:`m_{V}`: scalar mediator mass,
3. :math:`g_{V\chi}`: coupling of scalar mediator to dark matter,
4. :math:`\epsilon`: kinetic mixing parameter between the vector mediator
   and SM photon.

The generic couplings are obtained from these parameters through the
following relationships:

.. math::

    g_{Vf} = \epsilon e Q_{q}

where :math:`f=(u,d,s,e,\mu)`. For details on how to uses these classes,
see :ref:`basic_usage`.

Classes
-------

.. autoclass:: hazma.vector_mediator.VectorMediator

.. autoclass:: hazma.vector_mediator.KineticMixing

.. autoclass:: hazma.vector_mediator.QuarksOnly
