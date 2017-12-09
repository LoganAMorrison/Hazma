***********
Description
***********

Introduction
============

This package is used to compute the gamma ray spectra :math:`\dfrac{dN}{dE}` for light particles, such as, pions, kaon, electrons and muons, in an energy regime where the mass effects are important, i.e. is the MeV energy range. The code has been written in python/cython.

Decay spectra
=============

In this section, we describe how the radiative decay spectra are computed for the muon, charged pion and neutral pion.

Muon
----

The dominant contribution to the radiative decay of the muon comes from :math:`\mu^{\pm}\to e^{\pm}\nu\bar{\nu}\gamma`. The unpolarized differential branching fraction of this decay mode in the *muon rest frame* can be written as
[1]

.. math::
    \dfrac{dB}{dy \ d\cos\theta_{\gamma}^{R}} = \dfrac{1}{y}
    \dfrac{\alpha}{72\pi}(1-y)\left[
    12\left(3 - 2y(1-y)^2\right)\log\left(\dfrac{1-y}{r}\right)
    + y(1-y)(46 - 55y) - 102\right]

where :math:`r = (m_{e}/m_{\mu})^2`, :math:`0 \leq y = 2E_{\gamma}^{R\mu}/m_{\mu} \leq 1 - r`, (:math:`E_{\gamma}^{R\mu}` is the energy of the photon in the muon rest frame) and :math:`\theta_{\gamma}^{R}` is the angle the photon makes with respect to some axis in the muon rest frame.  In order to obtain the decay spectrum in the laboratory frame, we need to boost the above spectrum. In other words, we need to change variables from the gamma ray energy and angle in the muon rest frame to those in the lab frame. We then integrate over the angle to compute :math:`dN/dE_{\gamma}`. The Jacobian for this change of variables is

.. math::
    J = \dfrac{1}{2\gamma(1-\beta\cos\theta_{\gamma}^{L})}

where the boost parameters are

.. math::
    \gamma = E_{\mu} / m_{\mu}, \qquad \beta = \sqrt{1 - \left(\dfrac{m_{\mu}}{E_{\mu}}\right)^2}

Integrating over angles yields the gamma ray spectrum in the lab frame:

.. math::
    \dfrac{dN}{dE_{\gamma}^{L}} =
    \int_{-1}^{1}d\cos\theta_{\gamma}^{L}
    \dfrac{1}{2\gamma(1-\beta\cos\theta_{\gamma}^{L})}
    \dfrac{dB}{dE_{\gamma}^{R\mu}}

.. image:: figures/decaymuon.png
   :alt: Gamma ray spectrum fro radiative muon decay
   :align: center


Charged Pion
------------

To compute the gamma ray spectrum from a charged pion, one considers to possible decay modes. These decay modes are :math:`\pi^{\pm} \to \mu^{\pm}\nu_{\mu}\gamma` and :math:`\pi^{\pm} \to \mu^{\pm}\nu_{\mu} \to e^{\pm}\nu_{\mu}\nu_{\mu}\nu_{e}\gamma`. To compute the gamma ray spectrum from the first decay mode, one uses results from [2]. It turns out that the spectrum from this decay mode is roughly a factor of 100 times smaller than the spectrum from the second decay mode. We thus ignore the contributions from :math:`\pi^{\pm} \to \mu^{\pm}\nu_{\mu}\gamma`.

To compute the γ-ray spectrum from :math:`\pi^{\pm} \to \mu^{\pm}\nu_{\mu} \to e^{\pm}\nu_{\mu}\nu_{\mu}\nu_{e}\gamma`, we first take the muon decay spectra (see section on muon decay spectra) and boost the muon into the pion rest frame use the following:

.. math::
    \gamma_{1} = E_{R\mu}/m_{\mu} \qquad
    \beta_{1} = \sqrt{1-\left(\dfrac{m_{\mu}}{E_{R\mu}}\right)^2} \qquad  E_{R\mu} = \dfrac{m_{\pi}^2 - m_{\mu}^2}{m_{\pi}^2 + m_{\mu}^2}

where :math:`E_{R\mu}` is the energy of the muon in the pion rest frame. The photon spectrum in the charged pion rest frame, :math:`dN/dE_{\gamma}^{R\pi}`, is obtain by integrating the differential branching ratio times a Jacobian factor :math:`1/2\gamma_{1}(1-\beta_{1}\cos\theta_{\gamma}^{R\pi})` over the
angle the photon makes with the muon. Once this integration is completed, one then boosts into the laboratory frame of reference. The steps are nearly identical to boosting from the muon rest frame to the pion rest frame. The only thing that changes in the boost factor and the Jacobian. In going from the charged pion rest frame to the laboratory frame, the Jacobian and boost factor are

.. math::
    J = \dfrac{1}{2\gamma_{2}(1-\beta_{2}\cos\theta_{\gamma}^{L})} \qquad
    \gamma_{2} = E_{\pi} / m_{\pi} \qquad
    \beta_{2} = \sqrt{1 - \left(\dfrac{m_{\mu}}{E_{\pi}}\right)^2}

The gamma-ray spectrum in the laboratory frame will thus be

.. math::
    \dfrac{dN}{dE_{\gamma}^{L}} = \int_{-1}^{1} d\cos\theta_{\gamma}^{L} \dfrac{1}{2\gamma_{2}(1-\beta_{2}\cos\theta_{\gamma}^{L})} \times
    \left(\int_{-1}^{1}d\cos\theta_{\gamma}^{R\pi}
    \dfrac{1}{2\gamma_{1}(1-\beta_{1}\cos\theta_{\gamma}^{L})}
    \dfrac{dB}{dE_{\gamma}^{R\mu}}
    \right)

where

.. math::
    E_{\gamma}^{R\mu} = \gamma_{1} E_{\gamma}^{R\pi}\left(1 - \beta_{1}\cos\theta_{\gamma}^{R\pi}\right)
and

.. math::
    E_{\gamma}^{R\pi} = \gamma_{2} E_{\gamma}^{L}\left(1 - \beta_{2}\cos\theta_{\gamma}^{L}\right)

The limits on the photon energy are given by

.. math::
    0 \leq E_{\gamma}^{L} \leq \dfrac{m_{\mu}^2 - m_{e}^2}{2m_{\mu}}
    \gamma_{1}\gamma_{2}(1+\beta_{1})(1+\beta_{2})

Neutral Pion
------------
The dominant decay mode of the neutral pion is :math:`\pi^{0}\to\gamma\gamma`. In the laboratory frame, the spectrum is

.. math::
    \dfrac{dN}{dE_{\gamma}} = \dfrac{2}{m_{\pi}\gamma\beta}


References
----------

.. [1] Y. Kuno and Y. Okada, Reviews of Modern Physics 73, 151 (2001), ISSN 00346861.
.. [2] K. A. Olive, P. D. Group, et al., Chinese physics C 38, 090001 (2014).
