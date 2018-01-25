RAMBO (hazma.rambo)
===================
* Author - Logan A. Morrison and Adam Coogan
* Date - December 2017

Description
-----------
Sub-package for generating phases space points and computing phase space
integrals using a Monte Carlo algorithm called RAMBO. This algorithm was
originally implemented by [1].

This algorithm starts by generating random, massless four-momenta :math:`q_{i}`
with energies distributed according to :math:`q_{0}\exp(-q_{0})`. These
:math:`q_{i}`'s are then transformed into :math:`p_{i}`'s by boosting the
:math:`q_{i}`'s into the center-of-mass frame so that
:math:`\sum_{i}p_{i}^{\mu} = (E_{\text{CM}}, 0, 0, 0)`. Lastly, the
:math:`p_{i}`'s are rescaled into :math:`k_{i}`'s so that the :math:`k_{i}`'s
give the correct masses.

Details
-------
The RAMBO algorithm can be broken up into four steps:

* Randomly generate :math:`n` massless four-momenta: :math:`\{q_{i}\}`.
* Boost the set :math:`\{q_{i}\}` into :math:`\{p_{i}\}`.
* Transform the set :math:`\{p_{i}\}` into :math:`\{k_{i}\}`.
* Compute the weight of the phase space point.

Repeating these steps many times will generate the phase space.

Generating the :math:`\{q_{i}\}`
********************************

The first step is to generate the set :math:`\{q_{i}\}`, where
:math:`i\in\{1,\dots,n\}` and :math:`n` is the number of final state particles.
The :math:`q_{i}`'s will have energies distributed according to
:math:`q_{0}\exp(-q_{0})` and will be massless. Additionally, the
:math:`q_{i}`'s will be unphysical, i.e. will not have the correct
center-of-mass energy. To generate the :math:`\{q_{i}\}`, we first compute
:math:`4n` uniform random numbers, :math:`\rho_{1}^{i}, \rho_{2}^{i},
\rho_{3}^{i}, \rho_{4}^{i}`, on the interval :math:`(0,1)`. Then, we compute
the following:

.. math::
    c^{i} = 2\rho_{1}^{i} - 1.0 \qquad \phi^{i} = 2\pi\rho_{2}^{i}

Then, the components of :math:`q_{i}` will be:

.. math::
    q_{i}^{0} = -\log(\rho_{3}\rho_{4})\\
    q_{i}^{x} = q_{i}^{0}\sqrt{1-\left(c^{i}\right)^2}\cos\phi^{i}\\
    q_{i}^{y} = q_{i}^{0}\sqrt{1-\left(c^{i}\right)^2}\sin\phi^{i}\\
    q_{i}^{z} = q_{i}^{0}c^{i}


Generating the :math:`\{p_{i}\}`
********************************
Next, the :math:`\{q_{i}\}`'s need to be boosted to generate the
:math:`\{p_{i}\}`'s. This is done using the following:

.. math::
    p_{i}^{0} = x(\gamma q_{i}^{0} + \vec{b}\cdot\vec{q}_{i})\\
    \vec{p}_{i} = x\left(\vec{q}_{i}^{0} + q_{i}^{0}\vec{b} +
    a\left(\vec{b}\cdot\vec{q}_{i}\right)\vec{b}\right)

where

.. math::
    \vec{b} = -\vec{Q} / M, \quad x = E_{\text{CM}} / M, \quad a =
    \dfrac{1}{1+\gamma}

Here, :math:`Q` is a four-vector equal to the sum of the :math:`q_{i}`'s,
:math:`M = \sqrt{Q^2}` and :math:`\gamma = Q^{0}/M` is the relativistic boost
factor.

Generating the :math:`\{k_{i}\}`
********************************
At this point, we have the :math:`\{p_{i}\}`'s, which are four-momenta with the
correct center of mass energy, but which are still massless. Our next, step is
to transform the :math:`\{p_{i}\}`'s into :math:`\{k_{i}\}`'s, which have the
correct masses.

In order to get the correct masses, we need to rescale the :math:`\{p_{i}\}`'s.
This is done as follows:

.. math::
    \vec{k}_{i} = \xi\vec{p}_{i}\\
    k_{i}^{0} = \sqrt{m_{i}^2 + \xi^2 \left(p_{i}^0\right)^2}

where :math:`\xi` is the scaling factor. To compute the scaling factor, we
require that the sum of the :math:`k_{i}^{0}`'s gives the correct
center-of-mass energy. That is, that :math:`\xi` satisfies:

.. math::
    E_{\text{CM}} = \sum_{i=1}^{n}\sqrt{m_{i}^2 + \xi^2 \left(p_{i}^0\right)^2}

Thus, to compute the :math:`\{k_{i}\}`'s, we solve for :math:`\xi` an then
rescale the :math:`\{p_{i}\}`'s according to the above transformations.

Compute the phase space weight
******************************
The last step is to compute the phase space weight. The weights are such that,
when summing over all phase space points, weighted by the weights, we get the
correct phase space volume. The phase space weight is given by

.. math::
    W = E_{\text{CM}}^{2n-3}
    \left(\sum_{i=1}^{n}|\vec{k}_{i}|\right)^{2n-3}
    \left(\prod_{i=1}^{n}\dfrac{|\vec{k_{i}}|}{k_{i}^{0}}\right)
    \left(\sum_{i=1}^{n}\dfrac{|\vec{k}_{i}|^{2}}{k_{i}^{0}}\right)^{-1}
    \dfrac{(\pi/2)^{n-1}(2\pi)^{4 - 3n}}{\Gamma(n)\Gamma(n-1)}

Functions
---------

.. automodule:: hazma.rambo
    :members:
    :undoc-members:
    :show-inheritance:
