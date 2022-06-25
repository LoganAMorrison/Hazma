.. currentmodule:: hazma.form_factors


Form Factors (:mod:`hazma.form_factors`)
========================================

Overview
--------

``hazma`` has several form factors for computing matrix elements into mesonic
final states. Currently, ``hazma`` has all the important vector form factors
relevant for center of mass energies below the :math:`\tau` mass. In a future
release, we may include scalar form factors or form factors for other tensor
interactions.

Below, we describe how to use the form factors to compute cross-sections, widths
and distributions.

Vector Form Factors
~~~~~~~~~~~~~~~~~~~

Each of the form factor classes listed in the :ref:`class table <vff-table>`
follow a particular structure. They all have the following functions:


.. list-table::
    :header-rows: 1

    * - Method
      - Description

    * - ``form_factor(q,..., **kwargs)``
      - Compute the form factor (without Lorentz structure).
    * - ``integrated_form_factor(q, **kwargs)``
      - Compute the squared current integrated over phase space.
    * - ``width(mv, **kwargs)``
      - Compute the partial decay width of a massive vector.
    * - ``cross_section(q, mx, mv, gvxx, wv, **kwargs)``
      - Compute the annihilation cross-section of dark matter.

``q`` represents the center-of-mass energy in these functions.  The ``...`` is a
place-holder for additional arguments that depend on the type of form factor.

* **Two-body final states:**
  For two-body final states, ``q`` is the only needed argument (aside from
  additional arguments that depend on the specific final state.) 
* **Three-body final states:**
  For three-body final states, the squared invariant masses :math:`s=(p_2+p_3)^2` and
  :math:`t=(p_1+p_3)^2` are required, where :math:`p_1`, :math:`p_2` and
  :math:`p_3` are the momenta of particles 1, 2, and 3.
* **N>3-body final states:**
  Currently, the maximum number of final states is 4. However, for
  :math:`N`-body final states with :math:`N>3`, the form factors require the
  full momentum information of the final states. In these cases, the signature
  is ``form_factor(q, momenta, **kwargs)``, where ``momenta`` is a NumPy array
  containing the 4-momenta of all final state particles.

The ``**kwargs`` contains all the additional information needed for the specific
final state (e.g. couplings.) All of these functions are vectorized, so you can
pass an array of values to compute the quantity for each value at once.

The form factors are given as follows. Let :math:`J^{\mu} = \bra{\mathcal{H}}j^{\mu}\ket{0}` 
be the hadronic current for the given final state :math:`\mathcal{H}`. We decompose 
the current into the form

.. math::

    J^{\mu}_{\mathcal{H}} = \sum_{a=1}^{N}F^{a}_{\mathcal{H}}j^{a,\mu}_{\mathcal{H}}

where :math:`F^{a}_{\mathcal{H}}` is the form factor associated with current
:math:`j^{a,\mu}_{\mathcal{H}}`. For all the two- and three-body final states, there is only
one current (i.e. :math:`N=1`.) For these cases, the hadronic currents for final
states containing two pseudo-scalars, a pseudo-scalar and photon, a
pseudo-scalar and vector and three pseudo-scalars are given by:

.. math::

   J^{\mu}_{P_1P_2} 
   &= -(p_{1}^{\mu}-p_{2}^{\mu})F_{P_1P_2}(q^2), &
   q &=p_1+p_2\\
   J^{\mu}_{P\gamma} 
   &= \epsilon_{\mu\nu\alpha\beta}q^{\nu}\epsilon^{\alpha}_{\gamma}(p_{\gamma})p^{\beta}_{\gamma}F_{P\gamma}(q^2), &
   q &=p_{P}+p_{\gamma}\\
   J^{\mu}_{PV} 
   &= \epsilon_{\mu\nu\alpha\beta}q^{\nu}\epsilon^{\alpha}_{V}(p_{V})p^{\beta}_{P}F_{PV}(q^2), &
   q &=p_{P}+p_{V}\\
   J^{\mu}_{P_1P_2P_3} 
   &= \epsilon_{\mu\nu\alpha\beta}p_{1}^{\nu}p_{2}^{\alpha}p_{3}^{\beta}F_{P_1P_2P_3}(s,t)

In the two-body form factors, the momentum :math:`q` is given by the sum of the
final state momenta. In the three pseudo-scalar form factor,
:math:`s=(p_2+p_3)^2` and :math:`t=(p_1+p_3)^2`. 

In terms of these currents, the ``form_factor`` functions compute the :math:`F_{\mathcal{H}}`. 
The ``integrated_form_factor`` functions compute the following:

.. math::

    \mathcal{J}_{\mathcal{H}}(q^2) = -\frac{1}{3q^2}g_{\mu\nu}\int\dd{\Pi}_{\mathrm{LIPS}}
    J_{\mathcal{H}}^{\mu}\bar{J}_{\mathcal{H}}^{\nu}

From the integrated current, the widths and cross-sections are easily computed.
The expressions are:

.. math::

    \sigma_{\bar{\chi}\chi\to\mathcal{H}}(q^2) 
    &= 
    \frac{g_{V\chi\chi}^2(q^2+2m_{\chi}^2)}
    {\sqrt{q^2-4m_{\chi}^2}\qty((q^2-m_{V})^2+m_{V}^2\Gamma_{V}^2)}
    \frac{\sqrt{q^2}}{2}\mathcal{J}_{\mathcal{H}}(q^2)\\
    \Gamma_{V\to\mathcal{H}} &= \frac{m_{V}}{2}\mathcal{J}_{\mathcal{H}}(m_{V}^2)


Examples
~~~~~~~~

Partial Widths of Vector
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::
    :caption: :math:`V\to\pi^{+}\pi^{-}`

    import hazma.form_factors.vector as ffv

    # Compute the width V -> pi-pi for mv = 1 GeV with couplings of vector to
    # quarks gvuu = 2/3 and gvdd=-1/3
    ffv.VectorFormFactorPiPi().width(mv=1e3, gvuu=2.0/3.0, gvdd=-1.0/3.0)
    # output: [17.30309111083309]


.. code-block::
    :caption: :math:`V\to\pi^{+}K^{-}K^{0}`

    import hazma.form_factors.vector as ffv

    # Compute the width V -> pi-k-k0 for mv = 1.3 GeV with couplings of vector to
    # quarks gvuu = 2/3, gvdd=-1/3, gvss=-1/3
    ffv.VectorFormFactorPiKK0().width(mv=1.3e3, gvuu=2.0/3.0, gvdd=-1.0/3.0, gvss=-1.0/3.0)
    # output: 0.0004685155427290262 [MeV]


.. code-block::
    :caption: :math:`V\to\pi^{+}\pi^{-}\pi^{+}\pi^{-}`

    import hazma.form_factors.vector as ffv

    # Compute the width V -> pi-k-k0 for mv = 1.3 GeV with couplings of vector to
    # quarks gvuu = 2/3, gvdd=-1/3, gvss=-1/3
    ffv.VectorFormFactorPiPiPiPi().width(mv=1.3e3, gvuu=2.0/3.0, gvdd=-1.0/3.0)
    # output: 10.799920290416575 [MeV]


Dark Matter Annihilation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::
    :caption: :math:`\bar{\chi}\chi\to K^{+}K^{-}`

    import hazma.form_factors.vector as ffv

    mx = 300.0  # 300 MeV
    q = 1.2e3   # 1.2 GeV
    mv = 600.0  # 600 MeV
    wv = 1.0    # 1 MeV
    gvxx, gvuu, gvdd, gvss = 1.0, 2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0
    ffv.VectorFormFactorKK().cross_section(
        q=q, mx=mx, mv=mv, gvxx=gvxx, wv=wv, gvuu=gvuu, gvdd=gvss, gvss=gvss
    )
    # output: 5.235577288150283e-09 [MeV^-2]



API
---

.. list-table::
    :header-rows: 1
    :name: vff-table

    * - Class
      - Description
    
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPiPi`
      - :math:`\pi^{+}\pi^{-}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPi0Pi0`
      - :math:`\pi^{0}\pi^{0}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorKK`
      - :math:`K^{+}K^{-}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorK0K0`
      - :math:`K^{0}\bar{K}^{0}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPi0Gamma`
      - :math:`\pi^{0}\gamma` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPi0Omega`
      - :math:`\pi^{0}\omega` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPi0Phi`
      - :math:`\pi^{0}\phi` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorEtaGamma`
      - :math:`\eta\gamma` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorEtaOmega`
      - :math:`\eta\omega` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorEtaPhi`
      - :math:`\eta\phi` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPi0K0K0`
      - :math:`\pi^{0}K^{0}\bar{K}^{0}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPi0KpKm`
      - :math:`\pi^{0}K^{+}K^{-}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPiKK0`
      - :math:`\pi^{+}K^{-}K^{0}` or :math:`\pi^{-}K^{+}\bar{K}^{0}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPiPiEta`
      - :math:`\pi^{+}\pi^{-}\eta` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPiPiEtaPrime`
      - :math:`\pi^{+}\pi^{-}\eta'` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPiPiOmega`
      - :math:`\pi^{+}\pi^{-}\omega` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPi0Pi0Omega`
      - :math:`\pi^{0}\pi^{0}\omega` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPiPiPi0`
      - :math:`\pi^{+}\pi^{-}\pi^{0}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPiPiPi0Pi0`
      - :math:`\pi^{+}\pi^{-}\pi^{0}\pi^{0}` form factor.
    * - :py:meth:`~hazma.form_factors.vector.VectorFormFactorPiPiPiPi`
      - :math:`\pi^{+}\pi^{-}\pi^{+}\pi^{-}` form factor.


.. autoclass:: hazma.form_factors.vector.VectorFormFactorPiPi
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPi0Pi0
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorKK
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorK0K0
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPi0Gamma
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPi0Omega
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPi0Phi
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorEtaGamma
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorEtaOmega
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorEtaPhi
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPi0K0K0
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPi0KpKm
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPiKK0
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPiPiEta
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPiPiEtaPrime
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPiPiOmega
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPi0Pi0Omega
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPiPiPi0
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPiPiPi0Pi0
    :members:

.. autoclass:: hazma.form_factors.vector.VectorFormFactorPiPiPiPi
    :members:
