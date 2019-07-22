Gamma ray limits
================

Overview
--------

``hazma`` includes functionality for using existing gamma-ray data to
constrain theories and for projecting the discovery reach of proposed gamma-ray
detectors. For the first case, ``hazma`` defines a container class called
``FluxMeasurement`` for storing information about gamma-ray datasets, and
``Theory`` contains a method for using these to set limits. The second case
is also handled by a method in ``Theory`` which takes arguments specifying
various detector and target characteristics.

Limits from existing data
-------------------------

.. automethod:: hazma.theory.Theory.binned_limit

Discovery reach for upcoming detectors
--------------------------------------


.. automethod:: hazma.theory.Theory.unbinned_limit

Classes, functions and constants
--------------------------------

These data, functions and classes are relevant for setting constraints and
projecting discovery reach.

.. autoclass:: hazma.flux_measurement.FluxMeasurement

    .. automethod:: __init__

.. autoclass:: hazma.background_model.BackgroundModel

    .. automethod:: dPhi_dEdOmega

.. autofunction:: hazma.gamma_ray_parameters.energy_res_comptel

.. autodata:: hazma.gamma_ray_parameters.A_eff_comptel

.. autodata:: hazma.gamma_ray_parameters.comptel_diffuse

.. autofunction:: hazma.gamma_ray_parameters.energy_res_egret

.. autodata:: hazma.gamma_ray_parameters.A_eff_egret

.. autodata:: hazma.gamma_ray_parameters.egret_diffuse

.. autofunction:: hazma.gamma_ray_parameters.energy_res_fermi

.. autodata:: hazma.gamma_ray_parameters.A_eff_fermi

.. autodata:: hazma.gamma_ray_parameters.fermi_diffuse

.. autodata:: hazma.gamma_ray_parameters.energy_res_e_astrogam

.. autodata:: hazma.gamma_ray_parameters.A_eff_e_astrogam

.. autoclass:: hazma.gamma_ray_parameters.TargetParams

.. autofunction:: hazma.gamma_ray_parameters.solid_angle

