Gamma ray limits
================

Overview
--------

``hazma`` includes functionality for using existing gamma-ray data to constrain
theories and for projecting the discovery reach of proposed gamma-ray
detectors. For the first case, ``hazma`` defines a container class called
``FluxMeasurement`` for storing information about gamma-ray datasets, and
``TheoryGammaRayLimits`` contains a method for using these to set limits. The
second case is also handled by a method in ``TheoryGammaRayLimits`` which takes
arguments specifying various detector and target characteristics.

Limits from existing data
-------------------------

.. automethod:: hazma.theory.TheoryGammaRayLimits.binned_limit

Discovery reach for upcoming detectors
--------------------------------------

.. automethod:: hazma.theory.TheoryGammaRayLimits.unbinned_limit


Containers for measurements, background models and target parameters
--------------------------------------------------------------------

.. autoclass:: hazma.flux_measurement.FluxMeasurement

    .. automethod:: __init__

.. autoclass:: hazma.background_model.BackgroundModel

    .. automethod:: dPhi_dEdOmega


.. autoclass:: hazma.gamma_ray_parameters.TargetParams



Observation regions
-------------------

.. autodata:: hazma.gamma_ray_parameters.comptel_diffuse_targets
.. autodata:: hazma.gamma_ray_parameters.comptel_diffuse_targets_optimistic
.. autodata:: hazma.gamma_ray_parameters.egret_diffuse_targets
.. autodata:: hazma.gamma_ray_parameters.egret_diffuse_targets_optimistic
.. autodata:: hazma.gamma_ray_parameters.fermi_diffuse_targets
.. autodata:: hazma.gamma_ray_parameters.fermi_diffuse_targets_optimistic
.. autodata:: hazma.gamma_ray_parameters.integral_diffuse_targets
.. autodata:: hazma.gamma_ray_parameters.integral_diffuse_targets_optimistic
.. autodata:: hazma.gamma_ray_parameters.draco_targets
.. autodata:: hazma.gamma_ray_parameters.m31_targets
.. autodata:: hazma.gamma_ray_parameters.fornax_targets

Effective Areas
---------------

.. autofunction:: hazma.gamma_ray_parameters.effective_area_comptel
.. autofunction:: hazma.gamma_ray_parameters.effective_area_egret
.. autofunction:: hazma.gamma_ray_parameters.effective_area_fermi
.. autofunction:: hazma.gamma_ray_parameters.effective_area_adept
.. autofunction:: hazma.gamma_ray_parameters.effective_area_all_sky_astrogam
.. autofunction:: hazma.gamma_ray_parameters.effective_area_gecco
.. autofunction:: hazma.gamma_ray_parameters.effective_area_grams
.. autofunction:: hazma.gamma_ray_parameters.effective_area_mast
.. autofunction:: hazma.gamma_ray_parameters.effective_area_pangu


Energy Resolutions
------------------

.. autofunction:: hazma.gamma_ray_parameters.energy_res_adept
.. autofunction:: hazma.gamma_ray_parameters.energy_res_amego
.. autofunction:: hazma.gamma_ray_parameters.energy_res_comptel
.. autofunction:: hazma.gamma_ray_parameters.energy_res_all_sky_astrogam
.. autofunction:: hazma.gamma_ray_parameters.energy_res_egret
.. autofunction:: hazma.gamma_ray_parameters.energy_res_fermi
.. autofunction:: hazma.gamma_ray_parameters.energy_res_gecco
.. autofunction:: hazma.gamma_ray_parameters.energy_res_grams
.. autofunction:: hazma.gamma_ray_parameters.energy_res_integral
.. autofunction:: hazma.gamma_ray_parameters.energy_res_mast
.. autofunction:: hazma.gamma_ray_parameters.energy_res_pangu


