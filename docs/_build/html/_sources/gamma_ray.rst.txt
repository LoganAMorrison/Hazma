Gamma Ray Spectra Generator (hazma.gamma_ray)
=============================================

Description
-----------

Sub-package for generating gamma ray spectra given a multi-particle final state.

Hazma includes two different methods for computing gamma ray spectra. The first is done by specifying the final state of a process. Doing so, the particle decay spectra are then computed. The second method ``gamma_ray_rambo`` takes in the tree-level and radiative squared matrix elements and runs a Monte-Carlo to generate the gamma ray spectra.

Functions
---------

hazma.gamma_ray.gamma_ray
*************************

.. autofunction:: hazma.gamma_ray.gamma_ray

hazma.gamma_ray.gamma_ray_rambo
*******************************

.. autofunction:: hazma.gamma_ray.gamma_ray_rambo
