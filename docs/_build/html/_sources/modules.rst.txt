*******
Modules
*******

Gamma Ray Spectra Generator (hazma.gamma_ray)
=============================================

Description
-----------

Sub-package for generating gamma ray spectra given a multi-particle final state.

Hazma includes two different methods for computing gamma ray spectra. The first is done by specifying the final state of a process. Doing so, the particle decay spectra are then computed. The second method ``gamma_ray_rambo`` takes in the tree-level and radiative squared matrix elements and runs a Monte-Carlo to generate the gamma ray spectra.

Functions
---------

+-------------------------------------------+-----------------------------+
| Generate spectrum from builtin functions  | :ref:`func_gamma_ray`       |
+-------------------------------------------+-----------------------------+
| Generate spectrum using Monte Carlo       | :ref:`func_gamma_ray_rambo` |
+-------------------------------------------+-----------------------------+

RAMBO (hazma.rambo)
===================

Description
-----------
Sub-package for generating phases space points and computing phase space integrals using a Monte Carlo algorithm called RAMBO.

Functions
---------

+-------------------------+-------------------------------------------------+
| Computing annihilation  |  :ref:`func_compute_annihilation_cross_section` |
| cross sections          |                                                 |
+-------------------------+-------------------------------------------------+
| Computing decay widths  |  :ref:`func_compute_decay_width`                |
+-------------------------+-------------------------------------------------+
| Computing energy        |  :ref:`func_generate_energy_histogram`          |
| histograms for final    |                                                 |
| state particles         |                                                 |
+-------------------------+-------------------------------------------------+
| Compute a single        |  :ref:`func_generate_phase_space_point`         |
| relativistic phase      |                                                 |
| space point             |                                                 |
+-------------------------+-------------------------------------------------+
| Compute many            |  :ref:`func_generate_phase_space`               |
| relativistic phase      |                                                 |
| space points            |                                                 |
+-------------------------+-------------------------------------------------+


Decay (hazma.decay)
===================

Description
-----------

Functions
---------

+----------------+----------------------------------+
| Muon           |  :ref:`func_muon_decay`          |
+----------------+----------------------------------+
| Electron       |  :ref:`func_electron_decay`      |
+----------------+----------------------------------+
| Neutral Pion   |  :ref:`func_neutral_pion_decay`  |
+----------------+----------------------------------+
| Charged Pion   |  :ref:`func_charged_pion_decay`  |
+----------------+----------------------------------+
| Short Kaon     |  :ref:`func_short_kaon_decay`    |
+----------------+----------------------------------+
| Long Kaon      |  :ref:`func_long_kaon_decay`     |
+----------------+----------------------------------+
| Charged Kaon   |  :ref:`func_charged_kaon_decay`  |
+----------------+----------------------------------+
