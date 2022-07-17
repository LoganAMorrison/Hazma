"""
==============================
Spectra (:mod:`hazma.spectra`)
==============================

.. currentmodule:: hazma.spectra

The spectra module contains functions for computing photon, positron and
neutrino spectra from the decay of SM particles.

Available functions
===================


Altarelli-parisi
----------------

.. autosummary::

    dnde_photon_ap_fermion -- Approximate FSR from fermion.
    dnde_photon_ap_scalar  -- Approximate FSR from scalar.


Boost
-----

.. autosummary::
    :toctree:

    boost_delta_function        -- Boost a delta-function.
    double_boost_delta_function -- Perform two boosts of a delta-function.
    dnde_boost_array            -- Boost spectrum specified as an array.
    dnde_boost                  -- Boost spectrum specified as function.
    make_boost_function         -- Construct a boost function.

Photon
------

.. autosummary::
    :toctree:

    dnde_photon              -- Photon spectrum from decay/FSR of n-body final
                                states.
    dnde_photon_muon         -- Photon spectrum from decay of muon.
    dnde_photon_neutral_pion -- Photon spectrum from decay of neutral pion.
    dnde_photon_charged_pion -- Photon spectrum from decay of charged pion.
    dnde_photon_charged_kaon -- Photon spectrum from decay of charged kaon.
    dnde_photon_long_kaon    -- Photon spectrum from decay of long kaon.
    dnde_photon_short_kaon   -- Photon spectrum from decay of short kaon.
    dnde_photon_eta          -- Photon spectrum from decay of eta.
    dnde_photon_eta_prime    -- Photon spectrum from decay of eta'.
    dnde_photon_charged_rho  -- Photon spectrum from decay of charged rho.
    dnde_photon_neutral_rho  -- Photon spectrum from decay of neutral rho.
    dnde_photon_omega        -- Photon spectrum from decay of omega.
    dnde_photon_phi          -- Photon spectrum from decay of phi.

Positron
--------

.. autosummary::
    :toctree:

    dnde_positron              -- Positron spectrum from decay of n-body final
                                  states.
    dnde_positron_muon         -- Positron spectrum from decay of muon.
    dnde_positron_neutral_pion -- Positron spectrum from decay of neutral pion.
    dnde_positron_charged_pion -- Positron spectrum from decay of charged pion.
    dnde_positron_charged_kaon -- Positron spectrum from decay of charged kaon.
    dnde_positron_long_kaon    -- Positron spectrum from decay of long kaon.
    dnde_positron_short_kaon   -- Positron spectrum from decay of short kaon.
    dnde_positron_eta          -- Positron spectrum from decay of eta.
    dnde_positron_eta_prime    -- Positron spectrum from decay of eta'.
    dnde_positron_charged_rho  -- Positron spectrum from decay of charged rho.
    dnde_positron_neutral_rho  -- Positron spectrum from decay of neutral rho.
    dnde_positron_omega        -- Positron spectrum from decay of omega.
    dnde_positron_phi          -- Positron spectrum from decay of phi.

Neutrino
--------

.. autosummary::
    :toctree:

    dnde_neutrino              -- Neutrino spectrum from decay of n-body final
                                  states.
    dnde_neutrino_muon         -- Neutrino spectrum from decay of muon.
    dnde_neutrino_neutral_pion -- Neutrino spectrum from decay of neutral pion.
    dnde_neutrino_charged_pion -- Neutrino spectrum from decay of charged pion.
    dnde_neutrino_charged_kaon -- Neutrino spectrum from decay of charged kaon.
    dnde_neutrino_long_kaon    -- Neutrino spectrum from decay of long kaon.
    dnde_neutrino_short_kaon   -- Neutrino spectrum from decay of short kaon.
    dnde_neutrino_eta          -- Neutrino spectrum from decay of eta.
    dnde_neutrino_eta_prime    -- Neutrino spectrum from decay of eta'.
    dnde_neutrino_charged_rho  -- Neutrino spectrum from decay of charged rho.
    dnde_neutrino_neutral_rho  -- Neutrino spectrum from decay of neutral rho.
    dnde_neutrino_omega        -- Neutrino spectrum from decay of omega.
    dnde_neutrino_phi          -- Neutrino spectrum from decay of phi.

"""

from .altarelli_parisi import dnde_photon_ap_fermion, dnde_photon_ap_scalar

from ._nbody import dnde_photon, dnde_positron, dnde_neutrino

from .boost import (
    boost_delta_function,
    double_boost_delta_function,
    dnde_boost_array,
    make_boost_function,
    dnde_boost,
)

from ._photon import (
    dnde_photon_charged_kaon,
    dnde_photon_charged_pion,
    dnde_photon_charged_rho,
    dnde_photon_eta,
    dnde_photon_eta_prime,
    dnde_photon_long_kaon,
    dnde_photon_muon,
    dnde_photon_neutral_pion,
    dnde_photon_neutral_rho,
    dnde_photon_omega,
    dnde_photon_phi,
    dnde_photon_short_kaon,
)

from ._positron import (
    dnde_positron_charged_pion,
    dnde_positron_muon,
    dnde_positron_charged_kaon,
    dnde_positron_short_kaon,
    dnde_positron_long_kaon,
    dnde_positron_eta,
    dnde_positron_omega,
    dnde_positron_neutral_rho,
    dnde_positron_charged_rho,
    dnde_positron_eta_prime,
    dnde_positron_phi,
)

from ._neutrino import (
    dnde_neutrino_charged_pion,
    dnde_neutrino_muon,
    dnde_neutrino_charged_kaon,
    dnde_neutrino_long_kaon,
    dnde_neutrino_short_kaon,
    dnde_neutrino_eta,
    dnde_neutrino_omega,
    dnde_neutrino_eta_prime,
    dnde_neutrino_charged_rho,
    dnde_neutrino_neutral_rho,
    dnde_neutrino_phi,
)

__all__ = [
    # Photon
    "dnde_photon",
    "dnde_photon_muon",
    "dnde_photon_neutral_pion",
    "dnde_photon_charged_pion",
    "dnde_photon_charged_kaon",
    "dnde_photon_short_kaon",
    "dnde_photon_long_kaon",
    "dnde_photon_charged_rho",
    "dnde_photon_neutral_rho",
    "dnde_photon_eta",
    "dnde_photon_omega",
    "dnde_photon_eta_prime",
    "dnde_photon_phi",
    # Positron
    "dnde_positron",
    "dnde_positron_muon",
    "dnde_positron_charged_pion",
    "dnde_positron_charged_kaon",
    "dnde_positron_long_kaon",
    "dnde_positron_short_kaon",
    "dnde_positron_eta",
    "dnde_positron_omega",
    "dnde_positron_neutral_rho",
    "dnde_positron_charged_rho",
    "dnde_positron_eta_prime",
    "dnde_positron_phi",
    # Neutrino
    "dnde_neutrino",
    "dnde_neutrino_muon",
    "dnde_neutrino_charged_pion",
    "dnde_neutrino_charged_kaon",
    "dnde_neutrino_long_kaon",
    "dnde_neutrino_short_kaon",
    "dnde_neutrino_eta",
    "dnde_neutrino_omega",
    "dnde_neutrino_eta_prime",
    "dnde_neutrino_charged_rho",
    "dnde_neutrino_neutral_rho",
    "dnde_neutrino_phi",
    # Altarelli-Parisi
    "dnde_photon_ap_fermion",
    "dnde_photon_ap_scalar",
    "altarelli_parisi",
    # Boost
    "boost_delta_function",
    "double_boost_delta_function",
    "dnde_boost_array",
    "make_boost_function",
    "dnde_boost",
]
