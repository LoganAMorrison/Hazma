"""
This file contains parameters of the standard model for leptons and meson.

NOTE: All dimensionful parameters are in MeV.
"""

# MASSES (MeV)
electron_mass = 0.510998928   # electron
muon_mass = 105.6583715  # muon
neutral_pion_mass = 134.9766    # neutral pion
charged_pion_mass = 139.57018    # Charged pion
neutral_kaon_mass = 497.61       # neutral kaon
charged_kaon_mass = 493.68        # charged Kaon
eta_mass = 547.86      # eta
eta_prime_mass = 957.8      # eta prime
rho_mass = 775.3       # rho
omega_mass = 782.7     # omega

up_quark_mass = 2.3
down_quark_mass = 4.8

# MISC.
alpha_em = 1.0 / 137.04  # Fine structure constant.
GF = 1.1663787 * 10**-11.0  # Fermi constant in MeV**-2
vh = 246.22795 * 10**3  # Higgs VEV in MeV

# Low Energy constants
fpi0 = 91.924  # Neutral pion decay constant
fpi = 92.2138  # Charged pion decay constant
fk = 110.379  # Charged kaon decay constant
b0 = neutral_pion_mass**2 / (up_quark_mass + down_quark_mass)
G8 = 5.47
G27 = 0.392

# CKM Matrix Elements
Vud = 0.97417
Vus = 0.2248
