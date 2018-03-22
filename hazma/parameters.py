"""
This file contains parameters of the standard model for leptons and meson.

NOTE: All dimensionful parameters are in MeV.
"""

import numpy as np

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

# Quark masses in MS-bar scheme
up_quark_mass = 2.3
down_quark_mass = 4.8
strange_quark_mass = 95.0
charm_quark_mass = 1.275 * 10**3
bottom_quark_mass = 4.18 * 10**3
top_quark_mass = 160.0 * 10**3

# MISC.
alpha_em = 1.0 / 137.04  # Fine structure constant.
GF = 1.1663787 * 10**-11.0  # Fermi constant in MeV**-2
vh = 246.22795 * 10**3  # Higgs VEV in MeV
qe = np.sqrt(4.0 * np.pi * alpha_em)

# Low Energy constants
fpi0 = 91.924  # Neutral pion decay constant
fpi = 92.2138  # Charged pion decay constant
fk = 110.379  # Charged kaon decay constant
b0 = neutral_pion_mass**2 / (up_quark_mass + down_quark_mass)
G8 = 5.47
G27 = 0.392

# The following low energy constants are for NLO ChiPT, evaluated at mu = mrho.
nlo_lec_mu = rho_mass
Lr1 = 0.56 * 1.0e-3
Lr2 = 1.21 * 1.0e-3
L3 = -2.79 * 1.0e-3
Lr4 = -0.36 * 1.0e-3
Lr5 = 1.4 * 1.0e-3
Lr6 = 0.07 * 1.0e-3
L7 = -0.44 * 1.0e-3
Lr8 = 0.78 * 1.0e-3

# SU(2) LECs
Er = 0.029
Gr = 0.0073

LECS = {"1": Lr1, "2": Lr2, "3": L3, "4": Lr4, "5": Lr5, "6": Lr6,
        "7": L7, "8": Lr8, "SU2_Er": Er, "SU2_Gr": Gr}

# SU(2) LECs
Er = 0.029
Gr = 0.0073

# CKM Matrix Elements
Vud = 0.97417
Vus = 0.2248
