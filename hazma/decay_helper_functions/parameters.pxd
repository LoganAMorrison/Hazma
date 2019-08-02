"""
This file contains parameters of the standard model for leptons and meson.

NOTE: All dimensionful parameters are in MeV.
"""

# MASSES (MeV)
cdef double MASS_E = 0.510998928   # electron
cdef double MASS_MU = 105.6583715  # muon
cdef double MASS_PI0 = 134.9766    # neutral pion
cdef double MASS_PI = 139.57018    # Charged pion
cdef double MASS_K0 = 497.61       # neutral kaon
cdef double MASS_K = 493.68        # charged Kaon
cdef double MASS_ETA = 547.86      # eta
cdef double MASS_ETAP = 957.8      # eta prime
cdef double MASS_RHO = 775.3       # rho
cdef double MASS_OMEGA = 782.7     # omega

# BRANCHING RATIOS
cdef double BR_PI0_TO_GG = 0.9882         # Pi0   -> g   + g
cdef double BR_PI_TO_MUNU = 0.9998        # pi    -> mu  + nu
cdef double BR_PI_TO_ENU = 0.000123       # pi    -> e  + nu

cdef double BR_KS_TO_PIPI = 0.6920        # ks    -> pi  + pi
cdef double BR_KS_TO_PI0PI0 = 0.3069      # ks    -> pi0 + pi0

cdef double BR_KL_TO_PIENU = 0.4055       # kl    -> pi  + e   + nu
cdef double BR_KL_TO_PIMUNU = 0.2704      # kl    -> pi  + mu  + nu
cdef double BR_KL_TO_3PI0 = 0.1952        # kl    -> pi0 + pi0  + pi0
cdef double BR_KL_TO_2PIPI0 = 0.1254      # kl    -> pi  + pi  + pi0

cdef double BR_K_TO_MUNU = 0.6356         # k     -> mu  + nu
cdef double BR_K_TO_PIPI0 = 0.2067        # k     -> pi  + pi0
cdef double BR_K_TO_3PI = 0.05583         # k     -> pi  + pi  + pi
cdef double BR_K_TO_PI0ENU = 0.0507       # k     -> pi0 + e   + nu
cdef double BR_K_TO_PI0MUNU = 0.03352     # k     -> pi0 + mu  + nu
cdef double BR_K_TO_PI2PI0 = 0.01760      # k     -> pi  + pi0 + pi0

cdef double BR_ETA_TO_GG = 0.3941         # eta   -> g   + g
cdef double BR_ETA_TO_3PI0 = 0.3268       # eta   -> pi0 + pi0 + pi0
cdef double BR_ETA_TO_2PIPI0 = 0.2292     # eta   -> pi  + pi  + pi0
cdef double BR_ETA_TO_2PIG = 0.0422       # eta   -> pi  + pi  + g
cdef double BR_ETAP_TO_2PIETA = 0.429     # eta'  -> pi  + pi  + eta
cdef double BR_ETAP_TO_RHOG = 0.291       # eta'  -> rho + g
cdef double etap_BR_pi0_pi0_eta = 0.222   # eta'  -> pi0 + pi0 + eta
cdef double BR_ETAP_TO_OMEGAG = 0.0275    # eta'  -> omega + g
cdef double BR_ETAP_TO_GG = 0.0220        # eta'  -> g   + g
cdef double BR_ETAP_TO_3PI0 = 0.0214      # eta'  -> pi0 + pi0 + pi-
cdef double BR_ETAP_TO_MUMUG = 0.0108     # eta'  -> mu  + mu  + g

cdef double BR_OMEGA_TO_2PIPI0 = 0.892    # omega -> pi + pi   + pi0
cdef double BR_OMEGA_TO_PI0G = 0.0828     # omega -> pi0 + g
cdef double BR_OMEGA_TO_2PI = 0.0153      # omega -> pi + pi

# WIDTHS
cdef double WIDTH_K = 3.3406**-13.
cdef double WIDTH_PI = 2.528511206475808**-14.

# MISC.
cdef double ALPHA_EM = 1.0 / 137.0  # Fine structure constant.
cdef double RATIO_E_MU_MASS_SQ = (MASS_E / MASS_MU)**2.
cdef double F_A_PI = 0.0119
cdef double F_V_PI = 0.0254
cdef double F_V_PI_SLOPE = 0.1
cdef double F_A_K = 0.042
cdef double F_V_K = 0.096
cdef double DECAY_CONST_PI = 130.41  # PDG convention
cdef double DECAY_CONST_K = 156.1  # PDG convention
