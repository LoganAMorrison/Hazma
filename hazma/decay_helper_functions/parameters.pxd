"""
This file contains parameters of the standard model for leptons and meson.

NOTE: All dimensionful parameters are in MeV.
"""

# MASSES (MeV)
cdef float MASS_E = 0.510998928   # electron
cdef float MASS_MU = 105.6583715  # muon
cdef float MASS_PI0 = 134.9766    # neutral pion
cdef float MASS_PI = 139.57018    # Charged pion
cdef float MASS_K0 = 497.61       # neutral kaon
cdef float MASS_K = 493.68        # charged Kaon
cdef float MASS_ETA = 547.86      # eta
cdef float MASS_ETAP = 957.8      # eta prime
cdef float MASS_RHO = 775.3       # rho
cdef float MASS_OMEGA = 782.7     # omega

# BRANCHING RATIOS
cdef float BR_PI0_TO_GG = 0.9882         # Pi0   -> g   + g
cdef float BR_PI_TO_MUNU = 0.9998        # pi    -> mu  + nu

cdef float BR_KS_TO_PIPI = 0.6920        # ks    -> pi  + pi
cdef float BR_KS_TO_PI0PI0 = 0.3069      # ks    -> pi0 + pi

cdef float BR_KL_TO_PIENU = 0.4055       # kl    -> pi  + e   + nu
cdef float BR_KL_TO_PIMUNU = 0.2704      # kl    -> pi  + mu  + nu
cdef float BR_KL_TO_3PI0 = 0.1952        # kl    -> pi0 + pi0  + pi0
cdef float BR_KL_TO_2PIPI0 = 0.1254      # kl    -> pi  + pi  + pi0

cdef float BR_K_TO_MUNU = 0.6356         # k     -> mu  + nu
cdef float BR_K_TO_PIPI0 = 0.2067        # k     -> pi  + pi0
cdef float BR_K_TO_3PI = 0.05583         # k     -> pi  + pi  + pi
cdef float BR_K_TO_PI0ENU = 0.0507       # k     -> pi0 + e   + nu
cdef float BR_K_TO_PI0MUNU = 0.03352     # k     -> pi0 + mu  + nu
cdef float BR_K_TO_PI2PI0 = 0.01760      # k     -> pi  + pi0 + pi0

cdef float BR_ETA_TO_GG = 0.3941         # eta   -> g   + g
cdef float BR_ETA_TO_3PI0 = 0.3268       # eta   -> pi0 + pi0 + pi0
cdef float BR_ETA_TO_2PIPI0 = 0.2292     # eta   -> pi  + pi  + pi0
cdef float BR_ETA_TO_2PIG = 0.0422       # eta   -> pi  + pi  + g
cdef float BR_ETAP_TO_2PIETA = 0.429     # eta'  -> pi  + pi  + eta
cdef float BR_ETAP_TO_RHOG = 0.291       # eta'  -> rho + g
cdef float etap_BR_pi0_pi0_eta = 0.222   # eta'  -> pi0 + pi0 + eta
cdef float BR_ETAP_TO_OMEGAG = 0.0275    # eta'  -> omega + g
cdef float BR_ETAP_TO_GG = 0.0220        # eta'  -> g   + g
cdef float BR_ETAP_TO_3PI0 = 0.0214      # eta'  -> pi0 + pi0 + pi-
cdef float BR_ETAP_TO_MUMUG = 0.0108     # eta'  -> mu  + mu  + g

cdef float BR_OMEGA_TO_2PIPI0 = 0.892    # omega -> pi + pi   + pi0
cdef float BR_OMEGA_TO_PI0G = 0.0828     # omega -> pi0 + g
cdef float BR_OMEGA_TO_2PI = 0.0153      # omega -> pi + pi

# WIDTHS
cdef float WIDTH_K = 3.3406**-13.
cdef float WIDTH_PI = 2.528511206475808**-14.

# MISC.
cdef float ALPHA_EM = 1.0 / 137.0  # Fine structure constant.
cdef float RATIO_E_MU_MASS_SQ = (MASS_E / MASS_MU)**2.
cdef float A_PI = 0.0119
cdef float V_PI = 0.0254
cdef float A_K = 0.042
cdef float V_K = 0.096
cdef float DECAY_CONST_PI = 130.41
cdef float DECAY_CONST_K = 156.1
