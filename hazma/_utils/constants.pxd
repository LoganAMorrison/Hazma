# =========================================================
# ---- Masses in MeV --------------------------------------
# =========================================================

DEF MASS_E = 0.5109989461 # m[e-] = 0.5109989461 ± 3.1e-09
DEF MASS_MU = 105.6583745 # m[mu-] = 105.6583745 ± 2.4e-06
DEF MASS_TAU = 1776.86 # m[tau-] = 1776.86 ± 0.12
DEF MASS_PI0 = 134.9768 # m[pi0] = 134.9768 ± 0.0005
DEF MASS_PI = 139.57039 # m[pi+] = 139.57039 ± 0.00018
DEF MASS_ETA = 547.862 # m[eta] = 547.862 ± 0.017
DEF MASS_ETAP = 957.78 # m[eta'(958)] = 957.78 ± 0.06
DEF MASS_K = 493.677 # m[K+] = 493.677 ± 0.016
DEF MASS_K0 = 497.611 # m[K0] = 497.611 ± 0.013
DEF MASS_KL = 497.611 # m[K(L)0] = 497.611 ± 0.013
DEF MASS_KS = 497.611 # m[K(S)0] = 497.611 ± 0.013
DEF MASS_RHO = 775.26 # m[rho(770)0] = 775.26 ± 0.23
DEF MASS_OMEGA = 782.66 # m[omega(782)] = 782.66 ± 0.13
DEF MASS_PHI = 1019.461 # m[phi(1020)] = 1019.461 ± 0.016

# =========================================================
# ---- π⁺ Branching Ratios --------------------------------
# =========================================================

# BR(μ+, νμ) = (99.98770±0.00004) %
DEF BR_PI_TO_MU_NUMU = 0.9998770  
# BR(e+, νe) = ( 1.230±0.004  )×10−4
DEF BR_PI_TO_E_NUE = 1.230e-4

# =========================================================
# ---- π⁰ Branching Ratios --------------------------------
# =========================================================

# BR(γ, γ) = (98.823±0.034) %
DEF BR_PI0_TO_A_A = 98.823e-2
# BR(e+, e−, γ) = (1.174±0.035) %
DEF BR_PI0_TO_E_E_A = 1.174e-2
# BR(e+, e+, e−, e−) = (3.34±0.16 )×10−5
DEF BR_PI0_TO_E_E_E_E = 3.34e-5

# =========================================================
# ---- η Branching Ratios ---------------------------------
# =========================================================
                    
# BR(γ, γ) = (39.41±0.20) %
DEF BR_ETA_TO_A_A =39.41e-2 
# BR(π0, π0, π0) = (32.68±0.23) %
DEF BR_ETA_TO_PI0_PI0_PI0 =32.68e-2 
# BR(π0, γ, γ) = ( 2.56±0.22)×10−4
DEF BR_ETA_TO_PI0_A_A = 2.56e-4
# BR(π+, π−, π0) = (22.92±0.28) %
DEF BR_ETA_TO_PI_PI_PI0 =22.92e-2 
# BR(π+, π−, γ) = ( 4.22±0.08) %
DEF BR_ETA_TO_PI_PI_A =4.22e-2 
# BR(e+, e−, γ) = ( 6.9±0.4 )×10−3
DEF BR_ETA_TO_E_E_A = 6.9e-3
# BR(μ+, μ−, γ) = ( 3.1±0.4 )×10−4
DEF BR_ETA_TO_MU_MU_A = 3.1e-4
# BR(μ+, μ−) = ( 5.8±0.8 )×10−6
DEF BR_ETA_TO_MU_MU = 5.8e-6
# BR(π+, π−, e+, e−)  = ( 2.68±0.11)×10−4
DEF BR_ETA_TO_PI_PI_E_E = 2.68e-4
# BR(e+, e−, e+, e−) = ( 2.40±0.22)×10−5
DEF BR_ETA_TO_E_E_E_E = 2.40e-5

# =========================================================
# ---- ρ⁰ Branching Ratios --------------------------------
# =========================================================

DEF BR_RHO_TO_PI_PI = 0.9988447
# BR(π⁰, γ) = 4.7e-4
DEF BR_RHO_TO_PI0_A = 4.7e-4 
# BR(η, γ) = 3.00e-4
DEF BR_RHO_TO_ETA_A = 3.00e-4 
# BR(π⁺, π⁻, π⁰) = 1.01e-4
DEF BR_RHO_TO_PI_PI_PI0 = 1.01e-4 
# BR(e⁺, e⁻) = 4.72e-5
DEF BR_RHO_TO_E_E = 4.72e-5 
# BR(μ⁺, μ⁻) = 4.55e-5
DEF BR_RHO_TO_MU_MU = 4.55e-5 
# BR(π⁰, π⁰, γ) = 4.5e-5
DEF BR_RHO_TO_PI0_PI0_A = 4.5e-5 
# BR(π⁺, π⁻, π⁺, π⁻) = 1.8e-5
DEF BR_RHO_TO_PI_PI_PI_PI = 1.8e-5 
# BR(π⁺, π⁻, π⁰, π⁰) = 1.6e-5
DEF BR_RHO_TO_PI_PI_PI0_PI0 = 1.6e-5
# BR(π⁺, π⁻, γ) = 9.9e-3      
DEF BR_RHO_TO_PI_PI_A = 9.9e-3 

# =========================================================
# ---- ρ⁺ Branching Ratios --------------------------------
# =========================================================

DEF RHOP_TO_PI_PI0 = 0.9995502
# BR(π±, γ) = 4.5e-4
DEF RHOP_TO_PI_A = 4.5e-4

# =========================================================
# ---- K-Long Branching Ratios ----------------------------
# =========================================================

# BR(π0, π0, π0) = (19.52±0.12 ) %
DEF BR_KL_TO_PI0_PI0_PI0 =19.52e-2 
# BR(π+, π−, π0) = (12.54±0.05 ) %
DEF BR_KL_TO_PI_PI_PI0 =12.54e-2 
# BR(π±, e∓, νe) = (40.55±0.11 ) %
DEF BR_KL_TO_PI_E_NUE = 40.55e-2
# BR(π±, μ∓, νμ) =  (27.04±0.07 ) %
DEF BR_KL_TO_PI_MU_NUMU = 27.04e-2
# BR(π+, π−) = ( 1.967±0.010)×10−3
DEF BR_KL_TO_PI_PI = 1.967e-3
# BR(π0, π0) = ( 8.64±0.06 )×10−4
DEF BR_KL_TO_PI0_PI0 = 8.64e-4
# BR(γ, γ) = ( 5.47±0.04 )×10−4
DEF BR_KL_TO_A_A = 5.47e-4
# BR(π0, π±, e∓, ν) = ( 5.20±0.11 )×10−5
DEF BR_KL_TO_PI0_PI_E_NU = 5.20e-5
# BR(π±, e∓, ν, e+, e−) = ( 1.26±0.04 )×10−5
DEF BR_KL_TO_PI_E_E_E_NU = 1.26e-5

# =========================================================
# ---- K-Short Branching Ratios ---------------------------
# =========================================================

# BR(π+, π−) = (69.20±0.05) %
DEF BR_KS_TO_PI_PI = 69.20e-2 
# BR(π0, π0) = (30.69±0.05) %
DEF BR_KS_TO_PI0_PI0 = 30.69e-2
# BR(π+, π−, e+, e−) = ( 4.79±0.15)×10−5
DEF BR_KS_TO_PI_PI_E_E = 4.79e-5
# BR(π±, e∓, νe) =  ( 7.04±0.08)×10−4
DEF BR_KS_TO_PI_E_NUE = 7.04e-4
# BR(γ, γ) = ( 2.63±0.17)×10−6
DEF BR_KS_TO_A_A = 2.63e-6
# BR(π+, π−, π0) = ( 3.5+1.1−0.9)×10−7
DEF BR_KS_TO_PI_PI_PI0 = 3.5e-7
# BR(π+, π−, γ) = ( 1.79±0.05)×10−3
DEF BR_KS_TO_PI_PI_A = 1.79e-3
# BR(π0, γ, γ) =  ( 4.9±1.8 )×10−8
DEF BR_KS_TO_PI0_A_A = 4.9e-8
# BR(π0, e+, e−) = ( 3.0+1.5−1.2)×10−9
DEF BR_KS_TO_PI0_E_E = 3e-9
# BR(π0, μ+, μ−) = ( 2.9+1.5−1.2)×10−9
DEF BR_KS_TO_PI0_MU_MU = 2.9e-9


# =========================================================
# ---- K⁰ Branching Ratios --------------------------------
# =========================================================

DEF BR_K0_TO_KL = 0.5
DEF BR_K0_TO_KS = 0.5

# =========================================================
# ---- K⁰' (K⁰-star) Branching Ratios ---------------------
# =========================================================

# Taken from Pythia8306
DEF BR_K0STAR_TO_K_PI = 0.6649467
DEF BR_K0STAR_TO_K0_PI0 = 0.3326633
DEF BR_K0STAR_TO_K0_A = 0.0023900

# =========================================================
# ---- K⁺, K⁻ Branching Ratios ----------------------------
# =========================================================

# BR(μ+, νμ) = (63.56 ± 0.11) %
DEF BR_K_TO_MU_NUMU = 63.56e-2
# BR(e+, νe) = (1.582 ± 0.007)×10−5
DEF BR_K_TO_E_NUE = 1.582e-5
# BR(π+, π0) = (20.67 ± 0.08 ) %
DEF BR_K_TO_PI_PI0 = 20.67e-2 
# BR(π+, π+, π−) = (5.583 ± 0.024) %
DEF BR_K_TO_PI_PI_PI = 5.583e-2
# BR(π+, π0, π0) = (1.760 ± 0.023) %
DEF BR_K_TO_PI_PI0_PI0 =1.760e-2 
# BR(π0, e+, νe) = (5.07 ± 0.04) %
DEF BR_K_TO_E_NUE_PI0 = 5.07e-2
# BR(π0, μ+, νμ)   (3.352 ± 0.033) %
DEF BR_K_TO_MU_NUMU_PI0 = 3.352e-2
# BR(π0, π0, e+, νe) = (2.55 ± 0.04)×10−5
DEF BR_K_TO_E_NUE_PI0_PI0 = 2.55e-5
# BR(π+, π−, e+, νe) =  (4.247 ± 0.024)×10−5
DEF BR_K_TO_E_NUE_PI_PI = 4.247e-5
# Taken from Pythia8306 (can't find in PDG)
DEF BR_K_TO_MU_NUMU_PI0_PI0 = 0.0000140
# BR(π+, π−, μ+, νμ) =  (1.4 ± 0.9)×10−5
DEF BR_K_TO_MU_NUMU_PI_PI = 1.4e-5
# BR(e+, νe, e+, e−) =  (2.48 ± 0.20 )×10−8
DEF BR_K_TO_E_E_E_NUE = 2.48e-8
# BR(μ+, νμ, e+, e−) =  (7.06 ± 0.31 )×10−8
DEF BR_K_TO_MU_E_E_NUMU = 7.06e-8
# BR(e+, νe, μ+, μ−) =  (1.7 ± 0.5  )×10−8
DEF BR_K_TO_MU_MU_E_NUE = 1.7e-8
# BR(π+, e+, e−) = (3.00 ± 0.09 )×10−7
DEF BR_K_TO_PI_E_E = 3.00e-7
# BR(π+, μ+, μ−) = (9.4 ± 0.6  )×10−8
DEF BR_K_TO_PI_MU_MU = 9.4e-8

# =========================================================
# ---- K⁺' (K-star) Branching Ratios ----------------------
# =========================================================

DEF BR_KSTAR_TO_K0_PI = 0.6660067
DEF BR_KSTAR_TO_K_PI0 = 0.3330033
DEF BR_KSTAR_TO_K_A = 0.0009900

# =========================================================
# ---- Eta' Branching Ratios ------------------------------
# =========================================================

# BR(π⁺, π⁻, η) = (42.5 ± 0.5) %
DEF BR_ETAP_TO_PI_PI_ETA = 42.5e-2
# BR(ρ⁰, γ) = (29.5 ± 0.4) % (including non-resonant π+ + π− + γ)
DEF BR_ETAP_TO_RHO_A = 29.5e-2
# BR(π⁰, π⁰, η) = (22.4 ± 0.5) %
DEF BR_ETAP_TO_PI0_PI0_ETA = 22.4e-2
# BR(ω, γ) = ( 2.52 ± 0.07) %
DEF BR_ETAP_TO_OMEGA_A = 2.52e-2
# BR(γ, γ) = ( 2.307 ± 0.033) %
DEF BR_ETAP_TO_A_A = 2.307e-2
# BR(π⁰, π⁰, π⁰) = ( 2.50 ± 0.17 )×10−3
DEF BR_ETAP_TO_PI0_PI0_PI0 = 2.50e-3
# BR(μ⁺, μ⁻, γ) = (1.13 ± 0.28)×10−4
DEF BR_ETAP_TO_MU_MU_A = 1.13e-4
# BR(ω, e⁺, e⁻) = ( 2.0 ± 0.4  )×10−4
DEF BR_ETAP_TO_OMEGA_E_E = 2e-4
# BR(π⁺, π⁻, π⁰) = (3.61 ± 0.17)×10−3
# BR(π⁺, π⁻, π⁰) = (3.8 ± 0.5)×10−3 (S-wave)
DEF BR_ETAP_TO_PI_PI_PI0 = 3.61e-3
# BR(π∓, ρ±) = (7.4 ± 2.3)×10−4
DEF BR_ETAP_TO_PI_RHOP = 7.4e-4
# BR(π⁺, π⁻, π⁺, π⁻) = (8.4 ± 0.9)×10−5
DEF BR_ETAP_TO_PI_PI_PI_PI = 8.4e-5
# BR(π⁺, π⁻, π⁰, π⁰) = (1.8 ± 0.4)×10−4
DEF BR_ETAP_TO_PI_PI_PI0_PI0 = 1.8e-4
# BR(π⁺, π⁻, e⁺, e⁻) = (2.4 +1.3 −1.0)×10−3
DEF BR_ETAP_TO_PI_PI_E_E = 2.4e-3
# BR(γ, e⁺, e⁻) = (4.91 ± 0.27)×10−4
DEF BR_ETAP_TO_E_E_A = 4.91e-4
# BR(π⁰, γ, γ) = (3.20 ± 0.24)×10−3
# BR(π⁰, γ, γ) = (6.2 ± 0.9)×10−4 (non resonant)
DEF BR_ETAP_TO_PI0_A_A = 3.20e-3

# =========================================================
# ---- ω Branching Ratios ---------------------------------
# =========================================================

# BR(π⁺, π⁻, π⁰) = 89.2 ± 0.7 %
DEF BR_OMEGA_TO_PI_PI_PI0 = 89.2e-2
# BR(π⁰, γ) = 8.34 ± 0.26 %
DEF BR_OMEGA_TO_PI0_A = 8.34e-2
# BR(π⁺, π⁻) = 1.53 +0.11 −0.13 %
DEF BR_OMEGA_TO_PI_PI = 1.53e-2
# BR(η, γ) = 4.5e-4 ± 0.4e-4 
DEF BR_OMEGA_TO_ETA_A = 4.5e-4 
# BR(π⁰, e⁺, e⁻) = 7.7e-4 ± 0.6e-4 
DEF BR_OMEGA_TO_PI0_E_E = 7.7e-4 
# BR(π⁰, μ⁺, μ⁻) = 1.34e-4 ± 0.18e-4
DEF BR_OMEGA_TO_PI0_MU_MU = 1.34e-4 
# BR(e⁺, e⁻) = 7.39e-5 ± 0.19e-5
DEF BR_OMEGA_TO_E_E = 7.39e-5 
# BR(μ⁺, μ⁻) = 7.4e-5 ± 1.8e-5 
DEF BR_OMEGA_TO_MU_MU = 7.4e-5 
# BR(π⁰, π⁰, γ) = 6.7e-5 ± 1.1e-5 
DEF BR_OMEGA_TO_PI0_PI0_A = 6.7e-5 

# =========================================================
# ---- φ Branching Ratios ---------------------------------
# =========================================================

# BR(K⁺, K⁻) = (49.2 ± 0.5) %
DEF BR_PHI_TO_K_K = 49.2e-2
# BR(KL, KS) = (34.0 ± 0.4) %
DEF BR_PHI_TO_KL_KS = 34.0e-2
# PDG: BR(ρ, π⁰) + BR(ρ⁺, π⁻) + BR(ρ⁻, π⁺) +  BR(π⁺, π⁻, π⁰) = (15.24 ± 0.33) %
# The below is taken from Pythia8306
DEF BR_PHI_TO_RHOP_PI = 0.0420984
DEF BR_PHI_TO_RHO_PI0 = 0.0420984
DEF BR_PHI_TO_PI_PI_PI0 = 0.0270000
# BR(η, γ) = (1.303 ± 0.025) %
DEF BR_PHI_TO_ETA_A = 1.303e-2
# BR(π⁰, γ) = (1.32 ± 0.06)×10−3
DEF BR_PHI_TO_PI0_A = 1.32e-3
# BR(e⁺, e⁻) = (2.974 ± 0.034)×10−4
DEF BR_PHI_TO_E_E = 2.974e-4
# BR(μ⁺, μ⁻) = (2.86 ± 0.19)×10−4
DEF BR_PHI_TO_MU_MU = 2.86e-4
# BR(η, e⁺, e⁻) = (1.08 ± 0.04)×10−4
DEF BR_PHI_TO_ETA_E_E = 1.08e-4
# BR(π⁺, π⁻) = (7.3 ± 1.3)×10−5
DEF BR_PHI_TO_PI_PI = 7.3e-5
# BR(ω, π⁰) = (4.7 ± 0.5)×10−5
DEF BR_PHI_TO_OMEGA_PI0 = 4.7e-5
# BR(π⁺, π⁻, γ) = (4.1 ± 1.3)×10−5
DEF BR_PHI_TO_PI_PI_A = 4.1e-5
# BR(f₀(980), γ) = (3.22 ± 0.19)×10−4
DEF BR_PHI_TO_F0980_A = 3.22e-4
# BR(π⁰, π⁰, γ) = (1.12 ± 0.06)×10−4
DEF BR_PHI_TO_PI0_PI0_A = 1.12e-4
# BR(π⁺, π⁻, π⁺, π⁻) = (3.9 +2.8 −2.2)×10−6
DEF BR_PHI_TO_PI_PI_PI_PI = 3.9e-6
# BR(π⁰, e⁺, e⁻) = (1.33 +0.07 −0.10)×10−5
DEF BR_PHI_TO_PI0_E_E = 1.33e-5
# BR(π⁰, η, γ) = (7.27 ± 0.30)×10−5
DEF BR_PHI_TO_PI0_ETA_A = 7.27e-5
# BR(a₀(980), γ) = (7.6 ± 0.6)×10−5
DEF BR_PHI_TO_A0980_A = 7.6e-5
# BR(η'(958), γ) = (6.22 ± 0.21)×10−5
DEF BR_PHI_TO_ETAP_A = 6.22e-5
# BR(μ⁺, μ⁻, γ) = (1.4 ± 0.5)×10−5
DEF BR_PHI_TO_MU_MU_A = 1.4e-5

# =========================================================
# ---- Decay Widths ---------------------------------------
# =========================================================

DEF WIDTH_E = 0.0 # Γ[e-] = 0.0 ± 0.0
DEF WIDTH_MU = 2.9959837e-16 # Γ[mu-] = 2.9959837e-16 ± 3e-22
DEF WIDTH_TAU = 2.267e-09 # Γ[tau-] = 2.267e-09 ± 4e-12
DEF WIDTH_PI0 = 7.81e-06 # Γ[pi0] = 7.81e-06 ± 1.2e-07
DEF WIDTH_PI = 2.5284e-14 # Γ[pi+] = 2.5284e-14 ± 5e-18
DEF WIDTH_ETA = 0.00131 # Γ[eta] = 0.00131 ± 5e-05
DEF WIDTH_ETAP = 0.188 # Γ[eta'(958)] = 0.188 ± 0.006
DEF WIDTH_K = 5.317e-14 # Γ[K+] = 5.317e-14 ± 9e-17
# DEF WIDTH_K0 = None # Γ[K0] = None ± None
DEF WIDTH_KL = 1.287e-14 # Γ[K(L)0] = 1.287e-14 ± 5e-17
DEF WIDTH_KS = 7.3508e-12 # Γ[K(S)0] = 7.3508e-12 ± 2.9e-15
DEF WIDTH_RHO = 149.1 # Γ[rho(770)0] = 149.1 ± 0.8
DEF WIDTH_OMEGA = 8.68 # Γ[omega(782)] = 8.68 ± 0.13
DEF WIDTH_PHI = 4.249 # Γ[phi(1020)] = 4.249 ± 0.013


# =========================================================
# ---- Other Constants ------------------------------------
# =========================================================

# 1/137.035999084(21)
DEF ALPHA_EM = 1.0 / 137.035999084  # Fine structure constant.
DEF RATIO_E_MU_MASS_SQ = (MASS_E / MASS_MU)**2

# FA = 0.0119 ± 0.0001
DEF F_A_PI = 0.0119
# FV = 0.0254 ± 0.0017
DEF F_V_PI = 0.0254
# FV slope parameter a= 0.10 ± 0.06
DEF F_V_PI_SLOPE = 0.10

DEF F_A_K = 0.042
DEF F_V_K = 0.096

DEF DECAY_CONST_PI = 130.41  # PDG convention
DEF DECAY_CONST_K = 156.1  # PDG convention
