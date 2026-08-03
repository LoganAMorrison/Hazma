import numpy as np
from collections import namedtuple
from hazma.parameters import (
    charged_B_mass as mB,
    charged_kaon_mass as mk,
    charged_pion_mass as mpi,
    muon_mass as mmu,
    B_width,
    k_width,
    kl_width,
)

# Higgs decay information
higgs_width = 3.2  # MeV
# From CMS 2019 (https://arxiv.org/abs/1809.05937)
br_higgs_invis = 0.19

# Transverse extents of detectors in cm
# arxiv:0709
r_det_E949 = 145.0
# "Muon identification in the Belle experiment at KEKB", Abashian et al, 2002
r_det_belle = 331.0
# "The BABAR Detector", arxiv:0105044, figure 73, page 76
r_det_babar = 320.0

# Vertex resolutions of detectors in cm
# arXiv:hep-ex/0308044
r_vert_belle = 0.5
# arXiv:1405.7808: 13 microns transverse, 71 longitudinal, but 1310.6752,
# 1412.5174 find 5mm! Also need to use boost factor of 20...
r_vert_lhcb = 0.5
# Source: Jason Evans...
r_vert_NA482 = 3.0
# KTeV TDR, page 137, ECAL. TODO: what about 0.01, KTeV TDR, page 119, drift
# chambers???
r_vert_ktev = 0.1

"""Container for meson decay measurements.

Attributes
----------
width_bound : float
    Maximum contribution permitted to the width for this process.
r_max : float
    Mediator decays further than this distance (in cm) from the experiment's
    beampipe are treated as invisible.
s_bounds : list of (float, float)
    Pairs specifying the values of (p_mes - p_prods)^2 measurable by the
    experiment. This is necessary for handling experiments' kinematic cuts.
"""
RareDecayObs = namedtuple("RareDecayObs", ["width_bound", "r_max", "s_bounds"])

# Measurements and predictions for rare decay branching fractions
# B^+ -> K^+ invis, arxiv:1303.7465
br_B_k_invis_SM = (4.5 - 0.7) * 1.0e-6
# BABAR measurement, arxiv:1303.7465
br_B_k_invis_babar = 1.6e-5
B_k_invis_obs = RareDecayObs(
    width_bound=(br_B_k_invis_babar - br_B_k_invis_SM) * B_width,
    r_max=r_det_babar,
    s_bounds=[[0.0, 0.3 * mB**2]],
)

# K^+ -> pi^+ invis, arxiv:0709.1000
br_k_pi_invis_SM = (0.85 - 0.07) * 1.0e-10
# E949 measurement, arxiv:0709.1000
br_k_pi_invis_E949 = 1.73e-10 + 1.15e-10
# Bounds on pion momentum must be converted to bounds on s
k_pi_invis_obs = RareDecayObs(
    width_bound=(br_k_pi_invis_E949 - br_k_pi_invis_SM) * k_width,
    r_max=r_det_E949,
    s_bounds=[
        [max((mk**2 + mpi**2 - 2 * mk * np.sqrt(mpi**2 + p**2)), 0.0) for p in p_pis]
        for p_pis in [[229.0, 211.0], [195.0, 140.0]]
    ],
)

# B^+ -> K^+ mu mu, arXiv:hep-ph/0112300
br_B_k_mu_mu_SM = (3.5 - 1.2) * 1.0e-7
# Belle measurement, arxiv:0904.0770
br_B_k_mu_mu_belle = (4.8 + 0.6) * 1.0e-7
B_k_mu_mu_obs = RareDecayObs(
    width_bound=(br_B_k_mu_mu_belle - br_B_k_mu_mu_SM) * B_width,
    r_max=r_vert_belle,
    s_bounds=[[4 * mmu**2, 8.68e6], [10.09e6, 12.86e6], [14.18e6, (mB - mk) ** 2]],
)

# B^+ -> K^+ e^+ e^-, arXiv:hep-ph/0112300
br_B_k_e_e_SM = br_B_k_mu_mu_SM
# Belle measurement, arxiv:0904.0770
br_B_k_e_e_belle = (4.8 + 0.6) * 1.0e-7 / 1.03
B_k_e_e_obs = RareDecayObs(
    width_bound=(br_B_k_e_e_belle - br_B_k_e_e_SM) * B_width,
    r_max=r_vert_belle,
    s_bounds=[[8.11e6, 10.03e6], [12.15e6, 14.11e6]],
)

# K_L -> pi^0 mu^+ mu^-, arxiv:hep-ph/0404127
br_kl_pi0_mu_mu_SM = (1.5 - 0.3) * 1.0e-11
# KTeV upper bound, arxiv:hep-ex/0001006
br_kl_pi0_mu_mu_ktev = 3.8e-10
kl_pi0_mu_mu_obs = RareDecayObs(
    width_bound=(br_kl_pi0_mu_mu_ktev - br_kl_pi0_mu_mu_SM) * kl_width,
    r_max=r_vert_ktev,
    s_bounds=[[0.0, 350.0**2]],
)

# K_L -> pi^0 e^+ e^-, arxiv:hep-ph/0308008
br_kl_pi0_e_e_SM = (3.2 - 0.8) * 1.0e-11
# KTeV upper bound, arxiv:hep-ex/0309072
br_kl_pi0_e_e_ktev = 2.8e-10
kl_pi0_e_e_obs = RareDecayObs(
    width_bound=(br_kl_pi0_e_e_ktev - br_kl_pi0_e_e_SM) * kl_width,
    r_max=r_vert_ktev,
    s_bounds=[[140.0**2, 362.7**2]],
)

"""Container for information about beam dump experiments.

Attributes
----------
br_pN_k : float
    Inclusive branching fraction for pN -> K, where K is a charged or neutral
    kaon.
br_pN_B : float
    Inclusive branching fraction for pN -> K, where B is a charged or neutral
    B-meson.
mediator_energy : float
    Energy of particle produced in meson decays (MeV).
dist : float
    Distance to the detector (m).
len : float
    Length of the detector (m).
visible_fss : list of strings
    List of mediator decay final states that can be detected.
n_pot : float
    Number of protons on target.
n_dec : float
    Upper bound on the number of mediator decays.
"""
BeamDumpParams = namedtuple(
    "BeamDumpParams",
    [
        "br_pN_k",
        "br_pN_B",
        "mediator_energy",
        "dist",
        "length",
        "visible_fss",
        "n_pot",
        "n_dec",
    ],
)

# CHARM experiment
# Dolan et al (arXiv:1412.5174) use br_pN_k=3/7, which is a ~10% difference.
bd_charm = BeamDumpParams(
    br_pN_k=10.0 / 40.0,
    br_pN_B=3.6e-6 / 40.0,
    mediator_energy=10.0e3,
    dist=480.0,
    length=35.0,
    visible_fss=["g g", "e e", "mu mu"],
    n_pot=2.9e17,
    n_dec=2.3,
)

# SHiP experiment
bd_ship = BeamDumpParams(
    br_pN_k=10.0 / 40.0,
    br_pN_B=3.6e-3 / 40.0,
    mediator_energy=25.0e3,
    dist=70.0,
    length=55.0,
    visible_fss=["g g", "e e", "mu mu", "pi pi", "pi0 pi0", "k k", "k0 k0"],
    n_pot=2.0e20,
    n_dec=3.0,
)
