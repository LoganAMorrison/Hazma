from typing import List

from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import neutrino as _nu
from utils import charged_kaon as _k
from utils import make_msqrd_k_to_pi_l_nu, make_msqrd_k_to_ppp
from utils import DecayProcessInfo

LAM_PLUS_E_P = 2.59e-2
LAM_PLUS_E_PP = 0.186e-2

LAM_PLUS_MU_P = 24e-3
LAM_PLUS_MU_PP = 1.8e-3


msqrd_k_to_pi_pi_pi = make_msqrd_k_to_ppp(
    kaon=_k,
    masses=[_pi.mass] * 3,
    g=-0.21134,
    h=1.848e-2,
    k=4.63e-3,
)


msqrd_k_to_pi_pi0_pi0 = make_msqrd_k_to_ppp(
    kaon=_k,
    masses=[_pi.mass, _pi.mass, _pi0.mass],
    g=0.626,
    h=0.052,
    k=0.0054,
)


msqrd_k_to_pi0_e_nu = make_msqrd_k_to_pi_l_nu(
    kaon=_k,
    ml=_e.mass,
    lam_p=2.959e-2,
    lam_0=0.0,
    pi=_pi0,
)
msqrd_k_to_pi0_mu_nu = make_msqrd_k_to_pi_l_nu(
    kaon=_k,
    ml=_mu.mass,
    lam_p=2.959e-2,
    lam_0=1.76e-2,
    pi=_pi0,
)


processes = [
    # BR(K -> μ+, νμ) = (63.56 ± 0.11) %
    DecayProcessInfo(
        branching_fraction=63.56e-2,
        msqrd=None,
        final_states=[_mu, _nu],
    ),
    # BR(K -> π+, π0) = (20.67 ± 0.08 ) %
    DecayProcessInfo(
        branching_fraction=20.67e-2,
        msqrd=None,
        final_states=[_pi, _pi0],
    ),
    # BR(K -> π+, π+, π−) = (5.583 ± 0.024) %
    DecayProcessInfo(
        branching_fraction=5.583e-2,
        msqrd=msqrd_k_to_pi_pi_pi,
        final_states=[_pi, _pi, _pi],
    ),
    # BR(K -> π0, e+, νe) = (5.07 ± 0.04) %
    DecayProcessInfo(
        branching_fraction=5.07e-2,
        msqrd=msqrd_k_to_pi0_e_nu,
        final_states=[_pi0, _e, _nu],
    ),
    # BR(K -> π0, μ+, νμ)   (3.352 ± 0.033) %
    DecayProcessInfo(
        branching_fraction=3.352e-2,
        msqrd=msqrd_k_to_pi0_mu_nu,
        final_states=[_pi0, _mu, _nu],
    ),
    # BR(K -> π+, π0, π0) = (1.760 ± 0.023) %
    DecayProcessInfo(
        branching_fraction=1.760e-2,
        msqrd=msqrd_k_to_pi_pi0_pi0,
        final_states=[_pi, _pi0, _pi0],
    ),
    # BR(K -> e+, νe) = (1.582 ± 0.007)×10−5
    DecayProcessInfo(
        branching_fraction=1.582e-5,
        msqrd=None,
        final_states=[_e, _nu],
    ),
    # BR(K -> π0, π0, e+, νe) = (2.55 ± 0.04)×10−5
    DecayProcessInfo(
        branching_fraction=2.55e-5,
        msqrd=None,
        final_states=[_pi0, _pi0, _e, _nu],
    ),
    # BR(K -> π+, π−, e+, νe) =  (4.247 ± 0.024)×10−5
    DecayProcessInfo(
        branching_fraction=4.247e-5,
        msqrd=None,
        final_states=[_pi, _pi, _e, _nu],
    ),
    # from Pythia8306 (can't find in PDG)
    DecayProcessInfo(
        branching_fraction=0.0000140,
        msqrd=None,
        final_states=[_pi0, _pi0, _mu, _nu],
    ),
    # BR(K -> π+, π−, μ+, νμ) =  (1.4 ± 0.9)×10−5
    DecayProcessInfo(
        branching_fraction=1.4e-5,
        msqrd=None,
        final_states=[_pi, _pi, _mu, _nu],
    ),
    # BR(K -> e+, νe, e+, e−) =  (2.48 ± 0.20 )×10−8
    DecayProcessInfo(
        branching_fraction=2.48e-8,
        msqrd=None,
        final_states=[_e, _e, _e, _nu],
    ),
    # BR(K -> μ+, νμ, e+, e−) =  (7.06 ± 0.31 )×10−8
    DecayProcessInfo(
        branching_fraction=7.06e-8,
        msqrd=None,
        final_states=[_mu, _e, _e, _nu],
    ),
    # BR(K -> e+, νe, μ+, μ−) =  (1.7 ± 0.5  )×10−8
    DecayProcessInfo(
        branching_fraction=1.7e-8,
        msqrd=None,
        final_states=[_mu, _mu, _e, _nu],
    ),
    # BR(K -> π+, e+, e−) = (3.00 ± 0.09 )×10−7
    DecayProcessInfo(
        branching_fraction=3.00e-7,
        msqrd=None,
        final_states=[_pi, _e, _e],
    ),
    # BR(K -> π+, μ+, μ−) = (9.4 ± 0.6  )×10−8
    DecayProcessInfo(
        branching_fraction=9.4e-8,
        msqrd=None,
        final_states=[_pi, _mu, _mu],
    ),
]
