from utils import DecayProcessInfo
from utils import charged_pion as _pi
from utils import electron as _e
from utils import long_kaon as _kL
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import neutrino as _nu
from utils import photon as _a
from utils import make_msqrd_k_to_pi_l_nu, make_msqrd_k_to_ppp

LAM_PLUS_L_P = 2.40e-2
LAM_PLUS_L_PP = 0.20e-2

msqrd_kl_to_pi0_pi0_pi0 = make_msqrd_k_to_ppp(
    kaon=_kL,
    masses=[_pi0.mass] * 3,
    g=0.0,
    h=0.59e-3,
    k=0.0,
)

msqrd_kl_to_pi_pi_pi0 = make_msqrd_k_to_ppp(
    kaon=_kL,
    masses=[_pi.mass, _pi.mass, _pi0.mass],
    g=0.678,
    h=0.076,
    k=0.0099,
)


msqrd_kl_to_pi_e_nu = make_msqrd_k_to_pi_l_nu(
    kaon=_kL,
    ml=_e.mass,
    lam_p=2.82e-2,
    lam_0=0.0,
    pi=_pi0,
)
msqrd_kl_to_pi_mu_nu = make_msqrd_k_to_pi_l_nu(
    kaon=_kL,
    ml=_mu.mass,
    lam_p=2.82e-2,
    lam_0=1.38e-2,
    pi=_pi0,
)


processes = [
    # BR(π0, π0, π0) = (19.52±0.12 ) %
    DecayProcessInfo(
        branching_fraction=19.52e-2,
        msqrd=None,
        final_states=[_pi0, _pi0, _pi0],
    ),
    # BR(π+, π−, π0) = (12.54±0.05 ) %
    DecayProcessInfo(
        branching_fraction=12.54e-2,
        msqrd=msqrd_kl_to_pi_pi_pi0,
        final_states=[_pi, _pi, _pi0],
    ),
    # BR(π±, e∓, νe) = (40.55±0.11 ) %
    DecayProcessInfo(
        branching_fraction=40.55e-2,
        msqrd=msqrd_kl_to_pi_e_nu,
        final_states=[_pi, _e, _nu],
    ),
    # BR(π±, μ∓, νμ) =  (27.04±0.07 ) %
    DecayProcessInfo(
        branching_fraction=27.04e-2,
        msqrd=msqrd_kl_to_pi_mu_nu,
        final_states=[_pi, _mu, _nu],
    ),
    # BR(π+, π−) = ( 1.967±0.010)×10−3
    DecayProcessInfo(
        branching_fraction=0.001967,
        msqrd=None,
        final_states=[_pi, _pi],
    ),
    # BR(π0, π0) = ( 8.64±0.06 )×10−4
    DecayProcessInfo(
        branching_fraction=0.000864,
        msqrd=None,
        final_states=[_pi0, _pi0],
    ),
    # BR(γ, γ) = ( 5.47±0.04 )×10−4
    DecayProcessInfo(
        branching_fraction=0.000547,
        msqrd=None,
        final_states=[_a, _a],
    ),
    # BR(π0, π±, e∓, ν) = ( 5.20±0.11 )×10−5
    DecayProcessInfo(
        branching_fraction=0.0000520,
        msqrd=None,
        final_states=[_pi0, _pi, _e, _nu],
    ),
    # BR(π±, e∓, ν, e+, e−) = ( 1.26±0.04 )×10−5
    DecayProcessInfo(
        branching_fraction=0.0000126,
        msqrd=None,
        final_states=[_pi, _e, _e, _e, _nu],
    ),
]
