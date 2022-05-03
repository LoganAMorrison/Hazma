from utils import DecayProcessInfo
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import neutrino as _nu
from utils import photon as _a
from utils import short_kaon as _kS
from utils import make_msqrd_k_to_pi_l_nu


processes = [
    # BR(π+, π−) = (69.20±0.05) %
    DecayProcessInfo(
        branching_fraction=69.20e-2,
        msqrd=None,
        final_states=[_pi, _pi],
    ),
    # BR(π0, π0) = (30.69±0.05) %
    DecayProcessInfo(
        branching_fraction=30.69e-2,
        msqrd=None,
        final_states=[_pi0, _pi0],
    ),
    # BR(π+, π−, γ) = ( 1.79±0.05)×10−3
    DecayProcessInfo(
        branching_fraction=1.79e-3,
        msqrd=None,
        final_states=[_pi, _pi, _a],
    ),
    # BR(π±, e∓, νe) =  ( 7.04±0.08)×10−4
    DecayProcessInfo(
        branching_fraction=7.04e-4,
        msqrd=None,
        final_states=[_pi, _e, _nu],
    ),
    # BR(π+, π−, e+, e−) = ( 4.79±0.15)×10−5
    DecayProcessInfo(
        branching_fraction=4.79e-5,
        msqrd=None,
        final_states=[_pi, _pi, _e, _e],
    ),
    # BR(γ, γ) = ( 2.63±0.17)×10−6
    DecayProcessInfo(
        branching_fraction=2.63e-6,
        msqrd=None,
        final_states=[_a, _a],
    ),
    # BR(π+, π−, π0) = ( 3.5+1.1−0.9)×10−7
    DecayProcessInfo(
        branching_fraction=3.5e-7,
        msqrd=None,
        final_states=[_pi, _pi, _pi0],
    ),
    # BR(π0, γ, γ) =  ( 4.9±1.8 )×10−8
    DecayProcessInfo(
        branching_fraction=4.9e-8,
        msqrd=None,
        final_states=[_pi0, _a, _a],
    ),
    # BR(π0, e+, e−) = ( 3.0+1.5−1.2)×10−9
    DecayProcessInfo(
        branching_fraction=3e-9,
        msqrd=None,
        final_states=[_pi0, _e, _e],
    ),
    # BR(π0, μ+, μ−) = ( 2.9+1.5−1.2)×10−9
    DecayProcessInfo(
        branching_fraction=2.9e-9,
        msqrd=None,
        final_states=[_pi0, _mu, _mu],
    ),
]
