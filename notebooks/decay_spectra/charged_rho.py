from utils import DecayProcessInfo
from utils import charged_pion as _pi
from utils import neutral_pion as _pi0
from utils import photon as _a

processes = [
    DecayProcessInfo(
        branching_fraction=0.9995502,
        final_states=[_pi, _pi0],
        msqrd=None,
    ),
    # BR(π⁰, γ) = 4.7e-4
    DecayProcessInfo(
        branching_fraction=4.7e-4,
        final_states=[_pi0, _a],
        msqrd=None,
    ),
]
