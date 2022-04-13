from typing import Dict

from utils import DecayProcess, DecayProcesses
from utils import charged_pion as _pi
from utils import electron as _e
from utils import long_kaon as _kL
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import neutrino as _nu
from utils import photon as _a

processes = {
    # BR(π0, π0, π0) = (19.52±0.12 ) %
    "pi0 pi0 pi0": dict(
        branching_fraction=19.52e-2,
        msqrd=None,
        final_states=[_pi0, _pi0, _pi0],
    ),
    # BR(π+, π−, π0) = (12.54±0.05 ) %
    "pi pi pi0": dict(
        branching_fraction=12.54e-2,
        msqrd=None,
        final_states=[_pi, _pi, _pi0],
    ),
    # BR(π±, e∓, νe) = (40.55±0.11 ) %
    "pi e nu": dict(
        branching_fraction=40.55e-2,
        msqrd=None,
        final_states=[_pi, _e, _nu],
    ),
    # BR(π±, μ∓, νμ) =  (27.04±0.07 ) %
    "pi mu nu": dict(
        branching_fraction=27.04e-2,
        msqrd=None,
        final_states=[_pi0, _mu, _nu],
    ),
    # BR(π+, π−) = ( 1.967±0.010)×10−3
    "pi pi": dict(
        branching_fraction=0.001967,
        msqrd=None,
        final_states=[_pi, _pi],
    ),
    # BR(π0, π0) = ( 8.64±0.06 )×10−4
    "pi0 pi0": dict(
        branching_fraction=0.000864,
        msqrd=None,
        final_states=[_pi0, _pi0],
    ),
    # BR(γ, γ) = ( 5.47±0.04 )×10−4
    "a a": dict(
        branching_fraction=0.000547,
        msqrd=None,
        final_states=[_a, _a],
    ),
    # BR(π0, π±, e∓, ν) = ( 5.20±0.11 )×10−5
    "pi0 pi e nu": dict(
        branching_fraction=0.0000520,
        msqrd=None,
        final_states=[_pi0, _pi, _e, _nu],
    ),
    # BR(π±, e∓, ν, e+, e−) = ( 1.26±0.04 )×10−5
    "pi e e e nu": dict(
        branching_fraction=0.0000126,
        msqrd=None,
        final_states=[_pi, _e, _e, _e, _nu],
    ),
}


def make_processes(threshold=0.01) -> DecayProcesses:
    procs: Dict[str, DecayProcess] = dict()
    for name, process in processes.items():
        if process["branching_fraction"] > threshold:
            procs[name] = DecayProcess(
                parent=_kL,
                final_states=process["final_states"],
                msqrd=process["msqrd"],
                branching_fraction=process["branching_fraction"],
            )
    return DecayProcesses(parent=_kL, processes=procs)
