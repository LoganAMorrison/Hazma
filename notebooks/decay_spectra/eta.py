from typing import Dict

from utils import DecayProcess, DecayProcesses
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import photon as _a
from utils import eta as _eta


processes = {
    # BR(γ, γ) = (39.41±0.20) %
    "a a": dict(branching_fraction=69.20e-2, msqrd=None, final_states=[_a, _a]),
    # BR(π0, π0, π0) = (32.68±0.23) %
    "pi0 pi0 pi0": dict(
        branching_fraction=32.68e-2, msqrd=None, final_states=3 * [_pi0],
    ),
    # BR(π+, π−, π0) = (22.92±0.28) %
    "pi pi pi0": dict(
        branching_fraction=22.92e-2, msqrd=None, final_states=[_pi, _pi, _pi0],
    ),
    # BR(π+, π−, γ) = ( 4.22±0.08) %
    "pi pi a": dict(
        branching_fraction=4.22e-2, msqrd=None, final_states=[_pi, _pi, _a],
    ),
    # BR(π0, γ, γ) = ( 2.56±0.22)×10−4
    "pi0 a a": dict(
        branching_fraction=2.56e-4, msqrd=None, final_states=[_pi0, _a, _a],
    ),
    # BR(e+, e−, γ) = ( 6.9±0.4 )×10−3
    "e e a": dict(branching_fraction=6.9e-3, msqrd=None, final_states=[_e, _e, _a],),
    # BR(μ+, μ−, γ) = ( 3.1±0.4 )×10−4
    "mu mu a": dict(
        branching_fraction=3.1e-4, msqrd=None, final_states=[_mu, _mu, _a],
    ),
    # BR(μ+, μ−) = ( 5.8±0.8 )×10−6
    "mu mu": dict(branching_fraction=5.8e-6, msqrd=None, final_states=[_mu, _mu],),
    # BR(π+, π−, e+, e−)  = ( 2.68±0.11)×10−4
    "pi pi e e": dict(
        branching_fraction=2.68e-4, msqrd=None, final_states=[_pi0, _e, _e],
    ),
    # BR(e+, e−, e+, e−) = ( 2.40±0.22)×10−5
    "e e e e": dict(branching_fraction=2.40e-5, msqrd=None, final_states=4 * [_e]),
}


def make_processes(threshold=0.01) -> DecayProcesses:
    parent = _eta
    procs: Dict[str, DecayProcess] = dict()
    for name, process in processes.items():
        if process["branching_fraction"] > threshold:
            procs[name] = DecayProcess(
                parent=parent,
                final_states=process["final_states"],
                msqrd=process["msqrd"],
                branching_fraction=process["branching_fraction"],
            )
    return DecayProcesses(parent=parent, processes=procs)
