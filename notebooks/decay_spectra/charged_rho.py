from typing import Dict

from utils import DecayProcess, DecayProcesses
from utils import charged_pion as _pi
from utils import neutral_pion as _pi0
from utils import photon as _a
from utils import charged_rho as _rho

processes = {
    "pi pi": dict(branching_fraction=0.9995502, final_states=[_pi, _pi0], msqrd=None),
    # BR(π⁰, γ) = 4.7e-4
    "pi a": dict(branching_fraction=4.7e-4, final_states=[_pi0, _a], msqrd=None),
}


def make_processes(threshold=0.01) -> DecayProcesses:
    parent = _rho
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
