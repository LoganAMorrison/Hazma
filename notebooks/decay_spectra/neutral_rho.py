from typing import Dict

from utils import DecayProcess, DecayProcesses
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import photon as _a
from utils import eta as _eta
from utils import neutral_rho as _rho0

processes = {
    "pi pi": dict(branching_fraction=0.9988447, final_states=[_pi, _pi], msqrd=None),
    # BR(π⁰, γ) = 4.7e-4
    "pi0 a": dict(branching_fraction=4.7e-4, final_states=[_pi0, _a], msqrd=None),
    # BR(η, γ) = 3.00e-4
    "eta a": dict(branching_fraction=3.00e-4, final_states=[_eta, _a], msqrd=None),
    # BR(π⁺, π⁻, π⁰) = 1.01e-4
    "pi pi pi0": dict(
        branching_fraction=1.01e-4, final_states=[_pi, _pi, _pi0], msqrd=None
    ),
    # BR(e⁺, e⁻) = 4.72e-5
    "e e": dict(branching_fraction=4.72e-5, final_states=[_e, _e], msqrd=None),
    # BR(μ⁺, μ⁻) = 4.55e-5
    "mu mu": dict(branching_fraction=4.55e-5, final_states=[_mu, _mu], msqrd=None),
    # BR(π⁰, π⁰, γ) = 4.5e-5
    "pi0 pi0 a": dict(
        branching_fraction=4.5e-5, final_states=[_pi0, _pi0, _a], msqrd=None
    ),
    # BR(π⁺, π⁻, π⁺, π⁻) = 1.8e-5
    "pi pi pi pi": dict(
        branching_fraction=1.8e-5, final_states=[_pi, _pi, _pi], msqrd=None
    ),
    # BR(π⁺, π⁻, π⁰, π⁰) = 1.6e-5
    "pi pi pi0 pi0": dict(
        branching_fraction=1.6e-5, final_states=[_pi, _pi, _pi0, _pi0], msqrd=None
    ),
    # BR(π⁺, π⁻, γ) = 9.9e-3
    "pi pi a": dict(branching_fraction=9.9e-3, final_states=[_pi, _pi, _a], msqrd=None),
}


def make_processes(threshold=0.01) -> DecayProcesses:
    parent = _rho0
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
