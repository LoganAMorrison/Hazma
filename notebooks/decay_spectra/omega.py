from typing import Dict

from utils import DecayProcess, DecayProcesses
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import photon as _a
from utils import eta as _eta
from utils import omega as _omega

processes = {
    # BR(π⁺, π⁻, π⁰) = 89.2 ± 0.7 %
    "pi pi pi0": dict(
        branching_fraction=89.2e-2, final_states=[_pi, _pi, _pi0], msqrd=None
    ),
    # BR(π⁰, γ) = 8.34 ± 0.26 %
    "pi0 a": dict(branching_fraction=8.34e-2, final_states=[_pi0, _a], msqrd=None),
    # BR(π⁺, π⁻) = 1.53 +0.11 −0.13 %
    "pi pi": dict(branching_fraction=1.53e-2, final_states=[_pi, _pi], msqrd=None),
    # BR(η, γ) = 4.5e-4 ± 0.4e-4
    "eta a": dict(branching_fraction=4.5e-4, final_states=[_eta, _a], msqrd=None),
    # BR(π⁰, e⁺, e⁻) = 7.7e-4 ± 0.6e-4
    "pi0 e e": dict(branching_fraction=7.7e-4, final_states=[_pi0, _e, _e], msqrd=None),
    # BR(π⁰, μ⁺, μ⁻) = 1.34e-4 ± 0.18e-4
    "pi0 mu mu": dict(
        branching_fraction=1.34e-4, final_states=[_pi0, _mu, _mu], msqrd=None
    ),
    # BR(e⁺, e⁻) = 7.39e-5 ± 0.19e-5
    "e e": dict(branching_fraction=7.39e-5, final_states=[_e, _e], msqrd=None),
    # BR(μ⁺, μ⁻) = 7.4e-5 ± 1.8e-5
    "mu mu": dict(branching_fraction=7.4e-5, final_states=[_mu, _mu], msqrd=None),
    # BR(π⁰, π⁰, γ) = 6.7e-5 ± 1.1e-5
    "pi0 pi0 a": dict(
        branching_fraction=6.7e-5, final_states=[_pi0, _pi0, _a], msqrd=None
    ),
}


def make_processes(threshold=0.01) -> DecayProcesses:
    parent = _omega
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
