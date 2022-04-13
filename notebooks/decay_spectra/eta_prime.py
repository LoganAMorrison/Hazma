from typing import Dict

from utils import DecayProcess, DecayProcesses
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import photon as _a
from utils import eta_prime as _etap
from utils import eta as _eta
from utils import charged_rho as _rho
from utils import neutral_rho as _rho0
from utils import omega as _omega

processes = {
    # BR(π⁺, π⁻, η) = (42.5 ± 0.5) %
    "pi pi eta": dict(
        branching_fraction=42.5e-2, final_states=[_pi, _pi, _eta], msqrd=None
    ),
    # BR(ρ⁰, γ) = (29.5 ± 0.4) % (including non-resonant π+ + π− + γ)
    "rho0 a": dict(branching_fraction=29.5e-2, final_states=[_rho0, _a], msqrd=None),
    # BR(π⁰, π⁰, η) = (22.4 ± 0.5) %
    "pi0 pi0 eta": dict(
        branching_fraction=22.4e-2, final_states=[_pi0, _pi0, _eta], msqrd=None
    ),
    # BR(ω, γ) = ( 2.52 ± 0.07) %
    "omega a": dict(branching_fraction=2.52e-2, final_states=[_omega, _a], msqrd=None),
    # BR(γ, γ) = ( 2.307 ± 0.033) %
    "a a": dict(branching_fraction=2.307e-2, final_states=[_a, _a], msqrd=None),
    # BR(π⁰, π⁰, π⁰) = ( 2.50 ± 0.17 )×10−3
    "pi0 pi0 pi0": dict(
        branching_fraction=2.50e-3, final_states=[_pi0, _pi0, _pi0], msqrd=None
    ),
    # BR(μ⁺, μ⁻, γ) = (1.13 ± 0.28)×10−4
    "mu mu a": dict(
        branching_fraction=1.13e-4, final_states=[_mu, _mu, _a], msqrd=None
    ),
    # BR(ω, e⁺, e⁻) = ( 2.0 ± 0.4  )×10−4
    "omega e e": dict(
        branching_fraction=2e-4, final_states=[_omega, _e, _e], msqrd=None
    ),
    # BR(π⁺, π⁻, π⁰) = (3.61 ± 0.17)×10−3
    # BR(π⁺, π⁻, π⁰) = (3.8 ± 0.5)×10−3 (S-wave)
    "pi pi pi0": dict(
        branching_fraction=3.61e-3, final_states=[_pi, _pi, _pi0], msqrd=None
    ),
    # BR(π∓, ρ±) = (7.4 ± 2.3)×10−4
    "pi rho": dict(branching_fraction=7.4e-4, final_states=[_pi, _rho], msqrd=None),
    # BR(π⁺, π⁻, π⁺, π⁻) = (8.4 ± 0.9)×10−5
    "pi pi pi pi": dict(
        branching_fraction=8.4e-5, final_states=[_pi, _pi, _pi], msqrd=None
    ),
    # BR(π⁺, π⁻, π⁰, π⁰) = (1.8 ± 0.4)×10−4
    "pi pi pi0 pi0": dict(
        branching_fraction=1.8e-4, final_states=[_pi, _pi, _pi0, _pi0], msqrd=None
    ),
    # BR(π⁺, π⁻, e⁺, e⁻) = (2.4 +1.3 −1.0)×10−3
    "pi pi e e": dict(
        branching_fraction=2.4e-3, final_states=[_pi, _pi, _e, _e], msqrd=None
    ),
    # BR(γ, e⁺, e⁻) = (4.91 ± 0.27)×10−4
    "e e a": dict(branching_fraction=4.91e-4, final_states=[_e, _e, _a], msqrd=None),
    # BR(π⁰, γ, γ) = (3.20 ± 0.24)×10−3
    # BR(π⁰, γ, γ) = (6.2 ± 0.9)×10−4 (non resonant)
    "pi0 a a": dict(
        branching_fraction=3.20e-3, final_states=[_pi0, _a, _a], msqrd=None
    ),
}


def make_processes(threshold=0.01) -> DecayProcesses:
    parent = _etap
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
