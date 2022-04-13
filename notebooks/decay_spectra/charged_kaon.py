from typing import Dict

from utils import DecayProcess, DecayProcesses
from utils import charged_kaon as _k
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import neutrino as _nu


def msqrd_k_to_pi_pi_pi(s, t):
    return (2.0 / 3.0 * (_k.mass**2 + 3 * _pi.mass**2 - s - t)) ** 2


def msqrd_k_to_pi_pi0_pi0(s, _):
    return (1.0 / 3.0 * (_k.mass**2 + 3 * _pi.mass**2 - 2 * s)) ** 2


def msqrd_k_to_pi0_mu_nu(s, t):
    fp0 = 1.0
    fm0 = 1.0
    lp1 = 24e-3
    lp2 = 1.8e-3

    ml = _mu.mass
    mpi0 = _pi0.mass
    mk = _k.mass

    def fp(s):
        st = s / mpi0**2
        return fp0 * (1.0 + lp1 * st + 0.5 * lp2 * st**2)

    def fm(_):
        return fm0

    return -4 * (
        (ml**4 - ml**2 * s) * fm(s) ** 2
        + 2 * ml**2 * (ml**2 + 2 * mpi0**2 - s - 2 * t) * fm(s) * fp(s)
        + (
            ml**4
            + 4 * mk**2 * (mpi0**2 - t)
            + 4 * t * (-(mpi0**2) + s + t)
            - ml**2 * (s + 4 * t)
        )
        * fp(s) ** 2
    )


def msqrd_k_to_pi0_e_nu(s, t):
    fp0 = 1.0
    lp1 = 2.59e-2
    lp2 = 0.186e-2

    ml = _e.mass
    mpi0 = _pi0.mass
    mk = _k.mass

    def fp(s):
        st = s / mpi0**2
        return fp0 * (1.0 + lp1 * st + 0.5 * lp2 * st**2)

    def fm(_):
        return 0.0

    return -4 * (
        (ml**4 - ml**2 * s) * fm(s) ** 2
        + 2 * ml**2 * (ml**2 + 2 * mpi0**2 - s - 2 * t) * fm(s) * fp(s)
        + (
            ml**4
            + 4 * mk**2 * (mpi0**2 - t)
            + 4 * t * (-(mpi0**2) + s + t)
            - ml**2 * (s + 4 * t)
        )
        * fp(s) ** 2
    )


processes = {
    # BR(K -> μ+, νμ) = (63.56 ± 0.11) %
    "mu nu": dict(
        branching_fraction=63.56e-2,
        msqrd=None,
        final_states=[_mu, _nu],
    ),
    # BR(K -> π+, π0) = (20.67 ± 0.08 ) %
    "pi pi0": dict(
        branching_fraction=20.67e-2,
        msqrd=None,
        final_states=[_pi, _pi0],
    ),
    # BR(K -> π+, π+, π−) = (5.583 ± 0.024) %
    "pi pi pi": dict(
        branching_fraction=5.583e-2,
        msqrd=msqrd_k_to_pi_pi_pi,
        final_states=[_pi, _pi, _pi],
    ),
    # BR(K -> π0, e+, νe) = (5.07 ± 0.04) %
    "pi0 e nu": dict(
        branching_fraction=5.07e-2,
        msqrd=msqrd_k_to_pi0_e_nu,
        final_states=[_pi0, _e, _nu],
    ),
    # BR(K -> π0, μ+, νμ)   (3.352 ± 0.033) %
    "pi0 mu nu": dict(
        branching_fraction=3.352e-2,
        msqrd=msqrd_k_to_pi0_mu_nu,
        final_states=[_pi0, _mu, _nu],
    ),
    # BR(K -> π+, π0, π0) = (1.760 ± 0.023) %
    "pi pi0 pi0": dict(
        branching_fraction=1.760e-2,
        msqrd=msqrd_k_to_pi_pi0_pi0,
        final_states=[_pi, _pi0, _pi0],
    ),
    # BR(K -> e+, νe) = (1.582 ± 0.007)×10−5
    "e nu": dict(
        branching_fraction=1.582e-5,
        msqrd=None,
        final_states=[_e, _nu],
    ),
    # BR(K -> π0, π0, e+, νe) = (2.55 ± 0.04)×10−5
    "pi0 pi0 e nu": dict(
        branching_fraction=2.55e-5,
        msqrd=None,
        final_states=[_pi0, _pi0, _e, _nu],
    ),
    # BR(K -> π+, π−, e+, νe) =  (4.247 ± 0.024)×10−5
    "pi pi e nu": dict(
        branching_fraction=4.247e-5,
        msqrd=None,
        final_states=[_pi, _pi, _e, _nu],
    ),
    # from Pythia8306 (can't find in PDG)
    "pi0 pi0 mu nu": dict(
        branching_fraction=0.0000140,
        msqrd=None,
        final_states=[_pi0, _pi0, _mu, _nu],
    ),
    # BR(K -> π+, π−, μ+, νμ) =  (1.4 ± 0.9)×10−5
    "pi pi mu nu": dict(
        branching_fraction=1.4e-5,
        msqrd=None,
        final_states=[_pi, _pi, _mu, _nu],
    ),
    # BR(K -> e+, νe, e+, e−) =  (2.48 ± 0.20 )×10−8
    "e e e nu": dict(
        branching_fraction=2.48e-8,
        msqrd=None,
        final_states=[_e, _e, _e, _nu],
    ),
    # BR(K -> μ+, νμ, e+, e−) =  (7.06 ± 0.31 )×10−8
    "mu e e nu": dict(
        branching_fraction=7.06e-8,
        msqrd=None,
        final_states=[_mu, _e, _e, _nu],
    ),
    # BR(K -> e+, νe, μ+, μ−) =  (1.7 ± 0.5  )×10−8
    "mu mu e nu": dict(
        branching_fraction=1.7e-8,
        msqrd=None,
        final_states=[_mu, _mu, _e, _nu],
    ),
    # BR(K -> π+, e+, e−) = (3.00 ± 0.09 )×10−7
    "pi e e": dict(
        branching_fraction=3.00e-7,
        msqrd=None,
        final_states=[_pi, _e, _e],
    ),
    # BR(K -> π+, μ+, μ−) = (9.4 ± 0.6  )×10−8
    "pi mu mu": dict(
        branching_fraction=9.4e-8,
        msqrd=None,
        final_states=[_pi, _mu, _mu],
    ),
}


def make_processes(threshold=0.01) -> DecayProcesses:
    procs: Dict[str, DecayProcess] = dict()
    for name, process in processes.items():
        if process["branching_fraction"] > threshold:
            procs[name] = DecayProcess(
                parent=_k,
                final_states=process["final_states"],
                msqrd=process["msqrd"],
                branching_fraction=process["branching_fraction"],
            )
    return DecayProcesses(parent=_k, processes=procs)


# # BR(K -> μ+, νμ) = (63.56 ± 0.11) %
# BR_K_TO_MU_NU = 63.56e-2
# # BR(K -> π+, π0) = (20.67 ± 0.08 ) %
# BR_K_TO_PI_PI0 = 20.67e-2
# # BR(K -> π+, π+, π−) = (5.583 ± 0.024) %
# BR_K_TO_PI_PI_PI = 5.583e-2
# # BR(K -> π0, e+, νe) = (5.07 ± 0.04) %
# BR_K_TO_PI0_E_NU = 5.07e-2
# # BR(K -> π0, μ+, νμ)   (3.352 ± 0.033) %
# BR_K_TO_PI0_MU_NU = 3.352e-2
# # BR(K -> π+, π0, π0) = (1.760 ± 0.023) %
# BR_K_TO_PI_PI0_PI0 = 1.760e-2

# # BR(K -> e+, νe) = (1.582 ± 0.007)×10−5
# BR_K_TO_E_NUE = 1.582e-5
# # BR(K -> π0, π0, e+, νe) = (2.55 ± 0.04)×10−5
# BR_K_TO_E_NUE_PI0_PI0 = 2.55e-5
# # BR(K -> π+, π−, e+, νe) =  (4.247 ± 0.024)×10−5
# BR_K_TO_E_NUE_PI_PI = 4.247e-5
# # from Pythia8306 (can't find in PDG)
# BR_K_TO_MU_NUMU_PI0_PI0 = 0.0000140
# # BR(K -> π+, π−, μ+, νμ) =  (1.4 ± 0.9)×10−5
# BR_K_TO_MU_NUMU_PI_PI = 1.4e-5
# # BR(K -> e+, νe, e+, e−) =  (2.48 ± 0.20 )×10−8
# BR_K_TO_E_E_E_NUE = 2.48e-8
# # BR(K -> μ+, νμ, e+, e−) =  (7.06 ± 0.31 )×10−8
# BR_K_TO_MU_E_E_NUMU = 7.06e-8
# # BR(K -> e+, νe, μ+, μ−) =  (1.7 ± 0.5  )×10−8
# BR_K_TO_MU_MU_E_NUE = 1.7e-8
# # BR(K -> π+, e+, e−) = (3.00 ± 0.09 )×10−7
# BR_K_TO_PI_E_E = 3.00e-7
# # BR(K -> π+, μ+, μ−) = (9.4 ± 0.6  )×10−8
# BR_K_TO_PI_MU_MU = 9.4e-8

# k_to_mu_nu = DecayProcess(
#     parent=charged_kaon,
#     final_states=[muon, neutrino],
#     msqrd=lambda *_: BR_K_TO_MU_NU,
#     branching_fraction=BR_K_TO_MU_NU,
# )

# k_to_pi_pi0 = DecayProcess(
#     parent=charged_kaon,
#     final_states=[charged_pion, neutral_pion],
#     msqrd=lambda *_: BR_K_TO_MU_NU,
#     branching_fraction=BR_K_TO_PI_PI0,
# )

# k_to_pi_pi_pi = DecayProcess(
#     parent=charged_kaon,
#     final_states=[charged_pion, charged_pion, charged_pion],
#     msqrd=msqrd_k_to_pi_pi_pi,
#     branching_fraction=BR_K_TO_PI_PI_PI,
# )


# k_to_pi0_e_nu = DecayProcess(
#     parent=charged_kaon,
#     final_states=[neutral_pion, electron, neutrino],
#     msqrd=msqrd_k_to_pi0_e_nu,
#     branching_fraction=BR_K_TO_PI0_E_NU,
# )

# k_to_pi0_mu_nu = DecayProcess(
#     parent=charged_kaon,
#     final_states=[neutral_pion, muon, neutrino],
#     msqrd=msqrd_k_to_pi0_mu_nu,
#     branching_fraction=BR_K_TO_PI0_E_NU,
# )

# k_to_pi_pi0_pi0 = DecayProcess(
#     parent=charged_kaon,
#     final_states=[charged_pion, neutral_pion, neutral_pion],
#     msqrd=msqrd_k_to_pi_pi0_pi0,
#     branching_fraction=BR_K_TO_PI_PI0_PI0,
# )
