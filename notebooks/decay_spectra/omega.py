import numpy as np

from hazma.utils import lnorm_sqr

from utils import DecayProcessInfo
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import photon as _a
from utils import eta as _eta
from utils import omega as _omega

# Ref for omega -> pi + pi + pi0
# @article{PhysRevD.98.112007,
#   title = {Dalitz plot analysis of the decay $\ensuremath{\omega}\ensuremath{\rightarrow}{\ensuremath{\pi}}^{+}{\ensuremath{\pi}}^{\ensuremath{-}}{\ensuremath{\pi}}^{0}$},
#   author = {Ablikim, M. et.al},
#   collaboration = {BESIII Collaboration},
#   journal = {Phys. Rev. D},
#   volume = {98},
#   issue = {11},
#   pages = {112007},
#   numpages = {9},
#   year = {2018},
#   month = {Dec},
#   publisher = {American Physical Society},
#   doi = {10.1103/PhysRevD.98.112007},
#   url = {https://link.aps.org/doi/10.1103/PhysRevD.98.112007}
# }


def msqrd_omega_to_pi_pi_pi0(momenta):
    mw = _omega.mass
    mpi = _pi.mass
    mpi0 = _pi0.mass

    pp = momenta[:, 0]
    pm = momenta[:, 1]
    p0 = momenta[:, 2]

    s = lnorm_sqr(pp + pm)
    t = lnorm_sqr(pm + p0)
    u = lnorm_sqr(pp + p0)

    # |pp X pm|^2 / mw
    cross = (
        np.square(pm[3] * pp[2] - pm[2] * pp[3])
        - np.square(pm[3] * pp[1] - pm[1] * pp[3])
        + np.square(pm[2] * pp[1] - pm[1] * pp[2])
    )

    s0 = (s + t + u) / np.sqrt(3)
    rw = 2.0 / 3.0 * mw * (mw - 2 * mpi - mpi0)
    x = (t - u) / (np.sqrt(3) * rw)  # type: ignore
    y = (s - s0) / rw + 2 * (mpi - mpi0) / (mw - 2 * mpi - mpi0)
    z = x**2 + y**2
    phi = np.arctan2(y, x)

    alpha = 0.133
    beta = 0.037

    return cross / mw * (1.0 + 2 * alpha * z + 2 * beta * z**1.5 * np.sin(3 * phi))


processes = [
    # BR(π⁺, π⁻, π⁰) = 89.2 ± 0.7 %
    DecayProcessInfo(
        branching_fraction=89.2e-2,
        final_states=[_pi, _pi, _pi0],
        msqrd=msqrd_omega_to_pi_pi_pi0,
    ),
    # BR(π⁰, γ) = 8.34 ± 0.26 %
    DecayProcessInfo(
        branching_fraction=8.34e-2,
        final_states=[_pi0, _a],
        msqrd=None,
    ),
    # BR(π⁺, π⁻) = 1.53 +0.11 −0.13 %
    DecayProcessInfo(
        branching_fraction=1.53e-2,
        final_states=[_pi, _pi],
        msqrd=None,
    ),
    # BR(η, γ) = 4.5e-4 ± 0.4e-4
    DecayProcessInfo(
        branching_fraction=4.5e-4,
        final_states=[_eta, _a],
        msqrd=None,
    ),
    # BR(π⁰, e⁺, e⁻) = 7.7e-4 ± 0.6e-4
    DecayProcessInfo(
        branching_fraction=7.7e-4,
        final_states=[_pi0, _e, _e],
        msqrd=None,
    ),
    # BR(π⁰, μ⁺, μ⁻) = 1.34e-4 ± 0.18e-4
    DecayProcessInfo(
        branching_fraction=1.34e-4,
        final_states=[_pi0, _mu, _mu],
        msqrd=None,
    ),
    # BR(e⁺, e⁻) = 7.39e-5 ± 0.19e-5
    DecayProcessInfo(
        branching_fraction=7.39e-5,
        final_states=[_e, _e],
        msqrd=None,
    ),
    # BR(μ⁺, μ⁻) = 7.4e-5 ± 1.8e-5
    DecayProcessInfo(
        branching_fraction=7.4e-5,
        final_states=[_mu, _mu],
        msqrd=None,
    ),
    # BR(π⁰, π⁰, γ) = 6.7e-5 ± 1.1e-5
    DecayProcessInfo(
        branching_fraction=6.7e-5,
        final_states=[_pi0, _pi0, _a],
        msqrd=None,
    ),
]
