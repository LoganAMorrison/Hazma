import numpy as np

from hazma.utils import lnorm_sqr

from utils import DecayProcessInfo
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import photon as _a
from utils import eta as _eta
from utils import eta_prime as _etap
from utils import charged_rho as _rho
from utils import neutral_rho as _rho0
from utils import omega as _omega


def msqrd_etap_to_eta_pi_pi(momenta):
    s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
    t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])
    metap = _etap.mass
    meta = _eta.mass
    mpi = _pi.mass

    a = -0.056
    b = -0.049
    c = 0.0027
    d = -0.063

    ee = (metap**2 + meta**2 - s) / (2 * metap)
    ep = (metap**2 + mpi**2 - t) / (2 * metap)
    em = metap - ee - ep
    teta = ee - meta
    tp = ep - mpi
    tm = em - mpi

    q = metap - meta - 2 * mpi
    x = np.sqrt(3) * (tp - tm) / q
    y = (meta + 2 * mpi) / mpi * teta / q - 1

    return 1 + a * y + b * y**2 + c * x + d * x**2


def msqrd_etap_to_eta_pi0_pi0(momenta):
    """

    Parameters
    ----------
    s
    t

    Returns
    -------

    References
    ----------
    ..[1] https://journals-aps-org.oca.ucsc.edu/prd/pdf/10.1103/PhysRevD.97.012003

    """
    s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
    t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])

    metap = _etap.mass
    meta = _eta.mass
    mpi0 = _pi0.mass

    a = -0.087
    b = -0.073
    c = 0.0
    d = -0.074

    ee = (metap**2 + meta**2 - s) / (2 * metap)
    e1 = (metap**2 + mpi0**2 - t) / (2 * metap)
    e2 = metap - ee - e1
    teta = ee - meta
    t1 = e1 - mpi0
    t2 = e2 - mpi0

    q = metap - meta - 2 * mpi0
    x = np.sqrt(3) * np.abs(t1 - t2) / q
    y = (meta + 2 * mpi0) / mpi0 * teta / q - 1

    return 1 + a * y + b * y**2 + c * x + d * x**2


def msqrd_etap_to_pi0_pi0_pi0(momenta):
    """Compute the squared matrix element for η' -> π⁰ + π⁰ + π⁰.

    Uses the value from PDG particle listing of η'.
    """
    s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
    t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])

    metap = _etap.mass
    mpi0 = _pi0.mass

    e1 = (metap**2 + mpi0**2 - s) / (2 * metap)
    e2 = (metap**2 + mpi0**2 - t) / (2 * metap)
    e3 = metap - e1 - e2

    den = (metap - 3 * mpi0) ** 2
    z = 2.0 / 3.0 * sum([(3 * e - metap) ** 2 / den for e in [e1, e2, e3]])
    beta = -0.61

    return 1.0 + 2 * beta * z


processes = [
    # BR(π⁺, π⁻, η) = (42.5 ± 0.5) %
    DecayProcessInfo(
        branching_fraction=42.5e-2,
        final_states=[_pi, _pi, _eta],
        msqrd=msqrd_etap_to_eta_pi_pi,
    ),
    # BR(ρ⁰, γ) = (29.5 ± 0.4) % (including non-resonant π+ + π− + γ)
    DecayProcessInfo(
        branching_fraction=29.5e-2,
        final_states=[_rho0, _a],
        msqrd=None,
    ),
    # BR(π⁰, π⁰, η) = (22.4 ± 0.5) %
    DecayProcessInfo(
        branching_fraction=22.4e-2,
        final_states=[_pi0, _pi0, _eta],
        msqrd=msqrd_etap_to_eta_pi0_pi0,
    ),
    # BR(ω, γ) = ( 2.52 ± 0.07) %
    DecayProcessInfo(
        branching_fraction=2.52e-2,
        final_states=[_omega, _a],
        msqrd=None,
    ),
    # BR(γ, γ) = ( 2.307 ± 0.033) %
    DecayProcessInfo(
        branching_fraction=2.307e-2,
        final_states=[_a, _a],
        msqrd=None,
    ),
    # BR(π⁰, π⁰, π⁰) = ( 2.50 ± 0.17 )×10−3
    DecayProcessInfo(
        branching_fraction=2.50e-3,
        final_states=[_pi0, _pi0, _pi0],
        msqrd=msqrd_etap_to_pi0_pi0_pi0,
    ),
    # BR(μ⁺, μ⁻, γ) = (1.13 ± 0.28)×10−4
    DecayProcessInfo(
        branching_fraction=1.13e-4,
        final_states=[_mu, _mu, _a],
        msqrd=None,
    ),
    # BR(ω, e⁺, e⁻) = ( 2.0 ± 0.4  )×10−4
    DecayProcessInfo(
        branching_fraction=2e-4,
        final_states=[_omega, _e, _e],
        msqrd=None,
    ),
    # BR(π⁺, π⁻, π⁰) = (3.61 ± 0.17)×10−3
    # BR(π⁺, π⁻, π⁰) = (3.8 ± 0.5)×10−3 (S-wave)
    DecayProcessInfo(
        branching_fraction=3.61e-3,
        final_states=[_pi, _pi, _pi0],
        msqrd=None,
    ),
    # BR(π∓, ρ±) = (7.4 ± 2.3)×10−4
    DecayProcessInfo(
        branching_fraction=7.4e-4,
        final_states=[_pi, _rho],
        msqrd=None,
    ),
    # BR(π⁺, π⁻, π⁺, π⁻) = (8.4 ± 0.9)×10−5
    DecayProcessInfo(
        branching_fraction=8.4e-5,
        final_states=[_pi, _pi, _pi],
        msqrd=None,
    ),
    # BR(π⁺, π⁻, π⁰, π⁰) = (1.8 ± 0.4)×10−4
    DecayProcessInfo(
        branching_fraction=1.8e-4,
        final_states=[_pi, _pi, _pi0, _pi0],
        msqrd=None,
    ),
    # BR(π⁺, π⁻, e⁺, e⁻) = (2.4 +1.3 −1.0)×10−3
    DecayProcessInfo(
        branching_fraction=2.4e-3,
        final_states=[_pi, _pi, _e, _e],
        msqrd=None,
    ),
    # BR(γ, e⁺, e⁻) = (4.91 ± 0.27)×10−4
    DecayProcessInfo(
        branching_fraction=4.91e-4,
        final_states=[_e, _e, _a],
        msqrd=None,
    ),
    # BR(π⁰, γ, γ) = (3.20 ± 0.24)×10−3
    # BR(π⁰, γ, γ) = (6.2 ± 0.9)×10−4 (non resonant)
    DecayProcessInfo(
        branching_fraction=3.20e-3,
        final_states=[_pi0, _a, _a],
        msqrd=None,
    ),
]
