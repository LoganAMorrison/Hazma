import numpy as np

from hazma.utils import lnorm_sqr

from utils import DecayProcessInfo
from utils import charged_pion as _pi
from utils import electron as _e
from utils import muon as _mu
from utils import neutral_pion as _pi0
from utils import photon as _a
from utils import eta as _eta


def msqrd_eta_to_pi0_pi0_pi0(momenta):
    """
    Squared matrix element for eta -> 3pi0. Parameterization taken from eta
    listing in PDG.

    |M|^2 = 1 + 2*alpha*z, alpha = -0.0288

    Parameters
    ----------
    s:
        Mandelstam variable: s = (p2 + p3)^2.
    t:
        Mandelstam variable: t = (p1 + p3)^2.

    Returns
    -------
    msqrd:
        Squared matrix element.

    """
    s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
    t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])

    meta = _eta.mass
    m0 = _pi0.mass
    alpha = -0.0288
    e1 = (meta**2 + m0**2 - s) / (2 * meta)
    e2 = (meta**2 + m0**2 - t) / (2 * meta)
    e3 = meta - e1 - e2
    den = 1.0 / (_eta.mass - _pi0.mass) ** 2
    ee = _eta.mass / 3.0
    z = 6.0 * ((e1 - ee) ** 2 * den + (e2 - ee) ** 2 * den + (e3 - ee) ** 2 * den)
    return 1.0 + 2 * alpha * z


def msqrd_eta_to_pi_pi_pi0(momenta):
    """
    Squared matrix element for eta -> pi^+ + pi^- + pi^0. Parameters are taken
    from [1]_:
        - a     = -1.095 ± 0.003
        - b*10  = 1.454 ± 0.030
        - d*100 = 8.11 ± 0.33
        - f*10  = 1.41 ± 0.07
        - g*100 = −4.4 ± 0.9

    Parameters
    ----------
    s:
        Mandelstam variable: s = (p2 + p3)^2.
    t:
        Mandelstam variable: t = (p1 + p3)^2.

    Returns
    -------
    msqrd:
        Squared matrix element.

    References
    ----------
    ..[1] dell’Universita di Messina, Ambientali. "Precision measurement of
    the η→ π π− π0 Dalitz plot distribution with the KLOE detector."
    ..[2] https://link.springer.com/content/pdf/10.1007/JHEP05(2016)019.pdf

    """
    s = lnorm_sqr(momenta[:, 1] + momenta[:, 2])
    t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])

    meta = _eta.mass
    mpi0 = _pi0.mass
    mpi = _pi.mass

    ep = (meta**2 + mpi**2 - s) / (2 * meta)
    em = (meta**2 + mpi**2 - t) / (2 * meta)
    e0 = meta - ep - em
    tp = ep - mpi
    tm = em - mpi
    t0 = e0 - mpi0
    q = meta - 2 * mpi - mpi0
    x = np.sqrt(3) * (tp - tm) / q
    y = 3 * t0 / q - 1

    a = -1.095
    b = 1.454 / 10.0
    d = 8.11 / 100.0
    f = 1.41 / 10.0
    g = -4.4 / 100.0

    # Commented-out terms were set to zero.
    return (
        1.0
        + a * y
        + b * y**2
        # + c * x
        + d * x**2
        # + e * x * y
        + f * y**3
        + g * x**2 * y
        # + h * x * y**2
        # + l * x**3
    )


processes = [
    # BR(γ, γ) = (39.41±0.20) %
    DecayProcessInfo(
        branching_fraction=69.20e-2,
        msqrd=None,
        final_states=[_a, _a],
    ),
    # BR(π0, π0, π0) = (32.68±0.23) %
    DecayProcessInfo(
        branching_fraction=32.68e-2,
        msqrd=msqrd_eta_to_pi0_pi0_pi0,
        final_states=[_pi0, _pi0, _pi0],
    ),
    # BR(π+, π−, π0) = (22.92±0.28) %
    DecayProcessInfo(
        branching_fraction=22.92e-2,
        msqrd=msqrd_eta_to_pi_pi_pi0,
        final_states=[_pi, _pi, _pi0],
    ),
    # BR(π+, π−, γ) = ( 4.22±0.08) %
    DecayProcessInfo(
        branching_fraction=4.22e-2,
        msqrd=None,
        final_states=[_pi, _pi, _a],
    ),
    # BR(π0, γ, γ) = ( 2.56±0.22)×10−4
    DecayProcessInfo(
        branching_fraction=2.56e-4,
        msqrd=None,
        final_states=[_pi0, _a, _a],
    ),
    # BR(e+, e−, γ) = ( 6.9±0.4 )×10−3
    DecayProcessInfo(
        branching_fraction=6.9e-3,
        msqrd=None,
        final_states=[_e, _e, _a],
    ),
    # BR(μ+, μ−, γ) = ( 3.1±0.4 )×10−4
    DecayProcessInfo(
        branching_fraction=3.1e-4,
        msqrd=None,
        final_states=[_mu, _mu, _a],
    ),
    # BR(μ+, μ−) = ( 5.8±0.8 )×10−6
    DecayProcessInfo(
        branching_fraction=5.8e-6,
        msqrd=None,
        final_states=[_mu, _mu],
    ),
    # BR(π+, π−, e+, e−)  = ( 2.68±0.11)×10−4
    DecayProcessInfo(
        branching_fraction=2.68e-4,
        msqrd=None,
        final_states=[_pi0, _e, _e],
    ),
    # BR(e+, e−, e+, e−) = ( 2.40±0.22)×10−5
    DecayProcessInfo(
        branching_fraction=2.40e-5,
        msqrd=None,
        final_states=[_e, _e, _e, _e],
    ),
]
