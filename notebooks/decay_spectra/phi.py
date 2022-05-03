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
from utils import charged_kaon as _k
from utils import long_kaon as _kL
from utils import short_kaon as _kS


def rho_propagator(q2, rho_mass, rho_width):
    den = q2 / rho_mass**2 - 1 + 1j * np.sqrt(q2) * rho_width / rho_mass**2
    return 1.0 / den


def amplitude_rho_pi(pp, pm, p0, rho_mass, rho_width):
    """
    Amplitude -> pi^+ + pi^- + pi^0
    """

    q2_plus = lnorm_sqr(pp + p0)
    q2_minus = lnorm_sqr(pm + p0)
    q2_zero = lnorm_sqr(pp + pm)

    d1 = rho_propagator(q2_plus, rho_mass, rho_width)
    d2 = rho_propagator(q2_minus, rho_mass, rho_width)
    d3 = rho_propagator(q2_zero, rho_mass, rho_width)

    return d1 + d2 + d3


def make_msqrd_phi_to_pi_pi_pi0(
    a=0.101, phi=-2.91, rho_mass=775.26, rho_width=149.0, an=7.52
):
    def msqrd(momenta):
        pp = momenta[:, 0]
        pm = momenta[:, 1]
        p0 = momenta[:, 2]

        amp_rho_pi = amplitude_rho_pi(pp, pm, p0, rho_mass, rho_width)

        amp2 = np.abs(an * a * np.exp(1j * phi) + amp_rho_pi) ** 2
        cross = (
            np.square(pm[3] * pp[2] - pm[2] * pp[3])
            + np.square(-pm[3] * pp[1] + pm[1] * pp[3])
            + np.square(pm[2] * pp[1] - pm[1] * pp[2])
        )
        return amp2 * cross

    return msqrd


processes = [
    # BR(K⁺, K⁻) = (49.2 ± 0.5) %
    DecayProcessInfo(
        branching_fraction=49.2e-2,
        msqrd=None,
        final_states=[_k, _k],
    ),
    # BR(KL, KS) = (34.0 ± 0.4) %
    DecayProcessInfo(
        branching_fraction=34.0e-2,
        msqrd=None,
        final_states=[_kS, _kL],
    ),
    # PDG: BR(ρ, π⁰) + BR(ρ⁺, π⁻) + BR(ρ⁻, π⁺) +  BR(π⁺, π⁻, π⁰) = (15.24 ± 0.33) %
    DecayProcessInfo(
        branching_fraction=15.24e-2,
        msqrd=make_msqrd_phi_to_pi_pi_pi0(),
        final_states=[_pi, _pi, _pi0],
    ),
    # BR(η, γ) = (1.303 ± 0.025) %
    DecayProcessInfo(
        branching_fraction=1.303e-2,
        msqrd=None,
        final_states=[_eta, _a],
    ),
    # BR(π⁰, γ) = (1.32 ± 0.06)×10−3
    DecayProcessInfo(
        branching_fraction=1.32e-3,
        msqrd=None,
        final_states=[_pi0, _a],
    ),
    # BR(e⁺, e⁻) = (2.974 ± 0.034)×10−4
    DecayProcessInfo(
        branching_fraction=2.974e-4,
        msqrd=None,
        final_states=[_e, _e],
    ),
    # BR(μ⁺, μ⁻) = (2.86 ± 0.19)×10−4
    DecayProcessInfo(
        branching_fraction=2.86e-4,
        msqrd=None,
        final_states=[_mu, _mu],
    ),
    # BR(η, e⁺, e⁻) = (1.08 ± 0.04)×10−4
    DecayProcessInfo(
        branching_fraction=1.08e-4,
        msqrd=None,
        final_states=[_eta, _e, _e],
    ),
    # BR(π⁺, π⁻) = (7.3 ± 1.3)×10−5
    DecayProcessInfo(
        branching_fraction=7.3e-5,
        msqrd=None,
        final_states=[_pi, _pi],
    ),
    # BR(ω, π⁰) = (4.7 ± 0.5)×10−5
    DecayProcessInfo(
        branching_fraction=4.7e-5,
        msqrd=None,
        final_states=[_omega, _pi0],
    ),
    # BR(π⁺, π⁻, γ) = (4.1 ± 1.3)×10−5
    DecayProcessInfo(
        branching_fraction=4.1e-5,
        msqrd=None,
        final_states=[_pi, _pi, _a],
    ),
    # BR(f₀(980), γ) = (3.22 ± 0.19)×10−4
    # DecayProcessInfo(branching_fraction=3.22e-4,msqrd=None,final_states=[],),
    # BR(π⁰, π⁰, γ) = (1.12 ± 0.06)×10−4
    DecayProcessInfo(
        branching_fraction=1.12e-4,
        msqrd=None,
        final_states=[_pi0, _pi0, _a],
    ),
    # BR(π⁺, π⁻, π⁺, π⁻) = (3.9 +2.8 −2.2)×10−6
    DecayProcessInfo(
        branching_fraction=3.9e-6,
        msqrd=None,
        final_states=[_pi, _pi, _pi, _pi],
    ),
    # BR(π⁰, e⁺, e⁻) = (1.33 +0.07 −0.10)×10−5
    DecayProcessInfo(
        branching_fraction=1.33e-5,
        msqrd=None,
        final_states=[_pi0, _e, _e],
    ),
    # BR(π⁰, η, γ) = (7.27 ± 0.30)×10−5
    DecayProcessInfo(
        branching_fraction=7.27e-5,
        msqrd=None,
        final_states=[_pi0, _eta, _a],
    ),
    # BR(a₀(980), γ) = (7.6 ± 0.6)×10−5
    # DecayProcessInfo(branching_fraction=BR_PHI_TO_A0980_A = 7.6e-5,msqrd=None,final_states=[],),
    # BR(η'(958), γ) = (6.22 ± 0.21)×10−5
    DecayProcessInfo(
        branching_fraction=6.22e-5,
        msqrd=None,
        final_states=[_etap, _a],
    ),
    # BR(μ⁺, μ⁻, γ) = (1.4 ± 0.5)×10−5
    DecayProcessInfo(
        branching_fraction=1.4e-5,
        msqrd=None,
        final_states=[_mu, _mu, _a],
    ),
]
