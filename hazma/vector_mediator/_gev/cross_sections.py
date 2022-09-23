from typing import Callable, Dict, Sequence

import numpy as np

from hazma import parameters
from hazma.utils import RealArray
import hazma.form_factors.vector as vff


ME = parameters.electron_mass
MMU = parameters.muon_mass
MPI0 = parameters.neutral_pion_mass
MPI = parameters.charged_pion_mass
MK = parameters.charged_kaon_mass
MK0 = parameters.neutral_kaon_mass
META = parameters.eta_mass
METAP = parameters.eta_prime_mass
MPHI = parameters.phi_mass
MOMEGA = parameters.omega_mass


def _sigma_xx_to_call_wrapper(self, q, form_factor, **kwargs):
    return form_factor.cross_section(
        q=q,
        mx=self.mx,
        mv=self.mv,
        gvxx=self.gvxx,
        wv=self.width_v(),
        couplings=self._couplings,
        **kwargs,
    )


def cross_section_x_x_to_p_p(s, mx, mp, ff, mv, gamv):
    prop = mv**4 + s**2 + mv**2 * (gamv**2 - 2 * s)
    return ((-4 * mp**2 + s) ** 1.5 * (2 * mx**2 + s) * np.abs(ff) ** 2) / (
        48.0 * np.pi * s * np.sqrt(-4 * mx**2 + s) * prop
    )


def cross_section_x_x_to_p_a(s, mx, mp, ff, mv, gamv):
    prop = mv**4 + s**2 + mv**2 * (gamv**2 - 2 * s)
    return ((-(mp**2) + s) ** 3 * (2 * mx**2 + s) * np.abs(ff) ** 2) / (
        96.0 * np.pi * s * np.sqrt(s * (-4 * mx**2 + s)) * prop
    )


def cross_section_x_x_to_p_v(s, mx, mp, mvec, ff, mv, gamv):
    prop = mv**4 + s**2 + mv**2 * (gamv**2 - 2 * s)
    return (
        (2 * mx**2 + s)
        * ((mp**4 + (mvec**2 - s) ** 2 - 2 * mp**2 * (mvec**2 + s)) / s) ** 1.5
        * np.abs(ff) ** 2
    ) / (96.0 * np.pi * np.sqrt(-4 * mx**2 + s) * prop)


def __width_to_cs(self, cme):
    """
    Computes the factor needed to transform a width V -> X into a
    cross-section x+xbar -> X.

    (
        gvxx**2 * (2 * mx**2 + s)
    )/(
        np.sqrt(1 - (4 * mx**2)/s) * cme * ((M**2 - s)**2 + mv**2 * widthv**2)
    )
    """
    gvxx = self.gvxx
    mux2 = (self.mx / cme) ** 2
    mug2 = (self.width_v() / cme) ** 2
    muv2 = (self.mv / cme) ** 2

    num = gvxx**2 * (1 + 2 * mux2)
    den = (1.0 + (mug2 - 2.0) * muv2 + muv2**2) * np.sqrt(1.0 - 4.0 * mux2) * cme**3
    return num / den


def __sigma_from_width_fn(self, width: Callable, e_cm, fsp_masses: Sequence[float]):
    """
    Compute the DM annihilation cross section given a function to compute the
    vector mediator partial width.

    Parameters
    ----------
    width: Callable
        Unary function to compute vector mediator partial width.
    e_cm:
        Center-of-mass energy where the cross section should be computed.
    fsp_masses: list
        List of the final-state particle masses. Used for thresholds.
    """
    isp_thres = 2 * self.mx
    fsp_thres = sum(fsp_masses)

    if hasattr(e_cm, "__len__"):
        cme = np.array(e_cm, dtype=np.float64)
        mask = np.logical_and(cme > fsp_thres, cme > isp_thres)
        cs = np.zeros_like(cme)
        rescale = np.zeros_like(cme)

        rescale = __width_to_cs(self, cme[mask])
        cs[mask] = np.vectorize(width)(cme[mask]) * rescale

        return cs
    else:
        cs = 0.0
        if e_cm > fsp_thres and e_cm > isp_thres:
            cs = width(e_cm) * __width_to_cs(self, e_cm)
        return cs


def __sigma_xx_to_f_f(self, e_cm, gvff, mf):
    single = not hasattr(e_cm, "__len__")
    q = (
        np.array([e_cm], dtype=np.float64)
        if single
        else np.array(e_cm, dtype=np.float64)
    )

    mask = (q > 2 * self.mx) & (q > 2 * mf)

    mx = self.mx
    mv = self.mv
    widthv = self.width_v()
    gvxx = self.gvxx

    s = q[mask] ** 2
    result = np.zeros_like(q)

    result[mask] = (
        gvff**2
        * gvxx**2
        * np.sqrt(1 - (4 * mf**2) / s)
        * (2 * mf**2 + s)
        * (2 * mx**2 + s)
    ) / (
        12.0
        * np.pi
        * np.sqrt(1 - (4 * mx**2) / s)
        * s
        * (mv**4 + s**2 + mv**2 * (-2 * s + widthv**2))
    )

    if single:
        return result[0]

    return result


def sigma_xx_to_v_v(self, cme):
    """Compute the cross section for DM annhilating into two vector
    mediators [V V].

    Parameters
    ----------
    cme: array or float
        Center-of-mass energy.

    Returns
    -------
    sigma: array or float
        Cross section.
    """
    scalar = not hasattr(cme, "__len__")
    q = np.array([cme]) if scalar else np.array(cme)

    mask = (2 * self.mv < q) & (2 * self.mx < q)

    qq = q[mask]
    sigma = np.zeros_like(q)

    muv = self.mv / qq
    mux = self.mx / qq
    mux2 = mux**2
    muv2 = muv**2
    mux4 = mux2**2
    muv4 = muv2**2

    sqrtx = np.sqrt(1.0 - 4 * mux2)
    sqrtv = np.sqrt(1.0 - 4 * muv2)
    sqrtxv = sqrtx * sqrtv
    pre = self.gvxx**4 * sqrtv / (4.0 * np.pi * sqrtx * qq**2)

    sigma[mask] = pre * (
        -((2 * muv4 + mux2 + 4 * mux4) / (muv4 + mux2 - 4 * muv2 * mux2))
        - (
            2
            * (1 + 4 * muv4 + 4 * mux2 - 8 * muv2 * mux2 - 8 * mux4)
            * np.arctanh(sqrtxv / (1 - 2 * muv2))
        )
        / ((-1 + 2 * muv2) * sqrtxv)
    )

    if scalar:
        return sigma[0]

    return sigma


def sigma_xx_to_e_e(self, e_cm):
    """
    Compute the cross section for DM annhilating into two electrons
    [e⁺ e⁻].
    """
    return __sigma_xx_to_f_f(self, e_cm, self.gvee, ME)


def sigma_xx_to_mu_mu(self, e_cm):
    """
    Compute the cross section for DM annhilating into two muons [μ⁺ μ⁻].
    """
    return __sigma_xx_to_f_f(self, e_cm, self.gvmumu, MMU)


def sigma_xx_to_ve_ve(self, e_cm):
    """
    Compute the cross section for DM annhilating into two
    electron-neutrinos [νe νe].
    """
    return __sigma_xx_to_f_f(self, e_cm, self.gvveve, 0.0) / 2.0


def sigma_xx_to_vm_vm(self, e_cm):
    """
    Compute the cross section for DM annhilating into two muon-neutrinos
    [νμ νμ].
    """
    return __sigma_xx_to_f_f(self, e_cm, self.gvvmvm, 0.0) / 2.0


def sigma_xx_to_vt_vt(self, e_cm):
    """
    Compute the cross section for DM annhilating into two tau-neutrinos [ντ
    ντ].
    """
    return __sigma_xx_to_f_f(self, e_cm, self.gvvtvt, 0.0) / 2.0


def sigma_xx_to_pi_pi(self, e_cm):
    """
    Compute the cross section for DM annhilating into two charged pions
    [𝜋⁺ 𝜋⁻].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPiPi = self._ff_pi_pi
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_k0_k0(self, e_cm):
    """
    Compute the cross section for DM annhilating into into two neutral kaons
    [K⁰ K⁰].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorK0K0 = self._ff_k0_k0
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_k_k(self, e_cm):
    """
    Compute the cross section for DM annhilating into into two charged kaons
    [K⁺ K⁻].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorKK = self._ff_k_k
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_pi0_gamma(self, e_cm):
    """
    Compute the cross section for DM annhilating into into a neutral pion [𝜋⁰]
    and photon [𝛾].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPi0Gamma = self._ff_pi0_gamma
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_eta_gamma(self, e_cm):
    """
    Compute the cross section for DM annhilating into into a neutral pion [𝜋⁰]
    and photon [𝛾].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorEtaGamma = self._ff_eta_gamma
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_pi0_phi(self, e_cm):
    """
    Compute the cross section for DM annhilating into into an phi [𝜙(1020)] and
    neutral pion [𝜋⁰].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPi0Phi = self._ff_pi0_phi
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_eta_phi(self, e_cm):
    """
    Compute the cross section for DM annhilating into into an eta [𝜂] and phi
    [𝜙(1020)].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorEtaPhi = self._ff_eta_phi
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_eta_omega(self, e_cm):
    """
    Compute the cross section for DM annhilating into into an eta [𝜂] and omega
    [𝜔(782)].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorEtaOmega = self._ff_eta_omega
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_pi0_pi0_gamma(self, e_cm):
    """
    Compute the cross section for DM annhilating into into two neutral pions
    and a photon [𝜋⁰ 𝜋⁰ 𝛾].

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPi0Omega = self._ff_pi0_omega
    fsp_masses = [MPI0, MPI0, 0.0]

    def width(cme):
        pi0_omega = ff.width(mv=cme, couplings=self._couplings)
        br_omega_to_pi0_gamma = 8.34e-2

        return br_omega_to_pi0_gamma * pi0_omega

    return __sigma_from_width_fn(self, width, e_cm, fsp_masses)


def sigma_xx_to_pi_pi_pi0(self, e_cm, *, npts=10_000):
    """
    Compute the cross section for DM annhilating into into two charged pions
    and a neutral pion.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPiPiPi0 = self._ff_pi_pi_pi0
    return _sigma_xx_to_call_wrapper(self, e_cm, ff, npts=npts)


def sigma_xx_to_pi_pi_eta(self, e_cm):
    """
    Compute the cross section for DM annhilating into into two charged pions
    and an eta.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPiPiEta = self._ff_pi_pi_eta
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_pi_pi_etap(self, e_cm):
    """
    Compute the cross section for DM annhilating into into two charged pions
    and an eta-prime.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPiPiEtaPrime = self._ff_pi_pi_etap
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_pi_pi_omega(self, e_cm):
    """
    Compute the cross section for DM annhilating into into two charged pions
    and an eta-prime.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPiPiOmega = self._ff_pi_pi_omega
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_pi0_pi0_omega(self, e_cm):
    """
    Compute the cross section for DM annhilating into into two charged pions
    and an eta-prime.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPi0Pi0Omega = self._ff_pi0_pi0_omega
    return _sigma_xx_to_call_wrapper(self, e_cm, ff)


def sigma_xx_to_pi0_k0_k0(self, e_cm, *, npts: int = 50_000):
    """
    Compute the cross section for DM annhilating into into a neutral pion and
    two neutral kaons.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPi0K0K0 = self._ff_pi0_k0_k0
    return _sigma_xx_to_call_wrapper(self, e_cm, ff, npts=npts)


def sigma_xx_to_pi0_k_k(self, e_cm, *, npts: int = 50_000):
    """
    Compute the cross section for DM annhilating into into a neutral pion and
    two charged kaons.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPi0KpKm = self._ff_pi0_k_k
    return _sigma_xx_to_call_wrapper(self, e_cm, ff, npts=npts)


def sigma_xx_to_pi_k_k0(self, e_cm, *, npts: int = 50_000):
    """
    Compute the cross section for DM annhilating into into a neutral pion and
    two charged kaons.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    """
    ff: vff.VectorFormFactorPiKK0 = self._ff_pi_k_k0
    return _sigma_xx_to_call_wrapper(self, e_cm, ff, npts=npts)


def sigma_xx_to_pi_pi_pi_pi(self, e_cm, *, npts=1 << 14):
    """
    Compute the cross section for DM annhilating into into four charged pions.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    npts: int, optional
        Number of points to use for Monte-Carlo integration. Default
        is 1<<14 ~ 16_000.
    """
    ff: vff.VectorFormFactorPiPiPiPi = self._ff_pi_pi_pi_pi
    return _sigma_xx_to_call_wrapper(self, e_cm, ff, npts=npts)


def sigma_xx_to_pi_pi_pi0_pi0(self, e_cm, *, npts: int = 1 << 14):
    """
    Compute the cross section for DM annhilating into into two charged pion and
    two neutral pions.

    Parameters
    ----------
    e_cm: float or array like
        Center of mass energy.
    npts: int, optional
        Number of points to use for Monte-Carlo integration. Default
        is 1<<14 ~ 16_000.
    """
    ff: vff.VectorFormFactorPiPiPi0Pi0 = self._ff_pi_pi_pi0_pi0
    return _sigma_xx_to_call_wrapper(self, e_cm, ff, npts=npts)


def annihilation_cross_section_funcs(
    self,
) -> Dict[str, Callable[[float | RealArray], float | RealArray]]:
    return {
        "e e": self.sigma_xx_to_e_e,
        "mu mu": self.sigma_xx_to_mu_mu,
        "ve ve": self.sigma_xx_to_ve_ve,
        "vt vt": self.sigma_xx_to_vt_vt,
        "vm vm": self.sigma_xx_to_vm_vm,
        "pi pi": self.sigma_xx_to_pi_pi,
        "k0 k0": self.sigma_xx_to_k0_k0,
        "k k": self.sigma_xx_to_k_k,
        "pi0 gamma": self.sigma_xx_to_pi0_gamma,
        "eta gamma": self.sigma_xx_to_eta_gamma,
        "pi0 phi": self.sigma_xx_to_pi0_phi,
        "eta phi": self.sigma_xx_to_eta_phi,
        "eta omega": self.sigma_xx_to_eta_omega,
        "pi0 pi0 gamma": self.sigma_xx_to_pi0_pi0_gamma,
        "pi pi pi0": self.sigma_xx_to_pi_pi_pi0,
        "pi pi eta": self.sigma_xx_to_pi_pi_eta,
        "pi pi etap": self.sigma_xx_to_pi_pi_etap,
        "pi pi omega": self.sigma_xx_to_pi_pi_omega,
        "pi0 pi0 omega": self.sigma_xx_to_pi0_pi0_omega,
        "pi0 k0 k0": self.sigma_xx_to_pi0_k0_k0,
        "pi0 k k": self.sigma_xx_to_pi0_k_k,
        "pi k k0": self.sigma_xx_to_pi_k_k0,
        "pi pi pi pi": self.sigma_xx_to_pi_pi_pi_pi,
        "pi pi pi0 pi0": self.sigma_xx_to_pi_pi_pi0_pi0,
        "v v": self.sigma_xx_to_v_v,
    }
