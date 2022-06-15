"""Partial decay widths of a RH-neutrino.
"""

import numpy as np
from scipy import integrate

from hazma import parameters
from hazma.utils import kallen_lambda
from hazma.form_factors.vector import VectorFormFactorPiPi

from ._proto import SingleRhNeutrinoModel

_lepton_masses = [parameters.electron_mass, parameters.muon_mass, parameters.tau_mass]
_flavor_gen = {"e": 1, "mu": 2, "tau": 3}


# ============================================================================
# ---- N -> nu + neutral-meson -----------------------------------------------
# ============================================================================


def _width_v_hp(model: SingleRhNeutrinoModel, mp, fp):
    mx = model.mx
    if mx < mp:
        return 0.0

    x = mp / mx
    u = 0.5 * np.tan(2 * model.theta)
    return (
        u**2
        * parameters.GF**2
        * fp**2
        * mx**3
        * (1.0 - x**2) ** 2
        / (64.0 * np.pi)
    )


def width_v_pi0(model: SingleRhNeutrinoModel):
    """Partial decay width into a neutrino and neutral pion."""
    mh = parameters.neutral_pion_mass
    fh = parameters.fpi0
    return _width_v_hp(model, mh, fh)


def width_v_eta(model: SingleRhNeutrinoModel):
    """Partial decay width into a neutrino and eta."""
    mh = parameters.eta_mass
    fh = parameters.feta
    return _width_v_hp(model, mh, fh)


# ============================================================================
# ---- N -> nu + charged-meson -----------------------------------------------
# ============================================================================


def _width_l_hp(model: SingleRhNeutrinoModel, ml, mh, fh, ckm):
    mx = model.mx
    if mx < mh + ml:
        return 0.0

    xh = mh / mx
    xl = ml / mx
    u = 0.5 * np.tan(2 * model.theta)

    pre = u**2 * parameters.GF**2 * ckm**2 * fh**2 * mx**3 / (32 * np.pi)

    return (
        pre
        * ((1 - xl**2) ** 2 + xh**2 * (1 + xl**2) - 2 * xh**4)
        * np.sqrt(kallen_lambda(1, xh**2, xl**2))
    )


def width_l_pi(model: SingleRhNeutrinoModel, ml):
    """Partial decay width into a lepton and charged pion."""
    mh = parameters.charged_pion_mass
    fh = parameters.fpi
    ckm = parameters.Vud
    return _width_l_hp(model, ml, mh, fh, ckm)


def width_l_k(model: SingleRhNeutrinoModel, ml):
    """Partial decay width into a lepton and charged kaon."""
    mh = parameters.charged_kaon_mass
    fh = parameters.fk
    ckm = parameters.Vus
    return _width_l_hp(model, ml, mh, fh, ckm)


# ============================================================================
# ---- N -> nu + charged-vector-meson ----------------------------------------
# ============================================================================


def _width_ell_hv(model: SingleRhNeutrinoModel, ml, mh, gh, ckm):
    mx = model.mx
    xh = mh / mx
    xl = ml / mx
    u = 0.5 * np.tan(2 * model.theta)

    pre = (
        u**2
        * parameters.GF**2
        * ckm**2
        * gh**2
        * mx**3
        / (16 * np.pi * mh**2)
    )

    return (
        pre
        * ((1 - xl**2) ** 2 + xh**2 * (1 + xl**2) - 2 * xh**4)
        * np.sqrt(kallen_lambda(1, xh**2, xl**2))
    )


def _width_v_hv(model: SingleRhNeutrinoModel, mh, kh, gr):
    mx = model.mx
    xh = mh / mx
    u = 0.5 * np.tan(2 * model.theta)

    pre = (parameters.GF**2 * kh * gr * u / mh) ** 2 * mx**3 / (32 * np.pi)
    return pre * (1 + 2 * xh**2) * (1 - xh**2) ** 2


# ============================================================================
# ---- N -> nu + meson + meson -----------------------------------------------
# ============================================================================


def invariant_mass_distribution_l_pi0_pi(
    model: SingleRhNeutrinoModel, s, ml: float, ff: VectorFormFactorPiPi
):
    """Partial decay width into a charged lepton, a neutral pion and a charged
    pion."""
    mx = model.mx
    mpi = parameters.charged_pion_mass

    xl = ml / mx

    u = 0.5 * np.tan(2 * model.theta)
    vud2 = np.abs(parameters.Vud) ** 2
    pre = parameters.GF**2 * mx**3 * vud2 * u**2 / (384 * np.pi**3)
    gvuu = parameters.qe * parameters.Qu
    gvdd = parameters.qe * parameters.Qd

    single = np.isscalar(s)
    ss = np.atleast_1d(s).astype(np.float64)
    dist = np.zeros_like(ss)
    mask = (ss > 4 * mpi**2) & (ss < (mx - ml) ** 2)
    sm = ss[mask]

    qq = sm / mx**2
    t1 = (1 - xl**2) ** 2 + qq * (1 + xl**2) - 2 * qq**2
    t2 = np.sqrt(kallen_lambda(1.0, qq, xl**2))
    t3 = np.sqrt(1 - 4.0 * mpi**2 / sm) ** 3
    t4 = np.real(np.abs(ff.form_factor(q=np.sqrt(sm), gvuu=gvuu, gvdd=gvdd)) ** 2)

    dist[mask] = pre * t1 * t2 * t3 * t4

    if single:
        return dist[0]
    return dist


def width_l_pi0_pi(model: SingleRhNeutrinoModel, ml, ff: VectorFormFactorPiPi):
    """Partial decay width into a lepton, a charged pion and a neutral pion."""
    mx = model.mx
    mpi = parameters.charged_pion_mass
    if mx < ml + 2 * mpi:
        return 0.0

    def integrand(s):
        return invariant_mass_distribution_l_pi0_pi(model, s, ml, ff)

    return integrate.quad(integrand, 4 * mpi**2, (mx - ml) ** 2)[0]


def invariant_mass_distribution_v_pi_pi(
    model: SingleRhNeutrinoModel, s, ff: VectorFormFactorPiPi
):
    """Partial decay width into a neutrino and two charged pions."""
    mx = model.mx
    mpi = parameters.charged_pion_mass

    u = 0.5 * np.tan(2 * model.theta)
    sw = parameters.sin_theta_weak
    pre = (
        parameters.GF**2
        * mx**3
        * u**2
        / (768 * np.pi**3)
        * (1 - 2 * sw**2) ** 2
    )
    gvuu = parameters.qe * parameters.Qu
    gvdd = parameters.qe * parameters.Qd

    single = np.isscalar(s)
    ss = np.atleast_1d(s).astype(np.float64)
    dist = np.zeros_like(ss)
    mask = (ss > 4.0 * mpi**2) & (ss < mx**2)
    sm = ss[mask]

    qq = sm / mx**2
    t1 = (1.0 - qq) ** 2 * (1.0 + 2.0 * qq)
    t2 = np.sqrt(1.0 - 4.0 * mpi**2 / sm) ** 3
    t3 = np.abs(ff.form_factor(q=np.sqrt(sm), gvuu=gvuu, gvdd=gvdd)) ** 2
    dist[mask] = pre * t1 * t2 * np.real(t3)

    if single:
        return dist[0]
    return dist


def width_v_pi_pi(model: SingleRhNeutrinoModel, ff: VectorFormFactorPiPi):
    """Partial decay width into a neutrino and two charged pions."""
    mx = model.mx
    mpi = parameters.charged_pion_mass

    def integrand(s):
        return invariant_mass_distribution_v_pi_pi(model, s, ff)

    return integrate.quad(integrand, 4 * mpi**2, mx**2)[0]


# ============================================================================
# ---- N -> nu + f + f -------------------------------------------------------
# ============================================================================


def invariant_mass_distribution_v_f_f(model: SingleRhNeutrinoModel, s, mf: float):
    mx = model.mx
    z = 1 - s / mx**2
    x = mf / mx

    sw = parameters.sin_theta_weak
    cw = parameters.cos_theta_weak
    GF = parameters.GF

    return (
        GF**2
        * mx**3
        * z**2
        * np.sqrt(np.clip((1 - z) * (1 - 4 * x**2 - z), 0.0, None))
        * (
            3
            - 5 * z
            + 2 * x**2 * z
            + 2 * z**2
            + 4 * sw**2 * (1 + 2 * x**2 - z) * (-3 + 2 * z)
            - 8 * sw**4 * (1 + 2 * x**2 - z) * (-3 + 2 * z)
            + 4 * cw**4 * (3 + (-5 + 2 * x**2) * z + 2 * z**2)
            - 4
            * cw**2
            * (
                3
                - 5 * z
                + 2 * x**2 * z
                + 2 * z**2
                + 2 * sw**2 * (1 + 2 * x**2 - z) * (-3 + 2 * z)
            )
        )
        * np.sin(2 * model.theta) ** 2
    ) / (768.0 * cw**4 * np.pi**3 * (-1 + z) ** 2)


def _width_v_f_f(model: SingleRhNeutrinoModel, mf, nc, c1f, c2f):
    mx = model.mx
    if mx < 2 * mf:
        return 0.0

    u = 0.5 * np.tan(2 * model.theta)

    pre = nc * parameters.GF**2 * mx**5 / (192 * np.pi**3) * u**2

    x = mf / mx

    sx = np.sqrt(1.0 - 4 * x**2)
    num = 1.0 - 3.0 * x**2 - (1.0 - x**2) * sx
    if num < 1e-10:
        lx = 4 * np.log(x) + 4 * x**2 + 6 * x**4
    else:
        lx = np.log(num / (x**2 * (1.0 + sx)))

    t1_s = (1 - 14 * x**2 - 2 * x**4 - 12 * x**6) * sx
    t1_l = 12 * x**4 * (x**4 - 1) * lx
    t1 = t1_s + t1_l
    t2_s = x**2 * (2 + 10 * x**2 - 12 * x**4) * sx
    t2_l = 6 * x**4 * (1 - 2 * x**2 + 2 * x**4) * lx
    t2 = t2_s + t2_l

    return pre * (c1f * t1 + 4 * c2f * t2)


def _width_v_l_l(model: SingleRhNeutrinoModel, i, j):
    """Partial decay width into a neutrino and two charged leptons."""
    sw = parameters.sin_theta_weak
    if i == j:
        c1f = 0.25 * (1 + 4 * sw**2 + 8 * sw**4)
        c2f = 0.5 * sw**2 * (2 * sw**2 + 1)
    else:
        c1f = 0.25 * (1 - 4 * sw**2 + 8 * sw**4)
        c2f = 0.5 * sw**2 * (2 * sw**2 - 1)

    if j == 1:
        ml = parameters.electron_mass
    elif j == 2:
        ml = parameters.muon_mass
    elif j == 3:
        ml = parameters.tau_mass
    else:
        raise ValueError(f"Invalid value j = {j}. Must be 1, 2, or 3.")

    return _width_v_f_f(model, ml, 1.0, c1f, c2f)


def width_v_v_v(model: SingleRhNeutrinoModel, i: int, j: int):
    """Partial decay width into three neutrinos."""
    mx = model.mx
    u = 0.5 * np.tan(2 * model.theta)
    w = parameters.GF**2 * mx**5 / (768 * np.pi**3) * u**2

    if i == j:
        w = 2 * w

    return w


# ============================================================================
# ---- N -> l + u + d -----------------------------------------------------
# ============================================================================


def _width_l_u_d(model: SingleRhNeutrinoModel, nw, ml, mu, md):
    mx = model.mx

    if mx < ml + mu + md:
        return 0.0

    u = 0.5 * np.tan(2 * model.theta)

    pre = nw * parameters.GF**2 * mx**5 / (192 * np.pi**3) * u**2
    xl = ml / mx
    xu = mu / mx
    xd = md / mx

    def integrand(x):
        return (
            1.0
            / x
            * (x - xl**2 - xd**2)
            * (1 + xu**2 - x)
            * np.sqrt(kallen_lambda(x, xl**2, xd**2) * kallen_lambda(1, x, xu**2))
        )

    lb = (xd + xl) ** 2
    ub = (1 - xu) ** 2

    return pre * integrate.quad(integrand, lb, ub)[0]


def _width_l_v_l(model: SingleRhNeutrinoModel, i, j):
    ml = _lepton_masses[i - 1]
    nw = 1.0
    mu = 0.0
    md = _lepton_masses[j - 1]
    return _width_l_u_d(model, nw, ml, mu, md)


def width_v_l_l(model: SingleRhNeutrinoModel, i, j, k):
    """Partial decay width into a neutrino and two charged leptons."""
    gn = _flavor_gen[model.flavor]

    if gn == i:
        if j == k:
            return _width_v_l_l(model, i, j)
        return 0.0

    if gn == j:
        if i == k:
            return _width_l_v_l(model, j, k)
        return 0.0

    if gn == k:
        if i == j:
            return _width_l_v_l(model, k, j)
        return 0.0

    return 0.0


# ============================================================================
# ---- N -> l + u + d -----------------------------------------------------
# ============================================================================


def width_v_a(model: SingleRhNeutrinoModel) -> float:
    return (
        9
        * parameters.alpha_em
        * parameters.GF**2
        / (1024 * np.pi**4)
        * model.mx**5
        * np.sin(2 * model.theta) ** 2
    )
