"""Partial decay widths of a RH-neutrino.
"""

from typing import Optional, Tuple, NamedTuple

import numpy as np
from scipy import integrate

from hazma import parameters
from hazma.utils import kallen_lambda
from hazma.form_factors.vector import VectorFormFactorPiPi
from hazma.phase_space import ThreeBody, PhaseSpaceDistribution1D

from ._proto import SingleRhNeutrinoModel, Generation

_lepton_masses = [
    parameters.electron_mass,
    parameters.muon_mass,
    parameters.tau_mass,
]

ALPHA_EM = parameters.alpha_em
GF = parameters.GF
VUD = parameters.Vud
SW = parameters.sin_theta_weak
CW = parameters.cos_theta_weak
GVUU = 2.0 / 3.0
GVDD = -1.0 / 3.0

MPI = parameters.charged_pion_mass
MPI0 = parameters.neutral_pion_mass
MRHO = parameters.rho_mass
MOMEGA = parameters.omega_mass
MPHI = parameters.phi_mass


# See arXiv:1805.08567
_neutrino_vector_meson_constants = {
    "rho": dict(
        g=0.162e6,  # MeV^2
    ),
    "rho0": dict(
        g=0.162e6,  # MeV^2
        k=1.0 - 2 * SW**2,
    ),
    "omega": dict(
        g=0.153e6,  # MeV^2
        k=4 / 3.0 * SW**2,
    ),
    "phi": dict(
        g=0.234e6,  # MeV^2
        k=4 / 3.0 * SW**2 - 1,
    ),
}


def _get_quad_kwargs(**kwargs):
    quad_keys = [
        "args",
        "full_output",
        "epsabs",
        "epsrel",
        "limit",
        "points",
        "weight",
        "wvar",
        "wopts",
        "maxp1",
        "limlst",
    ]
    quad_kwargs = {}
    for key in quad_keys:
        if kwargs.get(key) is not None:
            quad_kwargs[key] = kwargs[key]
    return quad_kwargs


# ============================================================================
# ---- N -> nu + neutral-meson -----------------------------------------------
# ============================================================================


def _width_v_hp(model: SingleRhNeutrinoModel, mp, fp):
    mx = model.mx
    if mx < mp:
        return 0.0

    x = mp / mx
    u = 0.5 * np.tan(2 * model.theta)
    return u**2 * parameters.GF**2 * fp**2 * mx**3 * (1.0 - x**2) ** 2 / (64.0 * np.pi)


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

    pre = u**2 * parameters.GF**2 * abs(ckm) ** 2 * fh**2 * mx**3 / (32 * np.pi)

    return (
        pre
        * ((1 - xl**2) ** 2 + xh**2 * (1 + xl**2) - 2 * xh**4)
        * np.sqrt(kallen_lambda(1, xh**2, xl**2))
    )


def width_l_pi(model: SingleRhNeutrinoModel):
    """Partial decay width into a lepton and charged pion."""
    mh = parameters.charged_pion_mass
    fh = parameters.fpi
    ckm = parameters.Vud
    ml = _lepton_masses[model.gen]
    return _width_l_hp(model, ml, mh, fh, ckm)


def width_l_k(model: SingleRhNeutrinoModel):
    """Partial decay width into a lepton and charged kaon."""
    mh = parameters.charged_kaon_mass
    fh = parameters.fk
    ckm = parameters.Vus
    ml = _lepton_masses[model.gen]
    return _width_l_hp(model, ml, mh, fh, ckm)


# ============================================================================
# ---- N -> ell + charged-vector-meson ---------------------------------------
# ============================================================================


def _width_ell_hv(model: SingleRhNeutrinoModel, ml, mh, gh, ckm):
    mx = model.mx
    if mx < mh + ml:
        return 0.0

    xh = mh / mx
    xl = ml / mx
    u = 0.5 * np.tan(2 * model.theta)

    pre = u**2 * parameters.GF**2 * abs(ckm) ** 2 * gh**2 * mx**3 / (16 * np.pi * mh**2)

    return (
        pre
        * ((1 - xl**2) ** 2 + xh**2 * (1 + xl**2) - 2 * xh**4)
        * np.sqrt(kallen_lambda(1, xh**2, xl**2))
    )


def width_l_rho(model: SingleRhNeutrinoModel):
    r"""Compute the partial width for the decay of a right-handed neutrino into
    a charged lepton and charged rho.
    """
    ml = _lepton_masses[model.gen]
    params = _neutrino_vector_meson_constants["rho"]
    g = params["g"]
    return _width_ell_hv(model, ml, MRHO, g, VUD)


# ============================================================================
# ---- N -> nu + neutral-vector-meson ----------------------------------------
# ============================================================================


def _width_v_hv(model: SingleRhNeutrinoModel, mh, kh, gr):
    mx = model.mx
    if mx < mh:
        return 0.0

    xh = mh / mx
    u = 0.5 * np.tan(2 * model.theta)

    pre = (parameters.GF * kh * gr * u / mh) ** 2 * mx**3 / (32 * np.pi)
    return pre * (1 + 2 * xh**2) * (1 - xh**2) ** 2


def width_v_rho(model: SingleRhNeutrinoModel):
    r"""Compute the partial width for the decay of a right-handed neutrino into
    a neutrino and rho.
    """
    params = _neutrino_vector_meson_constants["rho0"]
    k, g = params["k"], params["g"]
    return _width_v_hv(model, MRHO, k, g)


def width_v_omega(model: SingleRhNeutrinoModel):
    r"""Compute the partial width for the decay of a right-handed neutrino into
    a neutrino and omega.
    """
    params = _neutrino_vector_meson_constants["omega"]
    k, g = params["k"], params["g"]
    return _width_v_hv(model, MOMEGA, k, g)


def width_v_phi(model: SingleRhNeutrinoModel):
    r"""Compute the partial width for the decay of a right-handed neutrino into
    a neutrino and phi.
    """
    params = _neutrino_vector_meson_constants["phi"]
    k, g = params["k"], params["g"]
    return _width_v_hv(model, MPHI, k, g)


# ============================================================================
# ---- N -> l + meson + meson -----------------------------------------------
# ============================================================================


def msqrd_l_pi0_pi(
    s, t, model: SingleRhNeutrinoModel, form_factor: VectorFormFactorPiPi
):
    """Squared matrix element for N -> l + pi0 + pi.

    Parameters
    ----------
    s: float or array-like
        Invariant mass of the pions.
    t: float or array-like
        Invariant mass of the lepton and one of the pions.
    form_factor: VectorFormFactorPiPi
        Pion electromagnetic form factor.
    """
    mx = model.mx
    ff = form_factor.form_factor(q=np.sqrt(s), couplings=(GVUU, GVDD))
    ml = _lepton_masses[model.gen]
    ckm = abs(VUD) ** 2
    u = 0.5 * np.tan(2 * model.theta)

    return (
        2
        * ckm
        * u**2
        * GF**2
        * (
            4 * MPI**4
            + (ml**2 + mx**2) * (ml**2 + mx**2 - s)
            - 4 * (ml**2 + mx**2 + 2 * MPI**2 - s) * t
            + 4 * t**2
        )
        * np.abs(ff) ** 2
    )


def energy_distributions_l_pi0_pi(
    model: SingleRhNeutrinoModel,
    form_factor: VectorFormFactorPiPi,
    nbins: int,
    method: str = "quad",
):
    def msqrd(s, t):
        return msqrd_l_pi0_pi(s, t, model, form_factor)

    ml = _lepton_masses[model.gen]
    tb = ThreeBody(model.mx, (ml, MPI, MPI), msqrd=msqrd)
    return tb.energy_distributions(nbins=nbins, method=method)


def invariant_mass_distributions_l_pi0_pi(
    model: SingleRhNeutrinoModel,
    form_factor: VectorFormFactorPiPi,
    nbins: int,
    method: str = "quad",
):
    def msqrd(s, t):
        return msqrd_l_pi0_pi(s, t, model, form_factor)

    ml = _lepton_masses[model.gen]
    tb = ThreeBody(model.mx, (ml, MPI, MPI), msqrd=msqrd)
    return tb.invariant_mass_distributions(nbins=nbins, method=method)


def width_l_pi0_pi(
    model: SingleRhNeutrinoModel, form_factor: VectorFormFactorPiPi, **kwargs
):
    r"""Partial decay width into a lepton, a charged pion and a neutral pion.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    form_factor: VectorFormFactorPiPi
        Pion electromagnetic form-factor.

    Returns
    -------
    width: float
        Partial width into a lepton, charged pion and neutral pion.
    """
    mx = model.mx
    ml = _lepton_masses[model.gen]
    if mx < ml + 2 * MPI:
        return 0.0

    xl2 = (ml / mx) ** 2
    u = 0.5 * np.tan(2 * model.theta)

    def integrand(s):
        z = s / mx**2
        poly = (1 - xl2) ** 2 + z * (1 + xl2) - 2 * z**2
        ff = form_factor.form_factor(q=np.sqrt(s), couplings=(GVUU, GVDD))
        beta = (1.0 - 4 * MPI**2 / s) ** 1.5
        p = np.sqrt(kallen_lambda(1.0, z, xl2))
        return poly * np.abs(ff) ** 2 * p * beta

    pre = u**2 * np.abs(VUD) ** 2 * GF**2 * mx**3 / (384.0 * np.pi**3)
    quad_kwargs = _get_quad_kwargs(**kwargs)
    return pre * integrate.quad(integrand, 4 * MPI**2, (mx - ml) ** 2, **quad_kwargs)[0]


# ============================================================================
# ---- N -> nu + meson + meson -----------------------------------------------
# ============================================================================


def msqrd_v_pi_pi(
    s, t, model: SingleRhNeutrinoModel, form_factor: VectorFormFactorPiPi
):
    """Squared matrix element for N -> l + pi0 + pi.

    Parameters
    ----------
    s: float or array-like
        Invariant mass of the pions.
    t: float or array-like
        Invariant mass of the lepton and one of the pions.
    form_factor: VectorFormFactorPiPi
        Pion electromagnetic form factor.
    """
    mx = model.mx
    ff = form_factor.form_factor(q=np.sqrt(s), couplings=(GVUU, GVDD))
    u = 0.5 * np.tan(2 * model.theta)

    return (
        u**2
        * GF**2
        * (1 - 2 * SW**2) ** 2
        * (mx**4 + 4 * MPI**4 + 4 * t * (-2 * MPI**2 + s + t) - mx**2 * (s + 4 * t))
        * np.abs(ff) ** 2
    )


def energy_distributions_v_pi_pi(
    model: SingleRhNeutrinoModel,
    form_factor: VectorFormFactorPiPi,
    nbins: int,
    method: str = "quad",
):
    def msqrd(s, t):
        return msqrd_v_pi_pi(s, t, model, form_factor)

    tb = ThreeBody(model.mx, (0.0, MPI, MPI), msqrd=msqrd)
    return tb.energy_distributions(nbins=nbins, method=method)


def invariant_mass_distributions_v_pi_pi(
    model: SingleRhNeutrinoModel,
    form_factor: VectorFormFactorPiPi,
    nbins: int,
    method: str = "quad",
):
    def msqrd(s, t):
        return msqrd_v_pi_pi(s, t, model, form_factor)

    tb = ThreeBody(model.mx, (0.0, MPI, MPI), msqrd=msqrd)
    return tb.invariant_mass_distributions(nbins=nbins, method=method)


def width_v_pi_pi(
    model: SingleRhNeutrinoModel, form_factor: VectorFormFactorPiPi, **kwargs
):
    """Partial decay width into a neutrino and two charged pions.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    form_factor: VectorFormFactorPiPi
        Pion electromagnetic form-factor.

    Returns
    -------
    width: float
        Partial width into a neutrino and two charged pions.
    """
    mx = model.mx
    if mx < 2.0 * MPI:
        return 0.0

    u = 0.5 * np.tan(2 * model.theta)

    def integrand(s):
        z = s / mx**2
        ff = form_factor.form_factor(q=np.sqrt(s), couplings=(GVUU, GVDD))
        beta = (1.0 - 4 * MPI**2 / s) ** 1.5
        poly = (1 - z) ** 2 * (1 + 2 * z)
        return np.abs(ff) ** 2 * beta * poly

    pre = u**2 * GF**2 * mx**3 * (1 - 2 * SW**2) ** 2 / (768 * np.pi**3)
    quad_kwargs = _get_quad_kwargs(**kwargs)
    return pre * integrate.quad(integrand, 4 * MPI**2, mx**2, **quad_kwargs)[0]


# ============================================================================
# ---- N -> v + l + l --------------------------------------------------------
# ============================================================================

# ---- N -> l + u + d --------------------------------------------------------
# For us, we are interested in the case where u = neutrino and d = charged
# lepton. These functions are valid when u and d have the same generation and
# when their generation is different from the RHN neutrino. Additionally, l must
# have the same generation as the RHN. Other cases are handled below.


def _msqrd_l_u_d(s, t, model: SingleRhNeutrinoModel, mu, md, ml, ckm):
    r"""Squared matrix element for N -> l + u + d, where u = (nu, up-type-quark)
    and d=(lep,down-type-quark). If u = nu and d=lep, they must have a different
    generation than l and N.
    """
    mx = model.mx
    u = 0.5 * np.tan(2 * model.theta)
    return (
        16.0
        * GF**2
        * (ml**2 + mx**2 - s - t)
        * (s + t - mu**2 - md**2)
        * u**2
        * abs(ckm) ** 2
    )


def _width_l_u_d(model: SingleRhNeutrinoModel, mu, md, ml, nw, **kwargs):
    r"""Partial width for N -> l + u + d, where u = (nu, up-type-quark)
    and d=(lep,down-type-quark). If u = nu and d=lep, they must have a different
    generation than l and N.
    """
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
            12.0
            / x
            * (x - xl**2 - xd**2)
            * (1 + xu**2 - x)
            * np.sqrt(kallen_lambda(x, xl**2, xd**2) * kallen_lambda(1, x, xu**2))
        )

    lb = (xd + xl) ** 2
    ub = (1.0 - xu) ** 2

    quad_kwargs = _get_quad_kwargs(**kwargs)
    return pre * integrate.quad(integrand, lb, ub, **quad_kwargs)[0]


# ---- N -> v + f + f --------------------------------------------------------
# For us, we are interested in the case where f = charged lepton.  These
# functions are valid when f has a generation different from the RHN neutrino.
# `v` must have the same generation as the RHN. The last case is handled below.


def _msqrd_v_f_f(s, t, model: SingleRhNeutrinoModel, mf: float):
    r"""Squared matrix element for N -> v + f + f. In the case where f is a
    charged lepton, the generation is assumed to differ from the RHN and LHN
    generations.
    """
    mx = model.mx
    u = 0.5 * np.tan(2 * model.theta)

    return (
        -2
        * u**2
        * GF**2
        * (
            2 * mf**4 * (1 - 4 * SW**2 + 8 * SW**4)
            + 2 * mf**2 * (mx**2 - s - 2 * (1 - 4 * SW**2 + 8 * SW**4) * t)
            + (1 - 4 * SW**2 + 8 * SW**4)
            * (s**2 + 2 * s * t + 2 * t**2 - mx**2 * (s + 2 * t))
        )
    )


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


# ---- N -> v + l + l --------------------------------------------------------
# Case where all particles belong to the same generation.


def _msqrd_v_l_l(s, t, model: SingleRhNeutrinoModel, ml: float):
    r"""Squared matrix element for N -> v + l + l, where all leptons have the
    same generation as the RHN.
    """
    mx = model.mx
    u = 0.5 * np.tan(2 * model.theta)
    return (
        -2
        * u**2
        * GF**2
        * (
            2 * ml**4 * (1 + 4 * SW**2 + 8 * SW**4)
            + 2 * ml**2 * (mx**2 - s - 2 * (1 + 4 * SW**2 + 8 * SW**4) * t)
            + (1 + 4 * SW**2 + 8 * SW**4)
            * (s**2 + 2 * s * t + 2 * t**2 - mx**2 * (s + 2 * t))
        )
    )


def _width_v_l_l(model: SingleRhNeutrinoModel, genl: Generation):
    """Partial decay width into a neutrino and two charged leptons."""

    sw = parameters.sin_theta_weak
    if model.gen == genl:
        c1f = 0.25 * (1 + 4 * sw**2 + 8 * sw**4)
        c2f = 0.5 * sw**2 * (2 * sw**2 + 1)
    else:
        c1f = 0.25 * (1 - 4 * sw**2 + 8 * sw**4)
        c2f = 0.5 * sw**2 * (2 * sw**2 - 1)

    ml = _lepton_masses[genl]
    return _width_v_f_f(model, ml, 1.0, c1f, c2f)


# ---- Remaining implementation using the above -------------------------------


def _make_msqrd_and_masses_v_l_l(
    model: SingleRhNeutrinoModel, genv: Generation, genl1: Generation, genl2: Generation
):
    r"""Construct the matrix element function and a tuple of the masses given the
    final state generations.
    """

    genn = model.gen
    ml1 = _lepton_masses[genl1]
    ml2 = _lepton_masses[genl2]

    if genn == genv == genl1 == genl2:
        masses = (0.0, ml1, ml1)

        def msqrd(s, t):
            return _msqrd_v_l_l(s, t, model, ml1)

    elif (genn == genv) and (genl1 == genl2):
        masses = (0.0, ml1, ml1)

        def msqrd(s, t):
            return _msqrd_v_f_f(s, t, model, ml1)

    elif (genn == genl1) and (genv == genl2):
        mu, md, ml = 0.0, ml2, ml1
        masses = (mu, md, ml)

        def msqrd(s, t):
            return _msqrd_l_u_d(s, t, model, mu, md, ml, 1.0)

    elif (genn == genl2) and (genv == genl1):
        mu, md, ml = 0.0, ml1, ml2
        masses = (mu, md, ml)

        def msqrd(s, t):
            return _msqrd_l_u_d(s, t, model, mu, md, ml, 1.0)

    else:
        masses = (0.0, ml1, ml2)

        def msqrd(s, t):
            return np.zeros_like(s)

    return msqrd, masses


def energy_distributions_v_l_l(
    model: SingleRhNeutrinoModel,
    genv: Generation,
    genl1: Generation,
    genl2: Generation,
    nbins: int,
    method: str = "quad",
):
    """Generate the energy distributions of the final states from N -> v + l + l.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    genv: Generation
        Generation of the final state left-handed neutrino.
    genl1, genl2: Generation
        Generations of the final state charged leptons.
    nbins: int
        Number of bins used to construct the distributions.
    method: str = "quad"
        Method used to integrate the squared matrix element.

    Returns
    -------
    dnde_v, dnde_l1, dnde_l2: PhaseSpaceDistribution1D
        The energy distributions of the final states.
    """
    msqrd, masses = _make_msqrd_and_masses_v_l_l(model, genv, genl1, genl2)
    tb = ThreeBody(model.mx, masses, msqrd=msqrd)
    return tb.energy_distributions(nbins=nbins, method=method)


def invariant_mass_distributions_v_l_l(
    model: SingleRhNeutrinoModel,
    genv: Generation,
    genl1: Generation,
    genl2: Generation,
    nbins: int,
    method: str = "quad",
):
    r"""Generate the invariant-mass distributions of the final states
    from N -> v + l + l.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    genv: Generation
        Generation of the final state left-handed neutrino.
    genl1, genl2: Generation
        Generations of the final state charged leptons.
    nbins: int
        Number of bins used to construct the distributions.
    method: str = "quad"
        Method used to integrate the squared matrix element.

    Returns
    -------
    dists: Dict[Tuple[int,int], PhaseSpaceDistribution1D]
        Dictionary containing the invariant-mass distributions of each pair of
        final state particles.
    """
    msqrd, masses = _make_msqrd_and_masses_v_l_l(model, genv, genl1, genl2)
    tb = ThreeBody(model.mx, masses, msqrd=msqrd)
    return tb.invariant_mass_distributions(nbins=nbins, method=method)


def width_v_l_l(
    model: SingleRhNeutrinoModel, genv: Generation, genl1: Generation, genl2: Generation
):
    r"""Partial decay width into a neutrino and two charged leptons.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    genv: Generation
        Generation of the final state left-handed neutrino.
    genl1, genl2: Generation
        Generations of the final state charged leptons.

    Returns
    -------
    pw: float
        Partial width for N -> v + l + l.
    """
    genn = model.gen

    if genn == genv:
        return _width_v_l_l(model, genl1)

    mu = 0.0
    md = _lepton_masses[genl2]
    ml = _lepton_masses[genl1]

    if (genn == genl1) and (genv == genl2):
        return _width_l_u_d(model, mu, md, ml, 1.0)

    if (genn == genl2) and (genv == genl1):
        md, ml = ml, md
        return _width_l_u_d(model, mu, md, ml, 1.0)

    return 0.0


# ============================================================================
# ---- N -> nu + nu + nu -----------------------------------------------------
# ============================================================================


def msqrd_v_v_v(s, t, model: SingleRhNeutrinoModel, gen: Generation):
    mx = model.mx
    u = 0.5 * np.tan(2 * model.theta)
    pre = 2.0 if gen == model.gen else 1.0
    return -pre * 16 * GF**2 * (s**2 + s * t + t**2 - mx**2 * (s + t)) * u


def energy_distributions_v_v_v(
    model: SingleRhNeutrinoModel,
    genv: Generation,
    nbins: int,
):
    r"""Generate the energy distributions of the final states
    from N -> v + v + v.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    genv: Generation
        Generations of the final state neutrinos 2 and 3. Generation of the
        first neutrino is assumed to be equal to the generation of the RHN.
    nbins: int
        Number of bins used to construct the distributions.

    Returns
    -------
    dists: Dict[Tuple[int,int], PhaseSpaceDistribution1D]
        Dictionary containing the invariant-mass distributions of each pair of
        final state particles.
    """

    def msqrd(s, t):
        return msqrd_v_v_v(s, t, model, genv)

    tb = ThreeBody(model.mx, (0, 0, 0), msqrd=msqrd)
    return tb.energy_distributions(nbins=nbins)


def invariant_mass_distributions_v_v_v(
    model: SingleRhNeutrinoModel,
    genv: Generation,
    nbins: int,
):
    r"""Generate the invariant-mass distributions of the final states
    from N -> v + l + l.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    genv: Generation
        Generations of the final state neutrinos 2 and 3. Generation of the
        first neutrino is assumed to be equal to the generation of the RHN.
    nbins: int
        Number of bins used to construct the distributions.

    Returns
    -------
    dists: Dict[Tuple[int,int], PhaseSpaceDistribution1D]
        Dictionary containing the invariant-mass distributions of each pair of
        final state particles.
    """

    def msqrd(s, t):
        return msqrd_v_v_v(s, t, model, genv)

    tb = ThreeBody(model.mx, (0, 0, 0), msqrd=msqrd)
    return tb.invariant_mass_distributions(nbins=nbins)


def width_v_v_v(model: SingleRhNeutrinoModel, genv: Generation):
    r"""Compute the partial width for N -> v + v + v.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.
    genv: Generation
        Generations of the final state neutrinos 2 and 3. Generation of the
        first neutrino is assumed to be equal to the generation of the RHN.

    Returns
    -------
    pw: float
        The partial width for N -> v + v + v.
    """
    mx = model.mx
    u = 0.5 * np.tan(2 * model.theta)
    w = parameters.GF**2 * mx**5 / (768 * np.pi**3) * u**2
    pre = 2 if genv == model.gen else 1.0
    return pre * w


# ============================================================================
# ---- N -> v + a ------------------------------------------------------------
# ============================================================================


def width_v_a(model: SingleRhNeutrinoModel) -> float:
    r"""Compute the partial width for the decay of a RHN into a left-handed
    neutrin and a photon N -> v + a.

    Parameters
    ----------
    model: SingleRhNeutrinoModel
        Object containing the model parameters. Should implement the
        `SingleRhNeutrinoModel` protocol.

    Returns
    -------
    pw: float
        The partial width for N -> v + a.
    """
    u = 0.5 * np.tan(2 * model.theta)
    return 9 * ALPHA_EM * GF**2 / (256 * np.pi**4) * model.mx**5 * u**2
