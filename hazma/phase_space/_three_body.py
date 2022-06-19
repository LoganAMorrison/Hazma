from typing import Callable, Any, Tuple, Dict, List, Union, overload

import numpy as np
from scipy import integrate

from hazma.utils import kallen_lambda, lnorm_sqr, RealArray

from ._rambo import Rambo
from ._utils import energy_limits, invariant_mass_limits
from ._dist import PhaseSpaceDistribution1D

ThreeBodyMsqrd = Callable[[Any, Any], Any]
ThreeBodyMasses = Tuple[float, float, float]
InvariantMassDists = Dict[Tuple[int, int], PhaseSpaceDistribution1D]


class _integrator:
    methods = {"quad", "trapz", "simps"}

    @staticmethod
    def _trapz(fn, a: float, b: float, npts: int):
        xs = np.linspace(a, b, npts)
        fs = np.array([fn(x) for x in xs])
        return integrate.trapz(fs, xs)

    @staticmethod
    def _simps(fn, a: float, b: float, npts: int):
        xs = np.linspace(a, b, npts)
        fs = np.array([fn(x) for x in xs])
        return integrate.simps(fs, xs)

    @staticmethod
    def _quad(fn, a: float, b: float):
        return integrate.quad(fn, a, b)[0]

    @staticmethod
    def integrate(fn, a: float, b: float, method: str = "quad", npts: int = 100):
        if method == "quad":
            return _integrator._quad(fn, a, b)
        elif method == "trapz":
            return _integrator._trapz(fn, a, b, npts)
        elif method == "simps":
            return _integrator._simps(fn, a, b, npts)
        else:
            raise ValueError(
                f"Invalid method {method}. Use 'quad', 'trapz' or 'simps'."
            )


def three_body_integration_bounds_t(s, m: float, m1: float, m2: float, m3: float):
    """Compute the integrate bounds on the 'Mandelstam' variable t = (p1 + p3)^2
    for a three-body final state.

    Parameters
    ----------
    s: float
        Invariant mass of particles 2 and 3.
    m: float
        Center-of-mass energy.
    m1, m2, m3: float
        Masses of the three final state particles.

    Returns
    -------
    lb: float
        Lower integration bound.
    ub: float
        Upper integration bound.
    """
    msqr = m**2
    m12 = m1**2
    m22 = m2**2
    m32 = m3**2

    pfac = -(msqr - m12) * (m22 - m32) + (msqr + m12 + m22 + m32) * s - s**2
    sfac = np.sqrt(kallen_lambda(s, m12, m22) * kallen_lambda(s, m22, m32))

    lb = 0.5 * (pfac - sfac) / s
    ub = 0.5 * (pfac + sfac) / s

    return lb, ub


def three_body_integration_bounds_s(m: float, m1: float, m2: float, m3: float):
    """Compute the integrate bounds on the 'Mandelstam' variable s = (p2 + p3)^2
    for a three-body final state.

    Parameters
    ----------
    m: float
        Center-of-mass energy.
    m1, m2, m3: float
        Masses of the three final state particles.

    Returns
    -------
    lb: float
        Lower integration bound.
    ub: float
        Upper integration bound.
    """
    lb = (m2 + m3) ** 2
    ub = (m - m1) ** 2
    return lb, ub


def three_body_integration_bounds_y(x, mu1: float, mu2: float, mu3: float):
    """Compute the integrate bounds on the scaled energy of particle 2: y = 2*E2/cme.

    Parameters
    ----------
    x: float
        Scaled energy of particle 1: x = 2*E1 / cme.
    mu1, mu2, mu3: float
        Scaled masses of the three final state particles, mu1=m1/cme,
        mu2=m2/cme, mu3=m3/cme.

    Returns
    -------
    lb: float
        Lower integration bound.
    ub: float
        Upper integration bound.
    """
    pfac = (2 - x) * (1 - x + mu1**2 + mu2**2 - mu3**2)
    sfac = np.sqrt(
        kallen_lambda(1, mu1**2, 1 - x + mu1**2)
        * kallen_lambda(1 - x + mu1**2, mu2**2, mu3**2)
    )

    lb = 0.5 * (pfac - sfac) / (1 - x + mu1**2)
    ub = 0.5 * (pfac + sfac) / (1 - x + mu1**2)
    return lb, ub


def three_body_integration_bounds_x(mu1: float, mu2: float, mu3: float):
    r"""Compute the integrate bounds on the scaled energy of particle 1: x = 2*E1/cme.

    Parameters
    ----------
    m1, m2, m3: float
        Masses of the three final state particles.

    Returns
    -------
    lb: float
        Lower integration bound.
    ub: float
        Upper integration bound.
    """
    lb = 2 * mu1
    ub = mu1**2 + (1 - (mu2 + mu3) ** 2)
    return lb, ub


def three_body_integral_prefactor(q: float):
    """Compute the phase-space integral pre-factor for a three-body final
    state.

    Parameters
    ----------
    q: float
        Center-of-mass energy.

    Returns
    -------
    pf: float
        Pre-factor.
    """
    return 1.0 / (128.0 * np.pi**3 * q**2)


def _partial_integration(
    msqrd: ThreeBodyMsqrd,
    q: float,
    e,
    masses: Tuple[float, float, float],
    i: int,
    method: str = "quad",
    npts: int = 100,
):
    r"""Compute the integral of the squared matrix element over the scaled
    energy of particle 2: y = 2 * E2 / q.

    Parameters
    ----------
    msqrd: callable
        Binary function taking in the squared invariant masses s=(p2+p3)^2, and
        t=(p1+p3)^2.
    q: float
        Center-of-mass energy.
    e: float or array-like
        Energy of particle 1.
    masses: (float, float, float)
        Masses of the final state particles.
    i: int
        Final state particle to integrate over.
    method: str
        Method used to integrate. Can be 'trapz', 'simps' or 'quad'. Default is
        'quad'.

    Returns
    -------
    dI/de: float
        Differential phase-space density with respect to the energy of particle 1.
    """
    assert 0 <= i <= 2, f"Invalid argument i = {i}. Must be 0,1 or 2."

    m1, m2, m3 = masses
    pre = q / (64.0 * np.pi**3)

    def msqrdxy(x, y):
        s = q**2 * (1.0 - x + (masses[0] / q) ** 2)
        t = q**2 * (1.0 - y + (masses[1] / q) ** 2)
        return msqrd(s, t)

    if i == 0:
        _m1, _m2, _m3 = m1, m2, m3

        def integrand(x, y):
            return msqrdxy(x, y)

    elif i == 1:
        # swap 1 and 2
        # x -> y, y -> x, z -> z
        _m1, _m2, _m3 = m2, m1, m3

        def integrand(x, y):
            return msqrdxy(y, x)

    elif i == 2:
        # swap 1 and 3
        # x -> z, y -> y, z -> z
        _m1, _m2, _m3 = m3, m2, m1

        def integrand(x, y):
            return msqrdxy(2 - (x + y), y)

    else:
        raise ValueError(f"Invalid argument i = {i}. Must be 0,1 or 2.")

    _mu1, _mu2, _mu3 = _m1 / q, _m2 / q, _m3 / q

    xmin, xmax = three_body_integration_bounds_x(_mu1, _mu2, _mu3)

    def bounds(x):
        return three_body_integration_bounds_y(x, _mu1, _mu2, _mu3)

    def integral(x):
        if x < xmin or xmax < x:
            return 0.0
        lb, ub = bounds(x)
        return pre * _integrator.integrate(
            lambda y: integrand(x, y), lb, ub, method=method, npts=npts
        )

    single = np.isscalar(e)
    es = np.atleast_1d(e)
    res = np.array([integral(2 * e / q) for e in es])

    if single:
        return res[0]
    return res


def _integrate_three_body_quad(
    msqrd: ThreeBodyMsqrd, q: float, masses: Tuple[float, float, float]
) -> Tuple[float, float]:
    r"""Compute the integral of the squared matrix element over three-body phase
    space.

    Parameters
    ----------
    msqrd: callable
        Binary function taking in the squared invariant masses s=(p2+p3)^2, and
        t=(p1+p3)^2.
    q: float
        Center-of-mass energy.
    masses: (float, float, float)
        Masses of the final state particles.

    Returns
    -------
    I: float
        Integral of the squared matrix element.
    """
    m1, m2, m3 = masses
    mu1, mu2, mu3 = m1 / q, m2 / q, m3 / q
    pre = q**2 / (128.0 * np.pi**3)

    def ymin(x):
        return three_body_integration_bounds_y(x, mu1, mu2, mu3)[0]

    def ymax(x):
        return three_body_integration_bounds_y(x, mu1, mu2, mu3)[1]

    def integrand(y, x):
        s = q**2 * (1.0 - x + mu1**2)
        t = q**2 * (1.0 - y + mu2**2)
        return msqrd(s, t)

    xmin, xmax = three_body_integration_bounds_x(mu1, mu2, mu3)
    res = integrate.dblquad(integrand, xmin, xmax, ymin, ymax)
    return pre * res[0], pre * res[1]


def _make_msqrd_rambo(msqrd):
    def msqrd_(ps):
        s = lnorm_sqr(ps[:, 1] + ps[:, 2])
        t = lnorm_sqr(ps[:, 0] + ps[:, 2])
        return msqrd(s, t)

    return msqrd_


def _integrate_three_body_rambo(
    msqrd: ThreeBodyMsqrd,
    q: float,
    masses: Tuple[float, float, float],
    npts: int = 10000,
) -> Tuple[float, float]:
    """Compute the integral of the squared matrix element over the squared
    invariant mass of particles 2 and 3.

    Parameters
    ----------
    msqrd: callable
        Binary function taking in the squared invariant masses s=(p2+p3)^2, and
        t=(p1+p3)^2.
    q: float
        Center-of-mass energy.
    masses: (float, float, float)
        Masses of the final state particles.
    """
    msqrd_ = _make_msqrd_rambo(msqrd)
    phase_space = Rambo(cme=q, masses=np.array(masses), msqrd=msqrd_)
    return phase_space.integrate(n=npts)


def integrate_three_body(
    msqrd: ThreeBodyMsqrd,
    q: float,
    masses: Tuple[float, float, float],
    method: str = "quad",
    npts: int = 10000,
) -> Tuple[float, float]:
    """Compute the integral of the squared matrix element over the squared
    invariant mass of particles 2 and 3.

    Parameters
    ----------
    msqrd: callable
        Two-argument function taking in the squared invariant masses
        s=(p2+p3)^2, and t=(p1+p3)^2.
    q: float
        Center-of-mass energy.
    masses: (float, float, float)
        Masses of the final state particles.
    method: str
        Method used to integrate over phase space. Can be:
            'quad': Numerical quadrature using scipy's 'dblquad',
            'rambo': Monte-Carlo integration.
    npts: int
        Number of phase-space points to use to integrate squared matrix
        element. Ignored unless method is 'rambo'. Default is 10000.

    Returns
    -------
    integral: float
        Integral of the squared matrix element.
    error: float
        Error estimate.
    """
    methods = {
        "quad": lambda *args: _integrate_three_body_quad(*args),
        "rambo": lambda *args: _integrate_three_body_rambo(*args, npts=npts),
    }

    if q < sum(masses):
        return (0.0, 0.0)

    meth = methods.get(method)
    if meth is None:
        ms = ",".join(methods.keys())
        raise ValueError(f"Invalid method: {method}. Use one of the following: {ms}.")

    return meth(msqrd, q, masses)


def _energy_distributions_quad(
    *,
    msqrd: ThreeBodyMsqrd,
    q: float,
    masses: ThreeBodyMasses,
    nbins: int,
):
    """Compute energy distributions using numerical integration."""

    def integral(e, i):
        return _partial_integration(msqrd=msqrd, q=q, e=e, masses=masses, i=i)

    elims = energy_limits(q, masses)

    dists = []
    for i, (emin, emax) in enumerate(elims):
        ebins = np.linspace(emin, emax, nbins)
        es = 0.5 * (ebins[1:] + ebins[:-1])

        dwde = integral(es, i)
        dists.append(PhaseSpaceDistribution1D(ebins, dwde))

    return dists


def _energy_distributions_rambo(
    *,
    msqrd: ThreeBodyMsqrd,
    q: float,
    masses: ThreeBodyMasses,
    nbins: int,
    npts: int = 10000,
):
    """Compute energy distributions using Monte-Carlo integration."""
    msqrd_ = _make_msqrd_rambo(msqrd)
    phase_space = Rambo(cme=q, masses=np.array(masses), msqrd=msqrd_)
    return phase_space.energy_distributions(n=npts, nbins=nbins)


def energy_distributions_three_body(
    msqrd: ThreeBodyMsqrd,
    q: float,
    masses: ThreeBodyMasses,
    nbins: int,
    method: str = "quad",
    npts: int = 10000,
) -> List[PhaseSpaceDistribution1D]:
    """Compute the energy distributions of the three final-state particles.

    Parameters
    ----------
    msqrd: callable
        Binary function taking in the squared invariant masses s=(p2+p3)^2, and
        t=(p1+p3)^2.
    q: float
        Center-of-mass energy.
    masses: (float, float, float)
        Masses of the final state particles.
    method: str
        Method used to integrate over phase space. Can be:
            'quad': Numerical quadrature using scipy's 'quad',
            'rambo': Monte-Carlo integration.
    npts: int
        Number of phase-space points to use to integrate squared matrix
        element. Ignored unless method is 'rambo'. Default is 10000.

    Returns
    -------
    dist1, dist2, dist3: (array, array)
        Distributions of the final-state particles.
    """
    methods = {
        "quad": lambda **kwargs: _energy_distributions_quad(**kwargs),
        "rambo": lambda **kwargs: _energy_distributions_rambo(**kwargs, npts=npts),
    }

    meth = methods.get(method)
    if meth is None:
        ms = ",".join(methods.keys())
        raise ValueError(f"Invalid method: {method}. Use one of the following: {ms}.")

    return meth(msqrd=msqrd, q=q, masses=masses, nbins=nbins)


def _invariant_mass_distributions_quad(
    *, msqrd: ThreeBodyMsqrd, q: float, masses: ThreeBodyMasses, nbins: int
) -> InvariantMassDists:
    """Compute invariant-mass distributions using numerical integration."""

    def integral(e, i):
        return _partial_integration(msqrd, q, e, masses, i)

    invmass_lims = invariant_mass_limits(q, masses)
    dists: Dict[Tuple[int, int], PhaseSpaceDistribution1D] = dict()

    for i, ((j, k), (mmin, mmax)) in enumerate(invmass_lims.items()):
        m1 = masses[i]

        mbins = np.linspace(mmin, mmax, nbins)
        ms = 0.5 * (mbins[1:] + mbins[:-1])
        es = (q**2 - ms**2 + m1**2) / (2 * q)

        dwdm = ms / q * integral(es, i)
        dists[(j, k)] = PhaseSpaceDistribution1D(mbins, dwdm)

    return dists


def _invariant_mass_distributions_rambo(
    *,
    msqrd: ThreeBodyMsqrd,
    q: float,
    masses: ThreeBodyMasses,
    nbins: int,
    npts: int = 10000,
) -> InvariantMassDists:
    """Compute invariant-mass distributions using Monte-Carlo integration."""
    msqrd_ = _make_msqrd_rambo(msqrd)
    phase_space = Rambo(cme=q, masses=np.array(masses), msqrd=msqrd_)
    return phase_space.invariant_mass_distributions(n=npts, nbins=nbins)


def invariant_mass_distributions_three_body(
    msqrd: ThreeBodyMsqrd,
    q: float,
    masses: ThreeBodyMasses,
    nbins: int,
    method: str = "quad",
    npts: int = 10000,
) -> InvariantMassDists:
    r"""Compute the invariant-mass distributions of the three pairs of
    final-state particles.

    Parameters
    ----------
    msqrd: callable
        Binary function taking in the squared invariant masses s=(p2+p3)^2, and
        t=(p1+p3)^2.
    q: float
        Center-of-mass energy.
    masses: (float, float, float)
        Masses of the final state particles.
    method: str
        Method used to integrate over phase space. Can be:
            'quad': Numerical quadrature using scipy's 'quad',
            'trapz': Trapiziod rule using scipy's 'trapz',
            'simps': Simpson's rule using scipy's 'simps',
            'rambo': Monte-Carlo integration.
    npts: int
        Number of phase-space points to use to integrate squared matrix
        element. Ignored unless method is 'rambo'. Default is 10000.

    Returns
    -------
    dist1, dist2, dist3: (array, array)
        Distributions of the final-state particles.

    Raises
    ------
    ValueError
        If the final-state particles masses are larger than center-of-mass
        energy.
    """
    methods = {
        "quad": lambda **kwargs: _invariant_mass_distributions_quad(**kwargs),
        "rambo": lambda **kwargs: _invariant_mass_distributions_rambo(
            **kwargs, npts=npts
        ),
    }
    if q < sum(masses):
        raise ValueError(
            "Center of mass energy is less than the sum of final-state particle masses."
        )

    meth = methods.get(method)
    if meth is None:
        ms = ",".join(methods.keys())
        raise ValueError(f"Invalid method: {method}. Use one of the following: {ms}.")

    return meth(msqrd=msqrd, q=q, masses=masses, nbins=nbins)
