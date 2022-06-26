"""
==================================
Boost (:mod:`hazma.spectra.boost`)
==================================

.. versionadded:: 2.0

Utilities for boosting differential energy spectra.
"""

from typing import Callable, Tuple
import inspect
import warnings

import numpy as np
from scipy import interpolate
from scipy import integrate

from hazma.utils import RealArray

# =============================================================================
# ---- Integrators ------------------------------------------------------------
# =============================================================================


# Default parameters for scipy integration methods
_integrate_defaults = {
    "quad": {
        key: p.default
        for key, p in inspect.signature(integrate.quad).parameters.items()
        if not p.default == p.empty
    },
    "quadrature": {
        key: p.default
        for key, p in inspect.signature(integrate.quadrature).parameters.items()
        if not p.default == p.empty
    },
    "trapz": {
        key: p.default
        for key, p in inspect.signature(integrate.trapezoid).parameters.items()
        if not p.default == p.empty
    },
    "simps": {
        key: p.default
        for key, p in inspect.signature(integrate.simpson).parameters.items()
        if not p.default == p.empty
    },
}


def _get_integrator_kwargs(method: str, **kwargs):
    meth_kwargs = _integrate_defaults[method]
    return {key: kwargs.get(key, default) for key, default in meth_kwargs.items()}


def _make_integrator(method, vectorized: bool, **kwargs):
    """Return a quadrature integrator."""
    integrate_kwargs = _get_integrator_kwargs(method, **kwargs)

    is_quad = method == "quad"
    is_quadrature = method == "quadrature"

    if is_quad or is_quadrature:
        integrator_ = integrate.quad if is_quad else integrate.quadrature

        def quad(integrand, emin, emax):
            return np.array(
                [
                    integrator_(integrand, a, b, **integrate_kwargs)[0]
                    for a, b in zip(emin, emax)
                ]
            )

        return quad

    is_trapz = method == "trapz"
    is_simps = method == "simps"

    if is_trapz or is_simps:
        del integrate_kwargs["x"]
        integrate_kwargs["axis"] = 1
        npts = kwargs.get("npts", 100)
        integrator_ = integrate.trapz if is_trapz else integrate.simps

        def fixed(f, a, b):
            es = np.array([np.linspace(a_, b_, npts) for a_, b_ in zip(a, b)])
            ff = f if vectorized else np.vectorize(f)
            integrands = np.array([ff(e) for e in es])
            return integrator_(integrands, es, **integrate_kwargs)

        return fixed

    raise ValueError(f"Invalid method {method}.")


def boost_delta_function(product_energy, e0: float, m: float, beta: float):
    """
    Boost a delta function of the form δ(e - e0) of a product of mass `m`
    with a boost parameter `beta`.

    Parameters
    ----------
    product_energy: float or array-like
        Energy of the product in the lab frame.
    e0: float
        Center of the dirac-delta spectrum in rest-frame
    m: float
        Mass of the product
    beta: float
        Boost velocity of the decaying particle
    """
    scalar = np.isscalar(product_energy)
    e = np.atleast_1d(product_energy)
    dnde = np.zeros_like(e)

    if 0.0 < beta < 1.0:
        gamma = 1.0 / np.sqrt(1.0 - beta * beta)

        mask = e >= m
        k = np.sqrt(e**2 - m**2)

        eminus = gamma * (e - beta * k)
        eplus = gamma * (e + beta * k)

        # - b * k0 < (e/g) - e0 < b * k0
        mask = mask & ((eminus < e0) & (e0 < eplus))
        dnde[mask] = 1.0 / (2.0 * gamma * beta * np.sqrt(e0 * e0 - m * m))

    if scalar:
        return dnde[0]
    return dnde


def double_boost_delta_function(
    product_energy, e0: float, m: float, beta1: float, beta2: float
):
    """
    Perform a double-boost of a delta function of the form δ(e - e0) of a
    product.

    Parameters
    ----------
    energies: float or array-like
        Energy of the product in the lab frame.
    e0: double
        Center of the dirac-δ spectrum in original rest-frame.
    m: double
        Mass of the product.
    beta1, beta2: double
        1st and 2nd boost velocities of the decaying particle.
    """
    scalar = np.isscalar(product_energy)
    e2 = np.atleast_1d(product_energy)
    dnde = np.zeros_like(e2)

    gamma1 = 1.0 / np.sqrt(1.0 - beta1**2)
    gamma2 = 1.0 / np.sqrt(1.0 - beta2**2)

    eps_m = gamma1 * (e0 - beta1 * np.sqrt(e0**2 - m**2))
    eps_p = gamma1 * (e0 + beta1 * np.sqrt(e0**2 - m**2))
    e_m = gamma2 * (e2 - beta2 * np.sqrt(e2**2 - m**2))
    e_p = gamma2 * (e2 + beta2 * np.sqrt(e2**2 - m**2))

    mask = np.logical_and(e_p > eps_m, e_m < eps_p)
    b = np.minimum(eps_p, e_p)[mask]
    a = np.maximum(eps_m, e_m)[mask]

    if m > 0.0:
        num = (a - np.sqrt(a**2 - m**2)) * (b + np.sqrt(b**2 - m**2))
        den = (a + np.sqrt(a**2 - m**2)) * (b - np.sqrt(b**2 - m**2))
        pre = 0.5
    else:
        num = b
        den = a
        pre = 1.0

    dnde[mask] = pre * np.log(num / den)
    dnde /= 4.0 * gamma1 * gamma2 * beta1 * beta2 * e0

    if scalar:
        return dnde[0]
    return dnde


def dnde_boost_array(dnde, energies, beta: float, mass: float = 0.0):
    """Boost a spectrum dN/dE given as a numeric array.

    Parameters
    ----------
    dnde: array
        Spectrum to boost.
    energies: array
        Energies corresponding to `dnde`.
    beta: float
        Boost velocity. If `beta` is outside [0,1), zeros are returned.
    mass: float
        Mass of the product of the spectrum (i.e. 0 for photon, electron-mass
        for positron).

    Notes
    -----
    The boosted spectrum is computed by creating a linear interpolating
    function from the data and using it to compute the integral.

    Returns
    -------
    dnde_boosted: array
        The boosted spectrum.
    """
    if beta < np.finfo(float).eps:
        return dnde

    if beta < 0 or beta > 1.0:
        return np.zeros_like(energies)

    gamma = 1.0 / np.sqrt(1 - beta**2)

    boosted = np.zeros_like(energies)
    mask = energies > mass
    es = energies[mask]
    k = np.sqrt(es**2 - mass**2)
    emax = gamma * (es + beta * k)
    emin = gamma * (es - beta * k)

    integrand = dnde[mask] / k
    pre = 1.0 / (2 * beta * gamma)

    spline = interpolate.InterpolatedUnivariateSpline(es, integrand, ext=1, k=1)
    boosted[mask] = np.array([spline.integral(a, b) for a, b in zip(emin, emax)])

    return pre * boosted


def make_boost_function(fn: Callable, mass: float, vectorized: bool = True):
    """Create a function to compute boost spectrum.

    Parameters
    ----------
    fn: Callable
        Spectrum function to transform.
    mass: float
        Mass of the product.
    vectorized: bool, optional
       If True, `fn` is assumed to be vectorized. Default is True.

    Notes
    -----
    The boosted spectrum is computed by creating a linear interpolating
    function from the data and using it to compute the integral.

    Returns
    -------
    boosted_fn: Callable
        The boosted function has the signature:

        .. code-block::

            def boosted(energies, beta: float, method="quad", **kwargs):
                ...

        where:

            * `energies`: Array of energies where boosted spectrum should be evaluated,
            * `beta`: Boost velocity. If outside [0, 1), zeros are returned,
            * `mass`: Mass of the product,
            * `method`: Method used to integrate. Can be one of the following:

                * 'quad': for adaptive quadrature using `scipy.integrate.quad`,
                * 'quadrature': for adaptive quadrature using `scipy.integrate.quadrature`,
                * 'trapz': trapizoid rule using `scipy.integrate.trapz`,
                * 'simps': Simpson's rule using `scipy.integrate.simps`.

            * `kwargs`: Keyword arguments to pass to underlying method. See
              SciPy's quad available arguments. For `trapizoid` or `simpson`,
              the number of points to sample from integrand can be specified
              through `npts`.
    """
    methods = ["quad", "trapz", "simps", "quadrature"]

    def kinematic_early_return(energies: RealArray, beta: float):
        # The function will return something if we should return early due to
        # kinematic constraints. The constraints are:
        #   1. beta ~ 0: In rest frame => no boost
        #   2. beta > 1 or beta < 0: Unphysical.

        if beta < np.finfo(energies.dtype).eps:
            return fn(energies)

        if beta < 0.0 or beta > 1.0:
            warnings.warn(f"Unphysical beta = {beta}. Returning zero.")
            return np.zeros_like(energies)

        return None

    def bounds_and_mask(energies: RealArray, gamma: float, beta: float):
        mask = energies > mass
        es = energies[mask]
        k = np.sqrt(es**2 - mass**2)
        emin = gamma * (es - beta * k)
        emax = gamma * (es + beta * k)

        return emin, emax, mask

    def check_method(method: str):
        if method not in methods:
            raise ValueError(f"Invalid method {method}. Use of one {methods}.")

    def make_integrand(*args):
        def integrand(e):
            return fn(e, *args) / np.sqrt(e**2 - mass**2)

        return integrand

    def fn_boosted(
        energies,
        beta: float,
        method: str = "quadrature",
        args: Tuple = tuple(),
        **kwargs,
    ):
        """Compute the boosted spectrum.

        Parameters
        ----------
        energies: array
            Array of energies where spectrum should be evaluated.
        beta: float
            Boost velocity. If beta < 0 or beta > 1, zeros are returned.
        method: str, optional
            Method to use to integrate. Can be 'quad', 'trapizoid' or
            'simpson'. Default is 'quadrature'.
        args: tuple, optional
            Additional arguments to pass to function.

        Returns
        -------
        boosted: array
            The boosted spectrum evaluated at the input energies.

        Other Parameters
        ----------------
        If `method` = 'trapz' or 'simps', the following keyword arguments
        are available:

        npts: int, optional
            Number of points to use in trapizoidal or simpson integration.
            Default is 100.

        If method = 'quad' or 'quadrature, any keyword arguments compatible with
        quad/quadrature can be specified.
        """
        check_method(method)
        early = kinematic_early_return(energies, beta)
        if early is not None:
            return early

        gamma = 1.0 / np.sqrt(1.0 - beta**2)
        boosted = np.zeros_like(energies)
        emin, emax, mask = bounds_and_mask(energies=energies, gamma=gamma, beta=beta)

        integrand = make_integrand(*args)
        integrator = _make_integrator(method, vectorized, **kwargs)

        boosted[mask] = integrator(integrand, emin, emax)
        return 0.5 * boosted / (beta * gamma)

    return fn_boosted


def dnde_boost(
    fn: Callable,
    energies,
    beta: float,
    mass: float = 0.0,
    method: str = "quadrature",
    vectorized: bool = True,
    args: Tuple = tuple(),
    **kwargs,
):
    """Boost a spectrum dN/dE given as a function.

    Parameters
    ----------
    energies: array
        Array of energies where spectrum should be evaluated.
    beta: float
        Boost velocity. If beta < 0 or beta > 1, zeros are returned.
    mass: float, optional
        Mass of the product. Default is zero (i.e. for a photon.)
    method: str, optional
        Method to use to integrate. Can be 'quad', 'trapizoid' or
        'simpson'. Default is 'quadrature'.
    vectorized: bool, optional
       If True, `fn` is assumed to be vectorized. Default is True.
    args: tuple, optional
        Additional arguments to pass to function.

    Notes
    -----
    The boosted spectrum is computed by creating a linear interpolating
    function from the data and using it to compute the integral.

    Returns
    -------
    dnde_boosted: array
        The boosted spectrum.
    """
    boosted = make_boost_function(fn, mass=mass, vectorized=vectorized)
    return boosted(energies, beta, mass=mass, method=method, args=args, **kwargs)
