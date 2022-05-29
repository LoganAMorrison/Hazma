from typing import Callable, Optional, Tuple
from inspect import signature
import warnings

import numpy as np
from scipy import interpolate
from scipy import integrate


def boost_delta_function(e, e0: float, m: float, beta: float):
    """
    Boost a delta function of the form δ(e - e0) of a product of mass `m`
    with a boost parameter `beta`.

    Parameters
    ----------
    e: double
        Energy of the product in the lab frame.
    e0: double
        Center of the dirac-delta spectrum in rest-frame
    m: double
        Mass of the product
    beta: double
        Boost velocity of the decaying particle
    """
    dnde = np.zeros_like(e)

    if beta > 1.0 or beta <= 0.0:
        return dnde

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    k = np.sqrt(e**2 - m**2)
    eminus = gamma * (e - beta * k)
    eplus = gamma * (e + beta * k)

    # - b * k0 < (e/g) - e0 < b * k0
    mask = np.logical_and(eminus < e0, e0 < eplus)
    dnde[mask] = 1.0 / (2.0 * gamma * beta * np.sqrt(e0 * e0 - m * m))

    return dnde


def double_boost_delta_function(e2, e0: float, m: float, beta1: float, beta2: float):
    """
    Perform a double-boost of a delta function of the form δ(e - e0) of a
    product.

    Parameters
    ----------
    e: double
        Energy of the product in the lab frame.
    e0: double
        Center of the dirac-δ spectrum in original rest-frame.
    m: double
        Mass of the product.
    beta1, beta2: double
        1st and 2nd boost velocities of the decaying particle.
    """
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

    res = np.zeros_like(e2)
    res[mask] = pre * np.log(num / den)
    res = res / (4.0 * gamma1 * gamma2 * beta1 * beta2 * e0)

    return res


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
    mask = energies < mass
    es = energies[mask]
    k = np.sqrt(es**2 - mass**2)
    emax = gamma * (es + beta * k)
    emin = gamma * (es - beta * k)

    integrand = dnde[mask] / k
    pre = 0.5 / (2 * beta * gamma)

    spline = interpolate.UnivariateSpline(es, integrand, ext=1, k=1)
    boosted[mask] = np.array([spline.integral(a, b) for a, b in zip(emin, emax)])

    return pre * boosted


def make_boost_function(fn: Callable):
    """Create a function to compute boost spectrum.

    Parameters
    ----------
    fn: Callable
        Spectrum function to transform.

    Notes
    -----
    The boosted spectrum is computed by creating a linear interpolating
    function from the data and using it to compute the integral.

    Returns
    -------
    boosted_fn: Callable
        The boosted function has the signature

            (energies, beta,mass=0,method="quad"),

        where:
            - `energies`: Array of energies where boosted spectrum should be evaluated,
            - `beta`: Boost velocity. If outside [0, 1), zeros are returned,
            - `mass`: Mass of the product,
            - `method`: Method used to integrate. Can be 'quad' for adaptive
              quadrature, 'trapz' for trapizoidal rule or 'simpz' for Simpson's
              rule.
            - `kwargs`: Keyword arguments to pass to underlying method. See
              SciPy's quad available arguments. For `trapizoid` or `simpson`,
              the number of points to sample from integrand can be specified
              through `npts`.

        That is, the returned function looks like:

            ```python
            def boosted(energies, beta: float, mass: float = 0.0):
                ...
            ```
    """
    methods = ["quad", "trapizoid", "simpson"]

    def bounds_and_mask(energies, gamma: float, beta: float, mass: float):
        mask = energies > mass
        es = energies[mask]
        k = np.sqrt(es**2 - mass**2)
        emin = gamma * (es - beta * k)
        emax = gamma * (es + beta * k)

        return emin, emax, mask

    def kinematic_early_return(energies, beta: float):
        if beta < np.finfo(float).eps:
            return fn(energies)

        if beta < 0.0 or beta > 1.0:
            warnings.warn(f"Unphysical beta = {beta}. Returning zero.")
            return np.zeros_like(energies)

        return None

    def check_method(method: str):
        if method not in methods:
            raise ValueError(f"Invalid method {method}. Use of one {methods}.")

    def get_quad_kwargs(**kwargs):
        positional = ["func", "a", "b"]

        def use(key):
            return key not in positional and kwargs.get(key) is not None

        return {
            key: kwargs.get(key, val)
            for key, val in signature(integrate.quad).parameters.items()
            if use(key)
        }

    def fn_boosted(
        energies,
        beta: float,
        mass: float = 0.0,
        method: str = "quad",
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
        mass: float, optional
            Mass of the product. Default is zero (i.e. for a photon.)
        method: str, optional
            Method to use to integrate. Can be 'quad', 'trapizoid' or
            'simpson'. Default is 'quad'.
        args: tuple, optional
            Additional arguments to pass to function.

        Returns
        -------
        boosted: array
            The boosted spectrum evaluated at the input energies.

        Other Parameters
        ----------------
        If `method` = 'trapizoid' or 'simpson', the following keyword arguments
        are available:

            npts: int, optional
                Number of points to use in trapizoidal or simpson integration.
                Default is 100.
            args: tuple, optional
                Additional arguments to pass to the wrapped spectrum function.

        If method = 'quad', any keyword arguments compatible with quad can be
        specified.

        """
        check_method(method)
        early = kinematic_early_return(energies, beta)
        if early is not None:
            return early

        gamma = 1.0 / np.sqrt(1.0 - beta**2)
        boosted = np.zeros_like(energies)
        emin, emax, mask = bounds_and_mask(
            energies=energies, gamma=gamma, beta=beta, mass=mass
        )

        if args is not None:

            def integrand(e):
                return fn(e, *args) / np.sqrt(e**2 - mass**2)

        else:

            def integrand(e):
                return fn(e) / np.sqrt(e**2 - mass**2)

        bounds = zip(emin, emax)

        if method == "quad":
            quad_kwargs = get_quad_kwargs(**kwargs)
            boosted[mask] = np.array(
                [integrate.quad(integrand, a, b, **quad_kwargs)[0] for a, b in bounds]
            )
        else:
            npts = kwargs.get("npts", 100)
            es = np.array([np.linspace(a, b, npts) for a, b in bounds])
            integrands = np.array([integrand(e) for e in es])

            if method == "trapizoid":
                integrator = integrate.trapezoid
            else:
                integrator = integrate.simpson

            boosted[mask] = integrator(integrands, es)

        return 0.5 * boosted / (beta * gamma)

    return fn_boosted


def dnde_boost(
    fn: Callable,
    energies,
    beta: float,
    mass: float = 0.0,
    method: str = "quad",
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
        'simpson'. Default is 'quad'.
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
    boosted = make_boost_function(fn)
    return boosted(energies, beta, mass=mass, method=method, args=args, **kwargs)
