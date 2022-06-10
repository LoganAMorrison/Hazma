"""
========================================
Convolve (:mod:`hazma.spectra.convolve`)
========================================

.. versionadded:: 2.0

Utilities for convolving differential energy spectra with energy distributions.
"""

from typing import Tuple, Any

import numpy as np
from scipy import integrate


def make_marginal_fn(
    dnde_fn,
    cmes,
    probs,
    *,
    method: str = "trapz",
    vectorized: bool = False,
    args: Tuple[Any, ...] = tuple(),
):
    """Construct a function to compute the marginalized differential energy
    spectrum given a probability distribution for the center-of-mass energy.

    Parameters
    ----------
    dnde_fn: Callable
        Function to compute the differential energy spectrum, dN/dE. Should
        have the signature `(ep, cme)` where `ep` is the energy of the product
        and `cme` is the center-of-mass energy.
    cmes: array
        Array for the center-of-mass energies.
    probs: array
        Array for the probabilities corresponding to the center-of-mass energies.
    method: str, optional
        Method to use to perform integration. Can be 'trapezoid' or 'simpson'.
        Default is 'trapezoid'.
    vectorized: bool, optional
        If True, the differential energy spectrum function is assume to be
        vectorized over the product energies.
    args: tuple, optional
        Tuple of additional arguments to pass differential spectum function.

    Returns
    -------
    marginalize: array
        Function to compute the differential energy spectum marginalized over
        the center-of-mass energy. The returned function has the signature
        `marginalize(product_energies)`.
    """
    methods = ["trapz", "simpson"]
    assert method in methods

    cmes_ = np.expand_dims(np.array(cmes), 1)
    probs_ = np.expand_dims(np.array(probs), 1)

    if vectorized:
        excluded = (0,) + tuple([i + 2 for i in range(len(args))])
    else:
        excluded = tuple([i + 2 for i in range(len(args))])

    integrand = np.vectorize(dnde_fn, excluded=excluded)

    if method == "trapz":
        integrator = np.trapz
    else:
        integrator = integrate.simpson

    def marginalize(product_energies):
        es = np.expand_dims(product_energies, 0)
        vals = probs_ * integrand(es, cmes_, *args)
        return integrator(vals, cmes, axis=0)

    return marginalize


def marginalize(
    dnde_fn,
    product_energies,
    cmes,
    probs,
    *,
    method: str = "trapz",
    vectorized: bool = True,
    args: Tuple[Any, ...] = tuple(),
):
    """Compute the marginalized differential energy spectrum given a
    probability distribution for the center-of-mass energy.

    Parameters
    ----------
    dnde_fn: Callable
        Function to compute the differential energy spectrum, dN/dE. Should
        have the signature `(ep, cme)` where `ep` is the energy of the product
        and `cme` is the center-of-mass energy.
    product_energies: array
        Array for the product energies.
    cmes: array
        Array for the center-of-mass energies.
    probs: array
        Array for the probabilities corresponding to the center-of-mass energies.
    method: str, optional
        Method to use to perform integration. Can be 'trapezoid' or 'simpson'.
        Default is 'trapezoid'.
    vectorized: bool, optional
        If True, the differential energy spectrum function is assume to be
        vectorized over the product energies.
    args: tuple, optional
        Tuple of additional arguments to pass differential spectum function.

    Returns
    -------
    dnde: array
        The differential energy spectum marginalized over the center-of-mass energy.

    """
    f = make_marginal_fn(
        dnde_fn, cmes, probs, method=method, vectorized=vectorized, args=args
    )
    return f(product_energies)
