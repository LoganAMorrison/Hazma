"""
Implementation of phase-space distribution objects.
"""

# pylint: disable=invalid-name

import logging
from typing import Callable, Tuple

import numpy as np
from scipy import integrate

from hazma.utils import RealArray

from ._base import AbstractPhaseSpaceDistribution


def normalize_distribution(probabilities, edges):
    """Normalize a probability density function.

    Parameters
    ----------
    probabilities: array-like
        The probabilities at each bin center.
    edges:
        The bin edges.

    Returns
    -------
    normalized: ndarray
        The normalized probabilities.
    """
    norm = np.sum([p * (edges[i + 1] - edges[i]) for i, p in enumerate(probabilities)])
    if norm <= 0.0:
        if np.min(probabilities) < 0.0:
            logging.warning("Negative probabilities encountered: %s", probabilities)
            return np.ones_like(probabilities) * np.nan
        return probabilities
    return probabilities / norm


class PhaseSpaceDistribution1D(AbstractPhaseSpaceDistribution):
    r"""Class for storing 1D probability distributions."""

    def __init__(self, x, y):
        """
        Parameters
        ----------
        x: array-like
            Independent variables.
        y: array-like
            The values of the distribution.
        """
        shape_x = np.shape(x)
        shape_y = np.shape(y)

        assert (
            len(shape_x) == 1
        ), f"Expected 1D array for independent variables. Got shape {shape_x}."
        assert (
            len(shape_y) == 1
        ), f"Expected 1D array for probabilities. Got shape {shape_x}."
        assert shape_x[0] == shape_y[0] + 1, (
            "Expected independent variables to have one more value than probabilities. "
            f"Found shapes {shape_x} and {shape_y}."
        )

        self._bins = np.array(x)
        self._bin_centers = 0.5 * (x[1:] + x[:-1])
        self._probabilities = np.array(normalize_distribution(y, x))

    def limits(self) -> Tuple[float, float]:
        r"""Return the limits on the independent variables."""
        return np.min(self._bins), np.max(self._bins)

    def __len__(self) -> int:
        return len(self._probabilities)

    @property
    def bin_centers(self) -> RealArray:
        """Return the central values of the bins."""
        return self._bin_centers

    @property
    def bins(self) -> RealArray:
        """Return the bin edges."""
        return self._bins

    @property
    def probabilities(self) -> RealArray:
        """Return the probabilities at each bin.."""
        return self._probabilities

    def _expect_fixed(self, fn, method) -> RealArray:
        xs = self._bin_centers
        ps = self._probabilities
        fs = np.array([fn(x) for x in xs])

        if len(fs.shape) > 1:
            integrands = np.expand_dims(ps, 1) * fs
        else:
            integrands = ps * fs

        expvals = method(integrands, xs, axis=0)
        return expvals

    def _expect_quad(self, fn) -> RealArray:
        def integrand(x):
            return self(x) * fn(x)

        xmin, xmax = self.limits()
        expvals = integrate.quad(integrand, xmin, xmax)[0]
        return expvals

    def expect(
        self, fn: Callable, method: str = "trapz", args: Tuple = tuple()
    ) -> RealArray:
        r"""Compute the expectation value of function.

        Parameters
        ----------
        fn: callable
            Function to compute expectation value of.
        method: str, optional
            Method used to integrate. Valid methods are 'trapz', 'simps' or
            'quad'. Default is 'trapz'.
        args: tuple, optional
            Additional arguments to pass to function.

        Returns
        -------
        expval: float or ndarray
            Expectation value of the function.
        """

        methods = {
            "trapz": lambda f: self._expect_fixed(f, np.trapz),
            "simps": lambda f: self._expect_fixed(f, integrate.simps),
            "quad": self._expect_quad,
        }

        meth = methods.get(method)
        if meth is None:
            raise ValueError(
                f"Invalid method {method}. Use one of {list(methods.keys())}"
            )

        if len(args) > 0:

            def f(x):
                return fn(x, *args)

        else:

            def f(x):
                return fn(x)

        expvals = meth(f)
        return expvals

    def __call__(self, x):
        """Return the probability at the input value."""
        return np.interp(x, self._bin_centers, self._probabilities, 0.0, 0.0)
