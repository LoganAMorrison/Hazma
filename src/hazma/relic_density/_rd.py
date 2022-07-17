from typing import Optional
import warnings

import numpy as np

from hazma.parameters import rho_crit, sm_entropy_density_today

from ._thermal_functions import yeq
from ._diffeq import solve_boltzmann
from ._approx import compute_xstar, compute_alpha


def relic_density(
    model,
    semi_analytic: bool = True,
    delta: Optional[float] = None,
    x0: float = 1.0,
    xf: Optional[float] = None,
    method: str = "Radau",
    rtol: float = 1e-5,
    atol: float = 1e-3,
) -> float:
    """
    Solves the Boltzmann equation and returns the relic density
    computed from the final dark matter comoving number density.

    Notes
    -----
    Uses SciPy's `solve_ivp` function to solve the Boltzmann
    equation.

    Parameters
    ----------
    model: Theory
        Dark matter model.
    semi_analytic: bool
        If `True`, the relic density is computed using semi-analyticall
        methods, otherwise the Boltzmann equation is numerically solved.
    delta: float, optional
        Ignored if 'semi_analytic' is True.
        Value of `delta` assumed for when DM begins to freeze out.
        Default value is the solution to:
            delta * (2 + delta) / (1 + delta) = 1,
        i.e.,
            delta = (sqrt(5) - 1) / 2 = 0.618033988749895.
        See Eqn.(13) of arXiv:1204.3622v3 for details and other used values of delta.
        Note: value of xstar is logarithmically sensitive to this number.
    x0: float, optional
        Initial value of x = mass / temperature. Default is `1.0`.
        Ignored if 'semi_analytic' is True.
    xf: float, optional
        Final value of x = mass / temperature. Default is `1000`
        times the initial starting value.
        Ignored if 'semi_analytic' is True.
    method: string, optional
        Method used to solve the Boltzmann equation. Default is
        'Radau'.
        Ignored if 'semi_analytic' is True.
    rtol: float, optional
        Relative tolerance used to solve the Boltzmann equation.
        Default is `1e-3`.
        Ignored if 'semi_analytic' is True.
    atol: float, optional
        Absolute tolerance used to solve the Boltzmann equation.
        Default is `1e-6`.
        Ignored if 'semi_analytic' is True.

    Returns
    -------
    rd: float
        Dark matter relic density.

    """
    if semi_analytic:
        # TODO: track down where these warnings are stemming from.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"divide by zero encountered in double_scalars"
            )
            warnings.filterwarnings("ignore", r"overflow encountered in double_scalars")
            xstar = compute_xstar(model, delta=delta)
            alpha = compute_alpha(model, xstar)
        ystar = yeq(model.mx / xstar, model.mx)
        Y0 = ystar / (1 + ystar * alpha)
    else:
        sol = solve_boltzmann(model, x0=x0, xf=xf, method=method, rtol=rtol, atol=atol)
        Y0 = np.exp(sol.y[0, -1])

    return Y0 * model.mx * sm_entropy_density_today / rho_crit
