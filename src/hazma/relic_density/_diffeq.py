from typing import Optional

import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import solve_ivp

from hazma.parameters import plank_mass
from hazma.utils import RealArray

from ._thermal_functions import weq
from ._thermal_functions import sm_sqrt_gstar
from ._thermal_functions import thermal_cross_section


# ----------------------------------------------- #
# Functions for solving the Bolzmann equation     #
# see arXiv:1204.3622v3 for detailed explaination #
# ------------------------------- --------------- #


def boltzmann_eqn(logx: float, w: RealArray, model) -> RealArray:
    """
    Compute the RHS of the Boltzmann equation. Here the RHS is
    given by dW/dlogx, with W = log(neq / sm_entropy_density).

    Parameters
    ----------
    logx: float
        Natural log of the particles mass divided by its
        temperature.
    w: array-like
        Array of the natural log of the dark matter comoving
        number density.
    model: Theory
        Dark matter model.

    Returns
    -------
    dw_dlogx: array-like
        Jacobian of the Boltzmann equation w.r.t. the comoving
        number density of the dark matter particle.
    """
    mx = model.mx
    x = np.exp(logx)
    T = mx / x
    pf = -np.sqrt(np.pi / 45) * plank_mass * mx * sm_sqrt_gstar(T) / x
    _weq = weq(T, mx, g=2.0)
    sv = thermal_cross_section(x, model)

    return np.array([pf * sv * (np.exp(w[0]) - np.exp(2.0 * _weq - w[0]))])


def jacobian_boltzmann_eqn(logx, w: RealArray, model) -> RealArray:
    """
    Compute the Jacobian of the RHS of the Boltzmann equation with
    respect to the log of the comoving equilibrium number density.

    Parameters
    ----------
    logx: float
        Natural log of the particles mass divided by its
        temperature.
    w: array-like
        Array of the natural log of the dark matter comoving
        number density.
    model: Theory
        Dark matter model.

    Returns
    -------
    jac: 2-D array
        Jacobian of the Boltzmann equation w.r.t. the comoving
        number density of the dark matter particle.
    """
    mx = model.mx
    x = np.exp(logx)
    T = mx / x
    pf = -np.sqrt(np.pi / 45) * plank_mass * mx * sm_sqrt_gstar(T) / x
    _weq = weq(T, mx, g=2.0)
    sv = thermal_cross_section(x, model)

    return np.array([[pf * sv * (np.exp(w[0]) + np.exp(2.0 * _weq - w[0]))]])


def solve_boltzmann(
    model,
    x0: float = 1.0,
    xf: Optional[float] = None,
    method: str = "Radau",
    rtol: float = 1e-5,
    atol: float = 1e-3,
) -> OdeResult:
    """
    Solve the Boltzmann equation for the log of the dark matter
    comoving number density as a function of `logx` - which is the
    log of the dark matter mass over its temperature.

    Notes
    -----
    Uses SciPy's `solve_ivp` function to solve the Boltzmann
    equation.

    Parameters
    ----------
    model: Theory
        Dark matter model.
    x0: float, optional
        Initial value of x = mass / temperature. Default is `1.0`.
    xf: float, optional
        Final value of x = mass / temperature. Default is `1000`
        times the initial starting value.
    method: string, optional
        Method used to solve the Boltzmann equation. Default is
        'Radau'.
    rtol: float, optional
        Relative tolerance used to solve the Boltzmann equation.
        Default is `1e-3`.
    atol: float, optional
        Absolute tolerance used to solve the Boltzmann equation.
        Default is `1e-6`.

    Returns
    -------
    sol: OdeSolution
        `OdeSolution` solution object created from scipy's
        `solve_ivp` function.

    """
    mx = model.mx
    T0 = mx / x0
    w0 = weq(T0, mx, g=2.0)
    logx0 = np.log(x0)
    logxf = logx0 + 7.0 if xf is None else np.log(xf)

    def f(logx, w):
        return boltzmann_eqn(logx, w, model)

    def jac(logx, w):
        return jacobian_boltzmann_eqn(logx, w, model)

    return solve_ivp(
        f,
        (logx0, logxf),
        [w0],
        method=method,
        jac=jac,
        vectorized=True,
        rtol=rtol,
        atol=atol,
    )
