from typing import Optional

import numpy as np
from scipy import optimize
from scipy import integrate

from hazma.parameters import plank_mass

from ._thermal_functions import sm_sqrt_gstar
from ._thermal_functions import yeq, yeq_derivx
from ._thermal_functions import thermal_cross_section

# ----------------------------------------------------------- #
# Functions for computing the relic density semi-analytically #
# see arXiv:1204.3622v3 for detailed explaination             #
# ----------------------------------------------------------- #


def xstar_root_eqn(xstar: float, model, delta: Optional[float] = None) -> float:
    r"""
    Returns residual of root equation used to solve for x_star.

    See Eqn.(14) of arXiv:1204.3622v3 for similar expressions. Note that our
    result is more exact since we do not assume `xstar` is large enough that
    Yeq ~ x^{3/2} e^{-x} / h_sm. This may cause a bit of a slow down.

    Parameters
    ----------
    xstar: float
        Current value of x_star, the value of mass / temperature at which DM
        begins to freeze out.
    model: Theory
        Dark matter model.
    delta: float, optional
        Value of `delta` assumed for when DM begins to freeze out. Default
        value is the solution to delta * (2 + delta) / (1 + delta) = 1, i.e.,
        delta = (sqrt(5) - 1) / 2 = 0.618033988749895. See Eqn.(13) of
        arXiv:1204.3622v3 for details and other used values of delta. Value of
        xstar is logarithmically sensitive to this number.

    Returns
    -------
    residual: float
        Residual of the root equation.
    """
    deltabar = 1.0 if delta is None else delta * (2.0 + delta) / (1.0 + delta)
    T = model.mx / xstar
    lam = np.sqrt(np.pi / 45.0) * model.mx * plank_mass * sm_sqrt_gstar(T)
    tcs = thermal_cross_section(xstar, model)
    _yeq = yeq(T, model.mx)
    dyeq = yeq_derivx(xstar, model.mx)
    return xstar**2 * dyeq + lam * deltabar * tcs * _yeq**2


def compute_xstar(model, delta: Optional[float] = None) -> float:
    r"""
    Computes to value of `xstar`: the value of dm_mass / temperature such that
    the DM begins to freeze out.

    Parameters
    ----------
    model: Theory
        Dark matter model.
    delta: float, optional
        Value of `delta` assumed for when DM begins to freeze out. Default
        value is the solution to delta * (2 + delta) / (1 + delta) = 1, i.e.,
        delta = (sqrt(5) - 1) / 2 = 0.618033988749895. See Eqn.(13) of
        arXiv:1204.3622v3 for details and other used values of delta. Value of
        xstar is logarithmically sensitive to this number.

    Returns
    -------
    xstar: float
        Value of mass / temperature at which DM begins to freeze-out.
    """
    return optimize.root_scalar(
        xstar_root_eqn, bracket=(0.01, 100.0), args=(model, delta)
    ).root


def compute_alpha(model, xstar: float) -> float:
    r"""
    Computes the value of the integral of RHS of the Boltzmann equation with
    Yeq set to zero from x_{\\star} to x_{\\mathrm{f.o.}}.

    Parameters
    ----------
    model: Theory
        Dark matter model.
    xstar: float
        Value of mass / temperature at which DM begins to freeze-out. See
        `compute_xstar` for more details.
    """
    pf = np.sqrt(np.pi / 45.0) * model.mx * plank_mass

    def integrand(x):
        return sm_sqrt_gstar(model.mx / x) * thermal_cross_section(x, model) / x**2

    return pf * integrate.quad(integrand, xstar, 100 * xstar)[0]
