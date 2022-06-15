from typing import List, Optional, NamedTuple, Callable

from scipy import special
from scipy import integrate

from hazma.relic_density import relic_density as rd

TWO_BODY = [
    "e e",
    "mu mu",
    "ve ve",
    "vt vt",
    "vm vm",
    "pi pi",
    "k0 k0",
    "k k",
    "pi0 gamma",
    "eta gamma",
    "pi0 phi",
    "eta phi",
    "eta omega",
]

THREE_BODY = [
    "pi0 pi0 gamma",
    "pi pi pi0",
    "pi pi eta",
    "pi pi etap",
    "pi pi omega",
    "pi0 pi0 omega",
    "pi0 k0 k0",
    "pi0 k k",
    "pi k k0",
]
FOUR_BODY = ["pi pi pi pi", "pi pi pi0 pi0"]


class VectorMediatorGeVRelicDensity(NamedTuple):
    """Small namedtuple to store needed components for computing the relic
    density.
    """

    mx: float
    thermal_cross_section: Callable


def _make_final_states(three_body: bool = False, four_body: bool = False) -> List[str]:
    final_states = TWO_BODY.copy()
    if three_body:
        final_states.extend(THREE_BODY)
    if four_body:
        final_states.extend(FOUR_BODY)
    return final_states


def relic_density(
    self,
    three_body: bool = False,
    four_body: bool = False,
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
    three_body: bool, optional
        If True, three-body final-states are included.
    four_body: bool, optional
        If True, four-body final-states are included.
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
    final_states = _make_final_states(three_body=three_body, four_body=four_body)

    acs_fns = {
        key: val
        for key, val in self.annihilation_cross_section_funcs().items()
        if key in final_states
    }

    def acs(cme):
        return sum(fn(cme) for fn in acs_fns.values())

    def integrand(z, x):
        sig = acs(self.mx * z)
        kernal = z**2 * (z**2 - 4.0) * special.k1(x * z)
        return sig * kernal

    def thermal_cross_section(x):
        # If x is really large, we will get divide by zero errors
        if x > 300:
            return 0.0

        pf = x / (2.0 * special.kn(2, x)) ** 2
        return pf * integrate.quad(integrand, 2.0, 50.0 / x, points=[2.0], args=(x,))[0]

    model = VectorMediatorGeVRelicDensity(
        mx=self.mx, thermal_cross_section=thermal_cross_section
    )

    return rd(
        model=model,
        semi_analytic=semi_analytic,
        delta=delta,
        x0=x0,
        xf=xf,
        method=method,
        rtol=rtol,
        atol=atol,
    )
