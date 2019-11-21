import numpy as np
from scipy.special import kn, k1
from scipy.integrate import quad  # simps
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from hazma.parameters import plank_mass, rho_crit, sm_entropy_density_today
import os
import warnings

_this_dir, _ = os.path.split(__file__)
_fname_sm_data = os.path.join(_this_dir, "smdof.dat")
_sm_data = np.genfromtxt(_fname_sm_data, delimiter=",", skip_header=1).T
_sm_tempetatures = _sm_data[0] * 1e3  # convert to MeV
_sm_sqrt_gstars = _sm_data[1]
_sm_heff = _sm_data[2]


# Interpolating function for SM's sqrt(g_star)
_sm_sqrt_gstar = UnivariateSpline(_sm_tempetatures, _sm_sqrt_gstars, s=0, ext=3)
# Interpolating function for SM d.o.f. stored in entropy: h_eff
_sm_heff = UnivariateSpline(_sm_tempetatures, _sm_heff, s=0, ext=3)
# derivative of SM d.o.f. in entropy w.r.t temperature
_sm_heff_deriv = _sm_heff.derivative(n=1)


def sm_dof_entropy(T):
    """
    Compute the d.o.f. stored in entropy of the Standard Model.

    Parameters
    ----------
    T: float
        Standard Model temperature.

    Returns
    -------
    heff: float
        d.o.f. stored in entropy
    """
    return _sm_heff(T)


def sm_sqrt_gstar(T):
    """
    Compute the square-root of g-star of the Standard Model.

    Parameters
    ----------
    T: float
        Standard Model temperature.

    Returns
    -------
    sqrt_gstar: float
        square-root of g-star of the Standard Model
    """
    return _sm_sqrt_gstar(T)


def sm_entropy_density(T):
    """
    Compute the entropy density of the Standard Model.

    Parameters
    ----------
    T: float
        Standard Model temperature.

    Returns
    -------
    s: float
        energy entropy of the Standard Model
    """
    return 2.0 * np.pi ** 2 / 45.0 * sm_dof_entropy(T) * T ** 3


def sm_entropy_density_deriv(T):
    """
    Compute the derivative of the entropy density of the Standard Model w.r.t.
    temperature.

    Parameters
    ----------
    T: float
        Standard Model temperature.

    Returns
    -------
    ds: float
        derivative of the entropy density of the Standard Model w.r.t
        temperature.
    """
    return (
        2.0
        * np.pi ** 2
        / 45.0
        * (_sm_heff_deriv(T) * T + 3.0 * sm_dof_entropy(T))
        * T ** 2
    )


def neq(Ts, mass, g=2.0, is_fermion=True):
    """
    Compute the equilibrium number density of a particle.

    Parameters
    ----------
    Ts : float or array-like
        Temperature of the particle.
    mass: float
        Mass of the particle.
    g: float, optional
        Internal d.o.f. of the particle. Default is spin 1/2 => g=2
    is_fermion: Bool, optional
        `True` if particle is a fermion, `False` if boson.

    Returns
    -------
    neq: float or array-like
        Equilibrium number density of particle at temperature `T`.
    """
    Ts = np.array(Ts) if hasattr(Ts, "__len__") else Ts
    if mass == 0:
        # if particle is massless, use analytic expression.
        # fermion: 7 / 8 zeta(3) / pi^2
        # boson: zeta(3) / pi^2
        nbar = 0.0913453711751798 if is_fermion else 0.121793828233573
    else:
        # use sum-over-bessel function representation of neq
        # nbar = x^2 sum_n (\pm 1)^{n+1}/n k_2(nx)
        eta = -1 if is_fermion else 1
        xs = mass / Ts
        ns = (
            np.array([1, 2, 3, 4, 5]).reshape(5, 1)
            if hasattr(Ts, "__len__")
            else np.array([1, 2, 3, 4, 5])
        )
        nbar = (
            xs ** 2
            * np.sum(eta ** (ns + 1) / ns * kn(2, ns * xs), axis=0)
            / (2.0 * np.pi ** 2)
        )
    return g * nbar * Ts ** 3


def neq_deriv(Ts, mass, g=2.0, is_fermion=True):
    """
    Compute the derivative of the equilibrium number density of a particle
    w.r.t. its temperature.

    Parameters
    ----------
    Ts : float or array-like
        Temperature of the particle.
    mass: float
        Mass of the particle.
    g: float, optional
        Internal d.o.f. of the particle. Default is spin 1/2 => g=2
    is_fermion: Bool, optional
        `True` if particle is a fermion, `False` if boson.

    Returns
    -------
    dneq: float or array-like
        Derivative of the quilibrium number density of particle w.r.t. its
        temperature at temperature `T`.
    """
    Ts = np.array(Ts) if hasattr(Ts, "__len__") else Ts
    if mass == 0:
        # if particle is massless, use analytic expression.
        dnbar = 0.0
        nbar = 0.0913453711751798 if is_fermion else 0.121793828233573
    else:
        # use sum-over-bessel function representation of neq
        # nbar = x^2 sum_n (\pm 1)^{n+1}/n k_2(nx)
        eta = -1 if is_fermion else 1
        xs = mass / Ts
        # perform a reshape is `x` is an array so we properly sum over ns
        ns = (
            np.array([1, 2, 3, 4, 5]).reshape(5, 1)
            if hasattr(Ts, "__len__")
            else np.array([1, 2, 3, 4, 5])
        )
        dnbar = xs ** 2 * np.sum(eta ** ns * k1(ns * xs), axis=0) / (2.0 * np.pi ** 2)
        nbar = (
            xs ** 2
            * np.sum(eta ** (ns + 1) / ns * kn(2, ns * xs), axis=0)
            / (2.0 * np.pi ** 2)
        )

    return g * Ts * (3.0 * Ts * nbar - mass * dnbar)


def yeq(T, mass, g=2.0, is_fermion=True):
    """
    Compute the equilibrium value of `Y`, the comoving number density
    `neq / s` where `s` is the SM entropy density.

    Parameters
    ----------
    T: float or array-like
        Temperature of the particle. Assumed to be the same
        temperature as the SM.
    mass: float
        Mass of the particle.
    g: float, optional
        Internal d.o.f. of the particle. Default is spin 1/2 => g=2
    is_fermion: Bool, optional
        `True` if particle is a fermion, `False` if boson.

    Returns
    -------
    yeq: float or array-like
        Equilibrium number density divided by the SM entropy density.
    """
    Ts = np.array(T) if hasattr(T, "__len__") else T
    s = sm_entropy_density(Ts)
    _neq = neq(Ts, mass, g=g, is_fermion=is_fermion)
    return _neq / s


def yeq_deriv(T, mass, g=2.0, is_fermion=True):
    """
    Compute the derivative of of `yeq` w.r.t. temperature.

    Parameters
    ----------
    T: float or array-like
        Temperature of the particle. Assumed to be the same
        temperature as the SM.
    mass: float
        Mass of the particle.
    g: float, optional
        Internal d.o.f. of the particle. Default is spin 1/2 => g=2
    is_fermion: Bool, optional
        `True` if particle is a fermion, `False` if boson.

    Returns
    -------
    dyeq: float or array-like
        Derivative of `yeq` w.r.t. temperature.
    """
    Ts = np.array(T) if hasattr(T, "__len__") else T
    s = sm_entropy_density(Ts)
    ds = sm_entropy_density_deriv(Ts)
    _neq = neq(Ts, mass, g=g, is_fermion=is_fermion)
    _dneq = neq_deriv(Ts, mass, g=g, is_fermion=is_fermion)
    return (_dneq * s - ds * _neq) / s ** 2


def yeq_derivx(x, mass, g=2.0, is_fermion=True):
    """
    Compute the derivative of of `yeq` w.r.t. x = `mass/temperature`.

    Parameters
    ----------
    x: float or array-like
        Mass of the particle divided by its temperature.
    mass: float
        Mass of the particle.
    g: float, optional
        Internal d.o.f. of the particle. Default is spin 1/2 => g=2
    is_fermion: Bool, optional
        `True` if particle is a fermion, `False` if boson.

    Returns
    -------
    dyeq_x: float or array-like
        Derivative of `yeq` w.r.t. `x`.
    """
    T = mass / x
    dyeq = yeq_deriv(T, mass, g=g, is_fermion=is_fermion)
    return -mass * dyeq / x ** 2


def weq(T, mass, g=2.0, is_fermion=True):
    """
    Compute the equilibrium value of `W`, the natural log of the
    comoving number density `Y` = `neq / s` where `s` is the
    SM entropy density.

    Parameters
    ----------
    T: float
        Temperature of the particle. Assumed to be the same
        temperature as the SM.
    mass: float
        Mass of the particle.
    g: float, optional
        Internal d.o.f. of the particle.
    is_fermion: Bool, optional
        `True` if particle is a fermion, `False` if boson.

    Returns
    -------
    weq: float
        Natural log of the equilibirum number density divided by
        the SM entropy density.
    """
    s = sm_entropy_density(T)
    _neq = neq(T, mass, g=g, is_fermion=is_fermion)
    return np.log(_neq / s) if _neq > 0.0 else -np.inf


def thermal_cross_section_integrand(z, x, model):
    """
    Compute the integrand of the thermally average cross section for the dark
    matter particle of the given model.

    Parameters
    ----------
    z: float
        Center of mass energy divided by DM mass.
    x: float
        Mass of the dark matter divided by its temperature.
    model: dark matter model
        Dark matter model, i.e. `ScalarMediator`, `VectorMediator`
        or any model with a dark matter particle.

    Returns
    -------
    integrand: float
        Integrand of the thermally-averaged cross-section.
    """
    sig = model.annihilation_cross_sections(model.mx * z)["total"]
    kernal = z ** 2 * (z ** 2 - 4.0) * k1(x * z)
    return sig * kernal


def thermal_cross_section(x, model):
    """
    Compute the thermally average cross section for the dark
    matter particle of the given model.

    Parameters
    ----------
    x: float
        Mass of the dark matter divided by its temperature.
    model: dark matter model
        Dark matter model, i.e. `ScalarMediator`, `VectorMediator`
        or any model with a dark matter particle.

    Returns
    -------
    tcs: float
        Thermally average cross section.
    """
    # If model implements 'thermal_cross_section', use that
    if hasattr(model, "thermal_cross_section"):
        return model.thermal_cross_section(x)

    # If x is really large, we will get divide by zero errors
    if x > 300:
        return 0.0

    pf = x / (2.0 * kn(2, x)) ** 2

    # Commented out code does not seem to work. It give about a two
    # orders-of-magnitude larger value that `quad`. I've tried `simps`,
    # `trapz`, `romb` and `lagguass` (after factoring out e^(-x)). All of them
    # seem to fail?
    # ss = np.linspace(2.0, 150, 500)
    # return simps(integrand(ss), ss) * numpf / den

    return (
        pf
        * quad(
            thermal_cross_section_integrand,
            2.0,
            50.0 / x,
            args=(x, model),
            points=[2.0],
        )[0]
    )


# ----------------------------------------------- #
# Functions for solving the Bolzmann equation     #
# see arXiv:1204.3622v3 for detailed explaination #
# ------------------------------- --------------- #


def boltzmann_eqn(logx, w, model):
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


def jacobian_boltzmann_eqn(logx, w, model):
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


def solve_boltzmann(model, x0=1.0, xf=None, method="Radau", rtol=1e-5, atol=1e-3):
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

    return solve_ivp(f, (logx0, logxf), [w0], method=method, jac=jac, vectorized=True)


# ----------------------------------------------------------- #
# Functions for computing the relic density semi-analytically #
# see arXiv:1204.3622v3 for detailed explaination             #
# ----------------------------------------------------------- #


def xstar_root_eqn(xstar, model, delta=None):
    """
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
    return xstar ** 2 * dyeq + lam * deltabar * tcs * _yeq ** 2


def compute_xstar(model, delta=None):
    """
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
    return root_scalar(xstar_root_eqn, bracket=(0.01, 100.0), args=(model, delta)).root


def compute_alpha(model, xstar):
    """
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
        return sm_sqrt_gstar(model.mx / x) * thermal_cross_section(x, model) / x ** 2

    return pf * quad(integrand, xstar, 100 * xstar)[0]


def relic_density(model, semi_analytic=True, **kwargs):
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
    kwargs: dict
        If `semi_analtyical` is `True`, the accepted kwargs are:
            delta: float, optional
                Value of `delta` assumed for when DM begins to freeze out.
                Default value is the solution to
                delta * (2 + delta) / (1 + delta) = 1, i.e.,
                delta = (sqrt(5) - 1) / 2 = 0.618033988749895. See Eqn.(13) of
                arXiv:1204.3622v3 for details and other used values of delta.
                Value of xstar is logarithmically sensitive to this number.

        If `semi_analytical` is `False`, accepted kwargs are:
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
            delta = kwargs["delta"] if ("delta" in kwargs) else None
            xstar = compute_xstar(model, delta=delta)
            alpha = compute_alpha(model, xstar)
        ystar = yeq(model.mx / xstar, model.mx)
        Y0 = ystar / (1 + ystar * alpha)
    else:
        x0 = kwargs["x0"] if ("x0" in kwargs) else 1.0
        xf = kwargs["xf"] if ("xf" in kwargs) else None
        method = kwargs["method"] if ("method" in kwargs) else "Radau"
        rtol = kwargs["rtol"] if ("rtol" in kwargs) else 1e-5
        atol = kwargs["atol"] if ("atol" in kwargs) else 1e-3
        sol = solve_boltzmann(model, x0=x0, xf=xf, method=method, rtol=rtol, atol=atol)

        Y0 = np.exp(sol.y[0, -1])

    return Y0 * model.mx * sm_entropy_density_today / rho_crit
