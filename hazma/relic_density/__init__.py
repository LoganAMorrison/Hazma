import numpy as np
from scipy.special import kn
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp
from hazma.parameters import plank_mass, rho_crit, sm_entropy_density_today
import os

_this_dir, _ = os.path.split(__file__)
_fname_sm_data = os.path.join(_this_dir, 'smdof.dat')
_sm_data = np.genfromtxt(_fname_sm_data, delimiter=',', skip_header=1).T
_sm_tempetatures = _sm_data[0] * 1e3  # convert to MeV
_sm_sqrt_gstars = _sm_data[1]
_sm_heff = _sm_data[2]


# Interpolating function for SM's sqrt(g_star)
_sm_sqrt_gstar = UnivariateSpline(
    _sm_tempetatures, _sm_sqrt_gstars, s=0, ext=3, )
# Interpolating function for SM d.o.f. stored in entropy: h_eff
_sm_heff = UnivariateSpline(_sm_tempetatures, _sm_heff, s=0, ext=3)


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
    return 2.0 * np.pi**2 / 45.0 * sm_dof_entropy(T) * T**3


def neq(T, mass, g=1.0, is_fermion=True):
    """
    Compute the equilibrium number density of a particle.

    Parameters
    ----------
    T : float
        Temperature of the particle.
    mass: float
        Mass of the particle.
    g: float, optional
        Internal d.o.f. of the particle.
    is_fermion: Bool, optional
        `True` if particle is a fermion, `False` if boson.

    Returns
    -------
    neq: float
        Equilibrium number density of particle at temperature `T`.
    """
    if mass == 0:
        # if particle is massless, use analytic expression.
        nbar = 0.0913453711751798 if is_fermion else 0.121793828233573
    else:
        eta = -1 if is_fermion else 1
        x = mass / T
        nbar = x**2 * np.sum(np.array([
            eta**(n + 1) / n * kn(2, n * x)
            for n in range(1, 6)])) / (2.0 * np.pi**2)
    return g * nbar * T**3


def weq(T, mass, g=1.0, is_fermion=True):
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
    if hasattr(model, 'thermal_cross_section'):
        return model.thermal_cross_section(x)
    mx = model.mx
    T = mx / x
    numpf = 2.0 * np.pi**2 * T
    den = (4.0 * np.pi * mx**2 * T * kn(2, x))**2

    # If the denominator `den` is zero, that means x is very
    # large. Thermal cross section will be zero in this case.
    if den == 0:
        return 0.0

    def integrand(s):
        """Integrand for the thermal cross section"""
        ecm = np.sqrt(s)
        sig = model.annihilation_cross_sections(ecm)['total']
        kernal = ecm * (s - 4.0 * mx**2) * kn(1, ecm / T)
        return sig * kernal

    lb = np.log10(4.0 * mx**2)
    ub = lb + 3.0
    ss = np.logspace(lb, ub, 500)
    return simps(integrand(ss), ss) * numpf / den


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

    return np.array([
        pf * sv * (np.exp(w[0]) - np.exp(2.0 * _weq - w[0]))])


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

    return np.array([[
        pf * sv * (np.exp(w[0]) + np.exp(2.0 * _weq - w[0]))]])


def solve_boltzmann(model, x0=1.0, xf=None, method='Radau',
                    rtol=1e-5, atol=1e-3):
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

    return solve_ivp(f, (logx0, logxf), [w0], method=method,
                     jac=jac, vectorized=True)


def relic_density(model, x0=1.0, xf=None, method='Radau',
                  rtol=1e-5, atol=1e-3):
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
    sol = solve_boltzmann(model, x0=x0, xf=xf, method=method,
                          rtol=rtol, atol=atol)

    Yf = np.exp(sol.y[0, -1])
    return Yf * model.mx * sm_entropy_density_today / rho_crit
