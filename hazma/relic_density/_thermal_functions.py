import os
from typing import overload

import numpy as np
from scipy.special import kn, k1
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

from hazma.utils import RealArray, RealOrRealArray

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


@overload
def sm_dof_entropy(T: float) -> float: ...


@overload
def sm_dof_entropy(T: RealArray) -> RealArray: ...


def sm_dof_entropy(T: RealOrRealArray) -> RealOrRealArray:
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
    return _sm_heff(T)  # type: ignore


@overload
def sm_sqrt_gstar(T: float) -> float: ...


@overload
def sm_sqrt_gstar(T: RealArray) -> RealArray: ...


def sm_sqrt_gstar(T: RealOrRealArray) -> RealOrRealArray:
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
    return _sm_sqrt_gstar(T)  # type: ignore


@overload
def sm_entropy_density(T: float) -> float: ...


@overload
def sm_entropy_density(T: RealArray) -> RealArray: ...


def sm_entropy_density(T: RealOrRealArray) -> RealOrRealArray:
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


@overload
def sm_entropy_density_deriv(T: float) -> float: ...


@overload
def sm_entropy_density_deriv(T: RealArray) -> RealArray: ...


def sm_entropy_density_deriv(T: RealOrRealArray) -> RealOrRealArray:
    """
    Compute the derivative of the entropy density of the Standard Model w.r.t.
    temperature.

    Parameters
    ----------
    T: float or array
        Standard Model temperature.

    Returns
    -------
    ds: float or array
        derivative of the entropy density of the Standard Model w.r.t
        temperature.
    """
    return (
        2.0
        * np.pi**2
        / 45.0
        * (_sm_heff_deriv(T) * T + 3.0 * sm_dof_entropy(T))  # type: ignore
        * T**2
    )


@overload
def neq(Ts: float, mass: float, g: float = ..., is_fermion: bool = ...) -> float: ...


@overload
def neq(
    Ts: RealArray, mass: float, g: float = ..., is_fermion: bool = ...
) -> RealArray: ...


def neq(
    Ts: RealOrRealArray, mass: float, g: float = 2.0, is_fermion: bool = True
) -> RealOrRealArray:
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
            xs**2
            * np.sum(eta ** (ns + 1) / ns * kn(2, ns * xs), axis=0)
            / (2.0 * np.pi**2)
        )
    return g * nbar * Ts**3


@overload
def neq_deriv(
    Ts: float, mass: float, g: float = ..., is_fermion: bool = ...
) -> float: ...


@overload
def neq_deriv(
    Ts: RealArray, mass: float, g: float = ..., is_fermion: bool = ...
) -> RealArray: ...


def neq_deriv(
    Ts: RealOrRealArray, mass: float, g: float = 2.0, is_fermion: bool = True
) -> RealOrRealArray:
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
        dnbar = xs**2 * np.sum(eta**ns * k1(ns * xs), axis=0) / (2.0 * np.pi**2)
        nbar = (
            xs**2
            * np.sum(eta ** (ns + 1) / ns * kn(2, ns * xs), axis=0)
            / (2.0 * np.pi**2)
        )

    return g * Ts * (3.0 * Ts * nbar - mass * dnbar)


@overload
def yeq(Ts: float, mass: float, g: float = ..., is_fermion: bool = ...) -> float: ...


@overload
def yeq(
    Ts: RealArray, mass: float, g: float = ..., is_fermion: bool = ...
) -> RealArray: ...


def yeq(
    Ts: RealOrRealArray, mass: float, g: float = 2.0, is_fermion: bool = True
) -> RealOrRealArray:
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
    Ts = np.array(Ts) if hasattr(Ts, "__len__") else Ts
    s = sm_entropy_density(Ts)
    _neq = neq(Ts, mass, g=g, is_fermion=is_fermion)
    return _neq / s


@overload
def yeq_deriv(
    Ts: float, mass: float, g: float = ..., is_fermion: bool = ...
) -> float: ...


@overload
def yeq_deriv(
    Ts: RealArray, mass: float, g: float = ..., is_fermion: bool = ...
) -> RealArray: ...


def yeq_deriv(
    Ts: RealOrRealArray, mass: float, g: float = 2.0, is_fermion: bool = True
) -> RealOrRealArray:
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
    Ts = np.array(Ts) if hasattr(Ts, "__len__") else Ts
    s = sm_entropy_density(Ts)
    ds = sm_entropy_density_deriv(Ts)
    _neq = neq(Ts, mass, g=g, is_fermion=is_fermion)
    _dneq = neq_deriv(Ts, mass, g=g, is_fermion=is_fermion)
    return (_dneq * s - ds * _neq) / s**2


@overload
def yeq_derivx(
    x: float, mass: float, g: float = ..., is_fermion: bool = ...
) -> float: ...


@overload
def yeq_derivx(
    x: RealArray, mass: float, g: float = ..., is_fermion: bool = ...
) -> RealArray: ...


def yeq_derivx(
    x: RealOrRealArray, mass: float, g: float = 2.0, is_fermion: bool = True
) -> RealOrRealArray:
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
    return -mass * dyeq / x**2


@overload
def weq(T: float, mass: float, g: float = ..., is_fermion: bool = ...) -> float: ...


@overload
def weq(
    T: RealArray, mass: float, g: float = ..., is_fermion: bool = ...
) -> RealArray: ...


def weq(
    T: RealOrRealArray, mass: float, g: float = 2.0, is_fermion: bool = True
) -> RealOrRealArray:
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


def thermal_cross_section_integrand(z: float, x: float, model) -> float:
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
    kernal = z**2 * (z**2 - 4.0) * k1(x * z)
    return sig * kernal


def thermal_cross_section(x: float, model) -> float:
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
