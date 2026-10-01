import os
from typing import Protocol, overload

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


def thermal_cross_section_upper_limit(x: float) -> float:
    """
    Compute the upper limit of the thermal average's integral over z.

    The integral runs over ``z = sqrt(s) / m`` from threshold, ``z = 2``,
    and its kernel falls off as ``K1(x z) ~ exp(-x z)``. The limit
    ``2 + 100 / x`` cuts it where the Bessel argument has run 100 past its
    threshold value ``2x``.
    The tail that drops is at most 3.0e-38 of the kernel
    ``z^2 (z^2 - 4) K1(x z)``'s full integral, measured at 30 digits for
    ``x`` from 0.01 to 300, so a cross section would have to grow by
    thirty decades across the tail to reach ``quad``'s default
    ``epsrel``. Half that interval is not enough: a channel that opens
    above threshold can grow by fifteen decades, as ``S S`` does in a
    `HiggsPortal` with ``stheta = 1e-4``, and a cut at ``2 + 50 / x``
    loses 1.2e-7 of that model's value at ``x = 12.6``.

    Counting from threshold assumes every channel that matters opens
    within the window. One that opens past ``2 + 100 / x`` is dropped
    whole: with ``stheta = 0`` that same model's average is ``0.0`` at
    ``x = 30``. The Rust mediator kernels count from their last channel
    threshold instead (``rust/src/kernels/thermal_window.rs``). The
    generic sites see only ``annihilation_cross_sections``, so they
    cannot; ``docs/followups/todo/python-thermal-sites-cannot-see-channel-thresholds.md``
    tracks giving them the thresholds.

    Because the interval scales with the decay length ``1 / x``, the
    integrator's first nodes also land where the integrand is not
    negligible. A fixed cut such as ``[2, 150]`` puts them in the tail at
    large ``x`` and loses 1e-4 of the value at ``x = 300``.

    Parameters
    ----------
    x: float
        Mass of the dark matter divided by its temperature.

    Returns
    -------
    z_max: float
        Upper limit of the integral, in units of the dark matter mass.
    """
    return 2.0 + 100.0 / x


class _DarkMatterModel(Protocol):
    """Any model with a dark matter mass ``mx`` in MeV."""

    mx: float


#: Ratio between successive break points bracketing a resonance, in units
#: of its width. See `thermal_cross_section_break_points`.
_RESONANCE_LADDER_RATIO = 4.0


def thermal_cross_section_break_points(
    x: float, model: _DarkMatterModel
) -> list[float]:
    """
    Compute the break points of the thermal average's integral over z.

    Each of the model's
    `hazma.theory.TheoryAnn.annihilation_resonances`, of mass ``m`` and
    width ``w``, contributes the ladder ``z_r +/- g 4^k`` for
    ``k = 0, 1, ...``, where ``z_r = m / mx`` and ``g = w / mx``, up to the
    length of the integration interval. The peak itself is never a break
    point. Only points inside the open interval from threshold to
    `thermal_cross_section_upper_limit` are kept. A model that does not
    define the method contributes none.

    Without break points near it, QUADPACK's error estimate misses the
    peak at isolated ``x``: for
    ``HiggsPortal(mx=200, ms=550, gsxx=1, stheta=1e-4)``, whose resonance
    is 7 MeV wide, the average came out 4.0e-4 low at ``x = 0.891``
    while ``x = 0.89`` is good to 6e-11. Breaking only *at* the peak, as
    the ``hazma._core`` kernels do, fixes that case but fails for narrow
    resonances. It leaves the peak on a subinterval's endpoint, where no
    Gauss-Kronrod node samples it. With ``gsxx=1e-2, stheta=1e-3`` the
    width is 7e-4 MeV, and the average then loses 13% at ``x = 1`` and
    all of it at ``x = 20``.
    The ladder puts the peak inside a subinterval two widths across, and
    each rung outward sees a tail that changes by a bounded factor.
    Against independently split references integrated to
    ``epsrel = 1e-12``, a ratio of 4 holds the error under 1.5e-6 for
    ``w / mx`` down to 7.8e-7 and under 6e-7 for widths of a few percent,
    across ``x`` from 0.1 to 300. A general split at decay lengths
    ``2 + k/x`` does not substitute for it. Such a split fixes the 7 MeV
    case but loses 99.7% of the narrow one at ``x = 3.487``, because it
    does not know where the peak is.

    Parameters
    ----------
    x: float
        Mass of the dark matter divided by its temperature.
    model: dark matter model
        Dark matter model with a mass ``mx`` in MeV.

    Returns
    -------
    points: list of float
        Sorted, distinct break points in units of the dark matter mass.
    """
    resonances = getattr(model, "annihilation_resonances", list)()
    z_min, z_max = 2.0, thermal_cross_section_upper_limit(x)
    points = set()
    for mass, width in resonances:
        z_res = mass / model.mx
        offset = width / model.mx
        while 0.0 < offset < z_max - z_min:
            points.update((z_res - offset, z_res + offset))
            offset *= _RESONANCE_LADDER_RATIO
    return sorted(z for z in points if z_min < z < z_max)


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

    # `epsabs=0.0` leaves the relative criterion as the binding one.
    # <sigma v> is of order 1e-27 here, twenty decades under scipy's
    # default `epsabs` of 1.49e-8, and QUADPACK returns as soon as
    # *either* criterion is met -- so at the default the first
    # Gauss-Kronrod pass clears it and the initial partition comes back
    # unrefined. Measured on the mediator kernels that share this defect,
    # that costs up to 100% of the value across the freeze-out region.
    #
    # `limit` keeps scipy's default 50 subdivisions free for refinement on
    # top of the intervals the break points already cut.
    points = thermal_cross_section_break_points(x, model)
    return (
        pf
        * quad(
            thermal_cross_section_integrand,
            2.0,
            thermal_cross_section_upper_limit(x),
            args=(x, model),
            points=[2.0, *points],
            epsabs=0.0,
            limit=50 + len(points),
        )[0]
    )
