import os
from collections.abc import Iterable
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


#: How far past the last feature the integral over ``z`` runs, in decay
#: lengths ``1/x``. See `thermal_cross_section_partition`. The
#: ``hazma._core`` kernels use the same value as
#: ``thermal_window::DECAY_LENGTHS_PAST_LAST``.
_DECAY_LENGTHS_PAST_LAST = 100.0

#: Break points past each channel opening, in decay lengths ``1/x``. The
#: piece after the last starts where the Boltzmann weight has fallen by
#: ``e^-50``. The ``hazma._core`` kernels use the same values as
#: ``thermal_window::SPLITS``.
_SPLITS = (1.0, 4.0, 16.0, 50.0)

#: Ratio between successive break points bracketing a resonance, in units
#: of its width. See `thermal_cross_section_partition`.
_RESONANCE_LADDER_RATIO = 4.0

#: How far above the threshold ``z = 2`` the first break point must sit.
#: The cross sections can be singular at ``z = 2`` itself, and the
#: Gauss-Kronrod nodes of a sliver ``[2, 2 + eps]`` round onto it. On a
#: piece this long the outermost 21-point Kronrod node lands 2.2e-12, about
#: 4900 ulps, past threshold. The ``hazma._core`` kernels apply the same
#: bound as ``thermal_window::MIN_THRESHOLD_PIECE``.
_MIN_THRESHOLD_PIECE = 1e-9


def thermal_cross_section_partition(
    x: float,
    mx: float,
    thresholds: Iterable[float],
    resonances: Iterable[tuple[float, float]],
) -> tuple[float, list[float]]:
    """
    Compute the upper limit and break points of the thermal average's integral.

    The integral runs over ``z = sqrt(s) / mx`` from the pair threshold
    ``z = 2``, and its Boltzmann kernel ``K1(x z) ~ exp(-x z)`` confines
    each channel's contribution to a few decay lengths ``1/x`` past the
    ``z`` at which it opens. The partition is therefore built from the
    *features* of the cross section: the threshold ``z = 2``, each channel
    threshold, and each resonance. It follows
    ``rust/src/kernels/thermal_window.rs``, the rule the ``hazma._core``
    mediator kernels use, except at resonances.

    - **The upper limit** is ``_DECAY_LENGTHS_PAST_LAST`` decay lengths
      past the last feature. The tail beyond it is at most 3.0e-38 of the
      kernel ``z^2 (z^2 - 4) K1(x z)``'s integral from that feature,
      measured at 30 digits for ``x`` from 0.01 to 300. Counting from
      ``z = 2`` alone drops a channel that opens past the window: the
      ``S S`` channel of ``HiggsPortal(mx=200, ms=550, gsxx=1)`` opens at
      ``z = 5.5``, and at ``x = 30`` a limit of ``2 + 100/x`` returns
      ``0.0`` for ``stheta = 0`` and loses 99.0% for ``stheta = 1e-20``.
      Half that many decay lengths is not enough either: at
      ``stheta = 1e-4`` the ``S S`` channel sits fifteen decades above the
      suppressed channels, and a cut 50 decay lengths past threshold loses
      1.2e-7 at ``x = 12.6``.
    - **Each channel opening**, and ``_SPLITS`` decay lengths past it, is a
      break point. A piece many decay lengths long whose integrand sits
      within ``1/x`` of its left end is where QUADPACK's first
      Gauss-Kronrod nodes all land in the tail and its error estimate
      misses the peak.
    - **Each resonance**, of mass ``m`` and width ``w``, contributes the
      ladder ``z_r +/- g 4^k`` for ``k = 0, 1, ...``, where
      ``z_r = m / mx`` and ``g = w / mx``, up to the length of the
      interval. The peak itself is never a break point: there it would sit
      on a subinterval's endpoint, where no Gauss-Kronrod node samples it.
      For ``HiggsPortal(mx=200, ms=550, gsxx=1e-2, stheta=1e-3)``, whose
      resonance is 7e-4 MeV wide, a break point at the peak on top of the
      splits above, as the ``hazma._core`` kernels place it, loses 87% of
      the average at ``x = 2`` and 99.9% at ``x = 3.487``. The splits
      alone do not know where the peak is, and lose 99.7% at
      ``x = 3.487`` and 99.95% at ``x = 5.818``. With the ladder, the
      average is within 1.2e-11 of an independently split reference
      integrated to ``epsrel = 1e-12``, on 80 points of ``x`` from 0.1 to
      300; it is within 8e-11 for ``KineticMixing(mx=200, mv=550,
      gvxx=1e-2, eps=1e-3)`` and 5.6e-9 for ``HiggsPortal(mx=100, ms=300,
      gsxx=1, stheta=0.1)``, whose width is 5% of ``mx``.

    A threshold below ``z = 2``, or within ``_MIN_THRESHOLD_PIECE`` above
    it, is moved onto it, and only break points more than
    ``_MIN_THRESHOLD_PIECE`` above threshold and below the upper limit are
    kept, so that no quadrature node rounds onto ``z = 2``. With no
    thresholds and no resonances the integral runs over
    ``[2, 2 + 100/x]``, split at decay lengths past ``z = 2``. That
    suffices for a cross section with no peak whose channels all open
    within the window, and drops any channel that opens past it.

    Parameters
    ----------
    x: float
        Mass of the dark matter divided by its temperature.
    mx: float
        Mass of the dark matter in MeV.
    thresholds: iterable of float
        Center-of-mass energies, in MeV, at which the annihilation channels
        open.
    resonances: iterable of (float, float)
        ``(mass, width)`` of each resonance, both in MeV. A resonance with
        zero width contributes no ladder.

    Returns
    -------
    z_max: float
        Upper limit of the integral, in units of the dark matter mass.
    points: list of float
        Sorted, distinct break points in units of the dark matter mass.
    """
    z_min = 2.0
    openings = {z_min}
    for e_cm in thresholds:
        z = e_cm / mx
        openings.add(z if z >= z_min + _MIN_THRESHOLD_PIECE else z_min)
    peaks = [(mass / mx, width / mx) for mass, width in resonances]
    last = max(openings | {z_res for z_res, _ in peaks})
    z_max = last + _DECAY_LENGTHS_PAST_LAST / x

    points = {z + k / x for z in openings for k in (0.0, *_SPLITS)}
    for z_res, width in peaks:
        offset = width
        while 0.0 < offset < z_max - z_min:
            points.update((z_res - offset, z_res + offset))
            offset *= _RESONANCE_LADDER_RATIO
    return z_max, sorted(z for z in points if z_min + _MIN_THRESHOLD_PIECE < z < z_max)


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
        or any model with a dark matter particle. A model without its own
        ``thermal_cross_section`` is integrated over a partition built
        from its ``annihilation_thresholds()`` and
        ``annihilation_resonances()``; see
        `thermal_cross_section_partition` for what is lost when it does
        not define them.

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
    #
    # A model need not derive from `hazma.theory.TheoryAnn`, so one that
    # lacks the threshold or resonance hook contributes no features.
    z_max, points = thermal_cross_section_partition(
        x,
        model.mx,
        getattr(model, "annihilation_thresholds", dict)().values(),
        getattr(model, "annihilation_resonances", list)(),
    )
    return (
        pf
        * quad(
            thermal_cross_section_integrand,
            2.0,
            z_max,
            args=(x, model),
            points=[2.0, *points],
            epsabs=0.0,
            limit=50 + len(points),
        )[0]
    )
