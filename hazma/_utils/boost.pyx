import cython
from libc.math cimport sqrt, fabs, fmin, fmax
from libc.float cimport DBL_EPSILON
from .boost cimport boost_gamma, boost_beta
import numpy as np
cimport numpy as np

@cython.cdivision(True)
cdef double boost_jac(double ep, double mp, double ed, double md, double zl):
    """
    Returns the Jacobian for boost integrals when boosting from the lab frame
    to the parent particle's rest frame.

    Parameters
    ----------
    ep: double
        Energy of the parent particle in lab-frame
    mp: double
        Mass of the parent particle
    ed: double
        Energy of the daughter particle in lab-frame
    mp: double
        Mass of the daughter particle
    zl: double
        Cosine of the angle the daughter particle makes wrt z-axis in
 
    Notes
    -----
    The Jacobian is given by:
        J = det({
            {    dER/dEL,    dER/dcostL }
            { dcostR/dEl, dcostR/dcostL }
        })
 
    where `ER` is the energy of the daughter particle in the parent particle's
    rest-frame, `costR` is the cosine of the angle the daughter particle makes
    w.r.t. the z-axis. The quantities with `L` are in the lab-frame.
    """
    cdef double b
    cdef double g
    cdef double kt

    b = boost_beta(ep, mp)
    g = boost_gamma(ep, mp)
    kt = sqrt(1 - (md / ed) ** 2)

    return kt / (g * (1.0 + b * kt * zl))


@cython.cdivision(True)
cdef double boost_eng(double ep, double mp, double ed, double md, double zl):
    """
    Compute the boosted energy of a daugther particle when boosted from the
    lab-frame to the rest-frame of the parent particle.

    Parameters
    ----------
    ep: double
        Energy of the parent particle.
    mp: double
        Mass of the parent particle.
    ed: double
        Energy of the daugther particle.
    md: double
        Mass of the daugther particle.
    zl: double
        Cosine of the angle daugther makes with z-axis.
    """
    cdef double b = boost_beta(ep, mp)
    cdef double g = boost_gamma(ep, mp)
    cdef double kt = sqrt(1 - (md / ed) ** 2)
    return g * ed * (1 + kt * b * zl)


@cython.cdivision(True)
cdef double boost_delta_function(double e0, double e, double m, double beta):
    """
    Boost a delta function of the form δ(e - e0) of a particle of mass `m`
    with a boost parameter `beta`.

    Parameters
    ----------
    e0: double
        Center of the dirac-delta spectrum in rest-frame
    e: double
        Cnergy of the product in the lab frame.
    m: double
        Mass of the product
    beta: double
        Boost velocity of the decaying particle
    """
    cdef double gamma
    cdef double k
    cdef double eminus
    cdef double eplus
    cdef double k0

    if beta > 1.0 or beta <= 0.0 or e < m:
        return 0.0

    gamma = 1.0 / sqrt(1.0 - beta * beta)
    k = sqrt(e * e - m * m)
    eminus = gamma * (e - beta * k)
    eplus = gamma * (e + beta * k)

    # - b * k0 < (e/g) - e0 < b * k0
    if eminus < e0 and e0 < eplus:
        k0 = sqrt(e0 * e0 - m * m)
        return 1.0 / (2.0 * gamma * beta * k0)

    return 0.0


# cdef double boost_dnde(boost_integrand integrand, double e2, double mass, double beta, double erf_min, double erf_max):
#     cdef:
#         double gamma
#         double k
#         double ep, em
#         double emin, emax
#         double jac

#     gamma = 1.0 / sqrt(1.0 - beta * beta)
#     k = sqrt(e2 * e2 - mass * mass)

#     em = gamma * (e2 - beta * k)
#     ep = gamma * (e2 + beta * k)

#     emin = fmax(em, erf_min)
#     emax = fmin(ep, erf_max)


cdef double integrate_linear_interp_edge(double x, double x1, double x2, double y1, double y2, int left):
    cdef double dx = x2 - x1
    cdef double m = (y2 - y1) / dx
    cdef double b = (x2 * y1 - x1 * y2) / dx

    if left == 1:
        return (x - x1) * (0.5 * m * (x + x1) + b)
    else:
        return (x2 - x) * (0.5 * m * (x2 + x) + b)


cdef (double, double) integration_bounds(double energy, double mass, double beta, double emax = -1.0):
    """
    Compute the integration bounds for the boost integral. If the arguments are invalid kinematically, 
    -1 is returned for both the upper and lower bounds.
    
    Parameters
    ----------
    energy:
        Energy of the product.
    mass
        Mass of the product.
    beta
        Boost velocity.
    emax: double, optional
        Maximum energy.

    Returns
    -------
    lb, ub: double
        Lower and upper bounds.
    """
    cdef double mu
    cdef double lb = -1.0
    cdef double ub = -1.0
    cdef double gamma = 1.0

    if mass < energy and 0.0 <= beta < 1.0:
        mu = mass / energy
        gamma = 1.0 / sqrt(1.0 - beta * beta)
        lb = fmax(gamma * energy * (1.0 - beta * sqrt(1.0 - mu * mu)), mass)
        ub = gamma * energy * (1.0 + beta * sqrt(1.0 - mu * mu))

        if emax > 0.0:
            ub = fmin(ub, emax)

    return lb, ub


@cython.cdivision(True)
@cython.wraparound(False)
cdef double boost_integrate_linear_interp(double photon_energy, double beta, np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    """
    Perform the boost integral given rest-frame spectrum data.

    Parameters
    ----------
    photon_energy:
        Energy to evaluate boosted spectrum at.
    beta:
        Boost velocity.
    x: np.ndarray
        Energies of the rest-frame spectrum.
    y: np.ndarray
        Spectrum values of the rest-frame spectrum.

    Returns
    -------
    boosted: double
        The boosted spectrum evaluated at `photon_energy`.
    """
    cdef:
        int npts
        double xmax
        double gamma
        double lb
        double ub
        double x0
        double y0
        double x1
        double x2
        double y2
        double y1
        double m
        double b
        double rat
        int ilow
        int ihigh
        double[:] yy
        double integral
    
    assert 0.0 < beta < 1.0
    npts = len(x)
    assert npts == len(y)

    xmax = x[npts - 1]
    x0 = x[0]
    y0 = y[0]

    gamma = 1.0 / sqrt(1.0 - beta * beta)
    lb = photon_energy * gamma * (1.0 - beta)
    ub = photon_energy * gamma * (1.0 + beta)


    if lb > xmax:
        return 0.0

    if ub < x0:
        return y0 * x0 / photon_energy

    integral = 0.0
    ilow = -1
    ihigh = -1

    if ub > xmax:
        ub = xmax
        ihigh = npts - 1

    if lb < x0:
        rat = (1.0 - beta) * photon_energy * gamma / x0
        integral += y0 * (1.0 - rat) / rat
        lb = x0
        ilow = 0

    yy = y / x

    if ilow == -1:
        ilow = np.flatnonzero(lb <= x)[0]
    if ihigh == -1:
        ihigh = np.flatnonzero(ub <= x)[0]
        if fabs(x[ihigh] - ub) > 1e-6:
            ihigh = ihigh - 1

    if ilow < ihigh:
        integral += np.trapz(yy[ilow:ihigh], x=x[ilow:ihigh])

    # Handle edges
    if ilow > 0 and fabs(x[ilow] - lb) > 1e-6:
        x2 = x[ilow]
        x1 = x[ilow-1]
        y2 = yy[ilow]
        y1 = yy[ilow-1]

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        integral += (x2 - lb) * (0.5 * m * (x2 + lb) + b)

    if ihigh < npts - 1 and fabs(ub - x[ihigh]) > 1e-6:
        x2 = x[ihigh+1]
        x1 = x[ihigh]
        y2 = yy[ihigh+1]
        y1 = yy[ihigh]

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        integral += (ub - x1) * (0.5 * m * (ub + x1) + b)

    return integral / (2.0 * gamma * beta)




@cython.cdivision(True)
@cython.wraparound(False)
cdef double boost_integrate_linear_interp_massive(
        double energy,
        double mass,
        double beta,
        np.ndarray[np.float64_t, ndim=1] x,
        np.ndarray[np.float64_t, ndim=1] y,
):
    """
    Perform the boost integral given rest-frame spectrum data.

    Parameters
    ----------
    energy:
        Energy to evaluate boosted spectrum at.
    mass:
        Mass of the product.
    beta:
        Boost velocity.
    x: np.ndarray
        Energies of the rest-frame spectrum.
    y: np.ndarray
        Spectrum values of the rest-frame spectrum.

    Returns
    -------
    boosted: double
        The boosted spectrum evaluated at `photon_energy`.
    """

    cdef Py_ssize_t npts

    cdef double x1
    cdef double x2
    cdef double y2
    cdef double y1
    cdef double m
    cdef double b

    assert 0.0 <= beta < 1.0

    cdef double gamma = 1.0 / sqrt(1.0 - beta * beta)

    npts = len(x)
    assert npts == len(y)

    if energy < mass:
        return 0.0

    cdef double xmax = x[npts - 1]
    cdef double x0 = x[0]
    cdef double y0 = y[0]

    cdef double lb
    cdef double ub
    lb, ub = integration_bounds(energy, mass, beta, xmax)

    if lb < 0.0 or ub < 0.0:
        return 0.0

    if lb > xmax or ub < x0:
        return 0.0

    cdef double integral = 0.0
    cdef Py_ssize_t il = 0
    cdef Py_ssize_t ih = npts - 1
    cdef Py_ssize_t ii = 0
    cdef int found_low = False
    cdef int found_high = False
    cdef int do_low = False
    cdef int do_high = False

    # Find il and ih
    for ii in range(npts - 1):
        x1 = x[ii]
        x2 = x[ii + 1]

        # In these checks, if lb or ub is sufficiently close to an interpolation point,
        # we use the main trapezoidal rule to integrate. Sufficiently close is such
        # that 1 / diff would yield a nan.
        # In other cases, we set il to index such that x[i] > a and ih s.t. x[j] > b

        if x1 <= lb < x2:
            if fabs(x1 / lb - 1.0) < DBL_EPSILON:
                il = ii
            elif fabs(x2 / lb - 1.0) < DBL_EPSILON:
                il = ii + 1
            else:
                il = ii + 1
                do_low = True
            found_low = True

        if x1 < ub <= x2:
            if fabs(x1 / ub - 1.0) < DBL_EPSILON:
                ih = ii + 1
            elif fabs(x2 / ub - 1.0) < DBL_EPSILON:
                ih = ii + 2
            else:
                ih = ii + 1
                do_high = True
            found_high = True

        if found_low and found_high:
            break

    # Edge cases: ih == il
    # If ih == il, then a and b are in the same interval. Use trapezoid rule
    # with special care to interval size: ub - lb = 2 * gamma * beta * k.
    # Thus, we can remove the 2 * gamma * beta since it is divided out by the
    # normalization.
    if ih == il or (il == 0 and ih == 1) or (il == npts - 2 and ih == npts - 1):
        x1 = x[il-1]
        x2 = x[il]
        y1 = y[il-1] / sqrt(x1**2 - mass**2)
        y2 = y[il] / sqrt(x2**2 - mass**2)

        m = (y2 - y1) / (x2 - x1)
        b = (x2 * y1 - x1 * y2) / (x2 - x1)
        k = sqrt(energy * energy - mass * mass)

        return k * (m * gamma * energy + b)

    # Edge case: ih == il + 1
    # In this case, we use Simpson's rule so that we can handle potentially
    # small (ub-lb).
    if ih == il + 1:
        x0 = lb
        x1 = x[il]
        x2 = ub

        y0 = y[il-1] / sqrt(x0**2 - mass**2)
        y1 = y[il] / sqrt(x1**2 - mass**2)
        y2 = y[il + 1] / sqrt(x2**2 - mass**2)

        k = sqrt(energy * energy - mass * mass)
        return k / 6.0 * (y2 + y0 + 4 * y1)


    # Perform bulk integral using trapezoid rule.
    for ii in range(il, ih):
        x0 = x[ii]
        x1 = x[ii + 1]
        y0 = y[ii] / sqrt(x0**2 - mass**2)
        y1 = y[ii + 1] / sqrt(x1**2 - mass**2)
        integral = integral + 0.5 * (x1 - x0) * (y1 + y0)

    # Handle left piece
    if do_low:
        x0 = x[il - 1]
        x1 = x[il]
        y0 = y[il] / sqrt(x0**2 - mass**2)
        y1 = y[il + 1] / sqrt(x1**2 - mass**2)

        integral = integral + integrate_linear_interp_edge(lb, x0, x1, y0, y1, True)

    # Handle right piece
    if do_high:
        x0 = x[ih - 1]
        x1 = x[ih]
        y0 = y[ih] / sqrt(x0**2 - mass**2)
        y1 = y[ih + 1] / sqrt(x1**2 - mass**2)

        integral = integral + integrate_linear_interp_edge(ub, x0, x1, y0, y1, False)

    return integral / (2.0 * gamma * beta)
