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
