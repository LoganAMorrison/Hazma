import cython
from libc.math cimport sqrt, fabs, fmin, fmax
from .boost cimport boost_gamma, boost_beta

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
    cdef double gaminv
    cdef double k0

    if beta > 1.0 or beta <= 0.0:
        return 0.0

    gaminv = sqrt(1.0 - beta ** 2)
    k0 = e0 * sqrt(1.0 - (m / e0)**2)

    # - b * k0 < (e/g) - e0 < b * k0
    if fabs(e * gaminv - e0) < beta * k0:
        return gaminv / (2 * beta * e0)

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
