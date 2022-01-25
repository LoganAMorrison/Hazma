import cython
from libc.math cimport sqrt, fabs

@cython.cdivision(True)
cdef double boost_gamma(double e, double m): 
    """
    Compute the gamma boost factor.

    Parameters
    ----------
    e: double
        Energy of the particle
    m: double
        Mass of the particle
    """
    return e / m


@cython.cdivision(True)
cdef double boost_beta(double e, double m): 
    """
    Compute the velocity of a particle given its energy and mass.

    Parameters
    ----------
    e: double
        Energy of the particle
    m: double
        Mass of the particle
    """
    return sqrt(1.0 - (m / e) ** 2)


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

    gaminv = sqrt(1.0 - beta ** 2);
    k0 = sqrt(e0 ** 2 - m ** 2);

    # - b * k0 < (e/g) - e0 < b * k0
    if fabs(e * gaminv - e0) < beta * k0:
        return gaminv / (2 * beta * e0)

    return 0.0

"""
/**
 * Boost the spectrum of a daughter particle in parent particles rest-frame into
 * the lab-frame.
 *
 * @param spec_rf unary function to compute spectrum in the rest-frame
 * @param ep energy of the parent particle in lab-frame
 * @param mp mass of the parent particle
 * @param ed energy of the daughter particle in lab-frame
 * @param mp mass of the daughter particle
 * @param ed_ub upper bound on the daughter energy. Default +infinity.
 * @param ed_ub lower bound on the daughter energy. Default -infinity.
 */
cdef double boost_spectrum(F spec_rf, double ep, double mp, double ed, double md,
                    double ed_ub = std::numeric_limits<double>::infinity(),
                    double ed_lb = -std::numeric_limits<double>::infinity())
    -> double {
  using boost::math::quadrature::gauss_kronrod;
  static constexpr unsigned int GK_N = 15;
  static constexpr unsigned int GK_MAX_LIMIT = 7;
  if (ep < mp) {
    return 0.0;
  }
  if (ep == mp) {
    return spec_rf(ed);
  }

  const auto integrand = [&](const double z) {
    const double eng = boost_eng(ep, mp, ed, md, z);
    const double jac = boost_jac(ep, mp, ed, md, z);
    return spec_rf(eng) * jac;
  };

  const double b = tools::beta(ep, mp);
  const double g = tools::gamma(ep, mp);
  const double den = b * sqrt(tools::sqr(ed) - tools::sqr(md));

  // Compute integration bounds
  const double lb =
      std::isfinite(ed_ub) ? std::max((ed - ed_ub / g) / den, -1.0) : -1.0;
  const double ub =
      std::isfinite(ed_lb) ? std::min((ed - ed_lb / g) / den, 1.0) : 1.0;
  if (lb >= 1.0 || ub <= -1.0) {
    return 0.0;
  }

  return 0.5 * gauss_kronrod<double, GK_N>::integrate(integrand, lb, ub,
                                                      GK_MAX_LIMIT, 1e-6);
}
"""
