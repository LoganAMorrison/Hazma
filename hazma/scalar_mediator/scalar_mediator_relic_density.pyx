import cython
import numpy as np
cimport numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import kv
from .get_path import get_dir_path
import os

from libc.math cimport M_PI, sqrt, log, log10, exp, atanh

include "../decay_helper_functions/parameters.pxd"

cdef double mmu = MASS_MU
cdef double me = MASS_E
cdef double mpi0 = 134.9766
cdef double mpi = 139.57018
cdef double qe = sqrt(4. * M_PI * ALPHA_EM)
cdef double muq = 2.3
cdef double mdq = 4.8
cdef double b0 = (mpi0 + mpi)**2 / (muq + mdq) / 4.0
cdef double vh = 246.22795e3
cdef double alpha_em = 1.0 / 137.04
cdef double mpl = 1.220910e19
cdef double rho_crit = 1.05375e-5  # units of h^2 GeV / cm^3
cdef double s_today = 2891.2  # units of cm^-3

sqrt_gstar_data_path = os.path.join(get_dir_path(),
                                    "..", "sm_dof_data", "sqrt_gstar.dat")
heff_data_path = os.path.join(get_dir_path(),
                              "..", "sm_dof_data", "heff.dat")

__sqrt_gstar_data = np.genfromtxt(sqrt_gstar_data_path, delimiter=',').T
__heff_data = np.genfromtxt(heff_data_path, delimiter=',').T

cdef double __sqrt_gstar_interp(double T):
    """
    Returns g^{1/2}_{*} of the standard model. 
    
    Notes
    -----
    g^{1/2}_{*} is defined by:
        g^{1/2}_{*} = h_{eff}/sqrt(g_{eff}) * (1 + T/(3h_{eff}) dh_{eff}/dT)
         
    Parameters
    ----------
    T: double
        Standard model temperature.

    Returns
    -------
    sqrt_gstar: double
        Square root of gstar.

    """
    return np.interp(log10(T), np.log10(__sqrt_gstar_data[0]),
                     __sqrt_gstar_data[1])

cdef double __heff_interp(double T):
    """
    Returns the effective number of degrees of freedom of the standard model 
    stored in entropy. 
         
    Parameters
    ----------
    T: double
        Standard model temperature.

    Returns
    -------
    heff: double
        Effective number of degrees of freedom of the standard model stored in 
        entropy.

    """
    return np.interp(log10(T), np.log10(__heff_data[0]),
                     __heff_data[1])

@cython.cdivision(True)
cdef double __sigmav_xx_to_s_to_ff(double eps, double mx, double ms,
                                   double gsxx, double gsff,
                                   double gsGG, double gsFF,
                                   double lam, double vs, double widths,
                                   double mf):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of fermions, *f* through a scalar mediator in
    the s-channel.
    
    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Deccay width of the scalar mediator. 
    mf: double
        FS fermion mass.
    
    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> f + fbar.

    """
    cdef double r = mf / mx
    cdef double gsll = gsff * mf / vh

    if r * r < 1 + eps and eps > 0:
        return ((eps * gsll**2 * gsxx**2 * mx**2 * (1 + eps - r**2)**1.5) / (
                2. * sqrt(1 + eps) * (1 + 2 * eps) * M_PI * (
                ms**4 + 16 * (1 + eps)**2 * mx**4 + ms**2 *
                (-8 * (1 + eps) * mx**2 + widths**2))))
    else:
        return 0.

cdef double __sigmav_xx_to_s_to_mumu(double eps, double mx, double ms,
                                     double gsxx, double gsff,
                                     double gsGG, double gsFF,
                                     double lam, double vs, double widths):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of muons through a scalar mediator in
    the s-channel.
    
    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Deccay width of the scalar mediator. 
    
    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> mu + mu.

    """
    return __sigmav_xx_to_s_to_ff(eps, mx, ms, gsxx, gsff, gsGG, gsFF, lam, vs,
                                  widths, mmu)

cdef double __sigmav_xx_to_s_to_ee(double eps, double mx, double ms,
                                   double gsxx, double gsff,
                                   double gsGG, double gsFF,
                                   double lam, double vs, double widths):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of electrons through a scalar mediator in
    the s-channel.
    
    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Deccay width of the scalar mediator. 
    
    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> e + e.

    """
    return __sigmav_xx_to_s_to_ff(eps, mx, ms, gsxx, gsff, gsGG, gsFF, lam, vs,
                                  widths, me)

@cython.cdivision(True)
cdef double __sigmav_xx_to_s_to_gg(double eps, double mx, double ms,
                                   double gsxx, double gsff,
                                   double gsGG, double gsFF,
                                   double lam, double vs, double widths):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of photons through a scalar mediator in the
    s-channel.

    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Deccay width of the scalar mediator. 

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> g + g.
    """
    cdef double r = 0;
    if 1 + eps >= r * r:

        return ((alpha_em**2 * eps * (1 + eps)**2 *
                 gsFF**2 * gsxx**2 * mx**4) / (
                        4. * (1 + 2 * eps) * lam**2 * M_PI**3 * (
                        ms**4 + 16 * (1 + eps)**2 * mx**4 + ms**2 *
                        (-8 * (1 + eps) * mx**2 + widths**2))))
    else:
        return 0.0

@cython.cdivision(True)
cdef double __sigmav_xx_to_s_to_pi0pi0(double eps, double mx, double ms,
                                       double gsxx, double gsff,
                                       double gsGG, double gsFF,
                                       double lam, double vs, double widths):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of neutral pion through a scalar mediator in
    the s-channel.

    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Deccay width of the scalar mediator. 

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> pi0 + pi0.
    """
    cdef double r = mpi0 / mx

    if 1 + eps > r * r:

        return ((eps * gsxx**2 * sqrt(1 - r**2 / (1 + eps)) * (
                324 * gsGG * lam**3 * mx**2 * (-2 * (1 + eps) + r**2) *
                vh**2 + b0 * (mdq + muq) * (
                        9 * lam + 4 * gsGG * vs) * (
                        27 * gsff**2 * lam**2 * vs *
                        (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 * (
                                27 * lam**2 - 30 * gsGG * lam * vs +
                                8 * gsGG**2 * vs**2) + gsff * (
                                -81 * lam**3 * vh +
                                48 * gsGG**2 * lam * vh * vs**2)))**2) / (
                        209952. * (1 + 2 * eps) * lam**6 * M_PI * vh**4 *
                        (9 * lam + 4 * gsGG * vs)**2 * (
                                ms**4 + 16 * (1 + eps)**2 * mx**4 + ms**2 *
                                (-8 * (1 + eps) * mx**2 + widths**2))))
    else:
        return 0.

@cython.cdivision(True)
cdef double __sigmav_xx_to_s_to_pipi(double eps, double mx, double ms,
                                     double gsxx, double gsff,
                                     double gsGG, double gsFF,
                                     double lam, double vs, double widths):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of charged pions through a scalar mediator in
    the s-channel.

    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Deccay width of the scalar mediator. 

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> M_PI + M_PI.
    """
    cdef double r = mpi / mx

    if r**2 < 1 + eps:

        return ((eps * gsxx**2 * sqrt(1 - r**2 / (1 + eps)) * (
                324 * gsGG * lam**3 * mx**2 * (-2 * (1 + eps) + r**2) *
                vh**2 + b0 * (mdq + muq) * (
                        9 * lam + 4 * gsGG * vs) * (
                        27 * gsff**2 * lam**2 * vs *
                        (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 * (
                                27 * lam**2 - 30 * gsGG * lam * vs +
                                8 * gsGG**2 * vs**2) + gsff * (
                                -81 * lam**3 * vh +
                                48 * gsGG**2 * lam * vh * vs**2)))**2) / (
                        104976. * (1 + 2 * eps) * lam**6 * M_PI * vh**4 *
                        (9 * lam + 4 * gsGG * vs)**2 * (
                                ms**4 + 16 * (1 + eps)**2 * mx**4 + ms**2 *
                                (-8 * (1 + eps) * mx**2 + widths**2))))
    else:
        return 0.

@cython.cdivision(True)
cdef double __sigmav_xx_to_ss(double eps, double mx, double ms,
                              double gsxx, double gsff,
                              double gsGG, double gsFF,
                              double lam, double vs, double widths):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of scalar mediator through the t and u
    channels.

    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2)
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Deccay width of the scalar mediator. 

    Returns
    -------
    sigma : float
        Cross section for x + x -> s + s.
    """
    cdef double r = ms / mx

    if r * r < 1 + eps:

        return (-(gsxx**4 *
                  (4 * sqrt(eps) + (2 * sqrt(eps) * (-4 + r**2)**2) /
                   (4 + 4 * eps - 4 * r**2 + r**4) +
                   ((8 * (3 + 6 * eps + eps**2) -
                     8 * (2 + eps) * r**2 + 3 * r**4) *
                    atanh((2 * sqrt((eps * (1 + eps - r**2)))) /
                          (2 + 2 * eps - r**2))) /
                   (sqrt(1 + eps - r**2) * (-2 * (1 + eps) + r**2)))) / (
                        128. * mx**2 * (M_PI + 2 * eps * M_PI) *
                        sqrt((eps * (1 + eps)) / (1 + eps - r**2))))
    else:
        return 0.

@cython.cdivision(True)
cdef double __integrand_thermal_cs(double eps, double x, double mx,
                                   double ms, double gsxx, double gsff,
                                   double gsGG, double gsFF, double lam,
                                   double vs, double widths):
    """
    Return the integrand of the thermal cross section.
    
    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Decay width of the scalar mediator.

    Returns
    -------
    integrand: double
        Thermal cross section integrand.
    

    """
    cdef double T = mx / x
    cdef double kernal = 0.0

    if x > 100.0:
        kernal = exp((2 - 2 * sqrt(1 + eps)) * x) * (
                (-3 * sqrt(eps) * (1 + 2 * eps) *
                 (-1 + 20 * sqrt(1 + eps)) * sqrt(x)) / (
                        8. * (1 + eps)**0.75 * sqrt(M_PI)) +
                (2 * sqrt(eps) * (1 + 2 * eps) * x**1.5) / (
                        (1 + eps)**0.25 * sqrt(M_PI)))
    else:
        kernal = ((2.0 * x) / kv(2, x)**2 * sqrt(eps) * (1 + 2 * eps) *
                  kv(1, 2 * x * sqrt(1 + eps)))

    cdef double tcs = (
            __sigmav_xx_to_ss(eps, mx, ms, gsxx, gsff, gsGG,
                              gsFF, lam, vs, widths) +
            __sigmav_xx_to_s_to_pipi(eps, mx, ms, gsxx, gsff, gsGG,
                                     gsFF, lam, vs, widths) +
            __sigmav_xx_to_s_to_pi0pi0(eps, mx, ms, gsxx, gsff, gsGG,
                                       gsFF, lam, vs, widths) +
            __sigmav_xx_to_s_to_mumu(eps, mx, ms, gsxx, gsff, gsGG,
                                     gsFF, lam, vs, widths) +
            __sigmav_xx_to_s_to_ee(eps, mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, vs, widths) +
            __sigmav_xx_to_s_to_gg(eps, mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, vs, widths))

    if np.isfinite(kernal * tcs):
        return kernal * tcs
    else:
        return 0.0

@cython.cdivision(True)
cdef double __thermal_cs(double x, double mx, double ms,
                         double gsxx, double gsff, double gsGG,
                         double gsFF, double lam, double vs,
                         double widths):
    """
    
    Parameters
    ----------
    x: double
        DM mass normalized to the temperature: x = mx/T.
    mx: double
        Dark matter mass.
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to the scalar mediator.
    gsff: double
        Coupling of DM to fermions divided by Yukawa.
    gsGG: double
        Effective coupling of DM to SM gluons.
    gsFF: double
        Effective coupling of DM to SM photons.
    lam: double
        Effective theory cut-off scale.
    vs: double
        Vacuum expectation value of the scalar mediator.
    widths: double
        Decay width of the scalar mediator.

    Returns
    -------
    thermal_cs: double
        Thermally averaged cross section for sum_{A} xx -> A.

    """
    cdef double res = quad(
        __integrand_thermal_cs, 0.0, np.inf,
        epsabs=1e-10, epsrel=1e-4,
        args=(x, mx, ms, gsxx, gsff, gsGG, gsFF, lam, vs, widths))[0]

    # The behavior of the bessel functions in the prefactor can yeild
    if np.isfinite(res):
        return res
    else:
        return 0.0

cdef double __yeq(double T, double m):
    """
    Returns the equilibrium number density normalized to the SM entropy 
    density.
    
    Parameters
    ----------
    T: double
        Standard model temperature.
    m: double
        Mass of the particle.

    Returns
    -------
    yeq: double
        Equilibrium number density normalized to the SM entropy density.

    """
    cdef double x = m / T
    return 45.0 / (4.0 * M_PI**4) * (2.0 / __heff_interp(T)) * x**2 * kv(2, x)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[double] __boltzmann(double logx, np.ndarray[double] w,
                                    double mx, double ms, double gsxx,
                                    double gsff, double gsGG, double gsFF,
                                    double lam, double vs, double widths):
    cdef double x = exp(logx)
    cdef double T = mx / x
    cdef double tcs = __thermal_cs(x, mx, ms, gsxx, gsff, gsGG, gsFF,
                                   lam, vs, widths)
    cdef double pf = -(sqrt(M_PI / 45.0) * mpl * T *
                       __sqrt_gstar_interp(T) * tcs)
    cdef double weq = log(__yeq(T, mx))
    cdef np.ndarray[double] dw = np.array([0.0])

    dw[0] = pf * (exp(w[0]) - exp(2.0 * weq - w[0]))
    return dw

@cython.cdivision(True)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[double, ndim=2] __jacobian(double logx, np.ndarray[double] w,
                                           double mx, double ms, double gsxx,
                                           double gsff, double gsGG, double gsFF,
                                           double lam, double vs, double widths):
    cdef double x = exp(logx)
    cdef double T = mx / x
    cdef double tcs = __thermal_cs(x, mx, ms, gsxx, gsff, gsGG, gsFF,
                                   lam, vs, widths)
    cdef double pf = -(sqrt(M_PI / 45.0) * mpl * T *
                       __sqrt_gstar_interp(T) * tcs)
    cdef double weq = log(__yeq(T, mx))
    cdef np.ndarray[double, ndim=2] jac = np.zeros((1, 1), dtype=np.float64)

    jac[0, 0] = pf * (exp(w[0]) + exp(2.0 * weq - w[0]))
    return jac

def solve_boltzmann(model):
    boltzmann = lambda logx, w: __boltzmann(logx, w, model.mx, model.ms,
                                            model.gsxx, model.gsff, model.gsGG,
                                            model.gsFF, model.lam, model.vs,
                                            model.width_s)
    jacobian = lambda logx, w: __jacobian(logx, w, model.mx, model.ms,
                                          model.gsxx, model.gsff, model.gsGG,
                                          model.gsFF, model.lam, model.vs,
                                          model.width_s)

    logx_span = (log(1.0), log(1000.0))
    w_init = np.array([log(__yeq(model.mx, model.mx))])
    sol = solve_ivp(boltzmann, logx_span, w_init, jac=jacobian, method='Radau')

    return sol.t, sol.y

def relic_density(model):
    _, w = solve_boltzmann(model)
    # 1e-3 to convert to GeV
    return model.mx * 1e-3 * np.exp(w[0][-1]) * s_today / rho_crit
