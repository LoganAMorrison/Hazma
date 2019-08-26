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
cdef double fpi = 92.2138;

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
cdef double __sigmav_xx_to_v_to_mumu(double eps, double mx, double mv,
                                     double gvxx, double gvuu, double gvdd,
                                     double gvss, double gvee, double gvmumu,
                                     double widthv):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of muons through a vector mediator in
    the s-channel.
    
    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to the vector mediator.
    gvuu: double
        Coupling of up-quark to vector mediator.
    gvdd: double
        Coupling of down-quark to vector mediator.
    gvss: double
        Coupling of strange-quark to vector mediator.
    gvee: double
        Coupling of electron to vector mediator.
    gvmumu: double
        Coupling of muon to vector mediator.
    widthv: double
        Deccay width of the vector mediator. 
    
    Returns
    -------
    sigma : float
        Cross section for x + x -> v* -> mu + mu.

    """
    cdef double r1 = mmu / mx

    if 1 + eps >= r1 * r1:
        return (((3 + 2 * eps) * gvmumu**2 * gvxx**2 * mx**2 *
                 (2 + 2 * eps + r1**2) * sqrt(1 - r1**2 / (1 + eps))) / (
                        6. * (1 + 2 * eps) * M_PI * (
                        mv**4 + 16 * (1 + eps)**2 * mx**4 + mv**2 *
                        (-8 * (1 + eps) * mx**2 + widthv**2))))
    else:
        return 0.0

cdef double __sigmav_xx_to_v_to_ee(double eps, double mx, double mv,
                                   double gvxx, double gvuu, double gvdd,
                                   double gvss, double gvee, double gvmumu,
                                   double widthv):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of electrons through a vector mediator in
    the s-channel.
    
    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to the vector mediator.
    gvuu: double
        Coupling of up-quark to vector mediator.
    gvdd: double
        Coupling of down-quark to vector mediator.
    gvss: double
        Coupling of strange-quark to vector mediator.
    gvee: double
        Coupling of electron to vector mediator.
    gvmumu: double
        Coupling of muon to vector mediator.
    widthv: double
        Deccay width of the vector mediator. 
    
    Returns
    -------
    sigma : float
        Cross section for x + x -> v* -> e + e.

    """
    cdef double r1 = me / mx

    if 1 + eps >= r1 * r1:
        return (((3 + 2 * eps) * gvee**2 * gvxx**2 * mx**2 *
                 (2 + 2 * eps + r1**2) * sqrt(1 - r1**2 / (1 + eps))) / (
                        6. * (1 + 2 * eps) * M_PI * (
                        mv**4 + 16 * (1 + eps)**2 * mx**4 + mv**2 *
                        (-8 * (1 + eps) * mx**2 + widthv**2))))
    else:
        return 0.0

@cython.cdivision(True)
cdef double __sigmav_xx_to_v_to_pi0v(double eps, double mx, double mv,
                                     double gvxx, double gvuu, double gvdd,
                                     double gvss, double gvee, double gvmumu,
                                     double widthv):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of photons through a vector mediator in the
    s-channel.

    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to the vector mediator.
    gvuu: double
        Coupling of up-quark to vector mediator.
    gvdd: double
        Coupling of down-quark to vector mediator.
    gvss: double
        Coupling of strange-quark to vector mediator.
    gvee: double
        Coupling of electron to vector mediator.
    gvmumu: double
        Coupling of muon to vector mediator.
    widthv: double
        Deccay width of the vector mediator. 

    Returns
    -------
    sigma : float
        Cross section for x + x -> v* -> g + g.
    """
    cdef double r1 = mpi0 / mx
    cdef double r2 = mv / mx
    if 1 + eps >= (r1 * r1 + r2 * r2) / 4.0:

        return (((gvdd - gvuu)**2 * (gvdd + gvuu)**2 * gvxx**2 * mx**2 *
                 sqrt(4 * (1 + eps) - (r1 - r2)**2) * sqrt(
                    4 * (1 + eps) - (r1 + r2)**2) *
                 (-3 * r1**4 * (-6 - 4 * eps + r2**2) + 4 * (1 + eps) * (
                         16 * (1 + eps) * (3 + 2 * eps) -
                         12 * (1 + eps) * r2**2 + 3 * r2**4) - r1**2 * (
                          16 * (1 + eps) * (6 + 5 * eps) - 12 * (
                          1 + eps) * r2**2 + 3 * r2**4))) / (
                        24576. * (1 + eps) *
                        (1 + 2 * eps) * fpi**2 * M_PI**5 * (
                                mx**2 * (-4 * (1 + eps) + r2**2)**2 +
                                r2**2 * widthv**2)))
    else:
        return 0.0

@cython.cdivision(True)
cdef double __sigmav_xx_to_v_to_pi0g(double eps, double mx, double mv,
                                     double gvxx, double gvuu, double gvdd,
                                     double gvss, double gvee, double gvmumu,
                                     double widthv):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of neutral pion through a vector mediator in
    the s-channel.

    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to the vector mediator.
    gvuu: double
        Coupling of up-quark to vector mediator.
    gvdd: double
        Coupling of down-quark to vector mediator.
    gvss: double
        Coupling of strange-quark to vector mediator.
    gvee: double
        Coupling of electron to vector mediator.
    gvmumu: double
        Coupling of muon to vector mediator.
    widthv: double
        Deccay width of the vector mediator. 

    Returns
    -------
    sigma : float
        Cross section for x + x -> v* -> pi0 + pi0.
    """
    cdef double r1 = mpi0 / mx
    cdef double r2 = 0.0
    if 1 + eps >= (r1 * r1 + r2 * r2) / 4.0:

        return (-((gvdd + 2 * gvuu)**2 * gvxx**2 * mx**4 * qe**2 *
                  (-4 * (1 + eps) + r1**2) * (
                          32 * (1 + eps)**2 * (3 + 2 * eps) - 8 *
                          (6 + 11 * eps + 5 * eps**2) * r1**2 + (
                                  9 + 6 * eps) * r1**4)) /
                (110592. * (1 + eps) * (1 + 2 * eps) * fpi**2 * M_PI**5 * (
                        mv**4 + 16 * (1 + eps)**2 * mx**4 + mv**2 *
                        (-8 * (1 + eps) * mx**2 + widthv**2))))
    else:
        return 0.0

@cython.cdivision(True)
cdef double __sigmav_xx_to_v_to_pipi(double eps, double mx, double mv,
                                     double gvxx, double gvuu, double gvdd,
                                     double gvss, double gvee, double gvmumu,
                                     double widthv):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of charged pions through a vector mediator in
    the s-channel.

    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to the vector mediator.
    gvuu: double
        Coupling of up-quark to vector mediator.
    gvdd: double
        Coupling of down-quark to vector mediator.
    gvss: double
        Coupling of strange-quark to vector mediator.
    gvee: double
        Coupling of electron to vector mediator.
    gvmumu: double
        Coupling of muon to vector mediator.
    widthv: double
        Deccay width of the vector mediator. 

    Returns
    -------
    sigma : float
        Cross section for x + x -> v* -> M_PI + M_PI.
    """
    cdef double r1 = mpi / mx

    if r1**2 < 1 + eps:

        return (((3 + 2 * eps) * (gvdd - gvuu)**2 * gvxx**2 *
                 mx**2 * (1 + eps - r1**2)**1.5) / (
                        12. * sqrt(1 + eps) * (1 + 2 * eps) * M_PI * (
                        mv**4 + 16 * (1 + eps)**2 * mx**4 +
                        mv**2 * (-8 * (1 + eps) * mx**2 + widthv**2))))
    else:
        return 0.

@cython.cdivision(True)
cdef double __sigmav_xx_to_vv(double eps, double mx, double mv, double gvxx,
                              double gvuu, double gvdd, double gvss,
                              double gvee, double gvmumu, double widthv):
    """Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of vector mediator through the t and u
    channels.

    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to the vector mediator.
    gvuu: double
        Coupling of up-quark to vector mediator.
    gvdd: double
        Coupling of down-quark to vector mediator.
    gvss: double
        Coupling of strange-quark to vector mediator.
    gvee: double
        Coupling of electron to vector mediator.
    gvmumu: double
        Coupling of muon to vector mediator.
    widthv: double
        Deccay width of the vector mediator. 

    Returns
    -------
    sigma : float
        Cross section for x + x -> s + s.
    """
    cdef double r1 = mv / mx

    if r1 * r1 < 1 + eps:

        return ((gvxx**4 * sqrt(1 - r1**2 / (1 + eps)) * (
                (-4 * (4 + 2 * eps + r1**4)) /
                (4 * eps + (-2 + r1**2)**2) - (
                        2 * (6 + 4 * eps * (3 + eps) - 2 * r1**2 + r1**4) *
                        atanh(
                            (2 * sqrt(eps * (1 + eps - r1**2))) /
                            (-2 * (1 + eps) + r1**2))) / (
                        sqrt(eps * (1 + eps - r1**2)) *
                        (2 + 2 * eps - r1**2)))) / (
                        16. * mx**2 * (M_PI + 2 * eps * M_PI)))
    else:
        return 0.

@cython.cdivision(True)
cdef double __integrand_thermal_cs(double eps, double x,
                                   double mx, double mv, double gvxx,
                                   double gvuu, double gvdd, double gvss,
                                   double gvee, double gvmumu, double widthv):
    """
    Return the integrand of the thermal cross section.
    
    Parameters
    ----------
    eps: double
        (s - 4mx^2) / (4mx^2).
    mx: double
        Dark matter mass.
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to the vector mediator.
    gvuu: double
        Coupling of up-quark to vector mediator.
    gvdd: double
        Coupling of down-quark to vector mediator.
    gvss: double
        Coupling of strange-quark to vector mediator.
    gvee: double
        Coupling of electron to vector mediator.
    gvmumu: double
        Coupling of muon to vector mediator.
    widthv: double
        Deccay width of the vector mediator. 

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
            __sigmav_xx_to_vv(eps, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                              gvmumu, widthv) +
            __sigmav_xx_to_v_to_pipi(eps, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                                     gvmumu, widthv) +
            __sigmav_xx_to_v_to_pi0g(eps, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                                     gvmumu, widthv) +
            __sigmav_xx_to_v_to_pi0v(eps, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                                     gvmumu, widthv) +
            __sigmav_xx_to_v_to_mumu(eps, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                                     gvmumu, widthv) +
            __sigmav_xx_to_v_to_ee(eps, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                                   gvmumu, widthv))

    if np.isfinite(kernal * tcs):
        return kernal * tcs
    else:
        return 0.0

@cython.cdivision(True)
cdef double __thermal_cs(double x, double mx, double mv, double gvxx,
                         double gvuu, double gvdd, double gvss,
                         double gvee, double gvmumu, double widthv):
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
        args=(x, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, widthv))[0]

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
                                    double mx, double mv, double gvxx,
                                    double gvuu, double gvdd, double gvss,
                                    double gvee, double gvmumu, double widthv):
    cdef double x = exp(logx)
    cdef double T = mx / x
    cdef double tcs = __thermal_cs(x, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                                   gvmumu, widthv)
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
                                           double mx, double mv, double gvxx,
                                           double gvuu, double gvdd, double gvss,
                                           double gvee, double gvmumu, double widthv):
    cdef double x = exp(logx)
    cdef double T = mx / x
    cdef double tcs = __thermal_cs(x, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                                   gvmumu, widthv)
    cdef double pf = -(sqrt(M_PI / 45.0) * mpl * T *
                       __sqrt_gstar_interp(T) * tcs)
    cdef double weq = log(__yeq(T, mx))
    cdef np.ndarray[double, ndim=2] jac = np.zeros((1, 1), dtype=np.float64)

    jac[0, 0] = pf * (exp(w[0]) + exp(2.0 * weq - w[0]))
    return jac

def solve_boltzmann(model):
    boltzmann = lambda logx, w: __boltzmann(logx, w, model.mx, model.mv,
                                            model.gvxx, model.gvuu, model.gvdd,
                                            model.gvss, model.gvee,
                                            model.gvmumu, model.width_v)
    jacobian = lambda logx, w: __jacobian(logx, w, model.mx, model.mv,
                                          model.gvxx, model.gvuu, model.gvdd,
                                          model.gvss, model.gvee,
                                          model.gvmumu, model.width_v)

    logx_span = (log(1.0), log(1000.0))
    w_init = np.array([log(__yeq(model.mx, model.mx))])
    sol = solve_ivp(boltzmann, logx_span, w_init, jac=jacobian, method='Radau')

    return sol.t, sol.y

def relic_density(model):
    _, w = solve_boltzmann(model)
    # 1e-3 to convert to GeV
    return model.mx * 1e-3 * np.exp(w[0][-1]) * s_today / rho_crit
