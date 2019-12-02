import cython
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport k1, kn
from scipy.integrate import quad

from libc.math cimport M_PI, sqrt, atanh, atan, log

cdef double me = 0.510998928
cdef double mmu = 105.6583715
cdef double mpi0 = 134.9766
cdef double mpi = 139.57018
cdef double fpi = 92.2138
cdef double alpha_em = 1.0 / 137.04


@cython.cdivision(True)
cdef double __sigma_xx_to_v_to_ff(double e_cm, double mx, double mv,
                                  double gvxx, double gvll,
                                  double width_v, double mf):

    if e_cm < 2.0 * mf or e_cm < 2.0 * mx:
        return 0.0

    return (
        gvll ** 2
        * gvxx ** 2
        * (2 * mf ** 2 + e_cm ** 2)
        * sqrt((-4 * mf ** 2 + e_cm ** 2) / (-4 * mx ** 2 + e_cm ** 2))
        * (2 * mx ** 2 + e_cm ** 2)
    ) / (12.0 * M_PI * e_cm ** 2 * ((mv ** 2 - e_cm ** 2) ** 2 + mv ** 2 * width_v ** 2))


@cython.cdivision(True)
cdef double __sigma_xx_to_v_to_pipi(double e_cm, double mx, double mv,
                                    double gvxx, double gvuu, double gvdd,
                                    double gvss, double gvee, double gvmumu, double width_v):
    if e_cm < 2.0 * mx or e_cm < 2.0 * mpi:
        return 0.0

    return (
        (gvdd - gvuu) ** 2
        * gvxx ** 2
        * (-4 * mpi ** 2 + e_cm ** 2) ** 1.5
        * (2 * mx ** 2 + e_cm ** 2)
    ) / (
        48.0
        * M_PI
        * e_cm ** 2
        * sqrt(-4 * mx ** 2 + e_cm ** 2)
        * ((mv ** 2 - e_cm ** 2) ** 2 + mv ** 2 * width_v ** 2)
    )


@cython.cdivision(True)
cdef double __sigma_xx_to_v_to_pi0g(double e_cm, double mx, double mv,
                                    double gvxx, double gvuu, double gvdd,
                                    double gvss, double gvee, double gvmumu, double width_v):
    if e_cm < mpi0 or e_cm < 2.0 * mx:
        return 0.0

    return (
        alpha_em
        * (gvdd + 2 * gvuu) ** 2
        * gvxx ** 2
        * (-mpi0 ** 2 + e_cm ** 2) ** 3
        * (2 * mx ** 2 + e_cm ** 2)
    ) / (
        3456.0
        * fpi ** 2
        * M_PI ** 4
        * e_cm ** 3
        * sqrt(-4 * mx ** 2 + e_cm ** 2)
        * ((mv ** 2 - e_cm ** 2) ** 2 + mv ** 2 * width_v ** 2)
    )


# TODO: UPDATE THIS!
@cython.cdivision(True)
cdef double __sigma_xx_to_v_to_pi0v(double e_cm, double mx, double mv,
                                    double gvxx, double gvuu, double gvdd,
                                    double gvss, double gvee, double gvmumu, double width_v):
    if e_cm < mpi0 + mv or e_cm < 2.0 * mx:
        return 0.0


    return (
        (gvdd - gvuu) ** 2
        * (gvdd + gvuu) ** 2
        * gvxx ** 2
        * (
            (mpi0 - mv - e_cm)
            * (mpi0 + mv - e_cm)
            * (mpi0 - mv + e_cm)
            * (mpi0 + mv + e_cm)
        )
        ** 1.5
        * (2 * mx ** 2 + e_cm ** 2)
    ) / (
        1536.0
        * fpi ** 2
        * M_PI ** 5
        * e_cm ** 3
        * sqrt(-4 * mx ** 2 + e_cm ** 2)
        * ((mv ** 2 - e_cm ** 2) ** 2 + mv ** 2 * width_v ** 2)
    )


@cython.cdivision(True)
cdef double __sigma_xx_to_vv(double e_cm, double mx, double mv,
                             double gvxx, double gvuu, double gvdd,
                             double gvss, double gvee, double gvmumu,
                             double width_v):
    if e_cm < 2.0 * mv or e_cm < 2.0 * mx:
        return 0.0

    return (
        gvxx ** 4
        * (
            (
                -2
                * sqrt(-4 * mv ** 2 + e_cm ** 2)
                * sqrt(-4 * mx ** 2 + e_cm ** 2)
                * (2 * mv ** 4 + 4 * mx ** 4 + mx ** 2 * e_cm ** 2)
            )
            / (mv ** 4 - 4 * mv ** 2 * mx ** 2 + mx ** 2 * e_cm ** 2)
            + (
                2
                * (
                    4 * mv ** 4
                    - 8 * mv ** 2 * mx ** 2
                    - 8 * mx ** 4
                    + 4 * mx ** 2 * e_cm ** 2
                    + e_cm ** 4
                )
                * log(
                    (
                        -2 * mv ** 2
                        + e_cm ** 2
                        + sqrt(-4 * mv ** 2 + e_cm ** 2)
                        * sqrt(-4 * mx ** 2 + e_cm ** 2)
                    )
                    / (
                        -2 * mv ** 2
                        + e_cm ** 2
                        - sqrt(-4 * mv ** 2 + e_cm ** 2)
                        * sqrt(-4 * mx ** 2 + e_cm ** 2)
                    )
                )
            )
            / (-2 * mv ** 2 + e_cm ** 2)
        )
    ) / (16.0 * M_PI * e_cm ** 2 * (-4 * mx ** 2 + e_cm ** 2))


@cython.cdivision(True)
cdef double __sigma_xx_to_all(double e_cm, double mx, double mv,
                              double gvxx, double gvuu, double gvdd,
                              double gvss, double gvee, double gvmumu,
                              double width_v):

    cdef double sig_e = __sigma_xx_to_v_to_ff(
        e_cm, mx, mv, gvxx, gvee, width_v, me)
    cdef double sig_mu = __sigma_xx_to_v_to_ff(
        e_cm, mx, mv, gvxx, gvmumu, width_v, mmu)
    cdef double sig_pi = __sigma_xx_to_v_to_pipi(
        e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)
    cdef double sig_pi0g = __sigma_xx_to_v_to_pi0g(
        e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)
    cdef double sig_pi0v = __sigma_xx_to_v_to_pi0v(
        e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)
    cdef double sig_v = __sigma_xx_to_vv(
        e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)

    return sig_e + sig_mu + sig_pi + sig_pi0g + sig_pi0v + sig_v


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_v_to_ff(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double mv,
    double gvxx, double gvll, double width_v, double mf):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_v_to_ff(
            e_cms[i], mx, mv, gvxx, gvll, width_v, mf)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_v_to_pipi(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double mv,
    double gvxx, double gvuu, double gvdd, double gvss, double gvee,
    double gvmumu, double width_v):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_v_to_pipi(
            e_cms[i], mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu,
            width_v)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_v_to_pi0g(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double mv,
    double gvxx, double gvuu, double gvdd, double gvss, double gvee,
    double gvmumu, double width_v):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_v_to_pi0g(
            e_cms[i], mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu,
            width_v)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_v_to_pi0v(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double mv,
    double gvxx, double gvuu, double gvdd, double gvss, double gvee,
    double gvmumu, double width_v):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_v_to_pi0v(
            e_cms[i], mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu,
            width_v)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_vv(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double mv,
    double gvxx, double gvuu, double gvdd, double gvss, double gvee,
    double gvmumu, double width_v):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_vv(
            e_cms[i], mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu,
            width_v)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_all(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double mv,
    double gvxx, double gvuu, double gvdd, double gvss, double gvee,
    double gvmumu, double width_v):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = sigma_xx_to_all(
            e_cms[i], mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu,
            width_v)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_v_to_ff(e_cms, double mx, double mv, double gvxx,
                        double gvll, double width_v, double ml):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of leptons, *f* through a vector mediator in
    the s-channel.

    Parameters
    ----------
    e_cms : double or array-like
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to vector mediator.
    gvll: double
        Coupling of vector mediator to lepton.
    ml: double
        Final state lepton mass.

    Returns
    -------
    sigma : double
        Cross section for x + x -> v* -> f + f.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_v_to_ff(
            np.array(e_cms), mx, mv, gvxx, gvll, width_v, ml)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_v_to_ff(
        e_cm, mx, mv, gvxx, gvll, width_v, ml)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_v_to_pipi(e_cms, double mx, double mv, double gvxx,
                          double gvuu, double gvdd, double gvss,
                          double gvee, double gvmumu, double width_v):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of charged pions through a vector mediator in
    the s-channel.

    Parameters
    ----------
    e_cms : double or array-like
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to vector mediator.
    gvuu: double
        Coupling of vector mediator to up-quarks.
    gvdd: double
        Coupling of vector mediator to down-quarks.
    gvss: double
        Coupling of vector mediator to strange-quarks.
    gvee: double
        Coupling of vector mediator to electrons.
    gvmumu: double
        Coupling of vector mediator to muons.
    width_v: double
        Full decay width of the vector mediator.
    mf: double
        Final state fermion mass.

    Returns
    -------
    sigma : double
        Cross section for x + x -> v* -> pi + pi.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_v_to_pipi(
            np.array(e_cms), mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
            gvmumu, width_v)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_v_to_pipi(
        e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_v_to_pi0g(e_cms, double mx, double mv, double gvxx,
                          double gvuu, double gvdd, double gvss,
                          double gvee, double gvmumu, double width_v):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a neutral pion and photon through a vector mediator
    in the s-channel.

    Parameters
    ----------
    e_cms : double or array-like
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to vector mediator.
    gvuu: double
        Coupling of vector mediator to up-quarks.
    gvdd: double
        Coupling of vector mediator to down-quarks.
    gvss: double
        Coupling of vector mediator to strange-quarks.
    gvee: double
        Coupling of vector mediator to electrons.
    gvmumu: double
        Coupling of vector mediator to muons.
    width_v: double
        Full decay width of the vector mediator.
    mf: double
        Final state fermion mass.

    Returns
    -------
    sigma : double
        Cross section for x + x -> v* -> pi0 + g.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_v_to_pi0g(
            np.array(e_cms), mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
            gvmumu, width_v)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_v_to_pi0g(
        e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
        gvmumu, width_v)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_v_to_pi0v(e_cms, double mx, double mv, double gvxx,
                          double gvuu, double gvdd, double gvss,
                          double gvee, double gvmumu, double width_v):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a neutral pion and vector mediator through a
    vector mediator in the s-channel.

    Parameters
    ----------
    e_cms : double or array-like
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to vector mediator.
    gvuu: double
        Coupling of vector mediator to up-quarks.
    gvdd: double
        Coupling of vector mediator to down-quarks.
    gvss: double
        Coupling of vector mediator to strange-quarks.
    gvee: double
        Coupling of vector mediator to electrons.
    gvmumu: double
        Coupling of vector mediator to muons.
    width_v: double
        Full decay width of the vector mediator.

    Returns
    -------
    sigma : double
        Cross section for x + x -> v* -> v + pi0.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_v_to_pi0v(
            np.array(e_cms), mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
            gvmumu, width_v)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_v_to_pi0v(
        e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
        gvmumu, width_v)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_vv(e_cms, double mx, double mv, double gvxx,
                   double gvuu, double gvdd, double gvss,
                   double gvee, double gvmumu, double width_v):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of vector mediators through the t and u
    channels.

    Parameters
    ----------
    e_cms : double or array-like
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to vector mediator.
    gvuu: double
        Coupling of vector mediator to up-quarks.
    gvdd: double
        Coupling of vector mediator to down-quarks.
    gvss: double
        Coupling of vector mediator to strange-quarks.
    gvee: double
        Coupling of vector mediator to electrons.
    gvmumu: double
        Coupling of vector mediator to muons.
    width_v: double
        Full decay width of the vector mediator.

    Returns
    -------
    sigma : double
        Cross section for x + x -> v + v.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_vv(
            np.array(e_cms), mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
            gvmumu, width_v)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_vv(e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                            gvmumu, width_v)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_all(e_cms, double mx, double mv, double gvxx,
                    double gvuu, double gvdd, double gvss,
                    double gvee, double gvmumu, double width_v):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into all availible final states.

    Parameters
    ----------
    e_cms : double or array-like
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to vector mediator.
    gvuu: double
        Coupling of vector mediator to up-quarks.
    gvdd: double
        Coupling of vector mediator to down-quarks.
    gvss: double
        Coupling of vector mediator to strange-quarks.
    gvee: double
        Coupling of vector mediator to electrons.
    gvmumu: double
        Coupling of vector mediator to muons.
    width_v: double
        Full decay width of the vector mediator.

    Returns
    -------
    sigma : double
        Cross section for x + x -> anything except x + x.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_all(
            np.array(e_cms), mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
            gvmumu, width_v)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_all(
        e_cm, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v)


@cython.cdivision(True)
cdef double __thermal_cross_section_integrand(
    double z, double x, double mx, double mv, double gvxx,
    double gvuu, double gvdd, double gvss,
    double gvee, double gvmumu, double width_v):

    cdef double sig = __sigma_xx_to_all(
        mx * z, mx, mv, gvxx, gvuu, gvdd, gvss, gvee, gvmumu, width_v
        )
    return sig * z**2 * (z**2 - 4.0) * k1(x * z)


@cython.cdivision(True)
def thermal_cross_section(double x, double mx, double mv, double gvxx,
                          double gvuu, double gvdd, double gvss,
                          double gvee, double gvmumu, double width_v):
    """
    Compute the thermally average cross section for scalar mediator model.

    Parameters
    ----------
    x: float
        Mass of the dark matter divided by its temperature.
    mx: double
        Dark matter mass
    mv: double
        Vector mediator mass.
    gvxx: double
        Coupling of DM to vector mediator.
    gvuu: double
        Coupling of vector mediator to up-quarks.
    gvdd: double
        Coupling of vector mediator to down-quarks.
    gvss: double
        Coupling of vector mediator to strange-quarks.
    gvee: double
        Coupling of vector mediator to electrons.
    gvmumu: double
        Coupling of vector mediator to muons.
    width_v: double
        Full decay width of the vector mediator.

    Returns
    -------
    tcs: float
        Thermally average cross section.
    """
    # If x is really large, we will get divide by zero errors
    # we clip x since the thermal cross section should tend
    # to a constant.
    # TODO: Probably should compute the asymptotic form of the
    # thermal cross section.
    cdef double xnew = x if x < 300.0 else 300
    cdef double pf = xnew / (2.0 * kn(2, xnew))**2

    # points at which integrand may have trouble are:
    #   1. endpoint
    #   2. when ss final state is accessible => z = 2 mv / mx
    #   3. when we hit mediator resonance => z = mv / mx
    return pf * quad(__thermal_cross_section_integrand, 2.0,
                     max(50.0 / xnew, 150.0),
                     args=(xnew, mx, mv, gvxx, gvuu, gvdd, gvss, gvee,
                           gvmumu, width_v),
                     points=[2.0, mv / mx, 2.0 * mv / mx])[0]
