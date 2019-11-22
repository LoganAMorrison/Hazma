import cython
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport k1, kn
from scipy.integrate import quad

from libc.math cimport M_PI, sqrt, atanh, atan, log

cdef double vh = 246.22795e3
cdef double alpha_em = 1.0 / 137.04
cdef double me = 0.510998928
cdef double mmu = 105.6583715
cdef double mpi0 = 134.9766
cdef double mpi = 139.57018
cdef double b0 = 2654.082197477761
cdef double muq = 2.3
cdef double mdq = 4.8

@cython.cdivision(True)
cdef double __sigma_xx_to_s_to_ff(double e_cm, double mx, double ms,
                                  double gsxx, double gsff, double gsGG,
                                  double gsFF, double lam, double width_s,
                                  double vs, double mf):
    if e_cm < 2.0 * mf or e_cm < 2.0 * mx:
        return 0.0

    return (
        gsff ** 2
        * gsxx ** 2
        * mf ** 2
        * (-4 * mf ** 2 + e_cm ** 2) ** 1.5
        * sqrt(-4 * mx ** 2 + e_cm ** 2)
    ) / (
        16.0
        * M_PI
        * e_cm ** 2
        * vh ** 2
        * ((ms ** 2 - e_cm ** 2) ** 2 + ms ** 2 * width_s ** 2)
    )


@cython.cdivision(True)
cdef double __sigma_xx_to_s_to_gg(double e_cm, double mx, double ms,
                                  double gsxx, double gsff, double gsGG,
                                  double gsFF, double lam, double width_s,
                                  double vs):
    if e_cm < 2.0 * mx:
        return 0.0

    return (
        alpha_em ** 2
        * gsFF ** 2
        * gsxx ** 2
        * e_cm ** 3
        * sqrt(-4 * mx ** 2 + e_cm ** 2)
    ) / (
        128.0
        * lam ** 2
        * M_PI ** 3
        * ((ms ** 2 - e_cm ** 2) ** 2 + ms ** 2 * width_s ** 2)
    )


@cython.cdivision(True)
cdef double __sigma_xx_to_s_to_pi0pi0(double e_cm, double mx, double ms,
                               double gsxx, double gsff, double gsGG,
                               double gsFF, double lam, double width_s,
                               double vs):
    if e_cm < 2.0 * mpi0 or e_cm < 2.0 * mx:
        return 0.0

    return (
        gsxx ** 2
        * sqrt((-4 * mpi0 ** 2 + e_cm ** 2) * (-4 * mx ** 2 + e_cm ** 2))
        * (
            162 * gsGG * lam ** 3 * (2 * mpi0 ** 2 - e_cm ** 2) * vh ** 2
            + b0
            * (mdq + muq)
            * (9 * lam + 4 * gsGG * vs)
            * (-3 * lam * vh + 3 * gsff * lam * vs + 2 * gsGG * vh * vs)
            * (
                2 * gsGG * vh * (9 * lam - 4 * gsGG * vs)
                + 9 * gsff * lam * (3 * lam + 4 * gsGG * vs)
            )
        )
        ** 2
    ) / (
        419904.0
        * lam ** 6
        * M_PI
        * e_cm ** 2
        * vh ** 4
        * (9 * lam + 4 * gsGG * vs) ** 2
        * ((ms ** 2 - e_cm ** 2) ** 2 + ms ** 2 * width_s ** 2)
    )


@cython.cdivision(True)
cdef double __sigma_xx_to_s_to_pipi(double e_cm, double mx, double ms,
                             double gsxx, double gsff, double gsGG,
                             double gsFF, double lam, double width_s,
                             double vs):
    if e_cm < 2.0 * mpi or e_cm < 2.0 * mx:
        return 0.0

    return (
        gsxx ** 2
        * sqrt((-4 * mpi ** 2 + e_cm ** 2) * (-4 * mx ** 2 + e_cm ** 2))
        * (
            162 * gsGG * lam ** 3 * (2 * mpi ** 2 - e_cm ** 2) * vh ** 2
            + b0
            * (mdq + muq)
            * (9 * lam + 4 * gsGG * vs)
            * (-3 * lam * vh + 3 * gsff * lam * vs + 2 * gsGG * vh * vs)
            * (
                2 * gsGG * vh * (9 * lam - 4 * gsGG * vs)
                + 9 * gsff * lam * (3 * lam + 4 * gsGG * vs)
            )
        )
        ** 2
    ) / (
        209952.0
        * lam ** 6
        * M_PI
        * e_cm ** 2
        * vh ** 4
        * (9 * lam + 4 * gsGG * vs) ** 2
        * ((ms ** 2 - e_cm ** 2) ** 2 + ms ** 2 * width_s ** 2)
    )



@cython.cdivision(True)
cdef double __sigma_xx_to_ss(double e_cm, double mx, double ms, double gsxx,
                      double gsff, double gsGG, double gsFF, double lam,
                      double width_s, double vs):
    if e_cm < 2.0 * ms or e_cm < 2.0 * mx:
        return 0.0

    return (
        gsxx ** 4
        * (
            -(
                (
                    sqrt((-4 * ms ** 2 + e_cm ** 2) * (-4 * mx ** 2 + e_cm ** 2))
                    * (
                        3 * ms ** 4
                        - 16 * ms ** 2 * mx ** 2
                        + 2 * mx ** 2 * (8 * mx ** 2 + e_cm ** 2)
                    )
                )
                / (ms ** 4 - 4 * ms ** 2 * mx ** 2 + mx ** 2 * e_cm ** 2)
            )
            + (
                (
                    6 * ms ** 4
                    - 32 * mx ** 4
                    + 16 * mx ** 2 * e_cm ** 2
                    + e_cm ** 4
                    - 4 * ms ** 2 * (4 * mx ** 2 + e_cm ** 2)
                )
                * log(
                    (
                        -2 * ms ** 2
                        + e_cm ** 2
                        + sqrt((-4 * ms ** 2 + e_cm ** 2) * (-4 * mx ** 2 + e_cm ** 2))
                    )
                    / (
                        -2 * ms ** 2
                        + e_cm ** 2
                        - sqrt((-4 * ms ** 2 + e_cm ** 2) * (-4 * mx ** 2 + e_cm ** 2))
                    )
                )
            )
            / (-2 * ms ** 2 + e_cm ** 2)
        )
    ) / (32.0 * M_PI * e_cm ** 2 * (-4 * mx ** 2 + e_cm ** 2))


@cython.cdivision(True)
cdef double __sigma_xx_to_all(double e_cm, double mx, double ms, double gsxx,
                              double gsff, double gsGG, double gsFF, double lam,
                              double width_s, double vs):

    cdef double sig_e = __sigma_xx_to_s_to_ff(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs, me)
    cdef double sig_mu = __sigma_xx_to_s_to_ff(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs, mmu)
    cdef double sig_g = __sigma_xx_to_s_to_gg(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)
    cdef double sig_pi0 = __sigma_xx_to_s_to_pi0pi0(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)
    cdef double sig_pi = __sigma_xx_to_s_to_pipi(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)
    cdef double sig_s = __sigma_xx_to_ss(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs)

    return sig_e + sig_mu + sig_g + sig_pi0 + sig_pi + sig_s


@cython.cdivision(True)
cdef double __sigma_xl_to_xl(double e_cm, double mx, double ms, double gsxx,
                             double gsff, double gsGG, double gsFF,
                             double lam, double width_s, double vs, double ml):

    if e_cm < mx + ml:
        return 0.0

    cdef double s = e_cm**2

    return 2.0 * (
        ((gsff * ml)**2 * gsxx**2 *
         ((-4 * ml**2 * (ms**2 - 4 * mx**2) + ms**2 *
           (ms**2 - 4 * mx**2 - width_s**2)) *
          atan(ms / width_s) +
          (4 * ml**2 * (ms**2 - 4 * mx**2) + ms**2 *
           (-ms**2 + 4 * mx**2 + width_s**2)) *
          atan((ms**2 - 4 * mx**2 + s) / (ms * width_s)) +
          ms * width_s *
          (4 * mx**2 - s + ms**2 * np.log(4) - ml**2 * log(16) - mx**2 *
           log(16) + (2 * ml**2 - ms**2 + 2 * mx**2) *
           log(4 * ms**2 * (ms**2 + width_s**2)) +
           (-2 * ml**2 + ms**2 - 2 * mx**2) *
           log(ms**4 + (-4 * mx**2 + s)**2 +
               ms**2 * (-8 * mx**2 + 2 * s + width_s**2))))) /
        (32.0 * ms * M_PI * e_cm**2 * (4 * mx**2 - s) * width_s))


@cython.cdivision(True)
cdef double __sigma_xpi_to_xpi(double e_cm, double mx, double ms, double gsxx,
                               double gsff, double gsGG, double gsFF,
                               double lam, double width_s, double vs):

    if e_cm < mx + mpi:
        return 0.0

    return 2.0 * (
        (gsxx**2 *
         (2 *
          (b0**2 * (mdq + muq)**2 * (ms**2 - 4 * mx**2) *
           (9 * lam + 4 * gsGG * vs)**2 *
           (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
            2 * gsGG * vh**2 *
            (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
            gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
           324 * b0 * gsGG * lam**3 * (mdq + muq) * vh**2 *
           (9 * lam + 4 * gsGG * vs) *
           (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
            2 * gsGG * vh**2 * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) + gsff *
            (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) *
           (2 * mpi**2 * (ms**2 - 4 * mx**2) + ms**2 *
            (-ms**2 + 4 * mx**2 + width_s**2)) +
           26244 * gsGG**2 * lam**6 * vh**4 *
           (ms**6 + 4 * mpi**4 * (ms**2 - 4 * mx**2) +
            4 * ms**2 * mx**2 * width_s**2 - 4 * mpi**2 * ms**2 *
            (ms**2 - 4 * mx**2 - width_s**2) -
            ms**4 * (4 * mx**2 + 3 * width_s**2))) * atan(ms / width_s) -
          2 * (b0**2 * (mdq + muq)**2 * (ms**2 - 4 * mx**2) *
               (9 * lam + 4 * gsGG * vs)**2 *
               (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                2 * gsGG * vh**2 *
                (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                gsff * (-81 * lam**3 * vh +
                        48 * gsGG**2 * lam * vh * vs**2))**2 +
               324 * b0 * gsGG * lam**3 * (mdq + muq) * vh**2 *
               (9 * lam + 4 * gsGG * vs) *
               (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                2 * gsGG * vh**2 *
                (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) *
               (2 * mpi**2 * (ms**2 - 4 * mx**2) +
                ms**2 * (-ms**2 + 4 * mx**2 + width_s**2)) +
               26244 * gsGG**2 * lam**6 * vh**4 *
               (ms**6 + 4 * mpi**4 * (ms**2 - 4 * mx**2) +
                4 * ms**2 * mx**2 * width_s**2 - 4 * mpi**2 * ms**2 *
                (ms**2 - 4 * mx**2 - width_s**2) -
                ms**4 * (4 * mx**2 + 3 * width_s**2))) *
          atan((ms**2 - 4 * mx**2 + e_cm**2) / (ms * width_s)) +
          ms * width_s *
          (-324 * gsGG * lam**3 * (4 * mx**2 - e_cm**2) * vh**2 *
           (81 * gsGG * lam**3 *
            (8 * mpi**2 - 4 * ms**2 + 4 * mx**2 + e_cm**2) * vh**2 +
            2 * b0 * (mdq + muq) * (9 * lam + 4 * gsGG * vs) *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
             gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))) -
           (648 * b0 * gsGG * lam**3 * (mdq + muq) *
            (mpi**2 - ms**2 + 2 * mx**2) * vh**2 *
            (9 * lam + 4 * gsGG * vs) *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
             gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) +
            b0**2 * (mdq + muq)**2 * (9 * lam + 4 * gsGG * vs)**2 *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs +
              8 * gsGG**2 * vs**2) + gsff *
             (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
            26244 * gsGG**2 * lam**6 * vh**4 *
            (4 * mpi**4 - 8 * mpi**2 * (ms**2 - 2 * mx**2) + ms**2 *
             (3 * ms**2 - 8 * mx**2 - width_s**2))) *
           log(ms**2 * (ms**2 + width_s**2)) +
           (648 * b0 * gsGG * lam**3 * (mdq + muq) *
            (mpi**2 - ms**2 + 2 * mx**2) * vh**2 *
            (9 * lam + 4 * gsGG * vs) *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs +
              8 * gsGG**2 * vs**2) + gsff *
             (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) +
            b0**2 * (mdq + muq)**2 * (9 * lam + 4 * gsGG * vs)**2 *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
             gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
            26244 * gsGG**2 * lam**6 * vh**4 *
            (4 * mpi**4 - 8 * mpi**2 * (ms**2 - 2 * mx**2) +
             ms**2 * (3 * ms**2 - 8 * mx**2 - width_s**2))) *
           log(ms**4 + (-4 * mx**2 + e_cm**2)**2 +
               ms**2 * (-8 * mx**2 + 2 * e_cm**2 + width_s**2))))) /
        (419904. * lam**6 * ms * M_PI * e_cm**2 *
         (-4 * mx**2 + e_cm**2) * vh**4 *
         (9 * lam + 4 * gsGG * vs)**2 * width_s))


@cython.cdivision(True)
cdef double __sigma_xpi0_to_xpi0(double e_cm, double mx, double ms, double gsxx,
                                 double gsff, double gsGG, double gsFF,
                                 double lam, double width_s, double vs):

    if e_cm < mx + mpi0:
        return 0.0

    return (
        (gsxx**2 *
         (2 *
          (b0**2 * (mdq + muq)**2 * (ms**2 - 4 * mx**2) *
           (9 * lam + 4 * gsGG * vs)**2 *
           (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
            2 * gsGG * vh**2 *
            (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
            gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
           324 * b0 * gsGG * lam**3 * (mdq + muq) * vh**2 *
           (9 * lam + 4 * gsGG * vs) *
           (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
            2 * gsGG * vh**2 *
            (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
            gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) *
           (2 * mpi0**2 * (ms**2 - 4 * mx**2) + ms**2 *
            (-ms**2 + 4 * mx**2 + width_s**2)) +
           26244 * gsGG**2 * lam**6 * vh**4 *
           (ms**6 + 4 * mpi0**4 * (ms**2 - 4 * mx**2) +
            4 * ms**2 * mx**2 * width_s**2 - 4 * mpi0**2 * ms**2 *
            (ms**2 - 4 * mx**2 - width_s**2) -
            ms**4 * (4 * mx**2 + 3 * width_s**2))) * atan(ms / width_s) -
          2 * (b0**2 * (mdq + muq)**2 * (ms**2 - 4 * mx**2) *
               (9 * lam + 4 * gsGG * vs)**2 *
               (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                2 * gsGG * vh**2 *
                (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                gsff *
                (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
               324 * b0 * gsGG * lam**3 * (mdq + muq) * vh**2 *
               (9 * lam + 4 * gsGG * vs) *
               (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
                2 * gsGG * vh**2 *
                (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
                gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) *
               (2 * mpi0**2 * (ms**2 - 4 * mx**2) + ms**2 *
                (-ms**2 + 4 * mx**2 + width_s**2)) +
               26244 * gsGG**2 * lam**6 * vh**4 *
               (ms**6 + 4 * mpi0**4 * (ms**2 - 4 * mx**2) +
                4 * ms**2 * mx**2 * width_s**2 - 4 * mpi0**2 * ms**2 *
                (ms**2 - 4 * mx**2 - width_s**2) -
                ms**4 * (4 * mx**2 + 3 * width_s**2))) *
          atan((ms**2 - 4 * mx**2 + e_cm**2) / (ms * width_s)) +
          ms * width_s *
          (-324 * gsGG * lam**3 * (4 * mx**2 - e_cm**2) * vh**2 *
           (81 * gsGG * lam**3 *
            (8 * mpi0**2 - 4 * ms**2 + 4 * mx**2 + e_cm**2) * vh**2 +
            2 * b0 * (mdq + muq) * (9 * lam + 4 * gsGG * vs) *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
             gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))) -
           (648 * b0 * gsGG * lam**3 * (mdq + muq) *
            (mpi0**2 - ms**2 + 2 * mx**2) * vh**2 *
            (9 * lam + 4 * gsGG * vs) *
            (27 * gsff**2 * lam**2 * vs *
             (3 * lam + 4 * gsGG * vs) - 2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
             gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) +
            b0**2 * (mdq + muq)**2 * (9 * lam + 4 * gsGG * vs)**2 *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
             gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
            26244 * gsGG**2 * lam**6 * vh**4 *
            (4 * mpi0**4 - 8 * mpi0**2 * (ms**2 - 2 * mx**2) +
             ms**2 * (3 * ms**2 - 8 * mx**2 - width_s**2))) *
           log(ms**2 * (ms**2 + width_s**2)) +
           (648 * b0 * gsGG * lam**3 * (mdq + muq) *
            (mpi0**2 - ms**2 + 2 * mx**2) * vh**2 *
            (9 * lam + 4 * gsGG * vs) *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
             gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)) +
            b0**2 * (mdq + muq)**2 * (9 * lam + 4 * gsGG * vs)**2 *
            (27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs) -
             2 * gsGG * vh**2 *
             (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2) +
             gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2))**2 +
            26244 * gsGG**2 * lam**6 * vh**4 *
            (4 * mpi0**4 - 8 * mpi0**2 * (ms**2 - 2 * mx**2) +
             ms**2 * (3 * ms**2 - 8 * mx**2 - width_s**2))) *
           log(ms**4 + (-4 * mx**2 + e_cm**2)**2 + ms**2 *
               (-8 * mx**2 + 2 * e_cm**2 + width_s**2))))) /
        (419904. * lam**6 * ms * M_PI * e_cm**2 *
         (-4 * mx**2 + e_cm**2) * vh**4 *
         (9 * lam + 4 * gsGG * vs)**2 * width_s))


@cython.cdivision(True)
cdef double __sigma_xg_to_xg(double e_cm, double mx, double ms, double gsxx,
                                 double gsff, double gsGG, double gsFF,
                                 double lam, double width_s, double vs):

    # for e_cm = 2mx there is complete destructive interference
    if e_cm < mx or e_cm == 2.0 * mx:
        return 0.0

    cdef double s = e_cm**2

    return ((alpha_em**2 * gsFF**2 * gsxx**2 *
             (-2 * (ms**5 + 4 * ms * mx**2 * width_s**2 -
                    ms**3 * (4 * mx**2 + 3 * width_s**2)) * atan(ms / width_s) +
              2 * (ms**5 + 4 * ms * mx**2 * width_s**2 -
                   ms**3 * (4 * mx**2 + 3 * width_s**2)) *
              atan((ms**2 - 4 * mx**2 + s) / (ms * width_s)) -
              width_s * ((4 * ms**2 - 4 * mx**2 - s) * (4 * mx**2 - s) +
                        ms**2 * (-3 * ms**2 + 8 * mx**2 + width_s**2) *
                        log(ms**2 * (ms**2 + width_s**2)) +
                        ms**2 * (3 * ms**2 - 8 * mx**2 - width_s**2) *
                        log(ms**4 + (-4 * mx**2 + s)**2 + ms**2 *
                            (-8 * mx**2 + 2 * s + width_s**2))))) /
            (128. * lam**2 * M_PI**3 * (4 * mx**2 - s) * s * width_s))


@cython.cdivision(True)
cdef double __sigma_xs_to_xs(double e_cm, double mx, double ms, double gsxx,
                             double gsff, double gsGG, double gsFF,
                             double lam, double width_s, double vs):
    if e_cm < mx + ms:
        return 0.0

    if e_cm == 2.0 * mx:
        return ((gsxx**4*(ms**4 - 10*ms**2*mx**2 + 9*mx**4)**2)/
                (36.0*mx**4*(2*ms**2 - 3*mx**2)**2*M_PI*e_cm**2))

    cdef double s = e_cm**2

    return (
        (gsxx**4*
         (4*ms**4 - 24*ms**2*mx**2 + 4*mx**4 +
          (4*(ms**2 - 4*mx**2)**2*(mx**2 - s)**2)/
          ((2*ms**2 - 3*mx**2)*(4*mx**2 - s)) -
          (4*(ms**2 - 4*mx**2)**2*(mx**2 - s)**2)/
          ((2*ms**2 + mx**2 - s)*(4*mx**2 - s)) - 8*ms**2*s +
          50*mx**2*s + 10*s**2 +
          (4*(mx**2 - s)*
           (-2*ms**4 - 16*ms**2*mx**2 +
            15*mx**4 + 18*mx**2*s - s**2)*
           log(abs((2*ms**2 - 3*mx**2)/(2*ms**2 + mx**2 - s))))/
          (4*mx**2 - s)))/(64.*M_PI*e_cm**2*(mx**2 - s)**2))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_s_to_ff(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx,
    double ms, double gsxx, double gsff, double gsGG, double gsFF,
    double lam, double width_s, double vs, double mf):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_s_to_ff(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                        gsFF, lam, width_s, vs, mf)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_s_to_gg(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_s_to_gg(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                        gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_s_to_pi0pi0(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms,
    double gsxx, double gsff, double gsGG, double gsFF, double lam,
    double width_s, double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_s_to_pi0pi0(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                            gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_s_to_pipi(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_s_to_pipi(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                          gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_ss(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xx_to_ss(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xx_to_all(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = sigma_xx_to_all(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                  gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xl_to_xl(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs, double mf):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xl_to_xl(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, width_s, vs, mf)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xpi_to_xpi(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xpi_to_xpi(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xpi0_to_xpi0(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xpi0_to_xpi0(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xg_to_xg(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xg_to_xg(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] __vec_sigma_xs_to_xs(
    np.ndarray[np.float64_t, ndim=1] e_cms, double mx, double ms, double gsxx,
    double gsff, double gsGG, double gsFF, double lam, double width_s,
    double vs):

    cdef int num_es = e_cms.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sigs = np.zeros(num_es, np.float64)

    cdef int i
    for i in range(num_es):
        sigs[i] = __sigma_xs_to_xs(e_cms[i], mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, width_s, vs)

    return sigs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_s_to_ff(e_cms, double mx, double ms, double gsxx,
                        double gsff, double gsGG, double gsFF,
                        double lam, double width_s, double vs,
                        double mf):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of fermions, *f* through a scalar mediator in
    the s-channel.

    Parameters
    ----------
    e_cms : double or array-like
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.
    mf: double
        Final state fermion mass.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s* -> f + f.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_s_to_ff(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs, mf)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_s_to_ff(e_cm, mx, ms, gsxx, gsff, gsGG,
                                 gsFF, lam, width_s, vs, mf)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_s_to_gg(e_cms, double mx, double ms, double gsxx,
                           double gsff, double gsGG, double gsFF,
                           double lam, double width_s, double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of photons through a scalar mediator in the
    s-channel.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s* -> g + g.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_s_to_gg(np.array(e_cms),
                                         mx, ms, gsxx, gsff, gsGG,
                                         gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_s_to_gg(e_cm, mx, ms, gsxx, gsff, gsGG,
                                 gsFF, lam, width_s, vs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_s_to_pi0pi0(e_cms, double mx, double ms,
                               double gsxx, double gsff, double gsGG,
                               double gsFF, double lam, double width_s,
                               double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of neutral pion through a scalar mediator in
    the s-channel.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s* -> pi0 + pi0.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_s_to_pi0pi0(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_s_to_pi0pi0(e_cm, mx, ms, gsxx, gsff, gsGG,
                                     gsFF, lam, width_s, vs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_s_to_pipi(e_cms, double mx, double ms,
                             double gsxx, double gsff, double gsGG,
                             double gsFF, double lam, double width_s,
                             double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of charged pions through a scalar mediator in
    the s-channel.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s* -> pi + pi.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_s_to_pipi(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_s_to_pipi(e_cm, mx, ms, gsxx, gsff, gsGG,
                                   gsFF, lam, width_s, vs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_ss(e_cms, double mx, double ms, double gsxx,
                   double gsff, double gsGG, double gsFF, double lam,
                   double width_s, double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of scalar mediator through the t and u
    channels.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s + s.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_ss(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_ss(e_cm, mx, ms, gsxx, gsff, gsGG,
                            gsFF, lam, width_s, vs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx_to_all(e_cms, double mx, double ms, double gsxx,
                   double gsff, double gsGG, double gsFF, double lam,
                   double width_s, double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into all availible final states.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> anything except x + x.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xx_to_all(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xx_to_all(e_cm, mx, ms, gsxx, gsff, gsGG,
                            gsFF, lam, width_s, vs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xl_to_xl(e_cms, double mx, double ms, double gsxx,
                   double gsff, double gsGG, double gsFF, double lam,
                   double width_s, double vs, double mf):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of scalar mediator through the t and u
    channels.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s + s.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xl_to_xl(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs, mf)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xl_to_xl(e_cm, mx, ms, gsxx, gsff, gsGG,
                            gsFF, lam, width_s, vs, mf)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xpi_to_xpi(e_cms, double mx, double ms, double gsxx,
                   double gsff, double gsGG, double gsFF, double lam,
                   double width_s, double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of scalar mediator through the t and u
    channels.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s + s.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xpi_to_xpi(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xpi_to_xpi(e_cm, mx, ms, gsxx, gsff, gsGG,
                            gsFF, lam, width_s, vs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xpi0_to_xpi0(e_cms, double mx, double ms, double gsxx,
                       double gsff, double gsGG, double gsFF, double lam,
                       double width_s, double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of scalar mediator through the t and u
    channels.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s + s.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xpi0_to_xpi0(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xpi0_to_xpi0(e_cm, mx, ms, gsxx, gsff, gsGG,
                                gsFF, lam, width_s, vs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xg_to_xg(e_cms, double mx, double ms, double gsxx,
                   double gsff, double gsGG, double gsFF, double lam,
                   double width_s, double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of scalar mediator through the t and u
    channels.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s + s.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xg_to_xg(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xg_to_xg(e_cm, mx, ms, gsxx, gsff, gsGG,
                            gsFF, lam, width_s, vs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xs_to_xs(e_cms, double mx, double ms, double gsxx,
                   double gsff, double gsGG, double gsFF, double lam,
                   double width_s, double vs):
    """
    Returns the spin-averaged, cross section for dark matter
    annihilating into a pair of scalar mediator through the t and u
    channels.

    Parameters
    ----------
    e_cm : double
        Center of mass energy(ies).
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    sigma : double
        Cross section for x + x -> s + s.
    """
    if hasattr(e_cms, '__len__') and e_cms.ndim > 0:
        return __vec_sigma_xs_to_xs(
            np.array(e_cms), mx, ms, gsxx, gsff, gsGG,
            gsFF, lam, width_s, vs)

    # e_cms is either a 0-d array or a scalar
    e_cm = e_cms.item() if hasattr(e_cms, "__len__") else e_cms

    return __sigma_xs_to_xs(e_cm, mx, ms, gsxx, gsff, gsGG,
                            gsFF, lam, width_s, vs)


@cython.cdivision(True)
cdef double __thermal_cross_section_integrand(double z, double x, double mx,
                                              double ms, double gsxx,
                                              double gsff, double gsGG, double gsFF,
                                              double lam, double width_s, double vs):
        cdef double sig = __sigma_xx_to_all(
            mx * z, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs
        )
        return sig * z**2 * (z**2 - 4.0) * k1(x * z)


@cython.cdivision(True)
def thermal_cross_section(double x, double mx, double ms, double gsxx,
                          double gsff, double gsGG, double gsFF,
                          double lam, double width_s, double vs):
    """
    Compute the thermally average cross section for scalar mediator model.

    Parameters
    ----------
    x: float
        Mass of the dark matter divided by its temperature.
    mx: double
        Dark matter mass
    ms: double
        Scalar mediator mass.
    gsxx: double
        Coupling of DM to scalar mediator.
    gsff: double
        Coupling of scalar mediator to fermions.
    gsGG: double
        Effective coupling of the scalar mediator to gluons.
    gsFF: double
        Effective coupling of the scalar mediator to photons.
    lam: double
        Cut-off scale of the SGG and SFF interactions.
    width_s: double
        Full decay width of the scalar mediator.
    vs: double
        Scalar mediator VEV.

    Returns
    -------
    tcs: float
        Thermally average cross section.
    """

    # If x is really large, we will get divide by zero errors
    if x > 300:
        return 0.0

    cdef double pf = x / (2.0 * kn(2, x))**2

    # points at which integrand may have trouble are:
    #   1. endpoint
    #   2. when ss final state is accessible => z = 2 ms / mx
    #   3. when we hit mediator resonance => z = ms / mx
    return pf * quad(__thermal_cross_section_integrand, 2.0,
                     max(50.0 / x, 100.0),
                     args=(x, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs),
                     points=[2.0, ms / mx, 2.0 * ms / mx])[0]
