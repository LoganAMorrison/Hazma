from cmath import sqrt, log, pi, atanh

from ..parameters import alpha_em
from ..parameters import pion_mass_chiral_limit as mPI
from ..parameters import kaon_mass_chiral_limit as mK
from ..parameters import rho_mass as mrho
from ..parameters import rho_width
from ..parameters import fpi, fv, gv, qe, vh

from ..unitarization.bethe_salpeter import amp_kk_to_kk_bse
from ..unitarization.bethe_salpeter import amp_pipi_to_kk_bse
from ..unitarization.bethe_salpeter import amp_pipi_to_pipi_bse
from ..unitarization.loops import bubble_loop

from ..field_theory_helper_functions.three_body_phase_space import E1_to_s
from ..field_theory_helper_functions.three_body_phase_space import t_integral
from ..field_theory_helper_functions.common_functions import \
    cross_section_prefactor
from ..field_theory_helper_functions.three_body_phase_space import \
    phase_space_prefactor, t_lim1, t_lim2

from .scalar_mediator_cross_sections import sigma_xx_to_s_to_pipi

cdef xx_s_pipig_no_rho_5_bub_E(double Q, double s, double t, params):
    cdef double gsxx = params.gsxx
    cdef double gsGG = params.gsGG
    cdef double gsff = params.gsff
    cdef double mx = params.mx
    cdef double ms = params.ms
    cdef double vs = params.vs
    return (complex(0., 0.0051440329218107) * gsxx *
            sqrt(-4. * mx**2 + Q**2) * qe *
            (6. * (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                                (9. * vh + 8. * gsGG * vs)**2) +
                   2. * gsGG * (27. * Q**2 * vh *
                                (3. * vh + 2. * gsGG * vs) +
                                mK**2 *
                                (81. * vh**2 - 144. * gsGG * vh * vs -
                                 64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**2 +
             12. * (9. * gsff *
                    (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                     (9. * vh + 8. * gsGG * vs)**2) +
                    2. * gsGG * (27. * Q**2 * vh *
                                 (3. * vh + 2. * gsGG * vs) +
                                 mPI**2 *
                                 (81. * vh**2 - 144. * gsGG * vh * vs -
                                  64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) *
             (-1. + amp_pipi_to_pipi_bse(Q) *
              bubble_loop(Q, mPI)) + bubble_loop(Q, mK) *
             (8. * sqrt(3.) *
              (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                            (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
              amp_pipi_to_kk_bse(Q) * bubble_loop(Q, mPI) +
              3. * (-2. * (9. * gsff *
                           (18. * gsGG * Q**2 * vh * vs +
                            mK**2 * (9. * vh + 8. * gsGG * vs)**2) +
                           2. * gsGG *
                           (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) +
                    sqrt(3.) * (9. * gsff *
                                (18. * gsGG * Q**2 * vh * vs +
                                 mPI**2 * (9. * vh + 8. * gsGG * vs)**2) +
                                2. * gsGG *
                                (27. * Q**2 * vh *
                                 (3. * vh + 2. * gsGG * vs) +
                                 mPI**2 *
                                 (81. * vh**2 - 144. * gsGG * vh * vs -
                                  64. * gsGG**2 * vs**2))) *
                    amp_pipi_to_kk_bse(Q) *
                    bubble_loop(Q, mPI))))) / \
        (sqrt(2.) * fpi**2 * (ms - Q) * (ms + Q) * vh *
         (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
         (9. * vh + 4. * gsGG * vs))


cdef xx_s_pipig_no_rho_fsr_bub_E(double Q, double s, double t, params):
    cdef double gsxx = params.gsxx
    cdef double gsGG = params.gsGG
    cdef double gsff = params.gsff
    cdef double mx = params.mx
    cdef double ms = params.ms
    cdef double vs = params.vs
    return (complex(0., 0.00205761316872428) * gsxx *
            sqrt(-4. * mx**2 + Q**2) * qe *
            (6. * (mPI**2 + 3. * Q**2 - t) *
             (9. * gsff *
              (18. * gsGG * Q**2 * vh * vs + mK**2 *
               (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG *
              (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
               mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                        64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mK) - 6. * (mPI**2 + 3. * Q**2 - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**2 -
             6. * (mPI**2 - 6. * Q**2 + 2. * t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) + 4. * sqrt(3.) *
             (mPI**2 - 6. * Q**2 + 2. * t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) -
             3. * sqrt(3.) * (mPI**2 + 3. * Q**2 - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) +
             6. * (mPI**2 - 6. * Q**2 + 2. * t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(Q) *
             bubble_loop(Q, mPI)**2)) / \
        (sqrt(2.) * fpi**2 * (ms - Q) * (ms + Q) * (mPI**2 - t) * vh *
         (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
         (9. * vh + 4. * gsGG * vs))


cdef xx_s_pipig_no_rho_no_bub_E(double Q, double s, double t, params):
    cdef double gsxx = params.gsxx
    cdef double gsGG = params.gsGG
    cdef double gsff = params.gsff
    cdef double mx = params.mx
    cdef double ms = params.ms
    cdef double vs = params.vs
    cdef double mrhoT = sqrt(mrho**2 - 2. * gsGG * vs / 9. / vh)
    return (complex(0., 0.012345679012345678) * gsxx *
            sqrt(-4. * mx**2 + Q**2) * qe *
            ((-2. * (-2. * fv * gsGG * gv * (mPI**2 - t) *
                     (mPI**2 + Q**2 - s - t) *
                     (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
                     (81. * vh**2 + 54. * gsGG * vh * vs +
                      8. * gsGG**2 * vs**2) +
                     3. * fpi**2 * mrhoT**2 * vh *
                     (9. * gsff *
                      (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                       (9. * vh + 8. * gsGG * vs)**2) +
                      2. * gsGG *
                      (27. * Q**2 * vh *
                       (3. * vh + 2. * gsGG * vs) +
                       mPI**2 *
                       (81. * vh**2 - 144. * gsGG * vh * vs -
                        64. * gsGG**2 * vs**2))))) /
             (mrhoT**2 * (mPI**2 - t) * vh**2 *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)) -
             (5184. * gsGG * mK**2 * pi**4 *
              (mPI**2 + Q**2 - s - t) * bubble_loop(0., mK)) /
             (Q**2 - s) - (10368. * gsGG * mPI**2 * pi**4 *
                           (mPI**2 + Q**2 - s - t) *
                           bubble_loop(0., mPI)) / (Q**2 - s) +
             (24. * pi**4 * (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
                (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                              (9. * vh + 8. * gsGG * vs)**2) +
                 2. * gsGG * (27. * Q**2 * vh *
                              (3. * vh + 2. * gsGG * vs) +
                              mK**2 *
                              (81. * vh**2 - 144. * gsGG * vh * vs -
                               64. * gsGG**2 * vs**2))) *
              bubble_loop(Q, mK)) /
             ((Q**2 - s)**2 * vh *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)) +
             (48. * pi**4 * (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
                (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                              (9. * vh + 8. * gsGG * vs)**2) +
                 2. * gsGG *
                 (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) + mPI**2 *
                  (81. * vh**2 - 144. * gsGG * vh * vs -
                   64. * gsGG**2 * vs**2))) *
                bubble_loop(Q, mPI)) /
             ((Q**2 - s)**2 * vh *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)) +
             (48. * pi**4 *
              (729. * gsff * mK**2 * s *
               (-mPI**2 - Q**2 + s + t) * vh**2 +
               128. * gsGG**3 * mK**2 * s *
               (mPI**2 + Q**2 - s - t) * vs**2 -
               18. * gsGG**2 * vs *
               (-(s *
                  (-6. * mPI**2 * Q**2 - 5. * Q**4 +
                   s**2 + Q**2 * (4. * s + 6. * t)) * vh
                  ) + 4. * mK**2 *
                (Q**4 * vh + Q**2 * s * (-6. * vh + 8. * gsff * vs) +
                 s * (5. * s * vh + 4. * t * vh -
                      8. * gsff * s * vs - 8. * gsff * t * vs -
                      4. * mPI**2 * (vh - 2. * gsff * vs)))) -
               27. * gsGG * vh *
               (-(s * (-6. * mPI**2 * Q**2 - 5. * Q**4 + s**2 +
                       Q**2 * (4. * s + 6. * t)) * (vh + gsff * vs)) +
                mK**2 * (-2. * Q**2 * s * (vh - 20. * gsff * vs) +
                         4. * Q**4 * (vh + gsff * vs) -
                         2. * s *
                         (-3. * mPI**2 * (vh + 8. * gsff * vs) +
                          3. * t * (vh + 8. * gsff * vs) + s *
                          (vh + 22. * gsff * vs))))) *
                bubble_loop(sqrt(s), mK)) /
             ((Q**2 - s)**2 * vh *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)) +
             (96. * pi**4 *
              (-729. * gsff * mPI**2 * s *
               (mPI**2 + Q**2 - s - t) * vh**2 +
               128. * gsGG**3 * mPI**2 * s *
               (mPI**2 + Q**2 - s - t) * vs**2 +
               18. * gsGG**2 * vs *
               (s * (-5. * Q**4 + s**2 + Q**2 *
                     (4. * s + 6. * t)) * vh +
                16. * mPI**4 * s * (vh - 2. * gsff * vs) -
                2. * mPI**2 *
                (2. * Q**4 * vh + Q**2 * s *
                 (-9. * vh + 16. * gsff * vs) +
                 2. * s * (5. * s * vh + 4. * t * vh -
                           8. * gsff * s * vs - 8. * gsff * t * vs))) -
               27. * gsGG * vh *
               (-(s * (-5. * Q**4 + s**2 + Q**2 *
                       (4. * s + 6. * t)) *
                  (vh + gsff * vs)) + 6. * mPI**4 * s *
                (vh + 8. * gsff * vs) +
                mPI**2 * (4. * Q**4 * (vh + gsff * vs) +
                          Q**2 * s * (4. * vh + 46. * gsff * vs) -
                          2. * s * (3. * t * (vh + 8. * gsff * vs) + s *
                                    (vh + 22. * gsff * vs))))) *
                bubble_loop(sqrt(s), mPI)) /
             ((Q**2 - s)**2 * vh *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)) -
             (3. * mK**2 * pi**2 * (mPI**2 + Q**2 - s - t) *
                (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                              (9. * vh + 8. * gsGG * vs)**2) +
                 2. * gsGG *
                 (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                  mK**2 * (81. * vh**2 -
                           144. * gsGG * vh * vs -
                           64. * gsGG**2 * vs**2))) *
                log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                    (2. * mK**2))**2) /
             ((Q**2 - s)**2 * vh *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)) -
             (6. * mPI**2 * pi**2 * (mPI**2 + Q**2 - s - t) *
                (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                              (9. * vh + 8. * gsGG * vs)**2) +
                 2. * gsGG *
                 (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                  mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                            64. * gsGG**2 * vs**2))) *
                log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                    (2. * mPI**2))**2) /
             ((Q**2 - s)**2 * vh *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)) +
             (3. * mK**2 * pi**2 * (mPI**2 + Q**2 - s - t) *
                (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                              (9. * vh + 8. * gsGG * vs)**2) +
                 2. * gsGG *
                 (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                  mK**2 * (81. * vh**2 -
                           144. * gsGG * vh * vs -
                           64. * gsGG**2 * vs**2))) *
                log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                    (2. * mK**2))**2) /
             ((Q**2 - s)**2 * vh *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)) +
             (6. * mPI**2 * pi**2 * (mPI**2 + Q**2 - s - t) *
                (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                              (9. * vh + 8. * gsGG * vs)**2) +
                 2. * gsGG *
                 (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                  mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                            64. * gsGG**2 * vs**2))) *
                log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                    (2. * mPI**2))**2) /
             ((Q**2 - s)**2 * vh *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs)))) / \
        (sqrt(2.) * fpi**2 * (ms - Q) * (ms + Q) *
         (9. * vh + 4. * gsGG * vs))


cdef xx_s_pipig_no_rho_triangle_bub_E(double Q, double s, double t, params):
    cdef double gsxx = params.gsxx
    cdef double gsGG = params.gsGG
    cdef double gsff = params.gsff
    cdef double mx = params.mx
    cdef double ms = params.ms
    cdef double vs = params.vs
    return (complex(0., 0.00017146776406035664) * gsxx * pi**2 *
            sqrt(-4. * mx**2 + Q**2) * qe *
            (1728. * mK**2 * pi**2 * (Q**2 - s) * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *

                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             bubble_loop(0., mK) * bubble_loop(Q, mK) +
             2304. * mPI**2 * pi**2 * (Q**2 - s) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *

                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             bubble_loop(0., mPI) * bubble_loop(Q, mK) -
             1296. * pi**2 * Q**2 * (Q**4 - s *
                                     (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *

                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mK)**2 - 1728. * mK**2 * pi**2 * (Q**2 - s) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(0., mK) *
             bubble_loop(Q, mK)**2 -
             2304. * mPI**2 * pi**2 * (Q**2 - s) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(0., mPI) *
             bubble_loop(Q, mK)**2 +
             1296. * pi**2 * Q**2 *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**3 +
             864. * mK**2 * pi**2 * (Q**2 - s) * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mPI**2 * (81. * vh**2 -
                                       144. * gsGG * vh * vs -
                                       64. * gsGG**2 * vs**2))) *
             bubble_loop(0., mK) * bubble_loop(Q, mPI) +
             4608. * mPI**2 * pi**2 *
             (Q**2 - s) * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 -
                                     144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(0., mPI) * bubble_loop(Q, mPI) -
             216. * pi**2 * Q**2 *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (8. * mK**2 * (9. * vh + 8. * gsGG * vs)**2 +
                           3. * (66. * gsGG * Q**2 * vh * vs + mPI**2 *
                                 (9. * vh + 8. * gsGG * vs)**2)) -
              2. * gsGG *
              (-8. * mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                              64. * gsGG**2 * vs**2) +
               3. * (-99. * Q**2 * vh *
                     (3. * vh + 2. * gsGG * vs) +
                     mPI**2 * (-81. * vh**2 +
                               144. * gsGG * vh * vs +
                               64. * gsGG**2 * vs**2)))) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) -
             576. * sqrt(3.) * mK**2 * pi**2 * (Q**2 - s) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) * bubble_loop(0., mK) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) -
             864. * sqrt(3.) * mK**2 * pi**2 * (Q**2 - s) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mPI**2 * (81. * vh**2 -
                                       144. * gsGG * vh * vs -
                                       64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) * bubble_loop(0., mK) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) -
             3072. * sqrt(3.) * mPI**2 * pi**2 * (Q**2 - s) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(0., mPI) * bubble_loop(Q, mK) *
             bubble_loop(Q, mPI) - 1152. * sqrt(3.) * mPI**2 * pi**2 *
             (Q**2 - s) * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mPI**2 * (81. * vh**2 -
                                       144. * gsGG * vh * vs -
                                       64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(0., mPI) * bubble_loop(Q, mK) *
             bubble_loop(Q, mPI) + 1728. * pi**2 * Q**2 *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) *
             bubble_loop(Q, mK)**2 * bubble_loop(Q, mPI) +
             432. * sqrt(3.) * pi**2 * Q**2 *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK)**2 * bubble_loop(Q, mPI) +
             648. * sqrt(3.) * pi**2 * Q**2 *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK)**2 * bubble_loop(Q, mPI) +
             1728. * pi**2 * (mPI**2 - 2. * Q**2) *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI)**2 - 864. * mK**2 * pi**2 * (Q**2 - s) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mPI**2 *
                             (81. * vh**2 - 144. * gsGG * vh * vs -
                              64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(
                 Q) * bubble_loop(0., mK) * bubble_loop(Q, mPI)**2 -
             4608. * mPI**2 * pi**2 *
             (Q**2 - s) * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(
                 Q) * bubble_loop(0., mPI) * bubble_loop(Q, mPI)**2 -
             1152. * sqrt(3.) * pi**2 * (mPI**2 - 2. * Q**2) *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI)**2 +
             864. * sqrt(3.) * pi**2 * Q**2 *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI)**2 +
             648. * pi**2 * Q**2 *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(
                 Q) * bubble_loop(Q, mK) * bubble_loop(Q, mPI)**2 -
             1728. * pi**2 * (mPI**2 - 2. * Q**2) *
             (Q**4 - s * (-2. * mPI**2 + s + 2. * t)) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(Q) * bubble_loop(Q, mPI)**3 +
             144. * pi**2 *
             (4. * mK**2 * (Q**2 - s)**2 -
              s * (-18. * mPI**2 * Q**2 - 17. * Q**4 + s**2 +
                   2. * Q**2 * (8. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mK) * bubble_loop(sqrt(s), mK) -
             144. * pi**2 *
             (4. * mK**2 * (Q**2 - s)**2 -
              s * (-18. * mPI**2 * Q**2 - 17. * Q**4 + s**2 +
                   2. * Q**2 * (8. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) *
             bubble_loop(Q, mK)**2 * bubble_loop(sqrt(s), mK) +
             72. * pi**2 * (4. * mK**2 * (Q**2 - s)**2 -
                            s * (-18. * mPI**2 * Q**2 - 17. * Q**4 + s**2 +
                                 2. * Q**2 * (8. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) * bubble_loop(sqrt(s), mK) -
             48. * sqrt(3.) * pi**2 *
             (4. * mK**2 * (Q**2 - s)**2 -
              s * (-18. * mPI**2 * Q**2 - 17. * Q**4 + s**2 + 2. * Q**2 *
                   (8. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             bubble_loop(sqrt(s), mK) -
             72. * sqrt(3.) * pi**2 *
             (4. * mK**2 * (Q**2 - s)**2 -
              s * (-18. * mPI**2 * Q**2 - 17. * Q**4 + s**2 + 2. * Q**2 *
                   (8. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG *
              (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
               mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                         64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) * bubble_loop(Q, mK) *
             bubble_loop(Q, mPI) * bubble_loop(sqrt(s), mK) -
             72. * pi**2 *
             (4. * mK**2 * (Q**2 - s)**2 -
              s * (-18. * mPI**2 * Q**2 - 17. * Q**4 + s**2 +
                   2. * Q**2 * (8. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(Q) * bubble_loop(Q, mPI)**2 *
             bubble_loop(sqrt(s), mK) +
             192. * pi**2 *
             (2. * mPI**2 * (2. * Q**4 + 5. * Q**2 * s + 2. * s**2) -
              s * (-17. * Q**4 + s**2 + 2. * Q**2 * (8. * s + 9. * t))) *
             (9. * gsff *
              (18. * gsGG * Q**2 * vh * vs + mK**2 *
               (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG *
              (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
               mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                        64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mK) * bubble_loop(sqrt(s), mPI) -
             192. * pi**2 *
             (2. * mPI**2 * (2. * Q**4 + 5. * Q**2 * s + 2. * s**2) -
              s * (-17. * Q**4 + s**2 + 2. * Q**2 * (8. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**2 *
             bubble_loop(sqrt(s), mPI) +
             384. * pi**2 * (-9. * mPI**4 * s - s *
                             (-17. * Q**4 + s**2 + 2. * Q**2 *
                              (8. * s + 9. * t)) +
                             mPI**2 * (4. * Q**4 + Q**2 * s + s *
                                       (13. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) * bubble_loop(sqrt(s), mPI) +
             256. * sqrt(3.) * pi**2 *
             (9. * mPI**4 * s +
              s * (-17. * Q**4 + s**2 + 2. * Q**2 * (8. * s + 9. * t)) -
              mPI**2 * (4. * Q**4 + Q**2 * s + s * (13. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG *
              (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
               mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                        64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) * bubble_loop(Q, mK) *
             bubble_loop(Q, mPI) * bubble_loop(sqrt(s), mPI) -
             96. * sqrt(3.) * pi**2 *
             (2. * mPI**2 * (2. * Q**4 + 5. * Q**2 * s + 2. * s**2) -
              s * (-17. * Q**4 + s**2 + 2. * Q**2 * (8. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             bubble_loop(sqrt(s), mPI) +
             384. * pi**2 * (9. * mPI**4 * s + s *
                             (-17. * Q**4 + s**2 + 2. * Q**2 *
                              (8. * s + 9. * t)) -
                             mPI**2 * (4. * Q**4 + Q**2 * s + s *
                                       (13. * s + 9. * t))) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(Q) * bubble_loop(Q, mPI)**2 *
             bubble_loop(sqrt(s), mPI) +
             162. * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mK) *
             log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                 (2. * mK**2))**2 - 162. * mK**2 * Q**2 *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**2 *
             log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                 (2. * mK**2))**2 +
             81. * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) *
             log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                 (2. * mK**2))**2 - 54. *
             sqrt(3.) * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                 (2. * mK**2))**2 -
             81. * sqrt(3.) * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                 (2. * mK**2))**2 -
             81. * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(Q) * bubble_loop(Q, mPI)**2 *
             log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                 (2. * mK**2))**2 +
             216. * mPI**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mK) *
             log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                 (2. * mPI**2))**2 - 216. * mPI**2 * Q**2 *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**2 *
             log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                 (2. * mPI**2))**2 -
             216. * mPI**2 * (mPI**2 - 2. * Q**2) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) *
             log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                 (2. * mPI**2))**2 + 144. * sqrt(3.) * mPI**2 *
             (mPI**2 - 2. * Q**2) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff *
              (18. * gsGG * Q**2 * vh * vs + mK**2 *
               (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) * bubble_loop(Q, mK) *
             bubble_loop(Q, mPI) *
             log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                 (2. * mPI**2))**2 -
             108. * sqrt(3.) * mPI**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                 (2. * mPI**2))**2 +
             216. * mPI**2 * (mPI**2 - 2. * Q**2) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(Q) * bubble_loop(Q, mPI)**2 *
             log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                 (2. * mPI**2))**2 -
             162. * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mK) *
             log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                 (2. * mK**2))**2 + 162. * mK**2 * Q**2 *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**2 *
             log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                 (2. * mK**2))**2 -
             81. * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) *
             log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                 (2. * mK**2))**2 + 54. * sqrt(3.) * mK**2 * Q**2 *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                 (2. * mK**2))**2 +
             81. * sqrt(3.) * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                 (2. * mK**2))**2 +
             81. * mK**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(Q) * bubble_loop(Q, mPI)**2 *
             log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                 (2. * mK**2))**2 -
             216. * mPI**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mK) *
             log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                 (2. * mPI**2))**2 + 216. * mPI**2 * Q**2 *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**2 *
             log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                 (2. * mPI**2))**2 +
             216. * mPI**2 * (mPI**2 - 2. * Q**2) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) *
             log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                 (2. * mPI**2))**2 - 144. * sqrt(3.) * mPI**2 *
             (mPI**2 - 2. * Q**2) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                 (2. * mPI**2))**2 +
             108. * sqrt(3.) * mPI**2 * Q**2 * (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mPI**2 *
                             (81. * vh**2 -
                              144. * gsGG * vh * vs -
                              64. * gsGG**2 * vs**2))) *
             amp_pipi_to_kk_bse(Q) *
             bubble_loop(Q, mK) * bubble_loop(Q, mPI) *
             log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                 (2. * mPI**2))**2 -
             216. * mPI**2 * (mPI**2 - 2. * Q**2) *
             (mPI**2 + Q**2 - s - t) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             amp_pipi_to_pipi_bse(Q) * bubble_loop(Q, mPI)**2 *
             log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                 (2. * mPI**2))**2)) / \
        (sqrt(2.) * fpi**4 * (ms - Q) * (ms + Q) * (Q**2 - s)**2 * vh *
         (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
         (9. * vh + 4. * gsGG * vs))


cdef xx_s_pipig_rho_4_bub_E(double Q, double s, double t, params):
    cdef double gsxx = params.gsxx
    cdef double gsGG = params.gsGG
    cdef double gsff = params.gsff
    cdef double mx = params.mx
    cdef double ms = params.ms
    cdef double vs = params.vs
    cdef double mrhoT = sqrt(mrho**2 - 2. * gsGG * vs / 9. / vh)
    return (complex(0., 0.0030864197530864196) * gsxx * gv *
            sqrt(-4. * mx**2 + Q**2) * qe * s *
            (mPI**2 + Q**2 - s - t) *
            (6. * (fv + 2. * gv) *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**2 +
             3. * (3. * fv + 8. * gv) *
             (9. * gsff *
              (18. * gsGG * Q**2 * vh * vs + mPI**2 *
               (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) *
             (-1. + amp_pipi_to_pipi_bse(Q) *
              bubble_loop(Q, mPI)) +
             bubble_loop(Q, mK) *
             (2. * sqrt(3.) * (3. * fv + 8. * gv) *
              (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                            (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
              amp_pipi_to_kk_bse(Q) * bubble_loop(Q, mPI) -
              3. * (fv + 2. * gv) *
              (2. * (9. * gsff *
                     (18. * gsGG * Q**2 * vh * vs + mK**2 *
                      (9. * vh + 8. * gsGG * vs)**2) +
                     2. * gsGG * (27. * Q**2 * vh *
                                  (3. * vh + 2. * gsGG * vs) +
                                  mK**2 * (81. * vh**2 -
                                           144. * gsGG * vh * vs -
                                           64. * gsGG**2 * vs**2))) -
               sqrt(3.) * (9. * gsff *
                           (18. * gsGG * Q**2 * vh * vs +
                            mPI**2 * (9. * vh + 8. * gsGG * vs)**2) +
                           2. * gsGG *
                           (27. * Q**2 * vh *
                            (3. * vh + 2. * gsGG * vs) +
                            mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
               amp_pipi_to_kk_bse(Q) *
               bubble_loop(Q, mPI))))) / \
        (sqrt(2.) * fpi**4 * mrhoT**2 * (ms - Q) * (ms + Q) *
         (mrho**2 - complex(0., 1.) * mrho * rho_width - s) * vh *
         (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
         (9. * vh + 4. * gsGG * vs))


cdef xx_s_pipig_rho_no_bub_E(double Q, double s, double t, params):
    cdef double gsxx = params.gsxx
    cdef double gsGG = params.gsGG
    cdef double gsff = params.gsff
    cdef double mx = params.mx
    cdef double ms = params.ms
    cdef double vs = params.vs
    cdef double mrhoT = sqrt(mrho**2 - 2. * gsGG * vs / 9. / vh)
    return (complex(0., 0.037037037037037035) * sqrt(2.) * gsxx * gv *
            sqrt(-4. * mx**2 + Q**2) * qe * s *
            (mPI**2 + Q**2 - s - t) *
            ((3. * fpi**2 * fv * gsGG) /
             (mrho**2 - complex(0., 1.) * mrho * rho_width - s) -
             (16. * gv * pi**4 * s**2 *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs + mK**2 *
                (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG *
               (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                         64. * gsGG**2 * vs**2))) *
              bubble_loop(Q, mK)) /
             ((Q**2 - s)**2 *
              (-mrho**2 + complex(0., 1.) * mrho * rho_width + s) *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
              (9. * vh + 4. * gsGG * vs)) -
             (32. * gv * pi**4 * s**2 *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG *
               (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                          64. * gsGG**2 * vs**2))) *
              bubble_loop(Q, mPI)) /
             ((Q**2 - s)**2 *
              (-mrho**2 + complex(0., 1.) * mrho * rho_width + s) *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
              (9. * vh + 4. * gsGG * vs)) +
             (16. * gv * pi**4 * s**2 *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs + mK**2 *
                (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
              bubble_loop(sqrt(s), mK)) /
             ((Q**2 - s)**2 * (-mrho**2 + complex(0., 1.) *
                               mrho * rho_width + s) *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
              (9. * vh + 4. * gsGG * vs)) +
             (32. * gv * pi**4 * s**2 *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
              bubble_loop(sqrt(s), mPI)) /
             ((Q**2 - s)**2 *
              (-mrho**2 + complex(0., 1.) * mrho * rho_width + s) *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
              (9. * vh + 4. * gsGG * vs)) +
             (gv * mK**2 * pi**2 * s *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs + mK**2 *
                (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
              log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                  (2. * mK**2))**2) /
             ((Q**2 - s)**2 *
              (-mrho**2 + complex(0., 1.) * mrho * rho_width + s) *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
              (9. * vh + 4. * gsGG * vs)) +
             (2. * gv * mPI**2 * pi**2 * s *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
              log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                  (2. * mPI**2))**2) /
             ((Q**2 - s)**2 *
              (-mrho**2 + complex(0., 1.) * mrho * rho_width + s) *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
              (9. * vh + 4. * gsGG * vs)) -
             (gv * mK**2 * pi**2 * s *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs + mK**2 *
                (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
              log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                  (2. * mK**2))**2) /
             ((Q**2 - s)**2 *
              (-mrho**2 + complex(0., 1.) * mrho * rho_width + s) *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
              (9. * vh + 4. * gsGG * vs)) -
             (2. * gv * mPI**2 * pi**2 * s *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
              log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                  (2. * mPI**2))**2) /
             ((Q**2 - s)**2 *
              (-mrho**2 + complex(0., 1.) * mrho * rho_width + s) *
              (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
              (9. * vh + 4. * gsGG * vs)))) / \
        (fpi**4 * mrhoT**2 * (ms - Q) * (ms + Q) * vh)


cdef xx_s_pipig_rho_triangle_bub_E(double Q, double s, double t, params):
    cdef double gsxx = params.gsxx
    cdef double gsGG = params.gsGG
    cdef double gsff = params.gsff
    cdef double mx = params.mx
    cdef double ms = params.ms
    cdef double vs = params.vs
    cdef double mrhoT = sqrt(mrho**2 - 2. * gsGG * vs / 9. / vh)
    return (complex(0., 0.0030864197530864196) * gsxx * gv**2 * pi**2 *
            sqrt(-4. * mx**2 + Q**2) * qe * s**2 *
            (mPI**2 + Q**2 - s - t) *
            (288. * pi**2 * Q**2 * s *
             (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                           (9. * vh + 8. * gsGG * vs)**2) +
              2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                           mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                    64. * gsGG**2 * vs**2))) *
             amp_kk_to_kk_bse(Q) * bubble_loop(Q, mK)**3 -
             3. * (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                                (9. * vh + 8. * gsGG * vs)**2) +
                   2. * gsGG *
                   (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                    mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                              64. * gsGG**2 * vs**2))) *
             bubble_loop(Q, mPI) *
             (-1. +
              amp_pipi_to_pipi_bse(Q) * bubble_loop(Q, mPI)) *
             (128. * pi**2 * (mPI**2 - 2. * Q**2) * s *
              bubble_loop(Q, mPI) +
              48. * pi**2 * Q**2 * s * bubble_loop(sqrt(s), mK) -
              128. * mPI**2 * pi**2 * s * bubble_loop(sqrt(s), mPI) +
              256. * pi**2 * Q**2 * s * bubble_loop(sqrt(s), mPI) +
              3. * mK**2 * Q**2 *
              log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                  (2. * mK**2))**2 - 8. * mPI**4 *
              log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                  (2. * mPI**2))**2 + 16. * mPI**2 * Q**2 *
              log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                  (2. * mPI**2))**2 -
              3. * mK**2 * Q**2 *
              log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                  (2. * mK**2))**2 + 8. * mPI**4 *
              log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                  (2. * mPI**2))**2 -
              16. * mPI**2 * Q**2 *
              log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                  (2. * mPI**2))**2) +
             6. * Q**2 * bubble_loop(Q, mK)**2 *
             (8. * pi**2 * s *
              (2. * sqrt(3.) *
               (9. * gsff *
                (18. * gsGG * Q**2 * vh * vs + mK**2 *
                 (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
               amp_pipi_to_kk_bse(Q) * bubble_loop(Q, mPI) +
               3. * (-2. * (9. * gsff *
                            (18. * gsGG * Q**2 * vh * vs +
                             mK**2 * (9. * vh + 8. * gsGG * vs)**2) +
                            2. * gsGG *
                            (27. * Q**2 * vh *
                             (3. * vh + 2. * gsGG * vs) +
                             mK**2 *
                             (81. * vh**2 - 144. * gsGG *
                              vh * vs - 64. * gsGG**2 * vs**2)
                             )) + sqrt(3.) *
                     (9. * gsff *
                      (18. * gsGG * Q**2 * vh * vs +
                       mPI**2 * (9. * vh + 8. * gsGG * vs)**2) +
                      2. * gsGG *
                      (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                       mPI**2 *
                       (81. * vh**2 - 144. * gsGG * vh * vs -
                        64. * gsGG**2 * vs**2))) *
                     amp_pipi_to_kk_bse(Q) *
                     bubble_loop(Q, mPI))) +
              (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                            (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                            mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                     64. * gsGG**2 * vs**2))) *
              amp_kk_to_kk_bse(Q) *
              (64. * pi**2 * s * bubble_loop(Q, mPI) -
               48. * pi**2 * s * bubble_loop(sqrt(s), mK) -
               64. * pi**2 * s * bubble_loop(sqrt(s), mPI) -
               3. * mK**2 *
               log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                   (2. * mK**2))**2 - 4. * mPI**2 *
               log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                   (2. * mPI**2))**2 + 3. * mK**2 *
               log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                   (2. * mK**2))**2 + 4. * mPI**2 *
               log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                   (2. * mPI**2))**2)) +
             bubble_loop(Q, mK) *
             (-16. * pi**2 * s *
              (16. * sqrt(3.) * (mPI**2 - 2. * Q**2) *
               (9. * gsff * (18. * gsGG * Q**2 * vh * vs +
                             mK**2 * (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
               amp_pipi_to_kk_bse(Q) -
               3. * Q**2 *
               (9. * gsff *
                (18. * gsGG * Q**2 * vh * vs +
                 mPI**2 * (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG *
                (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                 mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                           64. * gsGG**2 * vs**2))) *
               (4. * sqrt(3.) * amp_pipi_to_kk_bse(Q) +
                3. * amp_pipi_to_pipi_bse(Q))) *
              bubble_loop(Q, mPI)**2 +
              6. * Q**2 *
              (9. * gsff *
               (18. * gsGG * Q**2 * vh * vs +
                mK**2 * (9. * vh + 8. * gsGG * vs)**2) +
               2. * gsGG *
               (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                         64. * gsGG**2 * vs**2))) *
              (48. * pi**2 * s * bubble_loop(sqrt(s), mK) +
               64. * pi**2 * s * bubble_loop(sqrt(s), mPI) +
               3. * mK**2 *
               log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                   (2. * mK**2))**2 +
               4. * mPI**2 *
               log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                   (2. * mPI**2))**2 -
               3. * mK**2 *
               log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                   (2. * mK**2))**2 -
               4. * mPI**2 *
               log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                   (2. * mPI**2))**2) -
              bubble_loop(Q, mPI) *
              (2. * sqrt(3.) *
               (9. * gsff * (18. * gsGG * Q**2 * vh * vs + mK**2 *
                             (9. * vh + 8. * gsGG * vs)**2) +
                2. * gsGG * (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                             mK**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                                      64. * gsGG**2 * vs**2))) *
               amp_pipi_to_kk_bse(Q) *
               (48. * pi**2 * Q**2 * s * bubble_loop(sqrt(s), mK) -
                128. * pi**2 * (mPI**2 - 2. * Q**2) * s *
                bubble_loop(sqrt(s), mPI) +
                3. * mK**2 * Q**2 *
                log((2. * mK**2 +
                     Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                    (2. * mK**2))**2 -
                8. * mPI**4 *
                log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                    (2. * mPI**2))**2 +
                16. * mPI**2 * Q**2 *
                log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                    (2. * mPI**2))**2 -
                3. * mK**2 * Q**2 *
                log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                    (2. * mK**2))**2 +
                8. * mPI**4 *
                log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                    (2. * mPI**2))**2 -
                16. * mPI**2 * Q**2 *
                log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                    (2. * mPI**2))**2) +
               3. * Q**2 *
               (16. * pi**2 * s *
                (9. * gsff *
                 (8. * mK**2 * (9. * vh + 8. * gsGG * vs)**2 + 3. *
                  (66. * gsGG * Q**2 * vh * vs + mPI**2 * (
                      9. * vh + 8. * gsGG * vs)**2)) - 2. * gsGG *
                 (-8. * mK**2 *
                  (81. * vh**2 - 144. * gsGG * vh * vs -
                   64. * gsGG**2 * vs**2) +
                  3. * (-99. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                        mPI**2 *
                        (-81. * vh**2 + 144. * gsGG * vh * vs +
                         64. * gsGG**2 * vs**2)))) + sqrt(3.) *
                (9. * gsff *
                 (18. * gsGG * Q**2 * vh * vs + mPI**2 *
                  (9. * vh + 8. * gsGG * vs)**2) +
                 2. * gsGG *
                 (27. * Q**2 * vh * (3. * vh + 2. * gsGG * vs) +
                  mPI**2 * (81. * vh**2 - 144. * gsGG * vh * vs -
                            64. * gsGG**2 * vs**2))) *
                amp_pipi_to_kk_bse(Q) *
                (48. * pi**2 * s * bubble_loop(sqrt(s), mK) +
                 64. * pi**2 * s * bubble_loop(sqrt(s), mPI) +
                 3. * mK**2 *
                 log((2. * mK**2 + Q * (-Q + sqrt(-4. * mK**2 + Q**2))) /
                     (2. * mK**2))**2 +
                 4. * mPI**2 *
                 log((2. * mPI**2 + Q * (-Q + sqrt(-4. * mPI**2 + Q**2))) /
                     (2. * mPI**2))**2 -
                 3. * mK**2 *
                 log((2. * mK**2 - s + sqrt(s * (-4. * mK**2 + s))) /
                     (2. * mK**2))**2 -
                 4. * mPI**2 *
                 log((2. * mPI**2 - s + sqrt(s * (-4. * mPI**2 + s))) /
                     (2. * mPI**2))**2)))))) / \
        (sqrt(2.) * fpi**6 * mrhoT**2 *
         (ms - Q) * (ms + Q) * (Q**2 - s)**2 *
         (-mrho**2 + complex(0., 1.) * mrho * rho_width + s) * vh *
         (3. * vh + 3. * gsff * vs + 2. * gsGG * vs) *
         (9. * vh + 4. * gsGG * vs))


cdef xx_s_pipig_E(double Q, double s, double t, params):
    return xx_s_pipig_no_rho_5_bub_E(Q, s, t, params) + \
        xx_s_pipig_no_rho_fsr_bub_E(Q, s, t, params) + \
        xx_s_pipig_no_rho_no_bub_E(Q, s, t, params) + \
        xx_s_pipig_no_rho_triangle_bub_E(Q, s, t, params) + \
        xx_s_pipig_rho_4_bub_E(Q, s, t, params) + \
        xx_s_pipig_rho_no_bub_E(Q, s, t, params) + \
        xx_s_pipig_rho_triangle_bub_E(Q, s, t, params)


cdef msqrd_xx_to_s_to_pipig(double Q, double s, double t, params):
    """
    Returns the squared matrix element for two fermions annihilating
    into two charged pions and a photon.

    Parameters
    ----------
    Q : double
        Center of mass energy, or sqrt((ppip + ppim + pg)^2).
    s : double
        Mandelstam variable associated with the photon, defined as
        (P-pg)^2, where P = ppip + ppim + pg.
    t : double
        Mandelstam variable associated with the charged pion, defined as
        (P-ppip)^2, where P = ppip + ppim + pg.
    params: namedtuple
        Namedtuple of the model parameters.

    Returns
    -------
    Returns matrix element squared for
    :math:`\chi\bar{\chi}\to\pi^{+}\pi^{-}\gamma`.

    Notes
    -----
    The matrix element for this process, M, is related to the
    form factor by |M|^2. = s Re[E(s,t,u) E^*(s,u,t)] - m_PI^2.
    |E(s,t,u) + E(s,u,t)|^2.
    """

    cdef double ret_val = 0.0
    cdef double u = 0.0

    if 4. * mPI**2 < s < Q**2:
        if t_lim1(s, 0.0, mPI, mPI, Q) < t < t_lim2(s, 0.0, mPI, mPI, Q):

            u = Q**2 + 2. * mPI**2 - s - t

            E_t = xx_s_pipig_E(Q, s, t, params)
            E_u = xx_s_pipig_E(Q, s, u, params)

            ret_val = s * (E_t * E_u.conjugate()).real - \
                mPI**2 * abs(E_t + E_u)**2

    return ret_val


def c_dnde_xx_to_s_to_pipig(eng_gam, Q, params):
    """Unvectorized dnde_xx_to_s_to_pipig"""
    cdef double mx = params.mx

    cdef double s = Q**2 - 2. * Q * eng_gam

    cdef double ret_val = 0.0

    if 2. * mPI < Q and 4. * mPI**2 <= s <= Q**2:

        s = E1_to_s(eng_gam, 0., Q)

        mat_elem_sqrd = lambda t: msqrd_xx_to_s_to_pipig(Q, s, t, params)

        prefactor1 = phase_space_prefactor(Q)
        prefactor2 = 2. * Q / sigma_xx_to_s_to_pipi(Q, params)
        prefactor3 = cross_section_prefactor(mx, mx, Q)

        prefactor = prefactor1 * prefactor2 * prefactor3

        int_val, _ = t_integral(s, 0., mPI, mPI, Q, mat_elem_sqrd)

        ret_val = prefactor * int_val

    assert ret_val >= 0.

    return ret_val
