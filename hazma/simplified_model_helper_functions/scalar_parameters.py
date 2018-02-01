from ..parameters import vh, b0, fpi
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq

import numpy as np
from cmath import sqrt

from scipy.optimize import newton


def vs_eqn(vs, gsff, gsGG, ms):
    RHS = (27 * b0 * (3 * gsff + 2 * gsGG) * fpi**2 *
           (mdq + msq + muq) * vh) / \
        (ms**2 * (9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))

    return RHS - vs


def vs_solver(gsff, gsGG, ms):
    if ms == 0.0:
        return 27.0 * vh * (3.0 * gsff + 2.0 * gsGG) / \
            (16.0 * gsGG * (2.0 * gsGG - 9.0 * gsff))
    else:
        return newton(vs_eqn, 0.0, args=(gsff, gsGG, ms))


def mass_s(gsff, gsGG, ms, vs):

    ms_new_sqrd = ms**2 + 16.0 * b0 * fpi**2 * gsGG * (mdq + msq + muq) * \
        (9.0 * gsff - 2.0 * gsGG) / (8.0 * gsGG * vs + 9.0 * vh) / \
        (9.0 * gsff * vs - 2.0 * gsGG * vs + 9.0 * vh)

    return np.sqrt(ms_new_sqrd)


def scalar_pot_extrema(gsff, gsGG, ms):
    """
    Returns the locations of extrema for the scalar potental.

    Paramaters
    ----------
    gsff : double
        Coupling of the scalar to the standard model fermions. Note that the
        coupling to the standard model fermions comes from the scalar mixing
        with the Higgs, thus the coupling is :math:`c_{ffs} * m_{f} / v`.
    gsGG : double
        Coupling of the scalar to gluons.
    ms : double
        Mass of the scalar mediator.
    vs : double
        Vacuum expectation value of the scalar mediator.

    Returns
    -------
    extrema : numpy.array
        An array of the scalar potential extrema locations.
    """
    if ms == 0.0:
        return 27.0 * vh * (3.0 * gsff + 2.0 * gsGG) / \
            (16.0 * gsGG * (2.0 * gsGG - 9.0 * gsff))

    root1 = (3 * (-3 * (3 * gsff + 2 * gsGG) * ms**2 * vh +
                  (3 * (27 * gsff**2 -
                        36 * gsff * gsGG + 28 * gsGG**2) *
                   ms**4 * vh**2) /
                  (7776 * b0 * gsff**3 * gsGG**2 * fpi**2 *
                   mdq * ms**4 *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 * fpi**2 * mdq *
                   ms**4 * vh -
                   1920 * b0 * gsff * gsGG**4 * fpi**2 * mdq *
                   ms**4 * vh +
                   256 * b0 * gsGG**5 * fpi**2 * mdq * ms**4 * vh +
                   7776 * b0 * gsff**3 * gsGG**2 * fpi**2 *
                   ms**4 * msq *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 *
                   fpi**2 * ms**4 *
                   msq * vh -
                   1920 * b0 * gsff * gsGG**4 * fpi**2 *
                   ms**4 * msq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * msq * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * muq *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 *
                   fpi**2 * ms**4 *
                   muq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * muq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * muq * vh -
                   729 * gsff**3 * ms**6 * vh**3 +
                   1458 * gsff**2 * gsGG * ms**6 * vh**3 +
                   324 * gsff * gsGG**2 * ms**6 * vh**3 -
                   648 * gsGG**3 * ms**6 * vh**3 +
                   sqrt(ms**8 * vh**2 *
                        (-27 *
                         (27 * gsff**2 - 36 * gsff * gsGG +
                          28 * gsGG**2)**3 * ms**4 * vh**4 +
                         (3 * gsff + 2 * gsGG)**2 *
                         (32 * b0 *
                          (9 * gsff - 2 * gsGG)**2 * gsGG**2 *
                          fpi**2 * (mdq + msq + muq) -
                          81 *
                          (3 * gsff**2 -
                           8 * gsff * gsGG +
                           4 * gsGG**2) * ms**2 * vh**2)**2))) **
                  0.3333333333333333 +
                  (7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * mdq * ms**4 * vh +
                   1728 * b0 * gsff**2 * gsGG**3 *
                   fpi**2 * mdq * ms**4 *
                   vh - 1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * mdq * ms**4 *
                   vh + 256 * b0 * gsGG**5 * fpi**2 *
                   mdq * ms**4 * vh +
                   7776 * b0 * gsff**3 * gsGG**2 * fpi**2 *
                   ms**4 * msq *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 *
                   fpi**2 * ms**4 *
                   msq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * msq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * msq * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * muq *
                   vh + 1728 * b0 * gsff**2 *
                   gsGG**3 * fpi**2 * ms**4 *
                   muq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * muq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * muq * vh -
                   729 * gsff**3 * ms**6 * vh**3 +
                   1458 * gsff**2 * gsGG * ms**6 * vh**3 +
                   324 * gsff * gsGG**2 * ms**6 * vh**3 -
                   648 * gsGG**3 * ms**6 * vh**3 +
                   sqrt(ms**8 * vh**2 *
                        (-27 * (27 * gsff**2 - 36 * gsff * gsGG +
                                28 * gsGG**2)**3 * ms**4 * vh**4 +
                         (3 * gsff + 2 * gsGG)**2 *
                         (32 * b0 *
                          (9 * gsff - 2 * gsGG)**2 * gsGG**2 *
                          fpi**2 * (mdq + msq + muq) -
                          81 *
                          (3 * gsff**2 - 8 * gsff * gsGG +
                           4 * gsGG**2) *
                          ms**2 * vh**2)**2))) **
                  0.3333333333333333)) / \
        (8. * (9 * gsff - 2 * gsGG) * gsGG * ms**2)

    root2 = (3 * (-6 * (3 * gsff + 2 * gsGG) * ms**2 * vh -
                  ((np.complex(0, 3)) *
                   ((np.complex(0, -1)) + sqrt(3)) *
                   (27 * gsff**2 - 36 * gsff * gsGG + 28 * gsGG**2) *
                   ms**4 * vh**2) /
                  (7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * mdq * ms**4 *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 * fpi**2 * mdq *
                   ms**4 * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * mdq * ms**4 * vh +
                   256 * b0 * gsGG**5 * fpi**2 * mdq * ms**4 * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * msq *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 *
                   fpi**2 * ms**4 *
                   msq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * msq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * msq * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * muq *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 *
                   fpi**2 * ms**4 *
                   muq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * muq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * muq * vh -
                   729 * gsff**3 * ms**6 * vh**3 +
                   1458 * gsff**2 * gsGG * ms**6 * vh**3 +
                   324 * gsff * gsGG**2 * ms**6 * vh**3 -
                   648 * gsGG**3 * ms**6 * vh**3 +
                   sqrt(ms**8 * vh**2 *
                        (-27 *
                         (27 * gsff**2 - 36 * gsff * gsGG +
                          28 * gsGG**2)**3 * ms**4 * vh**4 +
                         (3 * gsff + 2 * gsGG)**2 *
                         (32 * b0 *
                          (9 * gsff - 2 * gsGG)**2 * gsGG**2 *
                          fpi**2 * (mdq + msq + muq) -
                          81 *
                          (3 * gsff**2 - 8 * gsff * gsGG +
                           4 * gsGG**2) * ms**2 * vh**2)**2))) **
                  0.3333333333333333 +
                  (np.complex(0, 1)) *
                  ((np.complex(0, 1)) + sqrt(3)) *
                  (7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * mdq * ms**4 *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 * fpi**2 * mdq *
                   ms**4 * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * mdq * ms**4 * vh +
                   256 * b0 * gsGG**5 * fpi**2 * mdq * ms**4 * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * msq *
                   vh + 1728 * b0 * gsff**2 *
                   gsGG**3 * fpi**2 * ms**4 *
                   msq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * msq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * msq * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * muq *
                   vh + 1728 * b0 * gsff**2 *
                   gsGG**3 * fpi**2 * ms**4 *
                   muq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * muq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * muq * vh -
                   729 * gsff**3 * ms**6 * vh**3 +
                   1458 * gsff**2 * gsGG * ms**6 * vh**3 +
                   324 * gsff * gsGG**2 * ms**6 * vh**3 -
                   648 * gsGG**3 * ms**6 * vh**3 +
                   sqrt(ms**8 * vh**2 *
                        (-27 *
                         (27 * gsff**2 - 36 * gsff * gsGG +
                          28 * gsGG**2)**3 * ms**4 * vh**4 +
                         (3 * gsff + 2 * gsGG)**2 *
                         (32 * b0 *
                          (9 * gsff - 2 * gsGG)**2 * gsGG**2 *
                          fpi**2 * (mdq + msq + muq) -
                          81 *
                          (3 * gsff**2 - 8 * gsff * gsGG +
                           4 * gsGG**2) * ms**2 * vh**2)**2))) **
                  0.3333333333333333)) / \
        (16. * (9 * gsff - 2 * gsGG) * gsGG * ms**2)

    root3 = (3 * (-6 * (3 * gsff + 2 * gsGG) * ms**2 * vh +
                  ((np.complex(0, 3)) * ((np.complex(0, 1)) +
                                         sqrt(3)) *
                   (27 * gsff**2 - 36 * gsff * gsGG + 28 * gsGG**2) *
                   ms**4 * vh**2) /
                  (7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * mdq * ms**4 *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 * fpi**2 * mdq *
                   ms**4 * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * mdq * ms**4 * vh +
                   256 * b0 * gsGG**5 * fpi**2 * mdq * ms**4 * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * msq *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 *
                   fpi**2 * ms**4 *
                   msq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * msq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * msq * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * muq *
                   vh + 1728 * b0 * gsff**2 *
                   gsGG**3 * fpi**2 * ms**4 *
                   muq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * muq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * muq * vh -
                   729 * gsff**3 * ms**6 * vh**3 +
                   1458 * gsff**2 * gsGG * ms**6 * vh**3 +
                   324 * gsff * gsGG**2 * ms**6 * vh**3 -
                   648 * gsGG**3 * ms**6 * vh**3 +
                   sqrt(ms**8 * vh**2 *
                        (-27 *
                         (27 * gsff**2 - 36 * gsff * gsGG +
                          28 * gsGG**2)**3 * ms**4 * vh**4 +
                         (3 * gsff + 2 * gsGG)**2 *
                         (32 * b0 *
                          (9 * gsff - 2 * gsGG)**2 * gsGG**2 *
                          fpi**2 * (mdq + msq + muq) -
                          81 *
                          (3 * gsff**2 - 8 * gsff * gsGG +
                           4 * gsGG**2) * ms**2 * vh**2)**2))) **
                  0.3333333333333333 -
                  (1 + (np.complex(0, 1)) * sqrt(3)) *
                  (7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * mdq * ms**4 *
                   vh + 1728 * b0 * gsff**2 * gsGG**3 * fpi**2 * mdq *
                   ms**4 * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * mdq * ms**4 * vh +
                   256 * b0 * gsGG**5 * fpi**2 * mdq * ms**4 * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * msq *
                   vh + 1728 * b0 * gsff**2 *
                   gsGG**3 * fpi**2 * ms**4 *
                   msq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * msq * vh +
                   256 * b0 * gsGG**5 * fpi**2 *
                   ms**4 * msq * vh +
                   7776 * b0 * gsff**3 * gsGG**2 *
                   fpi**2 * ms**4 * muq *
                   vh + 1728 * b0 * gsff**2 *
                   gsGG**3 * fpi**2 * ms**4 *
                   muq * vh -
                   1920 * b0 * gsff * gsGG**4 *
                   fpi**2 * ms**4 * muq * vh +
                   256 * b0 * gsGG**5 * fpi**2 * ms**4 * muq * vh -
                   729 * gsff**3 * ms**6 * vh**3 +
                   1458 * gsff**2 * gsGG * ms**6 * vh**3 +
                   324 * gsff * gsGG**2 * ms**6 * vh**3 -
                   648 * gsGG**3 * ms**6 * vh**3 +
                   sqrt(ms**8 * vh**2 *
                        (-27 *
                         (27 * gsff**2 - 36 * gsff * gsGG +
                          28 * gsGG**2)**3 * ms**4 * vh**4 +
                         (3 * gsff + 2 * gsGG)**2 *
                         (32 * b0 * (9 * gsff - 2 * gsGG)**2 *
                          gsGG**2 *
                          fpi**2 * (mdq + msq + muq) -
                          81 *
                          (3 * gsff**2 - 8 * gsff * gsGG +
                           4 * gsGG**2) * ms**2 * vh**2)**2))) **
                  0.3333333333333333)) / \
        (16. * (9 * gsff - 2 * gsGG) * gsGG * ms**2)

    return np.array([root1, root2, root3])
