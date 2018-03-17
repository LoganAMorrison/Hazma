import warnings
import numpy as np

from cmath import sqrt, log, pi

from ..parameters import alpha_em, GF, Vus, fpi
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import eta_mass as meta
from ..parameters import rho_mass as mrho
from ..parameters import electron_mass as me
from ..parameters import muon_mass as mmu

from ..field_theory_helper_functions.common_functions import minkowski_dot


mPI = (mpi + mpi0) / 2.
mK = (mk + mk0) / 2.

# ####################################
# LO Meson-Meson Scattering amplutudes
# ####################################


def amp_pipi_to_pipi_LO(s, t, u):
    return (s - mPI**2) / fpi**2


def amp_pipi_to_pipi_LO_I0(s, t):
    u = 4. * mPI**2 - s - t
    A2stu = amp_pipi_to_pipi_LO(s, t, u)
    A2tsu = amp_pipi_to_pipi_LO(s, t, u)
    A2uts = amp_pipi_to_pipi_LO(s, t, u)

    return 0.5 * (3. * A2stu + A2tsu + A2uts)


# #####################################
# NLO Meson-Meson Scattering amplutudes
# #####################################

mu = mrho
Lr1 = 0.56 * 1.0e-3  # (0.56 +- 0.10) * 1.0e-3
Lr2 = 1.21 * 1.0e-3  # (1.21 +- 0.10) * 1.0e-3
L3 = -2.79 * 1.0e-3  # (-2.79 +- 0.14) * 1.0e-3
Lr4 = -0.36 * 1.0e-3  # (-0.36 +- 0.17) * 1.0e-3
Lr5 = 1.4 * 1.0e-3  # (1.4 +- 0.5) * 1.0e-3
Lr6 = 0.07 * 1.0e-3  # (0.07 +- 0.08) * 1.0e-3
L7 = -0.44 * 1.0e-3  # (-0.44 +- 0.15) * 1.0e-3
Lr8 = 0.78 * 1.0e-3  # (0.78 +- 0.18) * 1.0e-3


def M(P):
    if P == "pi":
        return mPI
    if P == "k":
        return mK
    if P == "eta":
        return meta


def v(P, s):
    return sqrt(1. - 4. * M(P)**2 / s)


def kP(P):
    return (1. + log(M(P)**2 / mu**2)) / (32. * np.pi**2)


def Jbar(P, s):
    num = v(P, s) - 1. + 0.0j
    den = v(P, s) + 1. + 0.0j
    return (2. + v(P, s) * np.log(num / den)) / (16. * np.pi**2)


def Jr(P, s):
    return Jbar(P, s) - 2. * kP(P)


def Mr(P, s):
    return (1.0 / 12.) * Jbar(P, s) * v(P, s)**2 - \
        (1. / 6.0) * kP(P) + 1. / (288. * np.pi**2)


def B4(s, t, u):
    return ((mPI**4. / 18.) * Jr("eta", s) +
            (1. / 2.) * (s**2 - mPI**4) * Jr("pi", s) +
            (1. / 8.) * s**2 * Jr("k", s) +
            (1. / 4.) * (t - 2 * mPI**2)**2 * Jr("pi", t) +
            t * (s - u) * (Mr("pi", t) + 0.5 * Mr("k", t)) +
            (1. / 4.) * (u - 2 * mPI**2)**2 * Jr("pi", u) +
            u * (s - t) * (Mr("pi", u) + 0.5 * Mr("k", u))) / fpi**4


def C4(s, t, u):
    return (4. * ((8. * Lr6 + 4. * Lr8) * mPI**4. +
                  (4. * Lr4 + 2. * Lr5) * mPI**2. *
                  (s - 2. * mPI**2.) +
                  (L3 + 2. * Lr1) * (s - 2. * mPI**2)**2. +
                  Lr2 * ((t - 2. * mPI**2)**2 +
                         (u - 2. * mPI**2)**2))) / fpi**4


def A4(s, t, u):
    return B4(s, t, u) + C4(s, t, u)


def T4(s, t):
    u = 4. * mPI**2 - s - t
    return 0.5 * (3. * A4(s, t, u) + A4(t, s, u) + A4(u, t, s))


def mandlestam_t(x, s):
    return 0.5 * (4. * mPI**2 - s) * (1. - x)

# ####################
# Kaon Matrix Elements
# ####################


def msqrd_kl_to_pienu(moms):
    """
    Returns matrix element squared for kl -> pi  + e  + nu.

    Parameters
    ----------
    moms : nuMASS_PIy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.

    Returns
    -------
    mat_elem_sqrd : float
        Squared matrix element for kl -> pi  + e  + nu.
    """
    pp, pe, pn = moms

    pk = np.sum(moms, 0)

    # s, t, and u are defined as:
    # s = (p1 + p2)^2
    s = minkowski_dot(pk - pp, pk - pp)
    # t = (p1 + p3)^2
    t = minkowski_dot(pk - pe, pk - pe)
    # u = (p1 + p4)^2
    u = minkowski_dot(pk - pn, pk - pn)
    # where p1 = pk, p2 = -pp, p3 = -pe and p4 = -pn

    return GF**2 * (mk0**4 + (mpi**2 - s)**2 +
                    me**2 * (2 * (mk0**2 + mpi**2) - s) -
                    2 * mk0**2 * (mpi**2 + s) - (t - u)**2) * Vus**2


def msqrd_kl_to_pienug(moms):
    """
    Returns matrix element squared for kl -> pi  + e  + nu + gam.

    Parameters
    ----------
    moms : nuMASS_PIy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.

    Returns
    -------
    mat_elem_sqrd : float
        Squared matrix element for kl -> pi  + e  + nu + gam.
    """
    pp, pe, pn, pg = moms

    pk = np.sum(moms, 0)

    pkDOTpn = minkowski_dot(pk, pn)
    pkDOTpp = minkowski_dot(pk, pp)
    pgDOTpk = minkowski_dot(pk, pg)
    peDOTpk = minkowski_dot(pe, pk)
    peDOTpg = minkowski_dot(pe, pg)
    peDOTpn = minkowski_dot(pe, pn)
    pgDOTpn = minkowski_dot(pn, pg)
    pnDOTpp = minkowski_dot(pp, pn)
    peDOTpp = minkowski_dot(pp, pe)
    pgDOTpp = minkowski_dot(pp, pg)

    mat_elem_sqrd =\
        (8 * alpha_em * GF**2 * np.pi *
         (-2 * peDOTpg**3 *
          (mpi**2 * (pgDOTpn + pkDOTpn) + (mpi**2 - pgDOTpp) * pnDOTpp) +
          peDOTpg**2 *
          (2 * pgDOTpn *
           (-(mpi**2 * (peDOTpk + peDOTpp)) +
            (mpi**2 + peDOTpp) * pgDOTpp) -
           2 * (mpi**2 * (peDOTpk + peDOTpp) - mpi**2 * pgDOTpp +
                pgDOTpp**2) * pkDOTpn +
           2 * pgDOTpp * (pgDOTpn + pkDOTpn) * pkDOTpp +
           peDOTpn * (mpi**2 * (mk0**2 + mpi**2 + 2 * pgDOTpk) +
                      2 * (mpi**2 - pgDOTpp) * pkDOTpp) -
           (2 * mpi**2 *
            (peDOTpk + peDOTpp) +
            (mk0**2 - mpi**2 - 2 * peDOTpk - 2 * peDOTpp + 2 * pgDOTpk) *
            pgDOTpp + 4 * pgDOTpp**2) * pnDOTpp) +
          me**2 * pgDOTpp**2 *
          ((peDOTpn + pgDOTpn) *
           (mk0**2 + mpi**2 + 2 * pkDOTpp) -
           2 * (peDOTpk + peDOTpp + pgDOTpk + pgDOTpp) *
           (pkDOTpn + pnDOTpp)) +
          peDOTpg * pgDOTpp *
          (peDOTpn *
           (pgDOTpp * (mk0**2 + mpi**2 + 2 * pkDOTpp) +
            2 * peDOTpp * (mk0**2 + mpi**2 + pgDOTpk + pgDOTpp +
                           2 * pkDOTpp)) -
           2 * peDOTpp**2 * (pgDOTpn + 2 * (pkDOTpn + pnDOTpp)) +
           pgDOTpp * (-(pgDOTpn * (mk0**2 + mpi**2 + 2 * pkDOTpp)) +
                      2 * (me**2 - peDOTpk + pgDOTpk + pgDOTpp) *
                      (pkDOTpn + pnDOTpp)) +
           peDOTpp *
           (pgDOTpn *
            (mk0**2 + mpi**2 - 2 * peDOTpk + 2 * pkDOTpp) -
            2 * (2 * peDOTpk + pgDOTpk + 2 * pgDOTpp) * (pkDOTpn + pnDOTpp)))
          ) * Vus**2) / (peDOTpg**2 * pgDOTpp**2)

    return mat_elem_sqrd


def msqrd_kl_to_pimunu(moms):
    """
    Matrix element squared for kl -> pi  + mu  + nu.

    Parameters
    ----------
    moms : nuMASS_PIy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    """
    pp, pmu, pn = moms

    pk = np.sum(moms, 0)

    # s, t, and u are defined as:
    # s = (p1 + p2)^2
    s = minkowski_dot(pk - pp, pk - pp)
    # t = (p1 + p3)^2
    t = minkowski_dot(pk - pmu, pk - pmu)
    # u = (p1 + p4)^2
    u = minkowski_dot(pk - pn, pk - pn)
    # where p1 = pk, p2 = -pp, p3 = -pmu and p4 = -pn

    return GF**2 * (mk0**4 + (mpi**2 - s)**2 +
                    mmu**2 * (2 * (mk0**2 + mpi**2) - s) -
                    2 * mk0**2 * (mpi**2 + s) - (t - u)**2) * Vus**2


def msqrd_kl_to_pi0pi0pi0(moms):
    """
    Matrix element squared for kl -> pi0 + pi0  + pi0.

    Parameters
    ----------
    moms : nuMASS_PIy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    """
    warnings.warn("""kl -> pi0 + pi0  + pi0 matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0


def msqrd_kl_to_pipipi0(moms):
    """
    Matrix element squared for kl -> pi  + pi  + pi0.

    Parameters
    ----------
    moms : nuMASS_PIy.ndarray
        Array of four momenta of the final state particles. The first two must
        for momenta must be the fermions and the last must be the photon.
        moms must be in the form {{ke1, kx1, ky1, kz1}, ..., {keN, kxN, kyN,
        kzN}}.
    """
    warnings.warn("""kl -> pi  + pi  + pi0 matrix element not yet available.
                  Currently this returns 1.0.""")
    return 1.0
