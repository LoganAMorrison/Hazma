import numpy as np
from ..parameters import vh, b0
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq


def sigma_xx_to_s_to_k0k0(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of neutral kaons through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    cme : float
        Center of mass energy.
    mx : float
        Mass of incoming fermions.
    ms : float
        Mass of scalar mediator.
    gsxx : float
        Coupling of initial state fermions with the scalar mediator.
    gsGG : double
        Coupling of the scalar to gluons.


    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> k0 + k0.
    """

    s = cme**2

    if -2 * mk0 + np.sqrt(s) <= 0:
        return 0.0
    if -4 * mx**2 + s < 0:
        return 0.0

    sigma = (gsxx**2 * np.sqrt(-4 * mk0**2 + s) *
             np.sqrt(-4 * mx**2 + s) *
             (2 * gsGG * (2 * mk0**2 - s) *
              (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
              (9 * vh + 8 * gsGG * vs) + b0 * (mdq + msq) *
              (9 * vh + 4 * gsGG * vs) *
              (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
               (9 * vh + 16 * gsGG * vs)))**2) / \
        (32. * np.pi * (ms**2 - s)**2 * s *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    return sigma


def sigma_xx_to_s_to_kk(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of charged kaon through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    cme : float
        Center of mass energy.
    mx : float
        Mass of incoming fermions.
    ms : float
        Mass of scalar mediator.
    gsxx : float
        Coupling of initial state fermions with the scalar mediator.
    gsGG : double
        Coupling of the scalar to gluons.


    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> k+ + k-.
    """

    s = cme**2

    if -2 * mk + np.sqrt(s) <= 0:
        return 0.0
    if -4 * mx**2 + s < 0:
        return 0.0

    sigma = (gsxx**2 * np.sqrt(-4 * mk**2 + s) *
             np.sqrt(-4 * mx**2 + s) *
             (729 * b0 * gsff * (msq + muq) * vh**2 +
              32 * gsGG**3 * (2 * mk**2 - 4 * b0 * (msq + muq) - s) * vs**2 +
              36 * gsGG**2 * vs * (-2 * b0 * (msq + muq) *
                                   (vh - 8 * gsff * vs) -
                                   2 * mk**2 * (3 * vh + 4 * gsff * vs) + s *
                                   (3 * vh + 4 * gsff * vs)) +
              162 * gsGG * vh *
              (-2 * mk**2 * (vh + gsff * vs) + s * (vh + gsff * vs) +
               b0 * (msq + muq) * (3 * vh + 10 * gsff * vs)))**2) /\
        (32. * np.pi * (ms**2 - s)**2 * s *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    return sigma


def sigma_xx_to_s_to_pi0pi0(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs,
                            unit="BSE"):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of neutral pion through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    cme : float
        Center of mass energy.
    mx : float
        Mass of incoming fermions.
    ms : float
        Mass of scalar mediator.
    gsxx : float
        Coupling of initial state fermions with the scalar mediator.
    gsGG : double
        Coupling of the scalar to gluons.


    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> pi0 + pi0.
    """
    s = cme**2

    if -2 * mpi0 + np.sqrt(s) <= 0:
        return 0.0
    if -4 * mx**2 + s < 0:
        return 0.0

    sigma = (gsxx**2 * np.sqrt(-4 * mpi0**2 + s) *
             np.sqrt(-4 * mx**2 + s) *
             (2 * gsGG * (2 * mpi0**2 - s) *
              (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
              (9 * vh + 8 * gsGG * vs) + b0 * (mdq + muq) *
              (9 * vh + 4 * gsGG * vs) *
              (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
               (9 * vh + 16 * gsGG * vs)))**2) / \
        (64. * np.pi * (ms**2 - s)**2 * s *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    return sigma


def sigma_xx_to_s_to_pipi(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs,
                          unit="BSE"):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of charged pions through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    cme : float
        Center of mass energy.
    mx : float
        Mass of incoming fermions.
    ms : float
        Mass of scalar mediator.
    gsxx : float
        Coupling of initial state fermions with the scalar mediator.
    gsGG : double
        Coupling of the scalar to gluons.


    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> pi + pi.
    """

    s = cme**2

    if -2 * mpi + np.sqrt(s) <= 0:
        return 0.0
    if -4 * mx**2 + s < 0:
        return 0.0

    sigma = (gsxx**2 * np.sqrt(-4 * mpi**2 + s) *
             np.sqrt(-4 * mx**2 + s) *
             (2 * gsGG * (2 * mpi**2 - s) *
              (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
              (9 * vh + 8 * gsGG * vs) + b0 * (mdq + muq) *
              (9 * vh + 4 * gsGG * vs) *
              (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
               (9 * vh + 16 * gsGG * vs)))**2) / \
        (32. * np.pi * (ms**2 - s)**2 * s *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    return sigma
