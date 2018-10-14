"""Module containing cross sections for the scalar mediator.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

from cmath import sqrt, pi

from ..parameters import vh, b0, alpha_em
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import eta_mass as meta
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq


from ..unitarization import amp_bethe_salpeter_pipi_to_pipi
from ..unitarization import amp_bethe_salpeter_pipi_to_kk
from ..unitarization import amp_bethe_salpeter_kk_to_kk
from ..unitarization import loop_matrix


def msqrd_xx_s(s, mx, gsxx):
    return -(gsxx**2 * (4 * mx**2 - s)) / 2.


def __amp_s_PIPI(s, ms, gsxx, gsff, gsGG, gsFF, vs):
    return (__amp_s_pi0pi0(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit='LO') +
            2 * __amp_s_pipi(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit='LO')) / \
        sqrt(6.)


def __amp_s_KK(s, ms, gsxx, gsff, gsGG, gsFF, vs):
    return (__amp_s_k0k0(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit='LO') +
            __amp_s_kk(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit='LO')) / sqrt(2)


def __amp_s_k0k0(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit='BSE'):
    """Amplitude for S -> k0 k0bar"""

    amp_s_k0k0 = (-2 * gsGG * (-2 * mk0**2 + s)) / \
        (9 * vh + 4 * gsGG * vs) - \
        (b0 * (mdq + msq) * (54 * gsGG * vh - 32 * gsGG**2 * vs +
                             9 * gsff * (9 * vh + 16 * gsGG * vs))) / \
        ((9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))

    if unit == 'LO':
        return amp_s_k0k0

    if unit == 'BSE':
        loop_mat = loop_matrix(sqrt(s))

        amp_s_PIPI = __amp_s_PIPI(s, ms, gsxx, gsff, gsGG, gsFF, vs)
        amp_s_KK = __amp_s_KK(s, ms, gsxx, gsff, gsGG, gsFF, vs)

        amp_PIPI_k0k0 = amp_bethe_salpeter_pipi_to_kk(sqrt(s)) / sqrt(2)
        amp_KK_k0k0 = amp_bethe_salpeter_kk_to_kk(sqrt(s)) / sqrt(2)

        return amp_s_k0k0 - amp_s_PIPI * loop_mat[0, 0] * amp_PIPI_k0k0 - \
            amp_s_KK * loop_mat[1, 1] * amp_KK_k0k0

    else:
        raise ValueError(
            'Unitariazation method {} is not available. Use LO or' +
            'BSE'.format(unit))


def __amp_s_kk(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit='BSE'):
    """Amplitude for S -> k+ k-"""

    amp_s_kk = (-2 * gsGG * (-2 * mk**2 + s)) / \
        (9 * vh + 4 * gsGG * vs) - \
        (b0 * (msq + muq) *
         (54 * gsGG * vh - 32 * gsGG**2 * vs +
          9 * gsff * (9 * vh + 16 * gsGG * vs))) / \
        ((9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))

    if unit == 'LO':
        return amp_s_kk

    if unit == 'BSE':

        loop_mat = loop_matrix(sqrt(s))

        amp_s_PIPI = __amp_s_PIPI(s, ms, gsxx, gsff, gsGG, gsFF, vs)
        amp_s_KK = __amp_s_KK(s, ms, gsxx, gsff, gsGG, gsFF, vs)

        amp_PIPI_k0k0 = amp_bethe_salpeter_pipi_to_kk(sqrt(s)) / sqrt(2)
        amp_KK_k0k0 = amp_bethe_salpeter_kk_to_kk(sqrt(s)) / sqrt(2)

        return amp_s_kk - amp_s_PIPI * loop_mat[0, 0] * amp_PIPI_k0k0 - \
            amp_s_KK * loop_mat[1, 1] * amp_KK_k0k0

    else:
        raise ValueError(
            'Unitariazation method {} is not available. Use LO or' +
            'BSE'.format(unit))


def __amp_s_pi0pi0(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit='BSE'):
    """Amplitude for S -> pi0 pi0"""

    amp_s_pi0pi0 = (-2 * gsGG * (-2 * mpi0**2 + s)) / \
        (9 * vh + 4 * gsGG * vs) - \
        (b0 * (mdq + muq) *
         (54 * gsGG * vh - 32 * gsGG**2 * vs +
          9 * gsff * (9 * vh + 16 * gsGG * vs))) / \
        ((9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))

    if unit == 'LO':
        return amp_s_pi0pi0

    if unit == 'BSE':

        loop_mat = loop_matrix(sqrt(s))

        amp_s_PIPI = __amp_s_PIPI(s, ms, gsxx, gsff, gsGG, gsFF, vs)
        amp_s_KK = __amp_s_KK(s, ms, gsxx, gsff, gsGG, gsFF, vs)

        amp_PIPI_pi0pi0 = amp_bethe_salpeter_pipi_to_pipi(sqrt(s)) / sqrt(3)
        amp_KK_pi0pi0 = amp_bethe_salpeter_kk_to_kk(sqrt(s)) / sqrt(3)

        return amp_s_pi0pi0 - amp_s_PIPI * loop_mat[0, 0] * amp_PIPI_pi0pi0 - \
            amp_s_KK * loop_mat[1, 1] * amp_KK_pi0pi0

    else:
        raise ValueError(
            'Unitariazation method {} is not available. Use LO or' +
            'BSE'.format(unit))


def __amp_s_pipi(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit='BSE'):
    """Amplitude for S -> pi+ pi-"""

    amp_s_pipi = (-2 * gsGG * (-2 * mpi**2 + s)) / \
        (9 * vh + 4 * gsGG * vs) - \
        (b0 * (mdq + muq) *
         (54 * gsGG * vh - 32 * gsGG**2 * vs +
          9 * gsff * (9 * vh + 16 * gsGG * vs))) / \
        ((9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))

    if unit == 'LO':
        return amp_s_pipi

    if unit == 'BSE':

        loop_mat = loop_matrix(sqrt(s))

        amp_s_PIPI = __amp_s_PIPI(s, ms, gsxx, gsff, gsGG, gsFF, vs)
        amp_s_KK = __amp_s_KK(s, ms, gsxx, gsff, gsGG, gsFF, vs)

        amp_PIPI_pipi = amp_bethe_salpeter_pipi_to_pipi(sqrt(s)) / sqrt(3)
        amp_KK_pipi = amp_bethe_salpeter_kk_to_kk(sqrt(s)) / sqrt(3)

        return amp_s_pipi - amp_s_PIPI * loop_mat[0, 0] * amp_PIPI_pipi - \
            amp_s_KK * loop_mat[1, 1] * amp_KK_pipi

    else:
        raise ValueError(
            'Unitariazation method {} is not available. Use LO or' +
            'BSE'.format(unit))


def sigma_xx_to_s_to_etaeta(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of eta mesons through a
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
    gsff : float
        Coupling of final state fermions with the scalar mediator.
    gsGG : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.


    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> eta + eta.
    """

    s = cme**2

    if -2 * meta + abs(sqrt(s)) <= 0:
        return 0.0

    sigma = (gsxx**2 * sqrt(-4 * meta**2 + s) *
             sqrt(-4 * mx**2 + s) *
             (6 * gsGG * (2 * meta**2 - s) *
              (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
              (9 * vh + 8 * gsGG * vs) + b0 *
                 (mdq + 4 * msq + muq) * (9 * vh + 4 * gsGG * vs) *
                 (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
                  (9 * vh + 16 * gsGG * vs)))**2) / \
        (576. * pi * (ms**2 - s)**2 * s *
         (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 * (9 * vh + 8 * gsGG * vs)**2)

    return sigma


def sigma_xx_to_s_to_ff(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of fermions, *f* through a
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
    gsff : float
        Coupling of final state fermions with the scalar mediator.
    gsGG : double
        Coupling of the scalar to gluons.
    vs : double
        Vacuum expectation value of the scalar mediator.


    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> f + f.
    """

    s = cme**2

    if -2 * mf + abs(sqrt(s)) <= 0:
        return 0.0

    sigma = (gsff**2 * gsxx**2 * mf**2 * (-4 * mf**2 + s)**1.5 *
             sqrt(-4 * mx**2 + s)) / \
        (16. * pi * (ms**2 - s)**2 * s * vh**2)

    return sigma


def sigma_xx_to_s_to_gg(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of photons through a
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
    gsFF : double
        Coupling of the scalar to photons.


    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> g + g.
    """

    s = cme**2

    if -4 * mx**2 + s < 0:
        return 0.0

    sigma = (alpha_em**2 * gsFF**2 * gsxx**2 * s**1.5 *
             sqrt(-4 * mx**2 + s)) / \
        (512. * pi**3 * (ms**2 - s)**2 * vh**2)

    return sigma


def sigma_xx_to_s_to_k0k0(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs,
                          unit='BSE'):
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

    if cme < 2 * mk0:
        return 0.0

    s = cme**2

    amp_s_to_k0k0 = __amp_s_k0k0(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit=unit)

    msqrd = abs(amp_s_to_k0k0)**2 * \
        msqrd_xx_s(s, mx, gsxx) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mk0**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

    return sigma


def sigma_xx_to_s_to_kk(cme, mx, mf, ms, gsxx, gsff, gsGG, gsFF, vs,
                        unit='BSE'):
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

    if cme < 2 * mk:
        return 0.0

    s = cme**2

    amp_s_to_kk = __amp_s_kk(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit=unit)

    msqrd = abs(amp_s_to_kk)**2 * msqrd_xx_s(s, mx, gsxx) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mk**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

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

    if cme < 2 * mpi0:
        return 0.0

    s = cme**2

    amp_s_to_pi0pi0 = __amp_s_pi0pi0(
        s, ms, gsxx, gsff, gsGG, gsFF, vs, unit=unit)

    msqrd = abs(amp_s_to_pi0pi0)**2 * msqrd_xx_s(s, mx, gsxx) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mpi0**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

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

    if cme < 2. * mpi:
        return 0.0

    s = cme**2

    amp_s_to_pipi = __amp_s_pipi(s, ms, gsxx, gsff, gsGG, gsFF, vs, unit=unit)

    msqrd = abs(amp_s_to_pipi)**2 * msqrd_xx_s(s, mx, gsxx) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mpi**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

    return sigma
