from cmath import sqrt

from ..parameters import vh, b0
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq


from ..unitarization.bethe_salpeter import amp_pipi_to_pipi
from ..unitarization.bethe_salpeter import amp_pipi_to_kk
from ..unitarization.bethe_salpeter import amp_kk_to_kk
from ..unitarization.loops import loop_matrix


def __amp_s_PIPI(s, params):
    return (amp_s_pi0pi0(s, params, unit='LO') +
            2 * amp_s_pipi(s, params, unit='LO')) / sqrt(6.)


def __amp_s_KK(s, params):
    return (amp_s_k0k0(s, params, unit='LO') +
            amp_s_kk(s, params, unit='LO')) / sqrt(2)


def amp_s_k0k0(s, params, unit='BSE'):
    """Amplitude for S -> k0 k0bar"""

    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

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

        amp_s_PIPI = __amp_s_PIPI(s, params)
        amp_s_KK = __amp_s_KK(s, params)

        amp_PIPI_k0k0 = amp_pipi_to_kk(sqrt(s)) / sqrt(2)
        amp_KK_k0k0 = amp_kk_to_kk(sqrt(s)) / sqrt(2)

        return amp_s_k0k0 - amp_s_PIPI * loop_mat[0, 0] * amp_PIPI_k0k0 - \
            amp_s_KK * loop_mat[1, 1] * amp_KK_k0k0

    else:
        raise ValueError(
            'Unitariazation method {} is not available. Use LO or' +
            'BSE'.format(unit))


def amp_s_kk(s, params, unit='BSE'):
    """Amplitude for S -> k+ k-"""

    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

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

        amp_s_PIPI = __amp_s_PIPI(s, params)
        amp_s_KK = __amp_s_KK(s, params)

        amp_PIPI_k0k0 = amp_pipi_to_kk(sqrt(s)) / sqrt(2)
        amp_KK_k0k0 = amp_kk_to_kk(sqrt(s)) / sqrt(2)

        return amp_s_kk - amp_s_PIPI * loop_mat[0, 0] * amp_PIPI_k0k0 - \
            amp_s_KK * loop_mat[1, 1] * amp_KK_k0k0

    else:
        raise ValueError(
            'Unitariazation method {} is not available. Use LO or' +
            'BSE'.format(unit))


def amp_s_pi0pi0(s, params, unit='BSE'):
    """Amplitude for S -> pi0 pi0"""

    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

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

        amp_s_PIPI = __amp_s_PIPI(s, params)
        amp_s_KK = __amp_s_KK(s, params)

        amp_PIPI_pi0pi0 = amp_pipi_to_pipi(
            sqrt(s)) / sqrt(3)
        amp_KK_pi0pi0 = amp_kk_to_kk(sqrt(s)) / sqrt(3)

        return amp_s_pi0pi0 - amp_s_PIPI * loop_mat[0, 0] * \
            amp_PIPI_pi0pi0 - amp_s_KK * loop_mat[1, 1] * amp_KK_pi0pi0

    else:
        raise ValueError(
            'Unitariazation method {} is not available. Use LO or' +
            'BSE'.format(unit))


def amp_s_pipi(s, params, unit='BSE'):
    """Amplitude for S -> pi+ pi-"""

    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

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

        amp_s_PIPI = __amp_s_PIPI(s, params)
        amp_s_KK = __amp_s_KK(s, params)

        amp_PIPI_pipi = amp_pipi_to_pipi(sqrt(s)) / sqrt(3)
        amp_KK_pipi = amp_kk_to_kk(sqrt(s)) / sqrt(3)

        return amp_s_pipi - amp_s_PIPI * loop_mat[0, 0] * amp_PIPI_pipi - \
            amp_s_KK * loop_mat[1, 1] * amp_KK_pipi

    else:
        raise ValueError(
            'Unitariazation method {} is not available. Use LO or' +
            'BSE'.format(unit))
