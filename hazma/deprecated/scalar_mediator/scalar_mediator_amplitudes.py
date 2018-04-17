from cmath import sqrt

from ..parameters import vh, b0
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq


def amp_s_k0k0(s, params):
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

    return amp_s_k0k0


def amp_s_kk(s, params, unit='BSE'):
    """Amplitude for S -> k+ k-"""

    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

    return (-2 * gsGG * (-2 * mk**2 + s)) / \
        (9 * vh + 4 * gsGG * vs) - \
        (b0 * (msq + muq) *
         (54 * gsGG * vh - 32 * gsGG**2 * vs +
          9 * gsff * (9 * vh + 16 * gsGG * vs))) / \
        ((9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))


def amp_s_pi0pi0(s, params):
    """Amplitude for S -> pi0 pi0"""

    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

    return (-2 * gsGG * (-2 * mpi0**2 + s)) / \
        (9 * vh + 4 * gsGG * vs) - \
        (b0 * (mdq + muq) *
         (54 * gsGG * vh - 32 * gsGG**2 * vs +
          9 * gsff * (9 * vh + 16 * gsGG * vs))) / \
        ((9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))


def amp_s_pipi(s, params, unit='BSE'):
    """Amplitude for S -> pi+ pi-"""

    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

    return (-2 * gsGG * (-2 * mpi**2 + s)) / \
        (9 * vh + 4 * gsGG * vs) - \
        (b0 * (mdq + muq) *
         (54 * gsGG * vh - 32 * gsGG**2 * vs +
          9 * gsff * (9 * vh + 16 * gsGG * vs))) / \
        ((9 * vh + 9 * gsff * vs - 2 * gsGG * vs) *
         (9 * vh + 8 * gsGG * vs))
