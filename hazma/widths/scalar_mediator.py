"""Module containing decay widths for scalar mediator.

@author - Logan Morrison and Adam Coogan
@date - December 2017
"""

import numpy as np
from ..parameters import vh, b0, alpha_em
from ..parameters import charged_kaon_mass as mk
from ..parameters import neutral_kaon_mass as mk0
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import strange_quark_mass as msq


def width_s_to_gg(gsFF, ms):
    """
    Returns the partial decay width of the scalar decaying into photon.

    Parameters
    ----------
    gsFF : double
        Coupling of the scalar to gluons.
    ms : double
        Mass of the scalar mediator.
    """
    return (alpha_em**2 * gsFF**2 * (ms**2)**1.5 *
            np.heaviside(ms, 0.0)) / (256. * np.pi**3 * vh**2)


def width_s_to_k0k0(gsff, gsGG, ms, vs):
    """
    Returns the partial decay width of the scalar decaying into neutral kaon.

    Parameters
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
    """
    return (np.sqrt(-4 * mk0**2 + ms**2) *
            (2 * gsGG * (2 * mk0**2 - ms**2) *
             (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
             (9 * vh + 8 * gsGG * vs) +
             b0 * (mdq + msq) * (9 * vh + 4 * gsGG * vs) *
             (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
              (9 * vh + 16 * gsGG * vs)))**2 *
            np.heaviside(-2 * mk0 + ms)) / \
        (16. * ms**2 * np.pi * (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 *
         (9 * vh + 8 * gsGG * vs)**2)


def width_s_to_kk(gsff, gsGG, ms, vs):
    """
    Returns the partial decay width of the scalar decaying into charged kaon.

    Parameters
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
    """
    return (np.sqrt(-4 * mk**2 + ms**2) *
            (729 * b0 * gsff * (msq + muq) * vh**2 +
             32 * gsGG**3 * (2 * mk**2 - ms**2 - 4 * b0 * (msq + muq)) *
             vs**2 + 36 * gsGG**2 * vs *
             (-2 * b0 * (msq + muq) *
              (vh - 8 * gsff * vs) - 2 * mk**2 *
              (3 * vh + 4 * gsff * vs) + ms**2 *
              (3 * vh + 4 * gsff * vs)) + 162 * gsGG * vh *
             (-2 * mk**2 * (vh + gsff * vs) + ms**2 * (vh + gsff * vs) +
              b0 * (msq + muq) * (3 * vh + 10 * gsff * vs)))**2 *
            np.heaviside(-2 * mk + ms)) / \
        (16. * ms**2 * np.pi * (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 *
         (9 * vh + 8 * gsGG * vs)**2)


def width_s_to_pi0pi0(gsff, gsGG, ms, vs):
    """
    Returns the partial decay width of the scalar decaying into neutral pions.

    Parameters
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
    """
    return (np.sqrt(-4 * mpi0**2 + ms**2) *
            (2 * gsGG * (2 * mpi0**2 - ms**2) *
             (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
             (9 * vh + 8 * gsGG * vs) +
             b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
             (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
              (9 * vh + 16 * gsGG * vs)))**2 *
            np.heaviside(-2 * mpi0 + ms)) / \
        (32. * ms**2 * np.pi * (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 *
         (9 * vh + 8 * gsGG * vs)**2)


def width_s_to_pipi(gsff, gsGG, ms, vs):
    """
    Returns the partial decay width of the scalar decaying into charged pion.

    Parameters
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
    """
    return (np.sqrt(-4 * mpi**2 + ms**2) *
            (2 * gsGG * (2 * mpi**2 - ms**2) *
             (-9 * vh - 9 * gsff * vs + 2 * gsGG * vs) *
             (9 * vh + 8 * gsGG * vs) +
             b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
             (54 * gsGG * vh - 32 * gsGG**2 * vs + 9 * gsff *
              (9 * vh + 16 * gsGG * vs)))**2 *
            np.heaviside(-2 * mpi + ms)) / \
        (16. * ms**2 * np.pi * (9 * vh + 9 * gsff * vs - 2 * gsGG * vs)**2 *
         (9 * vh + 4 * gsGG * vs)**2 *
         (9 * vh + 8 * gsGG * vs)**2)


def width_s_to_xx(gsxx, mx, ms):
    """
    Returns the partial decay width of the scalar decaying into two fermions x.

    Parameters
    ----------
    gsxx : double
        Coupling of the initial state fermion to the scalar mediator.
    mx : double
        Mass of the initial state fermion.
    ms : double
        Mass of the scalar mediator.
    """
    return (gsxx**2 * (ms - 2 * mx) * (ms + 2 * mx) *
            np.sqrt(ms**2 - 4 * mx**2) *
            np.heaviside(ms - 2 * mx)) / (8. * ms**2 * np.pi)


def width_s_to_ff(gsff, mf, ms):
    """
    Returns the partial decay width of the scalar decaying into two fermions x.

    Parameters
    ----------
    gsxx : double
        Coupling of the initial state fermion to the scalar mediator.
    mx : double
        Mass of the initial state fermion.
    ms : double
        Mass of the scalar mediator.
    """
    return (gsff**2 * (ms - 2 * mf) * (ms + 2 * mf) *
            np.sqrt(ms**2 - 4 * mf**2) *
            np.heaviside(ms - 2 * mf)) / (8. * ms**2 * np.pi)
