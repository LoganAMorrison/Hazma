from cmath import sqrt, pi
import numpy as np

from ..parameters import vh, b0, alpha_em, fpi
from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import charged_pion_mass as mpi

from scipy.integrate import quad


def sigma_xx_to_p_to_ff(Q, mf, params):
    """
    Returns the cross section for two identical fermions "x" to two
    identical fermions "f".

    Parameters
    ----------
    Q : float
        Center of mass energy.
    mf : float
        Mass of final state fermions.
    params : object
        Object of the pseudo-scalar parameters class.

    Returns
    -------
    cross_section : float
        Cross section for x + x -> p -> f + f.
    """

    gpff = params.gpff
    gpxx = params.gpxx
    mp = params.mp
    mx = params.mx

    return (gpff**2 * gpxx**2 * Q**2 *
            sqrt(-4 * mf**2 + Q**2)) /\
        (16. * pi * (mp**2 - Q**2)**2 *
         sqrt(-4 * mx**2 + Q**2))


def sigma_xx_to_p_to_gg(Q, params):
    mx = params.mx

    if Q >= 2. * mx:
        gpFF = params.gpFF
        gpxx = params.gpxx
        mp = params.mp
        rx = mx / Q
        widthp = params.widthp

        ret = (alpha_em**2 * gpFF**2 * gpxx**2 * Q**4) / \
            (256. * np.pi**3 * np.sqrt(1 - 4 * rx**2) * vh**2 * ((mp**2 - Q**2)**2 +
                                                                 mp**2 * widthp**2))

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.


def sigma_xx_to_pp(Q, params):
    mx = params.mx
    mp = params.mp

    if Q > 2. * mp and Q >= 2. * mx:
        gpxx = params.gpxx
        rp = mp / Q
        rx = mx / Q

        ret = (gpxx**4 * ((-2 * np.sqrt((-1 + 4 * rp**2) * (-1 + 4 * rx**2)) *
                           (3 * rp**4 + 2 * rx**2 - 8 * rp**2 * rx**2)) /
                          (rp**4 + rx**2 - 4 * rp**2 * rx**2) +
                          (2 * (1 - 4 * rp**2 + 6 * rp**4) *
                           (-1j * np.pi +
                              2 * np.arctanh((-1 + 2 * rp**2) /
                                             np.sqrt((-1 + 4 * rp**2) *
                                                     (-1 + 4 * rx**2))))) /
                          (-1 + 2 * rp**2))) / (64. * Q**2 * np.pi * (1. - 4 * rx**2))

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.


def dsigma_ds_xx_to_p_to_pi0pipi(s, Q, params):
    mx = params.mx

    if Q > 2. * mx and Q >= 2. * mpi + mpi0:
        gpxx = params.gpxx
        gpuu = params.gpuu
        gpdd = params.gpdd
        gpGG = params.gpGG
        mp = params.mp
        widthp = params.widthp

        ret = (b0**2 * gpxx**2 * np.sqrt(s * (-4 * mpi**2 + s)) *
               np.sqrt(mpi0**4 + (Q**2 - s)**2 - 2 * mpi0**2 * (Q**2 + s)) *
               (gpGG * (mdq - muq) + (gpdd - gpuu) * vh)**2) / \
            (4608. * fpi**2 * np.pi**3 * Q * np.sqrt(-4 * mx**2 + Q**2) *
             s * vh**2 *
             (mp**4 + Q**4 + mp**2 * (-2 * Q**2 + widthp**2)))

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.


def sigma_xx_to_p_to_pi0pipi(Q, params):
    """
    Returns the dark matter annihilation cross section into a neutral pion and
    two charged pions through a pseudo-scalar mediator.

    Parameters
    ----------
    Q : float
        Center of mass energy.
    params : PseudoScalarMediator or PseudoScalarMediatorParameters object
        Object containing the parameters of the pseudo-scalar mediator
        model. Can be a PseudoScalarMediator or a
        PseudoScalarMediatorParameters object.

    Returns
    -------
    sigma : float
        The DM annihilation cross section into pi^0, pi^-, pi^+.
    """

    smax = (Q - mpi0)**2
    smin = 4. * mpi**2

    res = quad(dsigma_ds_xx_to_p_to_pi0pipi, smin, smax, args=(Q, params))

    return res[0]


def cross_sections(Q, params):
    """
    Compute the total cross section for two fermions annihilating through a
    pseudo-scalar mediator to mesons and leptons.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    cs : dict
        Total cross section.
    """
    muon_contr = sigma_xx_to_p_to_ff(Q, mmu, params)
    electron_contr = sigma_xx_to_p_to_ff(Q, me, params)

    total = muon_contr + electron_contr

    cross_secs = {'mu mu': muon_contr,
                  'e e': electron_contr,
                  'total': total}

    return cross_secs


def branching_fractions(Q, params):
    """
    Compute the branching fractions for two fermions annihilating through a
    scalar mediator to mesons and leptons.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    bfs : dictionary
        Dictionary of the branching fractions. The keys are 'total',
        'mu mu', 'e e', 'pi0 pi0', 'pi pi', 'k k', 'k0 k0'.
    """
    CSs = cross_sections(Q, params)

    bfs = {'mu mu': CSs['mu mu'] / CSs['total'],
           'e e': CSs['e e'] / CSs['total']}

    return bfs
