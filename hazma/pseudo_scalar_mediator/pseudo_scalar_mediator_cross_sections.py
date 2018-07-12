import numpy as np

from ..parameters import vh, b0, alpha_em, fpi
from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import charged_pion_mass as mpi

from scipy.integrate import quad


def sigma_xx_to_p_to_ff(Q, f, params):
    """
    Returns the cross section for two identical fermions "x" to two
    identical fermions "f".

    Parameters
    ----------
    Q : float
        Center of mass energy.
    f : string
        Name of final state fermions.
    params : object
        Object of the pseudo-scalar parameters class.

    Returns
    -------
    cross_section : float
        Cross section for x + x -> p -> f + f.
    """
    if f == "e":
        rf = me / Q
        gpff = params.gpee
    elif f == "mu":
        rf = mmu / Q
        gpff = params.gpmumu

    beta = params.beta
    gpxx = params.gpxx
    mp = params.mp
    width_p = params.width_p
    rx = params.mx / Q

    if 2.*rf < 1 and 2.*rx < 1:
        ret = ((1 - 2*beta**2)*gpff**2*gpxx**2*Q**2*np.sqrt(1 - 4*rf**2)) / \
            (16.*np.pi*np.sqrt(1 - 4*rx**2) *
             ((mp**2 - Q**2)**2 + mp**2*width_p**2))

        assert ret.imag == 0
        assert ret.real >= 0

        return ret.real
    else:
        return 0.


def sigma_xx_to_p_to_gg(Q, params):
    beta = params.beta
    mx = params.mx

    if Q >= 2. * mx:
        gpFF = params.gpFF
        gpxx = params.gpxx
        mp = params.mp
        rx = mx / Q
        width_p = params.width_p

        ret = (alpha_em**2*gpxx**2*Q**4*((1 - 2*beta**2)*gpFF**2 +
                                         2*beta*gpFF*vh + beta**2*vh**2)) / \
            (256.*np.pi**3*np.sqrt(1 - 4*rx**2)*vh**2 *
             ((mp**2 - Q**2)**2 + mp**2*width_p**2))

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.


def sigma_xx_to_pp(Q, params):
    mx = params.mx
    mp = params.mp
    beta = params.beta

    if Q > 2. * mp and Q >= 2. * mx:
        gpxx = params.gpxx
        rp = mp / Q
        rx = mx / Q

        ret = ((-1 + 2*beta**2)*gpxx**4 *
               ((2*np.sqrt((-1 + 4*rp**2)*(-1 + 4*rx**2)) *
                 (3*rp**4 + 2*rx**2 - 8*rp**2*rx**2)) /
                (rp**4 + rx**2 - 4*rp**2*rx**2) +
                (2*(1 - 4*rp**2 + 6*rp**4) *
                 (1j*np.pi +
                  2*np.arctanh((1 - 2*rp**2) /
                               np.sqrt((-1 + 4*rp**2)*(-1 + 4*rx**2))))) /
                (-1 + 2*rp**2))) / (64.*Q**2*np.pi*(1 - 4*rx**2))

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.


def dsigma_ds_xx_to_p_to_pi0pi0pi0(s, Q, params):
    mx = params.mx
    mpi0 = params.mpi0  # use shifted pion mass!

    if Q > 2. * mx and Q >= 3. * mpi0:
        beta = params.beta
        gpxx = params.gpxx
        gpuu = params.gpuu
        gpdd = params.gpdd
        gpGG = params.gpGG
        mp = params.mp
        width_p = params.width_p

        ret = -(b0**2*gpxx**2*np.sqrt(s*(-4*mpi0**2 + s)) *
                np.sqrt(mpi0**4 + (Q**2 - s)**2 - 2*mpi0**2*(Q**2 + s)) *
                (-(beta**2*(mdq + muq)**2*vh**2) +
                 2*beta*fpi*(mdq + muq)*vh*(gpGG*(mdq - muq) +
                                            (gpdd - gpuu)*vh) +
                 (-1 + 11*beta**2)*fpi**2*(gpGG*(mdq - muq) +
                                           (gpdd - gpuu)*vh)**2)) / \
            (512.*fpi**4*np.pi**3*Q*np.sqrt(-4*mx**2 + Q**2)*s*vh**2 *
             (mp**4 + Q**4 + mp**2*(-2*Q**2 + width_p**2)))

        assert ret.imag == 0
        assert ret.real >= 0

        return ret
    else:
        return 0.


def sigma_xx_to_p_to_pi0pi0pi0(Q, params):
    """
    Returns the dark matter annihilation cross section into three neutral pions
    through a pseudo-scalar mediator.

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
        The DM annihilation cross section into 3 pi^0.
    """

    mpi0 = params.mpi0  # use shifted pion mass!
    smax = (Q - mpi0)**2
    smin = 4. * mpi0**2

    res = quad(dsigma_ds_xx_to_p_to_pi0pi0pi0, smin, smax, args=(Q, params))

    return res[0]


def dsigma_ds_xx_to_p_to_pi0pipi(s, Q, params):
    mx = params.mx
    mpi0 = params.mpi0  # use shifted pion mass!

    if Q > 2. * mx and Q >= 2. * mpi + mpi0:
        beta = params.beta
        gpxx = params.gpxx
        gpuu = params.gpuu
        gpdd = params.gpdd
        gpGG = params.gpGG
        mp = params.mp
        width_p = params.width_p

        ret = (gpxx**2*np.sqrt(s*(-4*mpi**2 + s)) *
               np.sqrt(mpi0**4 + (Q**2 - s)**2 - 2*mpi0**2*(Q**2 + s)) *
               (beta**2*(2*mpi**2 + mpi0 - 3*s)**2*vh**2 +
                2*b0*beta*(2*mpi**2 + mpi0 - 3*s)*vh *
                (-(beta*(mdq + muq)*vh) + fpi*(gpGG*(mdq - muq) +
                                               (gpdd - gpuu)*vh)) +
                b0**2*(beta**2*(mdq + muq)**2*vh**2 -
                       2*beta*fpi*(mdq + muq)*vh*(gpGG*(mdq - muq) +
                                                  (gpdd - gpuu)*vh) -
                       (-1 + 5*beta**2)*fpi**2*(gpGG*(mdq - muq) +
                                                (gpdd - gpuu)*vh)**2))) / \
            (4608.*fpi**4*np.pi**3*Q*np.sqrt(-4*mx**2 + Q**2)*s*vh**2 *
             (mp**4 + Q**4 + mp**2*(-2*Q**2 + width_p**2)))

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

    mpi0 = params.mpi0  # use shifted pion mass!
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
    muon_contr = sigma_xx_to_p_to_ff(Q, 'mu', params)
    electron_contr = sigma_xx_to_p_to_ff(Q, 'e', params)
    photon_contr = sigma_xx_to_p_to_gg(Q, params)

    pi0pipi_contr = sigma_xx_to_p_to_pi0pipi(Q, params)
    pi0pi0pi0_contr = sigma_xx_to_p_to_pi0pi0pi0(Q, params)

    pp_contr = sigma_xx_to_pp(Q, params)

    total = (muon_contr + electron_contr + pi0pipi_contr + pi0pi0pi0_contr +
             photon_contr + pp_contr)

    cross_secs = {'mu mu': muon_contr,
                  'e e': electron_contr,
                  "pi0 pi pi": pi0pipi_contr,
                  "pi0 pi0 pi0": pi0pi0pi0_contr,
                  'g g': photon_contr,
                  'p p': pp_contr,
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

    bfs = {'e e': CSs['e e'] / CSs['total'],
           'mu mu': CSs['mu mu'] / CSs['total'],
           'g g': CSs['g g'] / CSs['total'],
           'p p': CSs['p p'] / CSs['total'],
           'pi0 pi pi': CSs['pi0 pi pi'] / CSs['total'],
           'pi0 pi0 pi0': CSs['pi0 pi0 pi0'] / CSs['total']}

    return bfs
