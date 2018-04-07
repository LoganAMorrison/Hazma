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
from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me

from .scalar_mediator_amplitudes import amp_s_k0k0, amp_s_kk
from .scalar_mediator_amplitudes import amp_s_pi0pi0, amp_s_pipi


def __msqrd_xx_s(s, params):
    """ Returns DM portion of amplitudes XX -> S -> anything """
    gsxx = params.gsxx
    mx = params.mx

    return -(gsxx**2 * (4 * mx**2 - s)) / 2.


def sigma_xx_to_s_to_etaeta(Q, params):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of eta mesons through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> eta + eta.
    """

    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs
    mx = params.mx
    gsxx = params.gsxx
    ms = params.ms

    s = Q**2

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

    return sigma.real


def sigma_xx_to_s_to_ff(Q, mf, params):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of fermions, *f* through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> f + f.
    """

    gsff = params.gsff
    mx = params.mx
    gsxx = params.gsxx
    ms = params.ms

    s = Q**2

    if -2 * mf + abs(sqrt(s)) <= 0:
        return 0.0

    sigma = (gsff**2 * gsxx**2 * mf**2 * (-4 * mf**2 + s)**1.5 *
             sqrt(-4 * mx**2 + s)) / \
        (16. * pi * (ms**2 - s)**2 * s * vh**2)

    return sigma.real


def sigma_xx_to_s_to_gg(Q, params):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of photons through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> g + g.
    """

    mx = params.mx
    gsFF = params.gsFF
    gsxx = params.gsxx
    ms = params.ms

    s = Q**2

    if -4 * mx**2 + s < 0:
        return 0.0

    sigma = (alpha_em**2 * gsFF**2 * gsxx**2 * s**1.5 *
             sqrt(-4 * mx**2 + s)) / \
        (512. * pi**3 * (ms**2 - s)**2 * vh**2)

    return sigma.real


def sigma_xx_to_s_to_k0k0(Q, params, unit='BSE'):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of neutral kaons through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> k0 + k0.
    """

    ms = params.ms
    mx = params.mx

    if Q < 2 * mk0:
        return 0.0

    s = Q**2

    amp_s_to_k0k0 = amp_s_k0k0(s, params, unit=unit)

    msqrd = abs(amp_s_to_k0k0)**2 * \
        __msqrd_xx_s(s, params) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mk0**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

    return sigma.real


def sigma_xx_to_s_to_kk(Q, params, unit='BSE'):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of charged kaon through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> k+ + k-.
    """
    ms = params.ms
    mx = params.mx

    if Q < 2 * mk:
        return 0.0

    s = Q**2

    amp_s_to_kk = amp_s_kk(s, params, unit=unit)

    msqrd = abs(amp_s_to_kk)**2 * __msqrd_xx_s(s, params) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mk**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

    return sigma.real


def sigma_xx_to_s_to_pi0pi0(Q, params, unit="BSE"):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of neutral pion through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> pi0 + pi0.
    """
    ms = params.ms
    mx = params.mx

    if Q < 2 * mpi0:
        return 0.0

    s = Q**2

    amp_s_to_pi0pi0 = amp_s_pi0pi0(s, params, unit=unit)

    msqrd = abs(amp_s_to_pi0pi0)**2 * __msqrd_xx_s(s, params) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mpi0**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

    return sigma.real


def sigma_xx_to_s_to_pipi(Q, params, unit="BSE"):
    """Returns the spin-averaged, cross section for a pair of fermions,
    *x*, annihilating into a pair of charged pions through a
    scalar mediator in the s-channel.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    sigma : float
        Cross section for x + x -> s* -> pi + pi.
    """
    ms = params.ms
    mx = params.mx

    if Q < 2. * mpi:
        return 0.0

    s = Q**2

    amp_s_to_pipi = amp_s_pipi(s, params, unit=unit)

    msqrd = abs(amp_s_to_pipi)**2 * __msqrd_xx_s(s, params) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mpi**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

    return sigma.real


def cross_sections(Q, params):
    """
    Compute the total cross section for two fermions annihilating through a
    scalar mediator to mesons and leptons.

    Parameters
    ----------
    cme : float
        Center of mass energy.

    Returns
    -------
    cs : float
        Total cross section.
    """
    eta_contr = sigma_xx_to_s_to_etaeta(Q, params)
    muon_contr = sigma_xx_to_s_to_ff(Q, mmu, params)
    electron_contr = sigma_xx_to_s_to_ff(Q, me, params)
    photon_contr = sigma_xx_to_s_to_gg(Q, params)
    NKaon_contr = sigma_xx_to_s_to_k0k0(Q, params)
    CKaon_contr = sigma_xx_to_s_to_kk(Q, params)
    NPion_contr = sigma_xx_to_s_to_pi0pi0(Q, params)
    CPion_contr = sigma_xx_to_s_to_pipi(Q, params)

    total = eta_contr + muon_contr + electron_contr + NKaon_contr + \
        CKaon_contr + NPion_contr + CPion_contr + photon_contr

    cross_secs = {'eta eta': eta_contr,
                  'mu mu': muon_contr,
                  'e e': electron_contr,
                  'g g': photon_contr,
                  'k0 k0': NKaon_contr,
                  'k k': CKaon_contr,
                  'pi0 pi0': NPion_contr,
                  'pi pi': CPion_contr,
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

    bfs = {'eta eta': CSs['eta eta'] / CSs['total'],
           'mu mu': CSs['mu mu'] / CSs['total'],
           'e e': CSs['e e'] / CSs['total'],
           'g g': CSs['g g'] / CSs['total'],
           'k0 k0': CSs['k0 k0'] / CSs['total'],
           'k k': CSs['k k'] / CSs['total'],
           'pi0 pi0': CSs['pi0 pi0'] / CSs['total'],
           'pi pi': CSs['pi pi'] / CSs['total']}

    return bfs
