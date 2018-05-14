from cmath import sqrt, pi, log

from ..parameters import vh, b0, alpha_em
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me


def __msqrd_xx_s(s, params):
    """ Returns DM portion of amplitudes XX -> S -> anything """
    gsxx = params.gsxx
    mx = params.mx

    return -(gsxx**2 * (4. * mx**2 - s)) / 2.


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

    if Q <= 2 * mf or Q < 2. * mx:
        return 0.0

    sigma = (gsff**2 * gsxx**2 * mf**2 * (-4. * mf**2 + s)**1.5 *
             sqrt(-4. * mx**2 + s)) / \
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

    if Q < 2. * mx:
        return 0.0

    sigma = (alpha_em**2 * gsFF**2 * gsxx**2 * s**1.5 *
             sqrt(-4. * mx**2 + s)) / \
        (512. * pi**3 * (ms**2 - s)**2 * vh**2)

    return sigma.real


def sigma_xx_to_s_to_pi0pi0(Q, params):
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
    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

    if Q < 2. * mpi0 or Q < 2. * mx:
        return 0.0

    s = Q**2

    amp_s_to_pi0pi0 = (-2. * gsGG * (-2. * mpi0**2 + s)) / \
        (9. * vh + 4. * gsGG * vs) - \
        (b0 * (mdq + muq) *
         (54. * gsGG * vh - 32. * gsGG**2 * vs +
          9. * gsff * (9. * vh + 16. * gsGG * vs))) / \
        ((9. * vh + 9. * gsff * vs - 2. * gsGG * vs) *
         (9. * vh + 8. * gsGG * vs))

    msqrd = abs(amp_s_to_pi0pi0)**2 * __msqrd_xx_s(s, params) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mpi0**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

    return sigma.real


def sigma_xx_to_s_to_pipi(Q, params):
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
    gsGG = params.gsGG
    gsff = params.gsff
    vs = params.vs

    if Q < 2. * mpi or Q < 2. * mx:
        return 0.0

    s = Q**2

    amp_s_to_pipi = (-2. * gsGG * (-2. * mpi**2 + s)) / \
        (9. * vh + 4. * gsGG * vs) - \
        (b0 * (mdq + muq) *
         (54. * gsGG * vh - 32. * gsGG**2 * vs +
          9. * gsff * (9. * vh + 16. * gsGG * vs))) / \
        ((9. * vh + 9. * gsff * vs - 2. * gsGG * vs) *
         (9. * vh + 8. * gsGG * vs))

    msqrd = abs(amp_s_to_pipi)**2 * __msqrd_xx_s(s, params) / (s - ms**2)**2

    pf = sqrt(1.0 - 4. * mpi**2 / s)
    pi = sqrt(1.0 - 4. * mx**2 / s)

    sigma = msqrd / 16. / pi * (pf / pi) / s

    return sigma.real


def sigma_xx_to_ss(Q, params):
    ms = params.ms

    if Q > 2. * ms:
        mx = params.mx
        gsxx = params.gsxx

        return (gsxx**4*sqrt(Q**2 - 4*ms**2) *
                ((4*(ms**2 - 4*mx**2)**2) /
                 (Q**2 - 2*ms**2 + sqrt((Q**2 - 4*ms**2)*(Q**2 - 4*mx**2))) +
                 (4*(ms**2 - 4*mx**2)**2) /
                 (-Q**2 + 2*ms**2 + sqrt((Q**2 - 4*ms**2)*(Q**2 - 4*mx**2))) -
                 2*(Q**2 - 2*ms**2 - 2*mx**2 +
                    sqrt((Q**2 - 4*ms**2)*(Q**2 - 4*mx**2))) -
                 2*(-Q**2 + 2*ms**2 + 2*mx**2 +
                    sqrt((Q**2 - 4*ms**2)*(Q**2 - 4*mx**2))) +
                 ((Q**2*(-Q**2 + 2*ms**2 + 8*mx**2) +
                   2*(Q**4 + 3*ms**4 + 4*Q**2*mx**2 - 16*mx**4 -
                      ms**2*(3*Q**2 + 8*mx**2))) *
                  log((Q**2 - 2*ms**2 +
                       sqrt((Q**2 - 4*ms**2)*(Q**2 - 4*mx**2)))**2 /
                      (-Q**2 + 2*ms**2 +
                       sqrt((Q**2 - 4*ms**2)*(Q**2 - 4*mx**2)))**2)) /
                 (Q**2 - 2*ms**2)))/(64.*Q**2*sqrt(Q**2 - 4*mx**2)*pi)
    else:
        return 0.


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
    muon_contr = sigma_xx_to_s_to_ff(Q, mmu, params)
    electron_contr = sigma_xx_to_s_to_ff(Q, me, params)
    photon_contr = sigma_xx_to_s_to_gg(Q, params)
    NPion_contr = sigma_xx_to_s_to_pi0pi0(Q, params)
    CPion_contr = sigma_xx_to_s_to_pipi(Q, params)

    total = (muon_contr + electron_contr + NPion_contr + CPion_contr +
             photon_contr)

    cross_secs = {'mu mu': muon_contr,
                  'e e': electron_contr,
                  'g g': photon_contr,
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
        'mu mu', 'e e', 'pi0 pi0', 'pi pi'
    """
    CSs = cross_sections(Q, params)

    if CSs['total'] == 0.0:
        return {'mu mu': 0.0,
                'e e': 0.0,
                'g g': 0.0,
                'pi0 pi0': 0.0,
                'pi pi': 0.0}
    else:
        return {'mu mu': CSs['mu mu'] / CSs['total'],
                'e e': CSs['e e'] / CSs['total'],
                'g g': CSs['g g'] / CSs['total'],
                'pi0 pi0': CSs['pi0 pi0'] / CSs['total'],
                'pi pi': CSs['pi pi'] / CSs['total']}
