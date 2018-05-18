from cmath import sqrt, pi, log

from ..parameters import vh, b0, alpha_em
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq
from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me


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
    mx = params.mx

    if Q > 2. * mf and Q >= 2. * mx:
        ms = params.ms
        gsff = params.gsff
        gsxx = params.gsxx
        width_s = params.width_s

        ret_val = (gsff**2 * gsxx**2 * mf**2 * (-2 * mx + Q) *
                   (2 * mx + Q) * (-4 * mf**2 + Q**2)**1.5) / \
            (16. * pi * Q**2 * sqrt(-4 * mx**2 + Q**2) * vh**2 *
             (ms**4 - 2 * ms**2 * Q**2 + Q**4 + ms**2 * width_s**2))

        assert ret_val.imag == 0
        assert ret_val.real >= 0

        return ret_val.real
    else:
        return 0.


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

    if Q >= 2. * mx:
        gsFF = params.gsFF
        gsxx = params.gsxx
        ms = params.ms
        width_s = params.width_s

        ret_val = (alpha_em**2 * gsFF**2 * gsxx**2 * Q**3 *
                   (-2 * mx + Q) * (2 * mx + Q)) / \
            (256. * pi**3 * sqrt(-4 * mx**2 + Q**2) * vh**2 *
             (ms**4 - 2 * ms**2 * Q**2 + Q**4 + ms**2 * width_s**2))

        assert ret_val.imag == 0
        assert ret_val.real >= 0

        return ret_val.real
    else:
        return 0.0


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
    mx = params.mx

    if Q > 2. * mpi0 and Q >= 2. * mx:
        gsxx = params.gsxx
        gsff = params.gsff
        gsGG = params.gsGG
        ms = params.ms
        vs = params.vs
        width_s = params.width_s

        ret_val = (gsxx**2 * (-2 * mx + Q) * (2 * mx + Q) * sqrt(-4 * mpi0**2 + Q**2) *
                   (54 * gsGG * (-2 * mpi0**2 + Q**2) * vh *
                    (3 * vh + 3 * gsff * vs + 2 * gsGG * vs) +
                    b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
                    (54 * gsGG * vh - 32 * gsGG**2 * vs +
                     9 * gsff * (9 * vh + 16 * gsGG * vs)))**2) / \
            (23328. * pi * Q**2 * sqrt(-4 * mx**2 + Q**2) * vh**2 *
             (3 * vh + 3 * gsff * vs + 2 * gsGG * vs)**2 * (9 * vh + 4 * gsGG * vs)**2 *
             (ms**4 + Q**4 + ms**2 * (-2 * Q**2 + width_s**2)))

        assert ret_val.imag == 0
        assert ret_val.real >= 0

        return ret_val.real
    else:
        return 0.


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
    mx = params.mx

    if Q > 2. * mpi and Q >= 2. * mx:
        gsxx = params.gsxx
        gsff = params.gsff
        gsGG = params.gsGG
        ms = params.ms
        vs = params.vs
        width_s = params.width_s

        ret_val = (gsxx**2 * (-2 * mx + Q) * (2 * mx + Q) *
                   sqrt(-4 * mpi**2 + Q**2) *
                   (54 * gsGG * (-2 * mpi**2 + Q**2) * vh *
                    (3 * vh + 3 * gsff * vs + 2 * gsGG * vs) +
                    b0 * (mdq + muq) * (9 * vh + 4 * gsGG * vs) *
                    (54 * gsGG * vh - 32 * gsGG**2 * vs +
                     9 * gsff * (9 * vh + 16 * gsGG * vs)))**2) / \
            (23328. * pi * Q**2 * sqrt(-4 * mx**2 + Q**2) * vh**2 *
             (3 * vh + 3 * gsff * vs + 2 * gsGG * vs)**2 *
             (9 * vh + 4 * gsGG * vs)**2 *
             (ms**4 + Q**4 + ms**2 * (-2 * Q**2 + width_s**2)))

        assert ret_val.imag == 0
        assert ret_val.real >= 0

        return ret_val.real
    else:
        return 0.


def sigma_xx_to_ss(Q, params):
    ms = params.ms
    mx = params.mx

    if Q > 2. * ms and Q >= 2. * mx:
        gsxx = params.gsxx

        ret_val = (gsxx**4 * sqrt(Q**2 - 4 * ms**2) *
                   ((4 * (ms**2 - 4 * mx**2)**2) /
                    (Q**2 - 2 * ms**2 +
                     sqrt((Q**2 - 4 * ms**2) *
                          (Q**2 - 4 * mx**2))) +
                    (4 * (ms**2 - 4 * mx**2)**2) /
                    (-Q**2 + 2 * ms**2 +
                     sqrt((Q**2 - 4 * ms**2) * (Q**2 - 4 * mx**2))) -
                    2 * (Q**2 - 2 * ms**2 - 2 * mx**2 +
                         sqrt((Q**2 - 4 * ms**2) * (Q**2 - 4 * mx**2))) -
                    2 * (-Q**2 + 2 * ms**2 + 2 * mx**2 +
                         sqrt((Q**2 - 4 * ms**2) * (Q**2 - 4 * mx**2))) +
                    ((Q**2 * (-Q**2 + 2 * ms**2 + 8 * mx**2) +
                      2 * (Q**4 + 3 * ms**4 + 4 * Q**2 * mx**2 - 16 * mx**4 -
                           ms**2 * (3 * Q**2 + 8 * mx**2))) *
                     log((Q**2 - 2 * ms**2 +
                          sqrt((Q**2 - 4 * ms**2) * (Q**2 - 4 * mx**2)))**2 /
                         (-Q**2 + 2 * ms**2 +
                          sqrt((Q**2 - 4 * ms**2) * (Q**2 - 4 * mx**2)))**2)) /
                    (Q**2 - 2 * ms**2))) / \
            (64. * Q**2 * sqrt(Q**2 - 4 * mx**2) * pi)

        assert ret_val.imag == 0
        assert ret_val.real >= 0

        return ret_val.real
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
    ss_contr = sigma_xx_to_ss(Q, params)

    total = (muon_contr + electron_contr + NPion_contr + CPion_contr +
             photon_contr + ss_contr)

    cross_secs = {'mu mu': muon_contr,
                  'e e': electron_contr,
                  'g g': photon_contr,
                  'pi0 pi0': NPion_contr,
                  'pi pi': CPion_contr,
                  's s': ss_contr,
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
                'pi pi': 0.0,
                's s': 0.0}
    else:
        return {'mu mu': CSs['mu mu'] / CSs['total'],
                'e e': CSs['e e'] / CSs['total'],
                'g g': CSs['g g'] / CSs['total'],
                'pi0 pi0': CSs['pi0 pi0'] / CSs['total'],
                'pi pi': CSs['pi pi'] / CSs['total'],
                's s': CSs['s s'] / CSs['total']}
