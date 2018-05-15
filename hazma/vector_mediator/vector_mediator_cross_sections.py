from cmath import sqrt, pi, log
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import fpi
from ..parameters import qe
from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me
from scipy.integrate import quad


def sigma_xx_to_v_to_ff(Q, f, params):
    """
    Returns the cross section for xbar x to fbar f.

    Parameters
    ----------
    Q : float
        Center of mass energy.
    f : float
        Name of final state fermion: "e" or "mu".
    params : object
        Class containing the vector mediator parameters.

    Returns
    -------
    cross_section : float
        Cross section for xbar + x -> v -> fbar + f.
    """
    if f == "e":
        mf = me
        gvll = params.gvee
    elif f == "mu":
        mf = mmu
        gvll = params.gvmumu

    mx = params.mx

    if Q >= 2. * mf and Q >= 2. * mx:
        gvxx = params.gvxx
        mv = params.mv
        width_v = params.width_v

        return (gvll**2*gvxx**2*sqrt(Q**2 - 4.*mf**2)*(Q**2 + 2.*mf**2) *
                (Q**2 + 2*mx**2)) / \
            (12.*Q**2*sqrt(Q**2 - 4.*mx**2)*pi *
             (Q**4 - 2.*Q**2*mv**2 + mv**4 + mv**2*width_v**2))
    else:
        return 0.


def sigma_xx_to_v_to_pipi(Q, params):
    """
    Returns the cross section for xbar x to pi+ pi-.

    Parameters
    ----------
    Q : float
        Center of mass energy.
    params : object
        Class containing the vector mediator parameters.

    Returns
    -------
    cross_section : float
        Cross section for xbar + x -> v -> f + f.
    """
    mx = params.mx

    if Q >= 2. * mpi and Q >= 2. * mx:
        gvuu = params.gvuu
        gvdd = params.gvdd
        gvxx = params.gvxx
        mv = params.mv
        width_v = params.width_v

        return ((gvdd - gvuu)**2*gvxx**2*(-4.*mpi**2 + Q**2)**1.5 *
                (2.*mx**2 + Q**2)) / \
            (48.*pi*Q**2*sqrt(-4.*mx**2 + Q**2) *
             (mv**4 - 2.*mv**2*Q**2 + Q**4 + mv**2*width_v**2))
    else:
        return 0.


def sigma_xx_to_v_to_pi0g(Q, params):
    """
    Returns the cross section for xbar x to pi0 g.

    Parameters
    ----------
    Q : float
        Center of mass energy.
    params : object
        Class containing the vector mediator parameters.

    Returns
    -------
    cross_section : float
        Cross section for xbar + x -> v -> pi0 g
    """
    mx = params.mx

    if Q >= mpi0 and Q >= 2. * mx:
        gvuu = params.gvuu
        gvdd = params.gvdd
        gvxx = params.gvxx
        mv = params.mv
        width_v = params.width_v

        return ((gvdd + 2.*gvuu)**2.*gvxx**2*(-mpi0**2 + Q**2)**3 *
                (2.*mx**2 + Q**2)*qe**2) / \
            (13824.*fpi**2*pi**5*Q**3*sqrt(-4.*mx**2 + Q**2) *
             (mv**4 - 2.*mv**2*Q**2 + Q**4 + mv**2*width_v**2))
    else:
        return 0.


def __sigma_t_integrated_xx_to_v_to_pi0pipi(s, Q, params):
    mx = params.mx

    if (Q > 2. * mpi + mpi0 and Q > 2. * mx and s > 4.*mpi**2
            and s < (Q - mpi0)**2):
        gvuu = params.gvuu
        gvdd = params.gvdd
        gvxx = params.gvxx
        mv = params.mv
        width_v = params.width_v

        ret_val = ((gvdd + gvuu)**2*gvxx**2*sqrt(s*(-4.*mpi**2 + s)) *
                   sqrt(Q**4 + (mpi0**2 - s)**2 - 2.*Q**2*(mpi0**2 + s)) *
                   (-24.*mpi**6*s + mpi**4 *
                    (-2.*mpi0**4 + 28.*mpi0**2*s + 22.*s**2) +
                    2.*mpi**2*(mpi0**6 - 4.*s**3) +
                    s*(-2.*mpi0**6 - 4.*mpi0**4*s - mpi0**2*s**2 + s**3) +
                    Q**4*(-2.*mpi**4 +
                          2.*mpi**2*(mpi0**2 - s) +
                          s*(-2.*mpi0**2 + s)) +
                    Q**2*(4.*mpi**4*(mpi0**2 + s) +
                          s*(4.*mpi0**4 + 5.*mpi0**2*s - 2.*s**2) -
                          4.*mpi**2*(mpi0**4 + 3.*mpi0**2*s - s**2)))) / \
            (294912.*fpi**6*pi**7*sqrt(Q**2)*sqrt(-4.*mx**2 + Q**2)*s**2 *
             (mv**4 - 2.*mv**2*Q**2 + Q**4 + mv**2*width_v**2))

        assert ret_val.imag == 0.

        return ret_val.real
    else:
        return 0.


def sigma_xx_to_v_to_pi0pipi(Q, params):
    s_min = 4. * mpi**2
    s_max = (Q - mpi0)**2

    return quad(__sigma_t_integrated_xx_to_v_to_pi0pipi, s_min, s_max,
                args=(Q, params))[0]


def sigma_xx_to_vv(Q, params):
    mx = params.mx
    mv = params.mv

    if Q >= 2. * mv and Q >= 2. * mx:
        gvxx = params.gvxx

        return (gvxx**4*sqrt(-4.*mv**2 + Q**2) *
                (-2.*sqrt((-4.*mv**2 + Q**2)*(-4.*mx**2 + Q**2)) -
                 (2.*(mv**2 + 2.*mx**2)**2*sqrt((-4.*mv**2 + Q**2) *
                                                (-4.*mx**2 + Q**2))) /
                 (mv**4 - 4.*mv**2*mx**2 + mx**2*Q**2) +
                 ((4.*mv**4 - 8.*mv**2*mx**2 -
                   8.*mx**4. + 4*mx**2*Q**2 + Q**4) *
                  log((-2.*mv**2 + Q**2 +
                       sqrt((-4.*mv**2 + Q**2)*(-4.*mx**2 + Q**2)))**2 /
                      (2.*mv**2 - Q**2 + sqrt((-4.*mv**2 + Q**2) *
                                              (-4.*mx**2 + Q**2)))**2)) /
                 (-2.*mv**2 + Q**2))) / (16.*pi*Q**2*sqrt(-4.*mx**2 + Q**2))
    else:
        return 0.0


def cross_sections(Q, params):
    """
    Compute the total cross section for two fermions annihilating through a
    vector mediator to mesons and leptons.

    Parameters
    ----------
    cme : float
        Center of mass energy.

    Returns
    -------
    cs : float
        Total cross section.
    """
    muon_contr = sigma_xx_to_v_to_ff(Q, "mu", params)
    electron_contr = sigma_xx_to_v_to_ff(Q, "e", params)
    pipi_contr = sigma_xx_to_v_to_pipi(Q, params)
    pi0g_contr = sigma_xx_to_v_to_pi0g(Q, params)
    pi0pipi_contr = sigma_xx_to_v_to_pi0pipi(Q, params)

    total = (muon_contr + electron_contr + pipi_contr + pi0g_contr +
             pi0pipi_contr)

    cross_secs = {'mu mu': muon_contr,
                  'e e': electron_contr,
                  'pi pi': pipi_contr,
                  'pi0 g': pi0g_contr,
                  'pi0 pi pi': pi0pipi_contr,
                  'total': total}

    return cross_secs


def branching_fractions(Q, params):
    """
    Compute the branching fractions for two fermions annihilating through a
    vector mediator to mesons and leptons.

    Parameters
    ----------
    Q : float
        Center of mass energy.

    Returns
    -------
    bfs : dictionary
        Dictionary of the branching fractions. The keys are 'total',
        'mu mu', 'e e', 'pi0 g', 'pi pi'.
    """
    CSs = cross_sections(Q, params)

    if CSs['total'] == 0.0:
        return {'mu mu': 0.0,
                'e e': 0.0,
                'pi pi': 0.0,
                'pi0 g': 0.0,
                "pi0 pi pi": 0.0}
    else:
        return {'mu mu': CSs['mu mu'] / CSs['total'],
                'e e': CSs['e e'] / CSs['total'],
                'pi pi': CSs['pi pi'] / CSs['total'],
                'pi0 g': CSs['pi0 g'] / CSs['total'],
                "pi0 pi pi": CSs["pi0 pi pi"] / CSs["total"]}
