# from ..parameters import muon_mass as mmu
# from ..parameters import electron_mass as me
from cmath import sqrt, pi
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0
from ..parameters import fpi
from ..parameters import qe
from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me


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

    gvxx = params.gvxx
    mx = params.mx
    mv = params.mv

    if Q >= 2. * mf and Q >= 2. * mx:
        return (gvll**2*gvxx**2*sqrt(Q**2 - 4*mf**2)*(Q**2 + 2*mf**2) *
                (Q**2 + 2*mx**2)) / \
            (12.*(Q**3 - Q*mv**2)**2*sqrt(Q**2 - 4*mx**2)*pi)
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
    gvuu = params.gvuu
    gvdd = params.gvdd
    gvxx = params.gvxx
    mx = params.mx
    mv = params.mv

    if Q >= 2. * mpi and Q >= 2. * mx:
        return ((gvdd - gvuu)**2*gvxx**2*(Q**2 - 4*mpi**2)**1.5 *
                (Q**2 + 2*mx**2)) / \
            (48.*(Q**3 - Q*mv**2)**2*sqrt(Q**2 - 4*mx**2)*pi)
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
    gvuu = params.gvuu
    gvdd = params.gvdd
    gvxx = params.gvxx
    mx = params.mx
    mv = params.mv

    if Q >= mpi0 and Q >= 2. * mx:
        return ((gvdd + 2*gvuu)**2*gvxx**2*(Q**2 - mpi0**2)**3 *
                (Q**2 + 2*mx**2)*qe**2) / \
            (13824.*Q**3*fpi**2*(Q**2 - mv**2)**2 *
             sqrt(Q**2 - 4*mx**2)*pi**5)
    else:
        return 0.


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

    total = muon_contr + electron_contr + pipi_contr + pi0g_contr

    cross_secs = {'mu mu': muon_contr,
                  'e e': electron_contr,
                  'pi pi': pipi_contr,
                  'pi0 g': pi0g_contr,
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
                'pi0 g': 0.0}
    else:
        return {'mu mu': CSs['mu mu'] / CSs['total'],
                'e e': CSs['e e'] / CSs['total'],
                'pi pi': CSs['pi pi'] / CSs['total'],
                'pi0 g': CSs['pi0 g'] / CSs['total']}
