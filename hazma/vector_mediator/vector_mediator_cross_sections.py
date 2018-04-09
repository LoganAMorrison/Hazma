from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me


def sigma_xx_to_v_to_ff(Q, mf, params):
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
        Class containing the vector mediator parameters.

    Returns
    -------
    cross_section : float
        Cross section for x + x -> v -> f + f.
    """

    pass


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
    # muon_contr = sigma_xx_to_v_to_ff(Q, mmu, params)
    # electron_contr = sigma_xx_to_v_to_ff(Q, me, params)

    # total = muon_contr + electron_contr

    # cross_secs = {'mu mu': muon_contr,
    #              'e e': electron_contr,
    #              'total': total}

    # return cross_secs
    pass


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
    # CSs = cross_sections(Q, params)

    # bfs = {'mu mu': CSs['mu mu'] / CSs['total'],
    #        'e e': CSs['e e'] / CSs['total']}

    # return bfs
    pass
