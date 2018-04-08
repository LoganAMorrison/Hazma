from cmath import sqrt, pi


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
