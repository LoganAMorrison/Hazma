from ..parameters import alpha_em

from cmath import sqrt, log, pi
import numpy as np


def __dnde_xx_to_v_to_ffg(egam, Q, mf, params):
    """Return the fsr spectra for fermions from decay of vector mediator.

    Computes the final state radiaton spectrum value dNdE from a vector
    mediator given a gamma ray energy of `egam`, center of mass energy `Q`
    and final state fermion mass `mf`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell vector mediator.
    mf : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from vector mediator.
    """
    e, m = egam / Q, mf / Q

    s = Q**2 - 2. * Q * egam

    ret_val = 0.0

    if 4. * mf**2 <= s <= Q**2:
        ret_val = -(alpha_em *
                    (4. * sqrt(1. - 2. * e - 4. * m**2) *
                     (1. - 2. * m**2 + 2. * e * (-1 + e + 2. * m**2)) +
                     sqrt(1. - 2. * e) * (1. + 2. * (-1 + e) * e -
                                          4. * e * m**2 - 4. * m**4) *
                     (log(1. - 2. * e) - 4. *
                      log(sqrt(1. - 2. * e) +
                          sqrt(1. - 2. * e - 4. * m**2)) +
                        2. * log((sqrt(1. - 2. * e) -
                                  sqrt(1. - 2. * e - 4. * m**2)) *
                                 (1. - sqrt(1. + (4. * m**2) /
                                            (-1 + 2. * e))))))) / \
            (2. * e * (1. + 2. * m**2) *
             sqrt((-1 + 2. * e) * (-1 + 4. * m**2)) * pi * Q)

        assert ret_val.imag == 0.

        ret_val = ret_val.real

        assert ret_val >= 0

    return ret_val


def dnde_xx_to_v_to_ffg(egam, Q, mf, params):
    """Return the fsr spectra for fermions from decay of vector mediator.

    Computes the final state radiaton spectrum value dNdE from a vector
    mediator given a gamma ray energy of `egam`, center of mass energy `Q`
    and final state fermion mass `mf`.

    Paramaters
    ----------
    egam : float
        Gamma ray energy.
    Q: float
        Center of mass energy of mass of off-shell vector mediator.
    mf : float
        Mass of the final state fermion.

    Returns
    -------
    spec_val : float
        Spectrum value dNdE from vector mediator.
    """
    if hasattr(egam, '__len__'):
        return np.array([__dnde_xx_to_v_to_ffg(e, Q, mf) for e in egam])
    else:
        return __dnde_xx_to_v_to_ffg(egam, Q, mf)
